"""Ultralytics YOLO model eğitimini başlatma aracı."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DATASET_YAML_PATH,
    MODEL_METADATA_PATH,
    MODEL_WEIGHTS_PATH,
    PROCESSED_DATA_DIR,
    TRAINING_OUTPUT_DIR,
)
from model_registry import load_model_metadata, model_sha256, publish_run_best  # noqa: E402


DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 5
DEFAULT_IMAGE_SIZE = 640
DEFAULT_BATCH_SIZE = 4
DEFAULT_DEVICE = "cpu"
DEFAULT_RUN_NAME = "pv_vision_yolov8n_v1"
DEFAULT_FRACTION = 1.0


@dataclass(frozen=True)
class TrainingPreset:
    """Sık kullanılan eğitim ayarlarını tutar."""

    model: str
    epochs: int
    image_size: int
    batch_size: int
    device: str
    run_name: str
    fraction: float
    patience: int
    workers: int
    cache: bool


TRAINING_PRESETS = {
    "smoke": TrainingPreset(
        model="yolov8n.pt",
        epochs=1,
        image_size=320,
        batch_size=4,
        device="cpu",
        run_name="pv_vision_yolov8n_v1_smoke",
        fraction=0.05,
        patience=5,
        workers=0,
        cache=False,
    ),
    "cpu-v1": TrainingPreset(
        model="yolov8n.pt",
        epochs=5,
        image_size=640,
        batch_size=4,
        device="cpu",
        run_name="pv_vision_yolov8n_v1_cpu",
        fraction=1.0,
        patience=5,
        workers=0,
        cache=False,
    ),
    "mps-v1": TrainingPreset(
        model="yolov8n.pt",
        epochs=10,
        image_size=640,
        batch_size=8,
        device="mps",
        run_name="pv_vision_yolov8n_mps_v1",
        fraction=1.0,
        patience=12,
        workers=2,
        cache=False,
    ),
    "quality": TrainingPreset(
        model=str(MODEL_WEIGHTS_PATH),
        epochs=60,
        image_size=640,
        batch_size=8,
        device="mps",
        run_name="pv_vision_yolov8n_quality",
        fraction=1.0,
        patience=15,
        workers=2,
        cache=False,
    ),
}


def validate_quality_start_model(
    model_path: Path = MODEL_WEIGHTS_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> None:
    """Kalite fine-tuning'inin yalnızca tamamlanmış ve kayıtlı aday modelden başlamasını sağlar."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Kalite eğitimi için tamamlanmış V1 best.pt bulunamadı: {model_path}"
        )
    metadata = load_model_metadata(metadata_path)
    if metadata is None or metadata.get("stage") not in {"candidate", "final"}:
        raise ValueError(
            "Kalite eğitimi yalnızca V1 eğitimi tamamlanıp model candidate/final "
            "aşamasına geldikten sonra başlatılabilir."
        )
    if metadata.get("sha256") != model_sha256(model_path):
        raise ValueError("Kalite başlangıç modeli model_info.json kaydıyla eşleşmiyor.")


def validate_training_inputs(dataset_yaml: Path) -> None:
    """Eğitim başlamadan önce YOLO veri seti dosyalarını kontrol eder."""
    required_paths = [
        dataset_yaml,
        PROCESSED_DATA_DIR / "images" / "train",
        PROCESSED_DATA_DIR / "images" / "val",
        PROCESSED_DATA_DIR / "labels" / "train",
        PROCESSED_DATA_DIR / "labels" / "val",
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        formatted = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(f"Eğitim için gerekli dosya/klasör eksik:\n{formatted}")

    for split_name in ("train", "val"):
        image_count = len(list((PROCESSED_DATA_DIR / "images" / split_name).glob("*.jpg")))
        label_count = len(list((PROCESSED_DATA_DIR / "labels" / split_name).glob("*.txt")))
        if image_count == 0 or label_count == 0:
            raise ValueError(f"{split_name} spliti boş görünüyor.")
        if image_count != label_count:
            raise ValueError(
                f"{split_name} splitinde görüntü/label sayısı eşleşmiyor: "
                f"{image_count} görüntü, {label_count} label"
            )


def publish_interim_weights(run_dir: Path, *, target_epochs: int) -> Path | None:
    """Kontrollü durdurmada mevcut en iyi ağırlığı ara model olarak yayımlar."""
    source = run_dir / "weights" / "best.pt"
    if not source.exists():
        print("Henüz yayımlanabilecek best.pt oluşmadı.")
        return None

    publish_run_best(
        run_dir,
        stage="interim",
        destination=MODEL_WEIGHTS_PATH,
        target_epochs=target_epochs,
    )
    print(f"En iyi ara model korundu: {MODEL_WEIGHTS_PATH}")
    return MODEL_WEIGHTS_PATH


def inspect_resume_checkpoint(checkpoint_path: Path, target_epochs: int) -> tuple[int, int, bool]:
    """Resume checkpoint'ini okur ve hedef epoch'un güvenli olduğunu doğrular."""
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint bulunamadı: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    completed_epochs = int(checkpoint.get("epoch", -1)) + 1
    train_args = checkpoint.get("train_args", {})
    original_epochs = int(train_args.get("epochs", 0)) if isinstance(train_args, dict) else 0
    has_optimizer = checkpoint.get("optimizer") is not None

    if not has_optimizer:
        raise ValueError(f"Checkpoint optimizer bilgisi içermiyor; gerçek resume güvenli değil: {checkpoint_path}")
    if completed_epochs <= 0:
        raise ValueError(f"Checkpoint epoch bilgisi okunamadı: {checkpoint_path}")
    if target_epochs <= completed_epochs:
        raise ValueError(
            f"Hedef epoch mevcut tamamlanan epoch'tan büyük olmalı. "
            f"Tamamlanan: {completed_epochs}, hedef: {target_epochs}"
        )

    return completed_epochs, original_epochs, has_optimizer


def create_resume_checkpoint_copy(
    checkpoint_path: Path,
    target_epochs: int,
    *,
    train_arg_overrides: dict[str, object] | None = None,
) -> Path:
    """Orijinal checkpoint'i bozmadan taşınabilir geçici resume kopyası üretir."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = dict(checkpoint.get("train_args", {}))
    train_args["epochs"] = target_epochs
    if train_arg_overrides:
        train_args.update(train_arg_overrides)
    checkpoint["train_args"] = train_args

    temp_dir = Path(tempfile.mkdtemp(prefix="pv_vision_resume_"))
    temp_checkpoint_path = temp_dir / checkpoint_path.name
    if train_arg_overrides:
        checkpoint["train_args"]["model"] = str(temp_checkpoint_path)
        checkpoint["train_args"]["resume"] = str(temp_checkpoint_path)
    torch.save(checkpoint, temp_checkpoint_path)
    return temp_checkpoint_path


def resume_training(
    *,
    checkpoint_path: Path,
    target_epochs: int,
    device: str | None,
    dataset_yaml: Path | None = None,
    project_dir: Path | None = None,
    run_name: str | None = None,
    workers: int | None = None,
) -> Path:
    """Ultralytics checkpoint'inden optimizer/scheduler durumunu koruyarak devam eder."""
    from ultralytics import YOLO

    resume_dataset = dataset_yaml or DATASET_YAML_PATH
    validate_training_inputs(resume_dataset)
    completed_epochs, original_epochs, has_optimizer = inspect_resume_checkpoint(checkpoint_path, target_epochs)
    run_dir = checkpoint_path.resolve().parent.parent
    train_arg_overrides: dict[str, object] = {"data": str(resume_dataset.resolve())}
    if device:
        train_arg_overrides["device"] = device
    if workers is not None:
        train_arg_overrides["workers"] = workers
    if project_dir is not None:
        resolved_project = project_dir.resolve()
        resolved_run_name = run_name or run_dir.name
        train_arg_overrides.update(
            {
                "project": str(resolved_project),
                "name": resolved_run_name,
                "save_dir": str(resolved_project / resolved_run_name),
            }
        )
        run_dir = resolved_project / resolved_run_name
    elif run_name is not None:
        raise ValueError("--resume-name yalnızca --resume-project ile birlikte kullanılabilir.")

    temp_checkpoint_path = create_resume_checkpoint_copy(
        checkpoint_path,
        target_epochs,
        train_arg_overrides=train_arg_overrides,
    )

    print("PV Vision AI YOLO resume başlıyor")
    print("---------------------------------")
    print(f"Orijinal checkpoint: {checkpoint_path}")
    print(f"Geçici resume checkpoint: {temp_checkpoint_path}")
    print(f"Tamamlanan epoch: {completed_epochs}")
    print(f"Checkpoint eski hedef epoch: {original_epochs}")
    print(f"Yeni toplam hedef epoch: {target_epochs}")
    print(f"Optimizer durumu var: {has_optimizer}")
    print(f"Device: {device or 'checkpoint ayarı'}")
    print(f"Dataset: {resume_dataset}")
    print(f"Çıktı klasörü: {run_dir}")
    print("Eğitim sıfırdan başlamayacak; mevcut optimizer, EMA ve epoch durumu korunacak.")

    try:
        model = YOLO(str(temp_checkpoint_path))
        train_kwargs = {"resume": True, "epochs": target_epochs}
        if device:
            train_kwargs["device"] = device

        results = model.train(**train_kwargs)
        run_dir = Path(results.save_dir)
    except KeyboardInterrupt as exc:
        print("\nEğitim kullanıcı tarafından kontrollü biçimde durduruldu.")
        publish_interim_weights(run_dir, target_epochs=target_epochs)
        print(f"Kaldığın yer: {run_dir / 'weights' / 'last.pt'}")
        raise SystemExit(130) from exc
    finally:
        shutil.rmtree(temp_checkpoint_path.parent, ignore_errors=True)

    publish_run_best(
        run_dir,
        stage="candidate",
        destination=MODEL_WEIGHTS_PATH,
        target_epochs=target_epochs,
    )
    copied_path = MODEL_WEIGHTS_PATH
    print("\nResume eğitimi tamamlandı.")
    print(f"Run klasörü: {run_dir}")
    print(f"Merkezi model ağırlığı: {copied_path}")
    return copied_path


def train_model(
    *,
    model_name: str,
    dataset_yaml: Path,
    epochs: int,
    image_size: int,
    batch_size: int,
    device: str,
    run_name: str,
    fraction: float,
    patience: int,
    workers: int,
    cache: bool,
) -> Path:
    """YOLO eğitimini çalıştırır ve merkezi best.pt dosyasını üretir."""
    from ultralytics import YOLO

    validate_training_inputs(dataset_yaml)
    TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("PV Vision AI YOLO eğitimi başlıyor")
    print("----------------------------------")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Epoch: {epochs}")
    print(f"Image size: {image_size}")
    print(f"Batch: {batch_size}")
    print(f"Device: {device}")
    print(f"Veri oranı: {fraction}")
    print(f"Patience: {patience}")
    print(f"Workers: {workers}")
    print(f"Cache: {cache}")
    print(f"Çıktı klasörü: {TRAINING_OUTPUT_DIR / run_name}")

    model = YOLO(model_name)
    run_dir = TRAINING_OUTPUT_DIR / run_name
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            device=device,
            project=str(TRAINING_OUTPUT_DIR),
            name=run_name,
            exist_ok=True,
            workers=workers,
            fraction=fraction,
            patience=patience,
            cache=cache,
        )
        run_dir = Path(results.save_dir)
    except KeyboardInterrupt as exc:
        print("\nEğitim kullanıcı tarafından kontrollü biçimde durduruldu.")
        publish_interim_weights(run_dir, target_epochs=epochs)
        print(f"Kaldığın yer: {run_dir / 'weights' / 'last.pt'}")
        raise SystemExit(130) from exc

    stage = "smoke" if fraction < 1.0 else "candidate"
    publish_run_best(
        run_dir,
        stage=stage,
        destination=MODEL_WEIGHTS_PATH,
        target_epochs=epochs,
    )
    copied_path = MODEL_WEIGHTS_PATH
    print("\nEğitim tamamlandı.")
    print(f"Run klasörü: {run_dir}")
    print(f"Merkezi model ağırlığı: {copied_path}")
    return copied_path


def main() -> None:
    """Model eğitimi giriş noktası."""
    parser = argparse.ArgumentParser(description="PV Vision AI için YOLOv8 eğitimini başlatır.")
    parser.add_argument(
        "--preset",
        choices=tuple(TRAINING_PRESETS),
        default=None,
        help="Hazır eğitim ayarı: smoke, cpu-v1, mps-v1 veya quality",
    )
    parser.add_argument("--model", default=None, help="Başlangıç YOLO modeli")
    parser.add_argument("--data", type=Path, default=DATASET_YAML_PATH, help="YOLO dataset.yaml yolu")
    parser.add_argument("--epochs", type=int, default=None, help="Eğitim epoch sayısı")
    parser.add_argument("--imgsz", type=int, default=None, help="Eğitim görüntü boyutu")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--device", default=None, help="Eğitim cihazı: cpu, cuda veya mps")
    parser.add_argument("--name", default=None, help="Ultralytics run adı")
    parser.add_argument("--fraction", type=float, default=None, help="Eğitimde kullanılacak veri oranı")
    parser.add_argument("--patience", type=int, default=None, help="Erken durdurma patience değeri")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader worker sayısı")
    parser.add_argument("--cache", action="store_true", help="Görüntüleri eğitim öncesi cache'e alır")
    parser.add_argument("--resume-from", type=Path, default=None, help="Devam edilecek last.pt checkpoint yolu")
    parser.add_argument("--resume-epochs", type=int, default=None, help="Resume için toplam hedef epoch sayısı")
    parser.add_argument("--resume-data", type=Path, default=None, help="Resume sırasında kullanılacak taşınabilir dataset.yaml yolu")
    parser.add_argument("--resume-project", type=Path, default=None, help="Resume çıktılarının yazılacağı proje klasörü")
    parser.add_argument("--resume-name", default=None, help="Resume çıktısının mevcut run adı")
    parser.add_argument("--resume-workers", type=int, default=None, help="Resume dataloader worker sayısı")
    args = parser.parse_args()

    if args.resume_from:
        if args.resume_epochs is None:
            raise ValueError("--resume-from kullanıldığında --resume-epochs verilmelidir.")
        resume_training(
            checkpoint_path=args.resume_from,
            target_epochs=args.resume_epochs,
            device=args.device,
            dataset_yaml=args.resume_data,
            project_dir=args.resume_project,
            run_name=args.resume_name,
            workers=args.resume_workers,
        )
        return

    preset = TRAINING_PRESETS.get(args.preset) if args.preset else None
    model_name = args.model or (preset.model if preset else DEFAULT_MODEL)
    if args.preset == "quality" and args.model is None:
        validate_quality_start_model(Path(model_name))
    epochs = args.epochs or (preset.epochs if preset else DEFAULT_EPOCHS)
    image_size = args.imgsz or (preset.image_size if preset else DEFAULT_IMAGE_SIZE)
    batch_size = args.batch or (preset.batch_size if preset else DEFAULT_BATCH_SIZE)
    device = args.device or (preset.device if preset else DEFAULT_DEVICE)
    run_name = args.name or (preset.run_name if preset else DEFAULT_RUN_NAME)
    fraction = args.fraction if args.fraction is not None else (preset.fraction if preset else DEFAULT_FRACTION)
    patience = args.patience if args.patience is not None else (preset.patience if preset else 10)
    workers = args.workers if args.workers is not None else (preset.workers if preset else 0)
    cache = args.cache or (preset.cache if preset else False)

    train_model(
        model_name=model_name,
        dataset_yaml=args.data,
        epochs=epochs,
        image_size=image_size,
        batch_size=batch_size,
        device=device,
        run_name=run_name,
        fraction=fraction,
        patience=patience,
        workers=workers,
        cache=cache,
    )


if __name__ == "__main__":
    main()
