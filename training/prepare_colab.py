"""Google Colab'da güvenli checkpoint resume için taşınabilir eğitim paketi üretir."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import IMAGE_EXTENSIONS, PROJECT_ROOT
from training.train import inspect_resume_checkpoint


DEFAULT_RUN_NAME = "pv_vision_yolov8n_mps_v1"
DEFAULT_TARGET_EPOCHS = 10
DEFAULT_COLAB_ROOT = Path("/content/pv_vision_ai")
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs/colab/pv_vision_colab.tar.gz"


@dataclass(frozen=True)
class ColabBundleSummary:
    """Üretilen paketin doğrulanabilir özetini tutar."""

    archive_path: str
    archive_size_bytes: int
    completed_epochs: int
    target_epochs: int
    train_images: int
    val_images: int
    train_labels: int
    val_labels: int
    checkpoint_sha256: str


def localize_dataset_yaml(
    dataset_path: Path,
    *,
    data_root: Path = DEFAULT_COLAB_ROOT / "data/processed",
) -> dict[str, object]:
    """dataset.yaml dosyasını Colab yoluna taşır ve test splitini eğitimden çıkarır."""
    payload = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset.yaml nesne biçiminde değil: {dataset_path}")
    payload["path"] = str(data_root)
    payload.pop("test", None)
    dataset_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return payload


def create_colab_bundle(
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path | None = None,
    target_epochs: int = DEFAULT_TARGET_EPOCHS,
    run_name: str = DEFAULT_RUN_NAME,
    project_root: Path = PROJECT_ROOT,
    force: bool = False,
) -> ColabBundleSummary:
    """Train/val verisini gerçek dosyalarla içeren, testten arındırılmış arşiv üretir."""
    project_root = project_root.resolve()
    run_dir = project_root / "outputs/training" / run_name
    checkpoint_path = (checkpoint_path or run_dir / "weights/last.pt").resolve()
    completed_epochs, _, _ = inspect_resume_checkpoint(checkpoint_path, target_epochs)

    train_images = _split_files(project_root, "images", "train", IMAGE_EXTENSIONS)
    val_images = _split_files(project_root, "images", "val", IMAGE_EXTENSIONS)
    train_labels = _split_files(project_root, "labels", "train", {".txt"})
    val_labels = _split_files(project_root, "labels", "val", {".txt"})
    _validate_split_counts(train_images, train_labels, "train")
    _validate_split_counts(val_images, val_labels, "val")

    required_files = [
        project_root / "config.py",
        project_root / "model_registry.py",
        project_root / "requirements.txt",
        project_root / "README.md",
        project_root / "data/processed/dataset.yaml",
        run_dir / "args.yaml",
        run_dir / "results.csv",
        run_dir / "weights/best.pt",
        checkpoint_path,
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Colab paketi için gerekli dosyalar eksik:\n{formatted}")

    output_path = output_path.expanduser().resolve()
    if output_path.exists() and not force:
        raise FileExistsError(f"Paket zaten var; yenilemek için --force kullan: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = _sha256(checkpoint_path)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name,
        "completed_epochs": completed_epochs,
        "target_epochs": target_epochs,
        "checkpoint_sha256": checkpoint_hash,
        "splits": {
            "train": {"images": len(train_images), "labels": len(train_labels)},
            "val": {"images": len(val_images), "labels": len(val_labels)},
        },
        "test_included": False,
    }

    localized_dataset = yaml.safe_load(
        (project_root / "data/processed/dataset.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(localized_dataset, dict):
        raise ValueError("dataset.yaml nesne biçiminde değil.")
    localized_dataset["path"] = str(DEFAULT_COLAB_ROOT / "data/processed")
    localized_dataset.pop("test", None)

    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=".pv-vision-colab-",
        suffix=".tar.gz",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        with tarfile.open(temporary_path, mode="w:gz", compresslevel=1, dereference=True) as archive:
            for path in _project_code_files(project_root):
                _add_file(archive, path, project_root)
            for path in train_images + val_images + train_labels + val_labels:
                _add_file(archive, path, project_root)
            for path in (run_dir / "args.yaml", run_dir / "results.csv", run_dir / "weights/best.pt"):
                _add_file(archive, path, project_root)
            _add_file(
                archive,
                checkpoint_path,
                project_root,
                arcname=f"outputs/training/{run_name}/weights/last.pt",
            )
            _add_bytes(
                archive,
                "data/processed/dataset.yaml",
                yaml.safe_dump(localized_dataset, sort_keys=False, allow_unicode=True).encode("utf-8"),
            )
            _add_bytes(
                archive,
                "colab_bundle_manifest.json",
                (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    summary = ColabBundleSummary(
        archive_path=str(output_path),
        archive_size_bytes=output_path.stat().st_size,
        completed_epochs=completed_epochs,
        target_epochs=target_epochs,
        train_images=len(train_images),
        val_images=len(val_images),
        train_labels=len(train_labels),
        val_labels=len(val_labels),
        checkpoint_sha256=checkpoint_hash,
    )
    verify_colab_bundle(output_path, expected=summary)
    return summary


def verify_colab_bundle(path: Path, *, expected: ColabBundleSummary | None = None) -> dict[str, object]:
    """Arşivde test/raw veri veya symlink bulunmadığını ve temel sayıları doğrular."""
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        links = [member.name for member in members if member.issym() or member.islnk()]
        forbidden = [
            name
            for name in names
            if name.startswith("data/raw/")
            or name.startswith("data/processed/images/test/")
            or name.startswith("data/processed/labels/test/")
        ]
        required = {
            "config.py",
            "model_registry.py",
            "training/train.py",
            "data/processed/dataset.yaml",
            "colab_bundle_manifest.json",
        }
        manifest_file = archive.extractfile("colab_bundle_manifest.json")
        if manifest_file is None:
            raise ValueError("Colab manifest dosyası okunamadı.")
        manifest = json.load(manifest_file)
        required.add(f"outputs/training/{manifest['run_name']}/weights/last.pt")
        missing = sorted(required - names)
        if links or forbidden or missing:
            raise ValueError(
                f"Geçersiz Colab paketi: links={links[:3]}, forbidden={forbidden[:3]}, missing={missing}"
            )

    if manifest.get("test_included") is not False:
        raise ValueError("Colab manifesti test verisini dışlamıyor.")
    if expected is not None:
        splits = manifest["splits"]
        if splits["train"] != {"images": expected.train_images, "labels": expected.train_labels}:
            raise ValueError("Train split sayıları manifest ile eşleşmiyor.")
        if splits["val"] != {"images": expected.val_images, "labels": expected.val_labels}:
            raise ValueError("Validation split sayıları manifest ile eşleşmiyor.")
    return manifest


def _project_code_files(project_root: Path) -> list[Path]:
    files = [
        project_root / "config.py",
        project_root / "model_registry.py",
        project_root / "requirements.txt",
        project_root / "README.md",
    ]
    files.extend(sorted((project_root / "training").glob("*.py")))
    notebook = project_root / "notebooks/PV_Vision_AI_Colab_Resume.ipynb"
    if notebook.exists():
        files.append(notebook)
    return files


def _split_files(project_root: Path, group: str, split: str, suffixes: set[str]) -> list[Path]:
    directory = project_root / "data/processed" / group / split
    if not directory.exists():
        raise FileNotFoundError(f"Split klasörü bulunamadı: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _validate_split_counts(images: list[Path], labels: list[Path], split: str) -> None:
    if not images or len(images) != len(labels):
        raise ValueError(
            f"{split} görüntü/label sayısı geçersiz: {len(images)} görüntü, {len(labels)} label"
        )
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if image_stems != label_stems:
        raise ValueError(f"{split} görüntü ve label dosya adları eşleşmiyor.")


def _add_file(
    archive: tarfile.TarFile,
    path: Path,
    project_root: Path,
    *,
    arcname: str | None = None,
) -> None:
    source = path.resolve(strict=True)
    destination = arcname or str(path.relative_to(project_root))
    archive.add(source, arcname=destination, recursive=False)


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = int(datetime.now(timezone.utc).timestamp())
    archive.addfile(info, io.BytesIO(payload))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="PV Vision AI için Google Colab eğitim paketi üretir.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Üretilecek .tar.gz yolu")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Paketlenecek last.pt yolu")
    parser.add_argument("--target-epochs", type=int, default=DEFAULT_TARGET_EPOCHS, help="Toplam hedef epoch")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Mevcut eğitim run adı")
    parser.add_argument("--force", action="store_true", help="Var olan paketin üzerine yaz")
    args = parser.parse_args()

    summary = create_colab_bundle(
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        target_epochs=args.target_epochs,
        run_name=args.run_name,
        force=args.force,
    )
    size_mb = summary.archive_size_bytes / (1024 * 1024)
    print("PV Vision AI Colab paketi hazır")
    print("--------------------------------")
    print(f"Paket: {summary.archive_path}")
    print(f"Boyut: {size_mb:.1f} MB")
    print(f"Checkpoint: {summary.completed_epochs}/{summary.target_epochs} epoch")
    print(f"Train: {summary.train_images} görüntü / {summary.train_labels} label")
    print(f"Validation: {summary.val_images} görüntü / {summary.val_labels} label")
    print("Test seti: dahil edilmedi")


if __name__ == "__main__":
    main()
