"""Eğitilmiş YOLO modelini değerlendirme aracı."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    CLASS_ID_TO_NAME_TR,
    DATASET_YAML_PATH,
    DEFECT_CLASSES_TR,
    MODEL_METADATA_PATH,
    MODEL_WEIGHTS_PATH,
    NUM_CLASSES,
    REPORTS_DIR,
    TRAINING_OUTPUT_DIR,
)
from model_registry import load_model_metadata, model_sha256  # noqa: E402


DEFAULT_IMAGE_SIZE = 640
DEFAULT_BATCH_SIZE = 4
DEFAULT_DEVICE = "cpu"


def validate_evaluation_inputs(model_path: Path, dataset_yaml: Path) -> None:
    """Değerlendirme için model ve dataset dosyalarını kontrol eder."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model ağırlığı bulunamadı: {model_path}")
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml bulunamadı: {dataset_yaml}")


def evaluate_model(
    *,
    model_path: Path,
    dataset_yaml: Path,
    split: str,
    image_size: int,
    batch_size: int,
    device: str,
    force_test_repeat: bool = False,
) -> dict[str, Any]:
    """Eğitilmiş YOLO modelini seçilen split üzerinde değerlendirir."""
    from ultralytics import YOLO

    validate_evaluation_inputs(model_path, dataset_yaml)
    validate_test_evaluation(
        model_path=model_path,
        split=split,
        force_repeat=force_test_repeat,
    )
    model = YOLO(str(model_path))
    model_hash = model_sha256(model_path)
    evaluation_name = f"evaluation_{split}_{model_hash[:12]}"

    print("PV Vision AI model değerlendirme")
    print("--------------------------------")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Veri bölümü: {split}")
    print(f"Device: {device}")

    with tempfile.TemporaryDirectory(prefix="pv_vision_evaluation_") as temporary_dir:
        localized_dataset_yaml = create_localized_dataset_yaml(
            dataset_yaml,
            Path(temporary_dir) / "dataset_tr.yaml",
        )
        metrics = model.val(
            data=str(localized_dataset_yaml),
            split=split,
            imgsz=image_size,
            batch=batch_size,
            device=device,
            project=str(TRAINING_OUTPUT_DIR),
            name=evaluation_name,
            exist_ok=True,
            workers=0,
            plots=True,
        )

    box = metrics.box
    print("\nTemel metrikler")
    print(f"Precision: {box.mp:.4f}")
    print(f"Recall: {box.mr:.4f}")
    print(f"mAP50: {box.map50:.4f}")
    print(f"mAP50-95: {box.map:.4f}")
    class_metrics = extract_class_metrics(metrics)
    if class_metrics:
        print("\nSınıf bazlı metrikler")
        for item in class_metrics:
            if not item["evaluated"]:
                print(f"- {item['class_name']}: bu veri bölümünde örnek yok, değerlendirilemedi")
            else:
                print(
                    f"- {item['class_name']}: "
                    f"P={item['precision']:.4f}, R={item['recall']:.4f}, "
                    f"mAP50={item['map50']:.4f}, mAP50-95={item['map50_95']:.4f}"
                )
    json_path, _ = write_evaluation_report(
        split=split,
        model_path=model_path,
        precision=box.mp,
        recall=box.mr,
        map50=box.map50,
        map5095=box.map,
        class_metrics=class_metrics,
        artifacts_dir=Path(metrics.save_dir),
    )
    return json.loads(json_path.read_text(encoding="utf-8"))


def write_evaluation_report(
    *,
    split: str,
    model_path: Path,
    precision: float,
    recall: float,
    map50: float,
    map5095: float,
    class_metrics: list[dict[str, Any]] | None = None,
    artifacts_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Değerlendirme metriklerini rapor klasörüne kaydeder."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "model_path": str(model_path),
        "model_sha256": model_sha256(model_path) if model_path.exists() else None,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "map50": round(float(map50), 6),
        "map50_95": round(float(map5095), 6),
        "class_metrics": class_metrics or [],
        "artifacts_dir": str(artifacts_dir) if artifacts_dir else None,
    }

    json_path = REPORTS_DIR / f"evaluation_{split}_metrics.json"
    txt_path = REPORTS_DIR / f"evaluation_{split}_summary.txt"
    model_hash = str(payload["model_sha256"] or "unknown")
    snapshot_stem = f"evaluation_{split}_{model_hash[:12]}"
    snapshot_json_path = REPORTS_DIR / f"{snapshot_stem}_metrics.json"
    snapshot_txt_path = REPORTS_DIR / f"{snapshot_stem}_summary.txt"
    report_json = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    json_path.write_text(report_json, encoding="utf-8")
    snapshot_json_path.write_text(report_json, encoding="utf-8")
    lines = [
        "PV Vision AI Model Değerlendirme Özeti",
        "--------------------------------------",
        f"Veri bölümü: {split}",
        f"Model: {model_path}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"mAP50: {map50:.4f}",
        f"mAP50-95: {map5095:.4f}",
    ]
    if artifacts_dir:
        lines.append(f"Grafik ve karışıklık matrisi klasörü: {artifacts_dir}")
    if class_metrics:
        lines.extend(["", "Sınıf bazlı metrikler", "----------------------"])
        lines.extend(_format_class_metric_line(item) for item in class_metrics)
    report_text = "\n".join(lines) + "\n"
    txt_path.write_text(report_text, encoding="utf-8")
    snapshot_txt_path.write_text(report_text, encoding="utf-8")
    print(f"\nRapor kaydedildi: {txt_path}")
    return json_path, txt_path


def validate_test_evaluation(
    *,
    model_path: Path,
    split: str,
    force_repeat: bool,
) -> None:
    """Test setini yalnızca tamamlanmış final adayı üzerinde ve varsayılan olarak bir kez açar."""
    if split != "test":
        return

    metadata = load_model_metadata(MODEL_METADATA_PATH)
    if metadata is None:
        raise ValueError("Test değerlendirmesi için model_info.json bulunamadı.")
    if metadata.get("stage") not in {"candidate", "final"}:
        raise ValueError(
            "Test seti yalnızca eğitimi tamamlanmış final adayı veya final model için kullanılabilir."
        )

    current_hash = model_sha256(model_path)
    if metadata.get("sha256") != current_hash:
        raise ValueError(
            "Değerlendirilecek model, models/weights altında kayıtlı final adayıyla eşleşmiyor."
        )

    existing_reports = (
        REPORTS_DIR / "evaluation_test_metrics.json",
        REPORTS_DIR / f"evaluation_test_{current_hash[:12]}_metrics.json",
    )
    if not force_repeat:
        for existing_report in existing_reports:
            if not existing_report.exists():
                continue
            try:
                previous = json.loads(existing_report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if previous.get("model_sha256") == current_hash:
                raise FileExistsError(
                    "Bu model test setinde daha önce değerlendirilmiş. "
                    "Test setini tekrar kullanmak için --force-test-repeat gerekir."
                )


def create_localized_dataset_yaml(source_path: Path, destination_path: Path) -> Path:
    """Evaluation grafiklerinde Türkçe sınıf adları kullanmak için geçici YAML üretir."""
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Geçersiz dataset.yaml içeriği: {source_path}")
    payload["names"] = DEFECT_CLASSES_TR
    payload["nc"] = len(DEFECT_CLASSES_TR)
    destination_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination_path


def extract_class_metrics(metrics: object) -> list[dict[str, Any]]:
    """Ultralytics metrik nesnesinden Türkçe sınıf bazlı sonuçları çıkarır."""
    class_indexes = getattr(metrics, "ap_class_index", [])
    instance_counts = getattr(metrics, "nt_per_class", [])
    result_index_by_class = {
        int(class_id_value): result_index
        for result_index, class_id_value in enumerate(class_indexes)
    }
    results: list[dict[str, Any]] = []
    for class_id in range(NUM_CLASSES):
        instances = int(instance_counts[class_id]) if len(instance_counts) > class_id else 0
        result_index = result_index_by_class.get(class_id)
        if result_index is None:
            precision = recall = map50 = map5095 = None
        else:
            precision, recall, map50, map5095 = metrics.class_result(result_index)
        results.append(
            {
                "class_id": class_id,
                "class_name": CLASS_ID_TO_NAME_TR.get(class_id, f"Bilinmeyen sınıf ({class_id})"),
                "instances": instances,
                "evaluated": result_index is not None and instances > 0,
                "precision": round(float(precision), 6) if precision is not None else None,
                "recall": round(float(recall), 6) if recall is not None else None,
                "map50": round(float(map50), 6) if map50 is not None else None,
                "map50_95": round(float(map5095), 6) if map5095 is not None else None,
            }
        )
    return results


def _format_class_metric_line(item: dict[str, Any]) -> str:
    if not item["evaluated"]:
        return (
            f"- {item['class_name']}: Değerlendirilemedi "
            f"(bu veri bölümünde {item['instances']} örnek)"
        )
    return (
        f"- {item['class_name']}: "
        f"Precision={item['precision']:.4f}, "
        f"Recall={item['recall']:.4f}, "
        f"mAP50={item['map50']:.4f}, "
        f"mAP50-95={item['map50_95']:.4f}, "
        f"Örnek={item['instances']}"
    )


def main() -> None:
    """Model değerlendirme giriş noktası."""
    parser = argparse.ArgumentParser(description="PV Vision AI YOLO modelini değerlendirir.")
    parser.add_argument("--model", type=Path, default=MODEL_WEIGHTS_PATH, help="Değerlendirilecek best.pt yolu")
    parser.add_argument("--data", type=Path, default=DATASET_YAML_PATH, help="YOLO dataset.yaml yolu")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="Değerlendirilecek veri split'i")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE, help="Değerlendirme görüntü boyutu")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Değerlendirme cihazı")
    parser.add_argument(
        "--force-test-repeat",
        action="store_true",
        help="Aynı final modeli test setinde bilinçli olarak yeniden değerlendir",
    )
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        dataset_yaml=args.data,
        split=args.split,
        image_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        force_test_repeat=args.force_test_repeat,
    )


if __name__ == "__main__":
    main()
