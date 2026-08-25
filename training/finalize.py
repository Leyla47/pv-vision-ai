"""Tamamlanan eğitimi değerlendirir ve kalite eşiğini geçen modeli final olarak yayımlar."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DATASET_YAML_PATH,
    MODEL_WEIGHTS_PATH,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from model_registry import model_sha256, publish_run_best  # noqa: E402
from training.analyze_dataset import (  # noqa: E402
    ClassDistribution,
    analyze_dataset,
    build_dataset_warnings,
    write_dataset_reports,
)
from training.evaluate import evaluate_model  # noqa: E402


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "pv_vision_yolov8n_mps_v1"
    / "weights"
    / "last.pt"
)
AGGREGATE_QUALITY_THRESHOLDS = {
    "precision": 0.40,
    "recall": 0.30,
    "map50": 0.35,
    "map50_95": 0.20,
}
SUPPORTED_CLASS_MIN_TRAIN_OBJECTS = 100
SUPPORTED_CLASS_MIN_MAP50 = 0.10
TEST_EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_test_metrics.json"


@dataclass(frozen=True)
class TrainingCompletion:
    """Finalizasyon için doğrulanmış eğitim çalışma bilgisi."""

    checkpoint_path: Path
    run_dir: Path
    best_path: Path
    completed_epochs: int
    target_epochs: int
    result_epochs: int
    checkpoint_stripped: bool
    stopped_early: bool


def validate_completed_training(
    checkpoint_path: Path,
    target_epochs: int,
    *,
    allow_early_stop: bool = False,
) -> TrainingCompletion:
    """Checkpoint ve results.csv dosyasının hedef epoch'u tamamladığını kanıtlar."""
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {checkpoint_path}")
    if target_epochs <= 0:
        raise ValueError("Hedef epoch pozitif olmalıdır.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    checkpoint_completed_epochs = checkpoint_epoch + 1
    checkpoint_stripped = checkpoint_epoch < 0 and checkpoint.get("optimizer") is None
    if not checkpoint_stripped and checkpoint_completed_epochs < target_epochs:
        raise ValueError(
            f"Eğitim henüz tamamlanmadı: "
            f"{checkpoint_completed_epochs}/{target_epochs} epoch. "
            "Finalizasyon çalıştırılmadı."
        )

    run_dir = checkpoint_path.resolve().parent.parent
    best_path = run_dir / "weights" / "best.pt"
    results_path = run_dir / "results.csv"
    if not best_path.exists():
        raise FileNotFoundError(f"En iyi model bulunamadı: {best_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Eğitim sonuçları bulunamadı: {results_path}")

    with results_path.open(encoding="utf-8", newline="") as results_file:
        rows = list(csv.DictReader(results_file))
    result_epochs = max(
        (int(float(row["epoch"])) for row in rows if row.get("epoch")),
        default=0,
    )
    if result_epochs <= 0:
        raise ValueError(f"results.csv tamamlanmış bir epoch içermiyor: {results_path}")
    if checkpoint_stripped:
        completed_epochs = result_epochs
        stopped_early = result_epochs < target_epochs
        if stopped_early and not allow_early_stop:
            raise ValueError(
                f"Eğitim hedef epoch'a ulaşmadan normal olarak sonlanmış: "
                f"{result_epochs}/{target_epochs}. Erken durdurmayı kabul etmek için "
                "--allow-early-stop kullanılmalıdır."
            )
    else:
        completed_epochs = checkpoint_completed_epochs
        stopped_early = False
        if result_epochs != completed_epochs:
            raise ValueError(
                "Checkpoint ve results.csv epoch bilgileri uyuşmuyor: "
                f"checkpoint={completed_epochs}, results.csv={result_epochs}."
            )

    if result_epochs < target_epochs and not stopped_early:
        raise ValueError(
            f"results.csv hedef epoch'u doğrulamıyor: {result_epochs}/{target_epochs}."
        )
    if not DATASET_YAML_PATH.exists():
        raise FileNotFoundError(f"dataset.yaml bulunamadı: {DATASET_YAML_PATH}")

    return TrainingCompletion(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        best_path=best_path,
        completed_epochs=completed_epochs,
        target_epochs=target_epochs,
        result_epochs=result_epochs,
        checkpoint_stripped=checkpoint_stripped,
        stopped_early=stopped_early,
    )


def assess_model_quality(
    evaluation: dict[str, Any],
    distributions: list[ClassDistribution],
) -> dict[str, Any]:
    """Aggregate ve yeterli örnekli sınıflar için V1 teknik kabul kontrolü yapar."""
    failures: list[str] = []
    for metric_name, threshold in AGGREGATE_QUALITY_THRESHOLDS.items():
        value = evaluation.get(metric_name)
        if value is None or float(value) < threshold:
            shown = "yok" if value is None else f"{float(value):.4f}"
            failures.append(
                f"{metric_name}={shown}; gerekli minimum {threshold:.2f}"
            )

    metric_by_class = {
        int(item["class_id"]): item
        for item in evaluation.get("class_metrics", [])
    }
    for distribution in distributions:
        if distribution.train_objects < SUPPORTED_CLASS_MIN_TRAIN_OBJECTS:
            continue
        item = metric_by_class.get(distribution.class_id)
        if item is None:
            failures.append(f"{distribution.class_name}: sınıf metriği raporda yok")
            continue
        if not item.get("evaluated"):
            failures.append(f"{item['class_name']}: değerlendirme örneği yok")
            continue
        map50 = item.get("map50")
        if map50 is None or float(map50) < SUPPORTED_CLASS_MIN_MAP50:
            shown = "yok" if map50 is None else f"{float(map50):.4f}"
            failures.append(
                f"{item['class_name']}: mAP50={shown}; gerekli minimum {SUPPORTED_CLASS_MIN_MAP50:.2f}"
            )

    return {
        "passed": not failures,
        "aggregate_thresholds": AGGREGATE_QUALITY_THRESHOLDS,
        "supported_class_min_train_objects": SUPPORTED_CLASS_MIN_TRAIN_OBJECTS,
        "supported_class_min_map50": SUPPORTED_CLASS_MIN_MAP50,
        "failures": failures,
    }


def load_reusable_test_evaluation(
    model_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any] | None:
    """Aynı model için tamamlanmış test raporu varsa testi tekrarlamadan yükler."""
    current_hash = model_sha256(model_path)
    report_paths = (
        (report_path,)
        if report_path is not None
        else (
            TEST_EVALUATION_REPORT_PATH,
            REPORTS_DIR / f"evaluation_test_{current_hash[:12]}_metrics.json",
        )
    )

    required_fields = {
        "precision",
        "recall",
        "map50",
        "map50_95",
        "class_metrics",
    }
    for candidate_path in report_paths:
        if not candidate_path.exists():
            continue
        try:
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("split") != "test":
            continue
        if payload.get("model_sha256") != current_hash:
            continue
        if not required_fields.issubset(payload):
            continue
        if not isinstance(payload["class_metrics"], list):
            continue
        return payload
    return None


def finalize_project(
    *,
    checkpoint_path: Path,
    target_epochs: int,
    device: str,
    batch_size: int,
    image_size: int,
    include_test: bool,
    check_only: bool,
    allow_early_stop: bool = False,
) -> dict[str, Any]:
    """Tamamlanan eğitimi aday/final model olarak değerlendirip raporlar."""
    completion = validate_completed_training(
        checkpoint_path,
        target_epochs,
        allow_early_stop=allow_early_stop,
    )
    print(
        f"Eğitim tamamlanma kontrolü başarılı: "
        f"{completion.completed_epochs}/{completion.target_epochs} epoch"
    )
    if check_only:
        return {
            "status": "ready",
            "completed_epochs": completion.completed_epochs,
            "target_epochs": completion.target_epochs,
            "best_path": str(completion.best_path),
            "stopped_early": completion.stopped_early,
        }

    distributions = analyze_dataset(PROCESSED_DATA_DIR / "labels")
    write_dataset_reports(distributions, REPORTS_DIR)
    publish_run_best(
        completion.run_dir,
        stage="candidate",
        destination=MODEL_WEIGHTS_PATH,
        target_epochs=target_epochs,
    )

    validation = evaluate_model(
        model_path=MODEL_WEIGHTS_PATH,
        dataset_yaml=DATASET_YAML_PATH,
        split="val",
        image_size=image_size,
        batch_size=batch_size,
        device=device,
    )
    validation_quality = assess_model_quality(validation, distributions)
    evaluation_metadata: dict[str, Any] = {
        "validation": _aggregate_metrics(validation),
        "test": None,
        "quality": validation_quality,
    }
    publish_run_best(
        completion.run_dir,
        stage="candidate",
        destination=MODEL_WEIGHTS_PATH,
        target_epochs=target_epochs,
        evaluation_metadata=evaluation_metadata,
    )

    test_evaluation: dict[str, Any] | None = None
    final_quality: dict[str, Any] | None = None
    final_stage = "candidate"
    if include_test and validation_quality["passed"]:
        test_evaluation = load_reusable_test_evaluation(MODEL_WEIGHTS_PATH)
        if test_evaluation is None:
            test_evaluation = evaluate_model(
                model_path=MODEL_WEIGHTS_PATH,
                dataset_yaml=DATASET_YAML_PATH,
                split="test",
                image_size=image_size,
                batch_size=batch_size,
                device=device,
            )
        else:
            print(
                "\nAynı model için tamamlanmış test raporu bulundu; "
                "test seti yeniden çalıştırılmadan sonuçlar kullanılıyor."
            )
        final_quality = assess_model_quality(test_evaluation, distributions)
        evaluation_metadata = {
            "validation": _aggregate_metrics(validation),
            "test": _aggregate_metrics(test_evaluation),
            "quality": final_quality,
        }
        if final_quality["passed"]:
            publish_run_best(
                completion.run_dir,
                stage="final",
                destination=MODEL_WEIGHTS_PATH,
                target_epochs=target_epochs,
                evaluation_metadata=evaluation_metadata,
            )
            final_stage = "final"
        else:
            publish_run_best(
                completion.run_dir,
                stage="candidate",
                destination=MODEL_WEIGHTS_PATH,
                target_epochs=target_epochs,
                evaluation_metadata=evaluation_metadata,
            )
    elif include_test:
        print(
            "\nValidation kalite eşiği geçilmediği için resmi test seti korunarak "
            "değerlendirme atlandı."
        )

    summary = {
        "status": final_stage,
        "model_path": str(MODEL_WEIGHTS_PATH),
        "model_sha256": model_sha256(MODEL_WEIGHTS_PATH),
        "completed_epochs": completion.completed_epochs,
        "target_epochs": completion.target_epochs,
        "stopped_early": completion.stopped_early,
        "validation": validation,
        "validation_quality": validation_quality,
        "test": test_evaluation,
        "final_quality": final_quality,
        "dataset_warnings": build_dataset_warnings(distributions),
    }
    write_finalization_summary(summary, REPORTS_DIR)
    return summary


def _aggregate_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    return {
        metric_name: float(evaluation[metric_name])
        for metric_name in ("precision", "recall", "map50", "map50_95")
    }


def write_finalization_summary(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Finalizasyon durumunu makine okunur JSON ve Türkçe Markdown olarak yazar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "finalization_summary.json"
    markdown_path = output_dir / "final_delivery_summary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    validation = summary["validation"]
    test = summary.get("test")
    quality = summary.get("final_quality") or summary["validation_quality"]
    lines = [
        "# PV Vision AI Finalizasyon Özeti",
        "",
        f"- Model durumu: {summary['status']}",
        f"- Tamamlanan eğitim: {summary['completed_epochs']}/{summary['target_epochs']} epoch",
        f"- Erken durdurma: {'Evet' if summary.get('stopped_early') else 'Hayır'}",
        f"- Model SHA-256: {summary['model_sha256']}",
        "",
        "## Validation Metrikleri",
        "",
        f"- Precision: {validation['precision']:.4f}",
        f"- Recall: {validation['recall']:.4f}",
        f"- mAP50: {validation['map50']:.4f}",
        f"- mAP50-95: {validation['map50_95']:.4f}",
    ]
    if test:
        lines.extend(
            [
                "",
                "## Test Metrikleri",
                "",
                f"- Precision: {test['precision']:.4f}",
                f"- Recall: {test['recall']:.4f}",
                f"- mAP50: {test['map50']:.4f}",
                f"- mAP50-95: {test['map50_95']:.4f}",
            ]
        )
    lines.extend(["", "## Teknik Kabul", ""])
    if quality["passed"]:
        lines.append("- V1 teknik kalite eşikleri geçildi.")
    else:
        lines.append("- V1 teknik kalite eşikleri henüz geçilmedi.")
        lines.extend(f"- {failure}" for failure in quality["failures"])
    lines.extend(["", "## Veri Seti Sınırlılıkları", ""])
    lines.extend(f"- {warning}" for warning in summary["dataset_warnings"])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tamamlanan PV Vision AI eğitimini değerlendirip final modeli yayımlar."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Validation sonrasında resmi test setini bir kez değerlendir",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Hiçbir model veya rapor değiştirmeden yalnızca tamamlanma kontrolü yap",
    )
    parser.add_argument(
        "--allow-early-stop",
        action="store_true",
        help="Normal bitmiş kalite eğitimini hedef epoch'tan önce durmuşsa kabul et",
    )
    args = parser.parse_args()

    try:
        summary = finalize_project(
            checkpoint_path=args.checkpoint,
            target_epochs=args.target_epochs,
            device=args.device,
            batch_size=args.batch,
            image_size=args.imgsz,
            include_test=args.include_test,
            check_only=args.check_only,
            allow_early_stop=args.allow_early_stop,
        )
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        print(f"HATA: {exc}")
        raise SystemExit(2) from None

    print("\nPV Vision AI finalizasyon durumu")
    print("--------------------------------")
    print(f"Durum: {summary['status']}")
    print(f"Tamamlanan epoch: {summary['completed_epochs']}/{summary['target_epochs']}")
    if not args.check_only:
        print(f"Finalizasyon raporu: {REPORTS_DIR / 'final_delivery_summary.md'}")


if __name__ == "__main__":
    main()
