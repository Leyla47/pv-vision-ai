"""PV Vision AI eğitim ve model durumunu tek komutla özetler."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    MODEL_METADATA_PATH,
    MODEL_WEIGHTS_PATH,
    REPORTS_DIR,
    TRAINING_OUTPUT_DIR,
)
from model_registry import load_model_metadata  # noqa: E402
from training.finalize import DEFAULT_CHECKPOINT  # noqa: E402


def inspect_project_status(checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Checkpoint, yayımlanmış model ve veri raporundan güncel durum üretir."""
    import torch

    metadata = load_model_metadata(MODEL_METADATA_PATH)
    checkpoint_path = _resolve_status_checkpoint(checkpoint_path, metadata)
    completed_epochs = 0
    target_epochs = 10
    checkpoint_ready = checkpoint_path.exists()
    optimizer_ready = False
    checkpoint_stripped = False
    stopped_early = False
    if checkpoint_ready:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        completed_epochs = checkpoint_epoch + 1
        train_args = checkpoint.get("train_args", {})
        if isinstance(train_args, dict):
            target_epochs = int(train_args.get("epochs", target_epochs))
        optimizer_ready = checkpoint.get("optimizer") is not None
        checkpoint_stripped = checkpoint_epoch < 0 and not optimizer_ready
        if checkpoint_stripped:
            completed_epochs = _read_result_epochs(
                checkpoint_path.resolve().parent.parent / "results.csv"
            )
            stopped_early = 0 < completed_epochs < target_epochs

    model_stage = metadata.get("stage") if metadata else "missing"
    model_available = MODEL_WEIGHTS_PATH.exists()
    dataset_report_path = REPORTS_DIR / "dataset_class_distribution.json"
    warning_count = 0
    if dataset_report_path.exists():
        try:
            dataset_report = json.loads(dataset_report_path.read_text(encoding="utf-8"))
            warning_count = len(dataset_report.get("warnings", []))
        except (OSError, json.JSONDecodeError):
            warning_count = 0

    if not checkpoint_ready:
        next_action = "Eğitim checkpoint'i bulunamadı; eğitim başlatılmalı."
        next_command = "python training/train.py --preset mps-v1"
    elif stopped_early:
        next_action = (
            f"Eğitim {completed_epochs}/{target_epochs} epoch'ta normal erken durdurmayla "
            "tamamlanmış; değerlendirme erken durdurma kabulüyle başlatılmalı."
        )
        try:
            displayed_checkpoint = checkpoint_path.relative_to(PROJECT_ROOT)
        except ValueError:
            displayed_checkpoint = checkpoint_path
        next_command = (
            "python training/finalize.py "
            f"--checkpoint {displayed_checkpoint} --target-epochs {target_epochs} "
            "--allow-early-stop --include-test --device mps --batch 8"
        )
    elif completed_epochs < target_epochs and optimizer_ready:
        next_action = (
            f"Eğitime {completed_epochs + 1}/{target_epochs} epoch'tan "
            "checkpoint ile devam edilmeli."
        )
        try:
            displayed_checkpoint = checkpoint_path.relative_to(PROJECT_ROOT)
        except ValueError:
            displayed_checkpoint = checkpoint_path
        next_command = (
            "python training/train.py "
            f"--resume-from {displayed_checkpoint} "
            f"--resume-epochs {target_epochs} --device mps"
        )
    elif completed_epochs < target_epochs:
        next_action = "Checkpoint devam ettirilebilir değil; results.csv ve eğitim çıktıları incelenmeli."
        next_command = "python training/status.py"
    elif model_stage != "final":
        next_action = "Eğitim tamamlandı; validation ve final test değerlendirmesi yapılmalı."
        try:
            displayed_checkpoint = checkpoint_path.relative_to(PROJECT_ROOT)
        except ValueError:
            displayed_checkpoint = checkpoint_path
        next_command = (
            "python training/finalize.py "
            f"--checkpoint {displayed_checkpoint} --target-epochs {target_epochs} "
            "--include-test --device mps --batch 8"
        )
    else:
        next_action = "Final model hazır; web uygulaması kullanılabilir."
        next_command = "streamlit run app/app.py"

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_ready": checkpoint_ready,
        "optimizer_ready": optimizer_ready,
        "checkpoint_stripped": checkpoint_stripped,
        "stopped_early": stopped_early,
        "completed_epochs": completed_epochs,
        "target_epochs": target_epochs,
        "model_available": model_available,
        "model_stage": model_stage,
        "dataset_warning_count": warning_count,
        "next_action": next_action,
        "next_command": next_command,
    }


def print_project_status(status: dict[str, Any]) -> None:
    print("PV Vision AI proje durumu")
    print("--------------------------")
    print(f"Checkpoint: {'hazır' if status['checkpoint_ready'] else 'yok'}")
    print(f"Optimizer: {'hazır' if status['optimizer_ready'] else 'yok'}")
    print(f"Tamamlanmış checkpoint: {'evet' if status['checkpoint_stripped'] else 'hayır'}")
    print(f"Eğitim: {status['completed_epochs']}/{status['target_epochs']} epoch")
    print(f"Uygulama modeli: {status['model_stage']}")
    print(f"Veri seti uyarısı: {status['dataset_warning_count']}")
    print(f"\nSıradaki işlem: {status['next_action']}")
    print(f"Komut: {status['next_command']}")


def _read_result_epochs(results_path: Path) -> int:
    if not results_path.exists():
        return 0
    try:
        with results_path.open(encoding="utf-8", newline="") as results_file:
            rows = csv.DictReader(results_file)
            return max(
                (int(float(row["epoch"])) for row in rows if row.get("epoch")),
                default=0,
            )
    except (OSError, ValueError, KeyError):
        return 0


def _resolve_status_checkpoint(
    requested_checkpoint: Path | None,
    metadata: dict[str, Any] | None,
) -> Path:
    if requested_checkpoint is not None:
        return requested_checkpoint
    source_run = metadata.get("source_run") if metadata else None
    if isinstance(source_run, str) and source_run and Path(source_run).name == source_run:
        model_checkpoint = TRAINING_OUTPUT_DIR / source_run / "weights" / "last.pt"
        if model_checkpoint.exists():
            return model_checkpoint
    return DEFAULT_CHECKPOINT


def main() -> None:
    parser = argparse.ArgumentParser(description="PV Vision AI güncel proje durumunu gösterir.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Belirtilmezse yayımlanmış modelin kaynak run checkpoint'i kullanılır",
    )
    parser.add_argument("--json", action="store_true", help="Durumu JSON olarak yazdır")
    args = parser.parse_args()

    status = inspect_project_status(args.checkpoint)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print_project_status(status)


if __name__ == "__main__":
    main()
