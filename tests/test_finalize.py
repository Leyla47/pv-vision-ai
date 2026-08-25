"""Final model kalite kapısı ve checkpoint tamamlanma testleri."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from training.analyze_dataset import ClassDistribution
from training.finalize import (
    assess_model_quality,
    load_reusable_test_evaluation,
    validate_completed_training,
    write_finalization_summary,
)
from model_registry import model_sha256


def distribution(class_id: int, train_objects: int) -> ClassDistribution:
    return ClassDistribution(
        class_id=class_id,
        class_name=f"Sınıf {class_id}",
        train_objects=train_objects,
        train_images=train_objects,
        val_objects=20,
        val_images=20,
        test_objects=20,
        test_images=20,
        training_support="Yeterli",
        validation_coverage="Kapsanıyor",
    )


class FinalizeTests(unittest.TestCase):
    def test_completed_checkpoint_and_results_are_both_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run"
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True)
            checkpoint_path = weights_dir / "last.pt"
            torch.save({"epoch": 9, "train_args": {"epochs": 10}}, checkpoint_path)
            (weights_dir / "best.pt").write_bytes(b"best")
            (run_dir / "results.csv").write_text(
                "epoch,metrics/mAP50(B)\n10,0.5\n",
                encoding="utf-8",
            )

            completion = validate_completed_training(checkpoint_path, 10)

            self.assertEqual(completion.completed_epochs, 10)
            self.assertEqual(completion.result_epochs, 10)

    def test_incomplete_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "last.pt"
            torch.save({"epoch": 3, "train_args": {"epochs": 10}}, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "4/10"):
                validate_completed_training(checkpoint_path, 10)

    def test_stripped_checkpoint_uses_results_csv_as_completion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run"
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True)
            checkpoint_path = weights_dir / "last.pt"
            torch.save(
                {"epoch": -1, "optimizer": None, "train_args": {"epochs": 10}},
                checkpoint_path,
            )
            (weights_dir / "best.pt").write_bytes(b"best")
            (run_dir / "results.csv").write_text(
                "epoch,metrics/mAP50(B)\n10,0.5\n",
                encoding="utf-8",
            )

            completion = validate_completed_training(checkpoint_path, 10)

            self.assertEqual(completion.completed_epochs, 10)
            self.assertTrue(completion.checkpoint_stripped)
            self.assertFalse(completion.stopped_early)

    def test_stripped_early_stop_requires_explicit_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run"
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True)
            checkpoint_path = weights_dir / "last.pt"
            torch.save(
                {"epoch": -1, "optimizer": None, "train_args": {"epochs": 60}},
                checkpoint_path,
            )
            (weights_dir / "best.pt").write_bytes(b"best")
            (run_dir / "results.csv").write_text(
                "epoch,metrics/mAP50(B)\n24,0.5\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "--allow-early-stop"):
                validate_completed_training(checkpoint_path, 60)

            completion = validate_completed_training(
                checkpoint_path,
                60,
                allow_early_stop=True,
            )
            self.assertEqual(completion.completed_epochs, 24)
            self.assertTrue(completion.stopped_early)

    def test_quality_gate_checks_aggregate_and_supported_classes(self) -> None:
        distributions = [distribution(class_id, 120 if class_id == 0 else 5) for class_id in range(12)]
        evaluation = {
            "precision": 0.60,
            "recall": 0.50,
            "map50": 0.55,
            "map50_95": 0.30,
            "class_metrics": [
                {
                    "class_id": 0,
                    "class_name": "Çatlak",
                    "evaluated": True,
                    "map50": 0.25,
                }
            ],
        }
        self.assertTrue(assess_model_quality(evaluation, distributions)["passed"])

        evaluation["recall"] = 0.10
        self.assertFalse(assess_model_quality(evaluation, distributions)["passed"])

    def test_completed_test_report_is_reused_only_for_the_same_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model_path = root / "best.pt"
            report_path = root / "evaluation_test_metrics.json"
            model_path.write_bytes(b"candidate model")
            payload = {
                "split": "test",
                "model_sha256": model_sha256(model_path),
                "precision": 0.5,
                "recall": 0.4,
                "map50": 0.45,
                "map50_95": 0.25,
                "class_metrics": [],
            }
            report_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(
                load_reusable_test_evaluation(model_path, report_path),
                payload,
            )

            model_path.write_bytes(b"different model")
            self.assertIsNone(load_reusable_test_evaluation(model_path, report_path))

    def test_hash_snapshot_recovers_when_canonical_test_report_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model_path = root / "best.pt"
            model_path.write_bytes(b"candidate model")
            model_hash = model_sha256(model_path)
            payload = {
                "split": "test",
                "model_sha256": model_hash,
                "precision": 0.5,
                "recall": 0.4,
                "map50": 0.45,
                "map50_95": 0.25,
                "class_metrics": [],
            }
            snapshot_path = root / f"evaluation_test_{model_hash[:12]}_metrics.json"
            snapshot_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch("training.finalize.REPORTS_DIR", root), patch(
                "training.finalize.TEST_EVALUATION_REPORT_PATH",
                root / "evaluation_test_metrics.json",
            ):
                self.assertEqual(load_reusable_test_evaluation(model_path), payload)

    def test_finalization_summary_files_are_written(self) -> None:
        summary = {
            "status": "candidate",
            "model_sha256": "abc",
            "completed_epochs": 10,
            "target_epochs": 10,
            "validation": {
                "precision": 0.5,
                "recall": 0.4,
                "map50": 0.45,
                "map50_95": 0.25,
            },
            "validation_quality": {"passed": False, "failures": ["örnek hata"]},
            "test": None,
            "final_quality": None,
            "dataset_warnings": ["örnek uyarı"],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            json_path, markdown_path = write_finalization_summary(
                summary,
                Path(temporary_dir),
            )
            self.assertEqual(json.loads(json_path.read_text())["status"], "candidate")
            report = markdown_path.read_text(encoding="utf-8")
            self.assertIn("örnek hata", report)
            self.assertIn("örnek uyarı", report)


if __name__ == "__main__":
    unittest.main()
