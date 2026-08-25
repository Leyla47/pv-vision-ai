"""Proje durum özetleyicisinin testleri."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from training.status import inspect_project_status


class StatusTests(unittest.TestCase):
    def test_default_status_follows_the_published_model_source_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            training_dir = root / "outputs" / "training"
            quality_checkpoint = training_dir / "quality_run" / "weights" / "last.pt"
            quality_checkpoint.parent.mkdir(parents=True)
            torch.save(
                {
                    "epoch": 7,
                    "optimizer": {"state": {}},
                    "train_args": {"epochs": 60},
                },
                quality_checkpoint,
            )
            metadata_path = root / "models" / "weights" / "model_info.json"
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text(
                json.dumps({"stage": "interim", "source_run": "quality_run"}),
                encoding="utf-8",
            )
            with patch("training.status.DEFAULT_CHECKPOINT", root / "missing_v1.pt"), patch(
                "training.status.TRAINING_OUTPUT_DIR",
                training_dir,
            ), patch(
                "training.status.MODEL_METADATA_PATH",
                metadata_path,
            ), patch(
                "training.status.MODEL_WEIGHTS_PATH",
                root / "missing_best.pt",
            ), patch(
                "training.status.REPORTS_DIR",
                root / "reports",
            ):
                status = inspect_project_status()

            self.assertEqual(status["checkpoint_path"], str(quality_checkpoint))
            self.assertEqual(status["completed_epochs"], 8)
            self.assertIn("quality_run", status["next_command"])

    def test_incomplete_checkpoint_returns_safe_resume_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint_path = root / "outputs" / "training" / "run" / "weights" / "last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            torch.save(
                {
                    "epoch": 3,
                    "optimizer": {"state": {}},
                    "train_args": {"epochs": 10},
                },
                checkpoint_path,
            )
            with patch("training.status.PROJECT_ROOT", root), patch(
                "training.status.MODEL_METADATA_PATH",
                root / "missing_model_info.json",
            ), patch(
                "training.status.MODEL_WEIGHTS_PATH",
                root / "missing_best.pt",
            ), patch(
                "training.status.REPORTS_DIR",
                root / "reports",
            ):
                status = inspect_project_status(checkpoint_path)

            self.assertEqual(status["completed_epochs"], 4)
            self.assertTrue(status["optimizer_ready"])
            self.assertIn("--resume-epochs 10", status["next_command"])

    def test_stripped_completed_checkpoint_uses_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            run_dir = root / "outputs" / "training" / "run"
            checkpoint_path = run_dir / "weights" / "last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            torch.save(
                {"epoch": -1, "optimizer": None, "train_args": {"epochs": 10}},
                checkpoint_path,
            )
            (run_dir / "results.csv").write_text(
                "epoch,metrics/mAP50(B)\n10,0.5\n",
                encoding="utf-8",
            )
            with patch("training.status.PROJECT_ROOT", root), patch(
                "training.status.MODEL_METADATA_PATH",
                root / "missing_model_info.json",
            ), patch(
                "training.status.MODEL_WEIGHTS_PATH",
                root / "missing_best.pt",
            ), patch(
                "training.status.REPORTS_DIR",
                root / "reports",
            ):
                status = inspect_project_status(checkpoint_path)

            self.assertEqual(status["completed_epochs"], 10)
            self.assertTrue(status["checkpoint_stripped"])
            self.assertFalse(status["stopped_early"])
            self.assertIn("training/finalize.py", status["next_command"])

    def test_stripped_early_stop_returns_safe_finalization_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            run_dir = root / "outputs" / "training" / "quality"
            checkpoint_path = run_dir / "weights" / "last.pt"
            checkpoint_path.parent.mkdir(parents=True)
            torch.save(
                {"epoch": -1, "optimizer": None, "train_args": {"epochs": 60}},
                checkpoint_path,
            )
            (run_dir / "results.csv").write_text(
                "epoch,metrics/mAP50(B)\n24,0.5\n",
                encoding="utf-8",
            )
            with patch("training.status.PROJECT_ROOT", root), patch(
                "training.status.MODEL_METADATA_PATH",
                root / "missing_model_info.json",
            ), patch(
                "training.status.MODEL_WEIGHTS_PATH",
                root / "missing_best.pt",
            ), patch(
                "training.status.REPORTS_DIR",
                root / "reports",
            ):
                status = inspect_project_status(checkpoint_path)

            self.assertEqual(status["completed_epochs"], 24)
            self.assertTrue(status["stopped_early"])
            self.assertIn("--allow-early-stop", status["next_command"])
            self.assertNotIn("--resume-from", status["next_command"])


if __name__ == "__main__":
    unittest.main()
