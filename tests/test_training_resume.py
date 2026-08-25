"""Checkpoint resume güvenlik kontrollerinin testleri."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from training.train import (
    TRAINING_PRESETS,
    create_resume_checkpoint_copy,
    inspect_resume_checkpoint,
    validate_quality_start_model,
)


class TrainingResumeTests(unittest.TestCase):
    def test_mps_v1_preset_targets_ten_epochs(self) -> None:
        preset = TRAINING_PRESETS["mps-v1"]
        self.assertEqual(preset.epochs, 10)
        self.assertEqual(preset.device, "mps")

    def test_quality_preset_starts_from_verified_candidate_model(self) -> None:
        preset = TRAINING_PRESETS["quality"]
        self.assertTrue(preset.model.endswith("models/weights/best.pt"))

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model_path = root / "best.pt"
            metadata_path = root / "model_info.json"
            model_path.write_bytes(b"candidate")
            metadata_path.write_text(
                json.dumps(
                    {
                        "stage": "candidate",
                        "sha256": hashlib.sha256(b"candidate").hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            validate_quality_start_model(model_path, metadata_path)

            metadata_path.write_text(
                json.dumps(
                    {
                        "stage": "interim",
                        "sha256": hashlib.sha256(b"candidate").hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "candidate/final"):
                validate_quality_start_model(model_path, metadata_path)

    def test_resume_copy_changes_only_the_target_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "last.pt"
            torch.save(
                {
                    "epoch": 3,
                    "optimizer": {"state": {}},
                    "ema": object(),
                    "train_args": {"epochs": 40, "device": "mps"},
                },
                checkpoint_path,
            )

            completed, original_target, has_optimizer = inspect_resume_checkpoint(
                checkpoint_path,
                target_epochs=10,
            )
            temporary_copy = create_resume_checkpoint_copy(checkpoint_path, target_epochs=10)

            original = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            copied = torch.load(temporary_copy, map_location="cpu", weights_only=False)
            self.assertEqual((completed, original_target, has_optimizer), (4, 40, True))
            self.assertEqual(original["train_args"]["epochs"], 40)
            self.assertEqual(copied["train_args"]["epochs"], 10)
            self.assertEqual(copied["epoch"], 3)
            self.assertIsNotNone(copied["optimizer"])

    def test_target_must_be_greater_than_completed_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "last.pt"
            torch.save(
                {"epoch": 3, "optimizer": {}, "train_args": {"epochs": 10}},
                checkpoint_path,
            )
            with self.assertRaises(ValueError):
                inspect_resume_checkpoint(checkpoint_path, target_epochs=4)

    def test_resume_copy_accepts_colab_path_overrides_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "last.pt"
            torch.save(
                {
                    "epoch": 5,
                    "optimizer": {"state": {}},
                    "train_args": {
                        "epochs": 10,
                        "data": "/Users/example/dataset.yaml",
                        "device": "mps",
                    },
                },
                checkpoint_path,
            )

            temporary_copy = create_resume_checkpoint_copy(
                checkpoint_path,
                target_epochs=10,
                train_arg_overrides={
                    "data": "/content/pv_vision_ai/data/processed/dataset.yaml",
                    "device": "0",
                    "project": "/content/drive/MyDrive/PV_Vision_AI_Colab/outputs/training",
                },
            )

            original = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            copied = torch.load(temporary_copy, map_location="cpu", weights_only=False)
            self.assertEqual(original["train_args"]["device"], "mps")
            self.assertEqual(copied["train_args"]["device"], "0")
            self.assertTrue(copied["train_args"]["data"].startswith("/content/"))
            self.assertEqual(copied["epoch"], 5)
            self.assertIsNotNone(copied["optimizer"])
            self.assertEqual(copied["train_args"]["model"], str(temporary_copy))
            self.assertEqual(copied["train_args"]["resume"], str(temporary_copy))


if __name__ == "__main__":
    unittest.main()
