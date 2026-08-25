"""Model yayımlama ve model kimlik bilgisi testleri."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_registry import load_model_metadata, publish_run_best


RESULTS_CSV = """epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)
1,0.40,0.30,0.35,0.20
2,0.50,0.45,0.52,0.31
3,0.48,0.44,0.49,0.29
"""


class ModelRegistryTests(unittest.TestCase):
    def test_publish_copies_best_and_records_best_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            run_dir = root / "sample_run"
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True)
            (weights_dir / "best.pt").write_bytes(b"sample-model")
            (run_dir / "results.csv").write_text(RESULTS_CSV, encoding="utf-8")
            (run_dir / "args.yaml").write_text(
                "model: yolov8n.pt\nepochs: 10\n",
                encoding="utf-8",
            )
            destination = root / "published" / "best.pt"
            metadata_path = root / "published" / "model_info.json"

            metadata = publish_run_best(
                run_dir,
                stage="interim",
                destination=destination,
                metadata_path=metadata_path,
            )

            self.assertEqual(destination.read_bytes(), b"sample-model")
            self.assertEqual(metadata["completed_epochs"], 3)
            self.assertEqual(metadata["best_epoch"], 2)
            self.assertEqual(metadata["metrics"]["map50_95"], 0.31)
            self.assertEqual(load_model_metadata(metadata_path)["stage"], "interim")
            self.assertEqual(
                json.loads(metadata_path.read_text(encoding="utf-8"))["source_run"],
                "sample_run",
            )

    def test_invalid_stage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(ValueError):
                publish_run_best(Path(temporary_dir), stage="unknown")

    def test_best_epoch_uses_ultralytics_detection_fitness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            run_dir = root / "fitness_run"
            (run_dir / "weights").mkdir(parents=True)
            (run_dir / "weights" / "best.pt").write_bytes(b"fitness-model")
            (run_dir / "results.csv").write_text(
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
                "1,0.4,0.3,1.0,0.30\n"
                "2,0.4,0.3,0.0,0.31\n",
                encoding="utf-8",
            )

            metadata = publish_run_best(
                run_dir,
                stage="candidate",
                destination=root / "best.pt",
                metadata_path=root / "model_info.json",
            )

            self.assertEqual(metadata["best_epoch"], 1)
            self.assertEqual(metadata["metrics"]["map50"], 1.0)
            self.assertEqual(metadata["metrics"]["map50_95"], 0.30)

    def test_candidate_stage_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            run_dir = root / "candidate_run"
            (run_dir / "weights").mkdir(parents=True)
            (run_dir / "weights" / "best.pt").write_bytes(b"candidate-model")
            (run_dir / "results.csv").write_text(RESULTS_CSV, encoding="utf-8")
            (run_dir / "args.yaml").write_text("epochs: 3\n", encoding="utf-8")

            metadata = publish_run_best(
                run_dir,
                stage="candidate",
                destination=root / "best.pt",
                metadata_path=root / "model_info.json",
                evaluation_metadata={
                    "validation": {"map50": 0.5},
                    "test": None,
                },
            )

            self.assertEqual(metadata["stage"], "candidate")
            self.assertEqual(metadata["evaluation"]["validation"]["map50"], 0.5)


if __name__ == "__main__":
    unittest.main()
