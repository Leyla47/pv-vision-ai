"""Değerlendirme raporu ve Türkçe sınıf metrikleri testleri."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from config import DEFECT_CLASSES_TR
from training.evaluate import (
    create_localized_dataset_yaml,
    extract_class_metrics,
    validate_test_evaluation,
    write_evaluation_report,
)
from model_registry import model_sha256


class FakeMetrics:
    ap_class_index = np.asarray([0, 2])
    nt_per_class = np.asarray([8, 0, 5])

    def class_result(self, index: int) -> tuple[float, float, float, float]:
        return [(0.8, 0.7, 0.75, 0.50), (0.6, 0.5, 0.55, 0.30)][index]


class EvaluateTests(unittest.TestCase):
    def test_class_metrics_use_turkish_names(self) -> None:
        metrics = extract_class_metrics(FakeMetrics())

        self.assertEqual(metrics[0]["class_name"], "Çatlak")
        self.assertEqual(metrics[2]["class_name"], "Siyah çekirdek")
        self.assertEqual(metrics[2]["instances"], 5)
        self.assertEqual(len(metrics), 12)
        self.assertFalse(metrics[1]["evaluated"])

    def test_localized_dataset_yaml_preserves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            source = directory / "dataset.yaml"
            destination = directory / "dataset_tr.yaml"
            source.write_text(
                "path: /tmp/data\ntrain: images/train\nval: images/val\nnc: 2\nnames: [a, b]\n",
                encoding="utf-8",
            )

            create_localized_dataset_yaml(source, destination)
            payload = yaml.safe_load(destination.read_text(encoding="utf-8"))

            self.assertEqual(payload["path"], "/tmp/data")
            self.assertEqual(payload["names"], DEFECT_CLASSES_TR)
            self.assertEqual(payload["nc"], 12)

    def test_report_contains_aggregate_and_class_metrics(self) -> None:
        from training import evaluate

        original_reports_dir = evaluate.REPORTS_DIR
        try:
            with tempfile.TemporaryDirectory() as temporary_dir:
                evaluate.REPORTS_DIR = Path(temporary_dir)
                class_metrics = extract_class_metrics(FakeMetrics())
                json_path, text_path = write_evaluation_report(
                    split="val",
                    model_path=Path("models/weights/best.pt"),
                    precision=0.7,
                    recall=0.6,
                    map50=0.65,
                    map5095=0.4,
                    class_metrics=class_metrics,
                    artifacts_dir=Path("outputs/training/evaluation_val"),
                )

                payload = json.loads(json_path.read_text(encoding="utf-8"))
                report = text_path.read_text(encoding="utf-8")
                self.assertEqual(len(payload["class_metrics"]), 12)
                self.assertIn("Sınıf bazlı metrikler", report)
                self.assertIn("Çatlak", report)
                self.assertIn("karışıklık matrisi", report)
                snapshot_paths = list(
                    Path(temporary_dir).glob("evaluation_val_*_metrics.json")
                )
                self.assertEqual(len(snapshot_paths), 1)
        finally:
            evaluate.REPORTS_DIR = original_reports_dir

    def test_test_split_requires_matching_candidate_and_is_not_repeated(self) -> None:
        from training import evaluate

        original_metadata_path = evaluate.MODEL_METADATA_PATH
        original_reports_dir = evaluate.REPORTS_DIR
        try:
            with tempfile.TemporaryDirectory() as temporary_dir:
                directory = Path(temporary_dir)
                model_path = directory / "best.pt"
                model_path.write_bytes(b"candidate")
                metadata_path = directory / "model_info.json"
                metadata_path.write_text(
                    json.dumps(
                        {
                            "stage": "candidate",
                            "sha256": model_sha256(model_path),
                        }
                    ),
                    encoding="utf-8",
                )
                evaluate.MODEL_METADATA_PATH = metadata_path
                evaluate.REPORTS_DIR = directory

                validate_test_evaluation(
                    model_path=model_path,
                    split="test",
                    force_repeat=False,
                )

                model_hash = model_sha256(model_path)
                (directory / f"evaluation_test_{model_hash[:12]}_metrics.json").write_text(
                    json.dumps({"model_sha256": model_sha256(model_path)}),
                    encoding="utf-8",
                )
                with self.assertRaises(FileExistsError):
                    validate_test_evaluation(
                        model_path=model_path,
                        split="test",
                        force_repeat=False,
                    )
        finally:
            evaluate.MODEL_METADATA_PATH = original_metadata_path
            evaluate.REPORTS_DIR = original_reports_dir


if __name__ == "__main__":
    unittest.main()
