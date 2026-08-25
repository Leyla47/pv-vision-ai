"""Veri seti sınıf dağılımı raporunun testleri."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.analyze_dataset import (
    analyze_dataset,
    build_dataset_warnings,
    write_dataset_reports,
)


class AnalyzeDatasetTests(unittest.TestCase):
    def test_distribution_and_coverage_warnings_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for split in ("train", "val", "test"):
                (root / "labels" / split).mkdir(parents=True)
            (root / "labels" / "train" / "a.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n5 0.5 0.5 0.1 0.1\n",
                encoding="utf-8",
            )
            (root / "labels" / "val" / "b.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            (root / "labels" / "test" / "c.txt").write_text(
                "5 0.5 0.5 0.1 0.1\n",
                encoding="utf-8",
            )

            distributions = analyze_dataset(root / "labels")
            warnings = build_dataset_warnings(distributions)
            csv_path, json_path, markdown_path = write_dataset_reports(
                distributions,
                root / "reports",
            )

            self.assertEqual(distributions[0].train_objects, 1)
            self.assertEqual(distributions[5].validation_coverage, "Değerlendirilemez")
            self.assertTrue(any("Köşe kusuru" in warning for warning in warnings))
            self.assertTrue(csv_path.exists())
            self.assertIn("Sınıf Dağılımı", markdown_path.read_text(encoding="utf-8"))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["class_count"], 12)


if __name__ == "__main__":
    unittest.main()
