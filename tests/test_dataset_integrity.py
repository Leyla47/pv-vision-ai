"""Hazırlanmış PVEL-AD veri setinin bütünlük testi."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import DATASET_YAML_PATH, NUM_CLASSES
from training.prepare_dataset import (
    _synchronize_generated_files,
    _validate_disjoint_split_stems,
    validate_processed_dataset,
)


class DatasetPreparationSafetyTests(unittest.TestCase):
    def test_stale_generated_files_are_removed_without_touching_other_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            keep = directory / "keep.jpg"
            stale = directory / "stale.jpg"
            unrelated = directory / "notes.md"
            keep.write_bytes(b"keep")
            stale.write_bytes(b"stale")
            unrelated.write_text("notes", encoding="utf-8")

            _synchronize_generated_files(directory, {keep.name}, suffix=".jpg")

            self.assertTrue(keep.exists())
            self.assertFalse(stale.exists())
            self.assertTrue(unrelated.exists())

    def test_overlapping_splits_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Veri sızıntısı"):
            _validate_disjoint_split_stems(
                {
                    "train": {"image-1", "image-2"},
                    "val": {"image-2"},
                    "test": set(),
                }
            )


@unittest.skipUnless(DATASET_YAML_PATH.exists(), "Hazırlanmış veri seti bulunmuyor.")
class DatasetIntegrityTests(unittest.TestCase):
    def test_all_splits_have_matching_valid_labels(self) -> None:
        summaries = validate_processed_dataset(NUM_CLASSES)

        self.assertEqual(summaries["train"].image_count, 3600)
        self.assertEqual(summaries["val"].image_count, 900)
        self.assertEqual(summaries["test"].image_count, 19150)
        for summary in summaries.values():
            self.assertEqual(summary.image_count, summary.label_count)
            self.assertGreater(summary.object_count, 0)


if __name__ == "__main__":
    unittest.main()
