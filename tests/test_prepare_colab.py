"""Google Colab eğitim paketi güvenlik testleri."""

from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from training.prepare_colab import create_colab_bundle, localize_dataset_yaml


class PrepareColabTests(unittest.TestCase):
    def test_localized_dataset_uses_colab_path_and_removes_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            dataset_path = Path(temporary_dir) / "dataset.yaml"
            dataset_path.write_text(
                "path: /Users/example/data\ntrain: images/train\nval: images/val\ntest: images/test\n",
                encoding="utf-8",
            )

            payload = localize_dataset_yaml(dataset_path, data_root=Path("/content/data"))

            self.assertEqual(payload["path"], "/content/data")
            self.assertNotIn("test", payload)
            self.assertNotIn("test", yaml.safe_load(dataset_path.read_text(encoding="utf-8")))

    def test_bundle_dereferences_images_and_excludes_test_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._create_minimal_project(root)
            output = root / "outputs/colab/pv_vision_colab.tar.gz"

            summary = create_colab_bundle(
                output_path=output,
                target_epochs=10,
                run_name="test_run",
                project_root=root,
            )

            self.assertEqual(summary.completed_epochs, 6)
            self.assertEqual((summary.train_images, summary.val_images), (1, 1))
            with tarfile.open(output, mode="r:gz") as archive:
                names = {member.name for member in archive.getmembers()}
                image = archive.getmember("data/processed/images/train/train.jpg")
                self.assertTrue(image.isfile())
                self.assertFalse(image.issym())
                self.assertNotIn("data/processed/images/test/test.jpg", names)
                self.assertNotIn("data/processed/labels/test/test.txt", names)

                dataset_file = archive.extractfile("data/processed/dataset.yaml")
                manifest_file = archive.extractfile("colab_bundle_manifest.json")
                self.assertIsNotNone(dataset_file)
                self.assertIsNotNone(manifest_file)
                dataset = yaml.safe_load(dataset_file.read())
                manifest = json.load(manifest_file)
                self.assertNotIn("test", dataset)
                self.assertFalse(manifest["test_included"])
                self.assertEqual(manifest["splits"]["train"]["images"], 1)

    @staticmethod
    def _create_minimal_project(root: Path) -> None:
        for relative, content in {
            "config.py": "# config\n",
            "model_registry.py": "# registry\n",
            "requirements.txt": "ultralytics==8.3.186\n",
            "README.md": "# Test\n",
            "training/__init__.py": "",
            "training/train.py": "# train\n",
            "data/processed/dataset.yaml": (
                "path: /Users/example/data\ntrain: images/train\nval: images/val\n"
                "test: images/test\nnc: 1\nnames: [defect]\n"
            ),
            "outputs/training/test_run/args.yaml": "epochs: 10\n",
            "outputs/training/test_run/results.csv": "epoch\n1\n",
        }.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        raw = root / "data/raw"
        raw.mkdir(parents=True)
        for split in ("train", "val", "test"):
            source = raw / f"{split}.jpg"
            source.write_bytes(f"image-{split}".encode())
            image = root / f"data/processed/images/{split}/{split}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.symlink_to(source)
            label = root / f"data/processed/labels/{split}/{split}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        weights = root / "outputs/training/test_run/weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"best")
        torch.save(
            {
                "epoch": 5,
                "optimizer": {"state": {}},
                "train_args": {"epochs": 10, "device": "mps"},
            },
            weights / "last.pt",
        )


if __name__ == "__main__":
    unittest.main()
