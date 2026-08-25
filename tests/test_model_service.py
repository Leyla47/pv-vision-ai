"""Model tahmin servisinin Türkçe çıktı testleri."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from app.services.model_service import build_analysis_summary, predict_image


class FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = FakeTensor([[10.0, 12.0, 70.0, 60.0]])
        self.conf = FakeTensor([0.875])
        self.cls = FakeTensor([0])

    def __len__(self) -> int:
        return 1


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    def __init__(self) -> None:
        self.source = None
        self.confidence = None

    def predict(self, *, source: object, conf: float, verbose: bool) -> list[FakeResult]:
        self.source = source
        self.confidence = conf
        self.verbose = verbose
        return [FakeResult()]


class ModelServiceTests(unittest.TestCase):
    def test_prediction_uses_turkish_labels_and_percentage_confidence(self) -> None:
        model = FakeModel()
        image = Image.new("RGB", (100, 80), color="white")

        result = predict_image(model, image, confidence=0.25)

        self.assertIsInstance(model.source, Image.Image)
        self.assertEqual(model.confidence, 0.25)
        self.assertEqual(result.annotated_image.shape, (80, 100, 3))
        self.assertEqual(result.detections.iloc[0]["Kusur sınıfı"], "Çatlak")
        self.assertEqual(result.detections.iloc[0]["Güven (%)"], 87.5)
        self.assertIn("Çatlak", result.summary)
        self.assertFalse(np.all(result.annotated_image == 255))

    def test_empty_detection_summary_is_clear(self) -> None:
        import pandas as pd

        summary = build_analysis_summary(
            pd.DataFrame(columns=["Kusur sınıfı", "Güven (%)"])
        )
        self.assertIn("kusur tespit edilmedi", summary)

    def test_invalid_confidence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            predict_image(FakeModel(), Image.new("RGB", (10, 10)), confidence=0.0)


if __name__ == "__main__":
    unittest.main()
