"""Kalite puanı, alan ve fiyat önerisi testleri."""

from __future__ import annotations

import unittest

import pandas as pd

from app.services.quality_service import assess_quality, quality_grade


def detections(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["Kusur sınıfı", "Güven (%)", "Sol", "Üst", "Sağ", "Alt"],
    )


class QualityServiceTests(unittest.TestCase):
    def test_empty_detection_is_full_quality_without_price(self) -> None:
        result = assess_quality(detections([]), (100, 100))

        self.assertEqual(result.score, 100.0)
        self.assertEqual(result.grade, "A")
        self.assertEqual(result.covered_area_percent, 0.0)
        self.assertIsNone(result.suggested_price)
        self.assertEqual(result.value_loss_percent, 5.0)
        self.assertIsNone(result.value_loss_amount)

    def test_overlapping_and_out_of_bounds_boxes_use_clipped_union_area(self) -> None:
        result = assess_quality(
            detections(
                [
                    {"Kusur sınıfı": "Çatlak", "Güven (%)": 100, "Sol": -10, "Üst": -10, "Sağ": 60, "Alt": 60},
                    {"Kusur sınıfı": "Çatlak", "Güven (%)": 100, "Sol": 40, "Üst": 40, "Sağ": 110, "Alt": 110},
                ]
            ),
            (100, 100),
        )

        self.assertEqual(result.covered_area_percent, 68.0)
        self.assertEqual(result.detailed_detections["Kutu alanı (%)"].tolist(), [36.0, 36.0])

    def test_invalid_rectangle_is_safe_and_has_zero_area(self) -> None:
        result = assess_quality(
            detections(
                [{"Kusur sınıfı": "Çizik", "Güven (%)": 80, "Sol": 70, "Üst": 20, "Sağ": 10, "Alt": 50}]
            ),
            (100, 100),
        )

        self.assertEqual(result.covered_area_percent, 0.0)
        self.assertEqual(result.detailed_detections.iloc[0]["Kutu alanı (%)"], 0.0)

    def test_critical_defect_penalizes_more_than_low_weight_defect(self) -> None:
        common = {"Güven (%)": 90, "Sol": 10, "Üst": 10, "Sağ": 40, "Alt": 40}
        crack = assess_quality(detections([{**common, "Kusur sınıfı": "Çatlak"}]), (100, 100))
        scratch = assess_quality(detections([{**common, "Kusur sınıfı": "Çizik"}]), (100, 100))

        self.assertLess(crack.score, scratch.score)

    def test_grade_boundaries(self) -> None:
        self.assertEqual(quality_grade(85), "A")
        self.assertEqual(quality_grade(84.9), "B")
        self.assertEqual(quality_grade(60), "B")
        self.assertEqual(quality_grade(59.9), "C")

    def test_grade_price_coefficients_and_currency(self) -> None:
        cases = [("Parmak izi kusuru", 0.95), ("Siyah çekirdek", 0.75), ("Çatlak", 0.45)]
        boxes = [(0, 0, 5, 5), (0, 0, 45, 45), (0, 0, 100, 100)]
        for (class_name, coefficient), (left, top, right, bottom) in zip(cases, boxes):
            with self.subTest(class_name=class_name):
                result = assess_quality(
                    detections(
                        [{"Kusur sınıfı": class_name, "Güven (%)": 100, "Sol": left, "Üst": top, "Sağ": right, "Alt": bottom}]
                    ),
                    (100, 100),
                    reference_price=1000,
                    currency="EUR",
                )
                self.assertEqual(result.price_coefficient, coefficient)
                self.assertEqual(result.suggested_price, 1000 * coefficient)
                self.assertAlmostEqual(
                    result.value_loss_percent,
                    (1 - coefficient) * 100,
                )
                self.assertEqual(result.value_loss_amount, 1000 - result.suggested_price)
                self.assertEqual(result.currency, "EUR")


if __name__ == "__main__":
    unittest.main()
