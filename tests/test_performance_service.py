"""Tahmini üretim performansı servisinin birim testleri."""

from __future__ import annotations

import unittest
from dataclasses import replace

import pandas as pd

from app.services.performance_service import estimate_production_performance
from app.services.quality_service import QualityAssessment, assess_quality


def assessment(score: float, grade: str = "A") -> QualityAssessment:
    empty_detections = pd.DataFrame(
        columns=["Kusur sınıfı", "Güven (%)", "Sol", "Üst", "Sağ", "Alt"]
    )
    base = assess_quality(empty_detections, (100, 100))
    return replace(base, score=score, grade=grade)


class PerformanceServiceTests(unittest.TestCase):
    def test_full_quality_has_full_performance(self) -> None:
        result = estimate_production_performance(assessment(100.0), 550)

        self.assertEqual(result.performance_percent, 100.0)
        self.assertEqual(result.performance_loss_percent, 0.0)
        self.assertEqual(result.estimated_power_w, 550.0)

    def test_planned_quality_example_rounds_to_one_decimal(self) -> None:
        result = estimate_production_performance(assessment(27.3, "C"), 550)

        self.assertEqual(result.performance_percent, 78.2)
        self.assertEqual(result.performance_loss_percent, 21.8)
        self.assertEqual(result.estimated_power_w, 430.1)

    def test_eighty_nine_percent_produces_expected_power(self) -> None:
        result = estimate_production_performance(
            assessment(63.33333333333333, "B"),
            550,
        )

        self.assertEqual(result.performance_percent, 89.0)
        self.assertEqual(result.estimated_power_w, 489.5)

    def test_invalid_reference_power_is_rejected(self) -> None:
        for value in (0, -1, float("inf"), float("nan"), "geçersiz"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                estimate_production_performance(assessment(90.0), value)  # type: ignore[arg-type]

    def test_quality_grade_is_preserved(self) -> None:
        for score, grade in ((90.0, "A"), (70.0, "B"), (40.0, "C")):
            with self.subTest(grade=grade):
                result = estimate_production_performance(assessment(score, grade), 550)
                self.assertEqual(result.quality_grade, grade)


if __name__ == "__main__":
    unittest.main()
