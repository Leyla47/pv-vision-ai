"""Panel sağlık skoru ve risk seviyesi testleri."""

from __future__ import annotations

import unittest

import pandas as pd

from app.services.health_service import (
    assess_panel_health,
    health_risk_level,
    health_status,
)
from app.services.quality_service import assess_quality


def detections(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["Kusur sınıfı", "Güven (%)", "Sol", "Üst", "Sağ", "Alt"],
    )


class HealthServiceTests(unittest.TestCase):
    def test_empty_detection_is_fully_healthy_and_low_risk(self) -> None:
        quality = assess_quality(detections([]), (100, 100))

        result = assess_panel_health(quality)

        self.assertEqual(result.score, 100.0)
        self.assertEqual(result.status, "Çok İyi")
        self.assertEqual(result.risk_level, "Düşük")
        self.assertEqual(result.class_effect, 0.0)

    def test_health_status_boundaries(self) -> None:
        cases = [
            (85.0, "Çok İyi"),
            (84.9, "İyi"),
            (70.0, "İyi"),
            (69.9, "Orta"),
            (50.0, "Orta"),
            (49.9, "Kritik"),
        ]
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(health_status(score), expected)

    def test_balanced_risk_rules(self) -> None:
        self.assertEqual(health_risk_level(90.0, 0, 0.0), "Düşük")
        self.assertEqual(health_risk_level(90.0, 1, 1.0), "Orta")
        self.assertEqual(health_risk_level(90.0, 2, 1.0), "Yüksek")
        self.assertEqual(health_risk_level(90.0, 1, 10.0), "Yüksek")
        self.assertEqual(health_risk_level(49.9, 0, 0.0), "Yüksek")

    def test_critical_defect_reduces_health_more_than_scratch(self) -> None:
        common = {"Güven (%)": 90, "Sol": 10, "Üst": 10, "Sağ": 40, "Alt": 40}
        crack = assess_panel_health(
            assess_quality(detections([{**common, "Kusur sınıfı": "Çatlak"}]), (100, 100))
        )
        scratch = assess_panel_health(
            assess_quality(detections([{**common, "Kusur sınıfı": "Çizik"}]), (100, 100))
        )

        self.assertLess(crack.score, scratch.score)
        self.assertEqual(crack.risk_level, "Orta")

    def test_breakdown_matches_health_formula(self) -> None:
        quality = assess_quality(
            detections(
                [
                    {"Kusur sınıfı": "Çatlak", "Güven (%)": 90, "Sol": 0, "Üst": 0, "Sağ": 20, "Alt": 20},
                    {"Kusur sınıfı": "Çizik", "Güven (%)": 80, "Sol": 30, "Üst": 30, "Sağ": 40, "Alt": 40},
                ]
            ),
            (100, 100),
        )

        result = assess_panel_health(quality)

        expected = round(
            max(0.0, 100.0 - result.quality_effect - result.area_effect - result.class_effect),
            1,
        )
        self.assertEqual(result.score, expected)
        self.assertAlmostEqual(
            sum(impact.points for impact in result.class_impacts),
            result.class_effect,
            places=2,
        )


if __name__ == "__main__":
    unittest.main()
