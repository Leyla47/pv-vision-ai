"""Risk ve kusur türüne göre bakım önerisi testleri."""

from __future__ import annotations

import unittest

import pandas as pd

from app.services.health_service import assess_panel_health
from app.services.maintenance_service import build_maintenance_plan
from app.services.quality_service import assess_quality


def detections(class_names: list[str]) -> pd.DataFrame:
    rows = [
        {
            "Kusur sınıfı": class_name,
            "Güven (%)": 90,
            "Sol": index * 10,
            "Üst": 0,
            "Sağ": index * 10 + 8,
            "Alt": 8,
        }
        for index, class_name in enumerate(class_names)
    ]
    return pd.DataFrame(
        rows,
        columns=["Kusur sınıfı", "Güven (%)", "Sol", "Üst", "Sağ", "Alt"],
    )


class MaintenanceServiceTests(unittest.TestCase):
    def test_low_risk_panel_gets_routine_advice(self) -> None:
        quality = assess_quality(detections([]), (100, 100))
        health = assess_panel_health(quality)

        plan = build_maintenance_plan(quality, health)

        self.assertEqual(plan.risk_level, "Düşük")
        self.assertIn("Rutin görsel kontrol ve planlı bakım yeterlidir.", plan.recommendations)

    def test_class_recommendations_are_limited_and_deduplicated(self) -> None:
        quality = assess_quality(
            detections(["Çatlak", "Yıldız çatlağı", "Parçalanma", "Köşe kusuru"]),
            (100, 100),
        )
        health = assess_panel_health(quality)

        plan = build_maintenance_plan(quality, health)

        self.assertLessEqual(len(plan.considered_classes), 3)
        self.assertEqual(len(plan.recommendations), len(set(plan.recommendations)))
        self.assertTrue(any("çatlak tespit edilmiştir" in item for item in plan.recommendations))

    def test_black_core_gets_thermal_and_iv_advice(self) -> None:
        quality = assess_quality(detections(["Siyah çekirdek"]), (100, 100))
        health = assess_panel_health(quality)

        plan = build_maintenance_plan(quality, health)

        self.assertTrue(
            any("sıcak nokta" in item and "IV performansı" in item for item in plan.recommendations)
        )


if __name__ == "__main__":
    unittest.main()
