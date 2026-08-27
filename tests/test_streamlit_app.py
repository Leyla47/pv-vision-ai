"""Streamlit uygulamasının açılış testi."""

from __future__ import annotations

import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

from app import app as streamlit_app
from app.services.health_service import assess_panel_health
from app.services.maintenance_service import build_maintenance_plan
from app.services.model_service import DetectionResult
from app.services.performance_service import estimate_production_performance
from app.services.quality_service import assess_quality
from config import (
    APP_DIR,
    DEFAULT_REFERENCE_PRICE,
    MODEL_WEIGHTS_PATH,
    PROJECT_ROOT,
    REPORTS_DIR,
)


class StreamlitAppTests(unittest.TestCase):
    def test_app_import_defers_ultralytics_and_torch(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import app.app; "
                    "assert 'ultralytics' not in sys.modules; "
                    "assert 'torch' not in sys.modules"
                ),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_app_opens_without_exception(self) -> None:
        app = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30)
        app.run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.slider[0].label, "Güven eşiği")
        self.assertEqual(app.number_input[0].label, "Referans panel gücü (W)")
        self.assertEqual(app.number_input[0].value, 550)
        self.assertEqual(app.toggle[0].label, "Fiyat önerisini hesapla")
        self.assertEqual(app.button[0].label, "Analiz Et")
        self.assertTrue(app.button[0].disabled)
        rendered_html = "\n".join(markdown.value for markdown in app.markdown)
        self.assertIn("PV Vision AI", rendered_html)
        self.assertIn("Güneş Hücrelerinde", rendered_html)
        self.assertIn('id="analiz"', rendered_html)
        self.assertIn('href="#model"', rendered_html)
        if (REPORTS_DIR / "dataset_class_distribution.json").exists():
            self.assertTrue(
                any(
                    expander.label == "Model kapsamı ve veri sınırlılıkları"
                    for expander in app.expander
                )
            )

    def test_hero_uses_optimized_webp_asset(self) -> None:
        self.assertEqual(streamlit_app.HERO_IMAGE_PATH.suffix, ".webp")
        self.assertLessEqual(streamlit_app.HERO_IMAGE_PATH.stat().st_size, 350_000)
        with Image.open(streamlit_app.HERO_IMAGE_PATH) as hero:
            self.assertEqual(hero.size, (1774, 887))
        self.assertTrue(streamlit_app._hero_data_uri().startswith("data:image/webp;base64,"))

    def test_candidate_and_final_stages_have_distinct_status_tones(self) -> None:
        cases = [
            ("candidate", "Aday model", "info"),
            ("final", "Final model", "success"),
        ]
        for stage, expected_label, expected_tone in cases:
            with self.subTest(stage=stage), patch.object(
                streamlit_app,
                "load_model_metadata",
                return_value={
                    "stage": stage,
                    "completed_epochs": 10,
                    "target_epochs": 10,
                    "best_epoch": 8,
                    "metrics": {"map50": 0.5, "map50_95": 0.3},
                },
            ):
                view = streamlit_app._get_model_view()
                self.assertEqual(view["stage_label"], expected_label)
                self.assertEqual(view["tone"], expected_tone)

    def test_optional_price_controls_appear_when_enabled(self) -> None:
        app = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30)
        app.run()

        app.toggle[0].set_value(True)
        app.run()

        self.assertEqual(app.number_input[0].label, "Referans panel gücü (W)")
        self.assertEqual(app.number_input[1].label, "Referans panel fiyatı")
        self.assertEqual(app.number_input[1].value, DEFAULT_REFERENCE_PRICE)
        self.assertEqual(app.selectbox[0].label, "Para birimi")
        self.assertEqual(app.selectbox[0].options, ["TRY", "USD", "EUR"])

    @unittest.skipUnless(MODEL_WEIGHTS_PATH.exists(), "best.pt bulunmuyor.")
    def test_real_model_is_reused_from_resource_cache(self) -> None:
        model_version = MODEL_WEIGHTS_PATH.stat().st_mtime_ns
        streamlit_app.get_cached_model.clear()
        try:
            first = streamlit_app.get_cached_model(str(MODEL_WEIGHTS_PATH), model_version)
            second = streamlit_app.get_cached_model(str(MODEL_WEIGHTS_PATH), model_version)
        finally:
            streamlit_app.get_cached_model.clear()

        self.assertIs(first, second)

    def test_text_report_contains_health_maintenance_and_loss_results(self) -> None:
        detections = pd.DataFrame(
            [
                {
                    "Kusur sınıfı": "Çatlak",
                    "Güven (%)": 90,
                    "Sol": 0,
                    "Üst": 0,
                    "Sağ": 20,
                    "Alt": 20,
                }
            ]
        )
        result = DetectionResult(
            annotated_image=np.zeros((100, 100, 3), dtype=np.uint8),
            detections=detections,
            summary="Bir çatlak tespit edildi.",
        )
        quality = assess_quality(detections, (100, 100), reference_price=1000)
        performance = estimate_production_performance(quality, 550)
        health = assess_panel_health(quality)
        maintenance = build_maintenance_plan(quality, health)

        report = streamlit_app._build_text_report(
            result,
            quality,
            performance,
            health,
            maintenance,
        )

        self.assertIn("Panel sağlık skoru:", report)
        self.assertIn("Risk seviyesi:", report)
        self.assertIn("Tahmini performans kaybı:", report)
        self.assertIn("Tahmini değer kaybı:", report)
        self.assertIn("Bakım ve kontrol önerileri:", report)
        self.assertIn(streamlit_app.ANALYSIS_DISCLAIMER, report)


if __name__ == "__main__":
    unittest.main()
