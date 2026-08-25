"""Kalite değerlendirmesinden tahmini panel performansı üretir."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from app.services.quality_service import QualityAssessment
from config import PERFORMANCE_QUALITY_LOSS_FACTOR


@dataclass(frozen=True)
class PerformanceEstimate:
    """Görüntü tabanlı performans ve güç tahmininin değişmez sonucu."""

    quality_score: float
    quality_grade: str
    performance_percent: float
    performance_loss_percent: float
    reference_power_w: float
    estimated_power_w: float


def estimate_production_performance(
    assessment: QualityAssessment,
    reference_power_w: float,
) -> PerformanceEstimate:
    """Kalite puanını açıklanabilir bir V1 performans tahminine dönüştürür."""
    try:
        nominal_power = float(reference_power_w)
    except (TypeError, ValueError) as exc:
        raise ValueError("Referans panel gücü pozitif ve sonlu bir sayı olmalıdır.") from exc
    if not isfinite(nominal_power) or nominal_power <= 0:
        raise ValueError("Referans panel gücü pozitif ve sonlu bir sayı olmalıdır.")

    quality_score = float(assessment.score)
    if not isfinite(quality_score) or not 0.0 <= quality_score <= 100.0:
        raise ValueError("Kalite puanı 0 ile 100 arasında olmalıdır.")

    performance_loss = (100.0 - quality_score) * PERFORMANCE_QUALITY_LOSS_FACTOR
    performance_percent = round(
        min(100.0, max(0.0, 100.0 - performance_loss)),
        1,
    )
    estimated_power_w = round(
        nominal_power * performance_percent / 100.0,
        1,
    )
    return PerformanceEstimate(
        quality_score=quality_score,
        quality_grade=assessment.grade,
        performance_percent=performance_percent,
        performance_loss_percent=round(100.0 - performance_percent, 1),
        reference_power_w=nominal_power,
        estimated_power_w=estimated_power_w,
    )
