"""Kalite sonucundan açıklanabilir panel sağlık ve risk değerlendirmesi üretir."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from app.services.quality_service import QualityAssessment
from config import (
    CRITICAL_DEFECT_NAMES_TR,
    HEALTH_AREA_PENALTY_CAP,
    HEALTH_AREA_WEIGHT,
    HEALTH_BREAKDOWN_CLASS_LIMIT,
    HEALTH_CLASS_EFFECT_FACTOR,
    HEALTH_CLASS_PENALTY_CAP,
    HEALTH_HIGH_RISK_CRITICAL_AREA_PERCENT,
    HEALTH_HIGH_RISK_CRITICAL_COUNT,
    HEALTH_HIGH_RISK_SCORE,
    HEALTH_MEDIUM_RISK_SCORE,
    HEALTH_QUALITY_WEIGHT,
    HEALTH_STATUS_THRESHOLDS,
)


@dataclass(frozen=True)
class HealthImpact:
    """Sağlık skorundan düşülen açıklanabilir tek etki."""

    label: str
    points: float


@dataclass(frozen=True)
class HealthAssessment:
    """Panelin görüntü tabanlı sağlık ve risk sonucu."""

    score: float
    status: str
    risk_level: str
    quality_effect: float
    area_effect: float
    class_effect: float
    critical_defect_count: int
    critical_area_percent: float
    impacts: tuple[HealthImpact, ...]
    class_impacts: tuple[HealthImpact, ...]


def assess_panel_health(assessment: QualityAssessment) -> HealthAssessment:
    """Mevcut kalite ayrıntılarını birleşik ve açıklanabilir sağlık skoruna çevirir."""
    quality_score = float(assessment.score)
    covered_area = float(assessment.covered_area_percent)
    if not isfinite(quality_score) or not 0.0 <= quality_score <= 100.0:
        raise ValueError("Kalite puanı 0 ile 100 arasında olmalıdır.")
    if not isfinite(covered_area) or not 0.0 <= covered_area <= 100.0:
        raise ValueError("Kusurlu alan oranı 0 ile 100 arasında olmalıdır.")

    quality_effect = (100.0 - quality_score) * HEALTH_QUALITY_WEIGHT
    area_effect = min(HEALTH_AREA_PENALTY_CAP, covered_area * HEALTH_AREA_WEIGHT)
    raw_class_effects: dict[str, float] = {}
    for row in assessment.detailed_detections.to_dict(orient="records"):
        class_name = str(row.get("Kusur sınıfı", "Bilinmeyen kusur"))
        severity = _finite_or_default(row.get("Önem ağırlığı"), 1.0)
        confidence = _clamp(
            _finite_or_default(row.get("Güven (%)"), 0.0) / 100.0,
            0.0,
            1.0,
        )
        confidence_effect = 0.5 + 0.5 * confidence
        raw_class_effects[class_name] = raw_class_effects.get(class_name, 0.0) + (
            severity * confidence_effect * HEALTH_CLASS_EFFECT_FACTOR
        )

    raw_class_total = sum(raw_class_effects.values())
    class_effect = min(HEALTH_CLASS_PENALTY_CAP, raw_class_total)
    scale = class_effect / raw_class_total if raw_class_total > 0 else 0.0
    class_impacts = tuple(
        HealthImpact(label=class_name, points=round(points * scale, 3))
        for class_name, points in sorted(
            raw_class_effects.items(), key=lambda item: (-item[1], item[0])
        )
    )

    score = round(
        max(0.0, 100.0 - quality_effect - area_effect - class_effect),
        1,
    )
    critical_count = int(
        assessment.detailed_detections["Kusur sınıfı"]
        .isin(CRITICAL_DEFECT_NAMES_TR)
        .sum()
    )
    critical_area = round(
        min(
            100.0,
            float(
                assessment.class_summary.loc[
                    assessment.class_summary["Kusur sınıfı"].isin(CRITICAL_DEFECT_NAMES_TR),
                    "Birleşik alan (%)",
                ].sum()
            ),
        ),
        3,
    )

    return HealthAssessment(
        score=score,
        status=health_status(score),
        risk_level=health_risk_level(score, critical_count, critical_area),
        quality_effect=round(quality_effect, 3),
        area_effect=round(area_effect, 3),
        class_effect=round(class_effect, 3),
        critical_defect_count=critical_count,
        critical_area_percent=critical_area,
        impacts=_build_visible_impacts(quality_effect, area_effect, class_impacts),
        class_impacts=class_impacts,
    )


def health_status(score: float) -> str:
    """0-100 sağlık puanını kullanıcıya gösterilecek duruma dönüştürür."""
    numeric_score = float(score)
    if not isfinite(numeric_score) or not 0.0 <= numeric_score <= 100.0:
        raise ValueError("Sağlık puanı 0 ile 100 arasında olmalıdır.")
    for label, threshold in HEALTH_STATUS_THRESHOLDS.items():
        if numeric_score >= threshold:
            return label
    return "Kritik"


def health_risk_level(
    score: float,
    critical_defect_count: int,
    critical_area_percent: float,
) -> str:
    """Sağlık ve kritik kusur yükünden dengeli risk seviyesini belirler."""
    if (
        score < HEALTH_HIGH_RISK_SCORE
        or critical_defect_count >= HEALTH_HIGH_RISK_CRITICAL_COUNT
        or critical_area_percent >= HEALTH_HIGH_RISK_CRITICAL_AREA_PERCENT
    ):
        return "Yüksek"
    if score < HEALTH_MEDIUM_RISK_SCORE or critical_defect_count >= 1:
        return "Orta"
    return "Düşük"


def _build_visible_impacts(
    quality_effect: float,
    area_effect: float,
    class_impacts: tuple[HealthImpact, ...],
) -> tuple[HealthImpact, ...]:
    impacts = [
        HealthImpact("Kalite puanı etkisi", round(quality_effect, 3)),
        HealthImpact("Kusurlu alan etkisi", round(area_effect, 3)),
    ]
    visible_classes = class_impacts[:HEALTH_BREAKDOWN_CLASS_LIMIT]
    impacts.extend(
        HealthImpact(f"{impact.label} etkisi", impact.points)
        for impact in visible_classes
    )
    remaining_points = sum(
        impact.points for impact in class_impacts[HEALTH_BREAKDOWN_CLASS_LIMIT:]
    )
    if remaining_points > 0:
        impacts.append(HealthImpact("Diğer kusurlar", round(remaining_points, 3)))
    return tuple(impacts)


def _finite_or_default(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if isfinite(numeric) else default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))
