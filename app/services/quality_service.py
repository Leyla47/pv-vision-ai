"""Kusur tespitlerinden açıklanabilir kalite ve fiyat sonucu üretir."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import pandas as pd

from config import (
    DEFECT_SEVERITY_BY_NAME_TR,
    QUALITY_GRADE_LABELS,
    QUALITY_GRADE_THRESHOLDS,
    QUALITY_PRICE_COEFFICIENTS,
    SUPPORTED_PRICE_CURRENCIES,
)


DETAIL_COLUMNS = [
    "Kusur sınıfı",
    "Güven (%)",
    "Sol",
    "Üst",
    "Sağ",
    "Alt",
    "Kutu alanı (%)",
    "Önem ağırlığı",
]
SUMMARY_COLUMNS = [
    "Kusur sınıfı",
    "Tespit sayısı",
    "Birleşik alan (%)",
    "En yüksek güven (%)",
    "Önem ağırlığı",
]
REQUIRED_COLUMNS = {"Kusur sınıfı", "Güven (%)", "Sol", "Üst", "Sağ", "Alt"}


@dataclass(frozen=True)
class QualityAssessment:
    """Tek görüntünün puan, sınıf, alan ve fiyat değerlendirmesi."""

    score: float
    grade: str
    grade_label: str
    covered_area_percent: float
    area_penalty: float
    count_penalty: float
    price_coefficient: float
    reference_price: float | None
    suggested_price: float | None
    value_loss_percent: float
    value_loss_amount: float | None
    currency: str
    detailed_detections: pd.DataFrame
    class_summary: pd.DataFrame


def assess_quality(
    detections: pd.DataFrame,
    image_size: tuple[int, int],
    *,
    reference_price: float | None = None,
    currency: str = "TRY",
) -> QualityAssessment:
    """YOLO kutularını açıklanabilir bir 0-100 kalite sonucuna dönüştürür."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Görüntü boyutları pozitif olmalıdır.")
    if not REQUIRED_COLUMNS.issubset(detections.columns):
        missing = ", ".join(sorted(REQUIRED_COLUMNS - set(detections.columns)))
        raise ValueError(f"Tespit tablosunda zorunlu sütunlar eksik: {missing}")
    if currency not in SUPPORTED_PRICE_CURRENCIES:
        raise ValueError(f"Desteklenmeyen para birimi: {currency}")
    if reference_price is not None and (
        not isfinite(float(reference_price)) or float(reference_price) <= 0
    ):
        raise ValueError("Referans fiyat pozitif ve sonlu bir sayı olmalıdır.")

    image_area = float(width * height)
    detailed = detections.copy()
    box_areas: list[float] = []
    severities: list[float] = []
    rectangles: list[tuple[float, float, float, float]] = []
    rectangles_by_class: dict[str, list[tuple[float, float, float, float]]] = {}
    raw_area_penalty = 0.0
    raw_count_penalty = 0.0

    for row in detailed.to_dict(orient="records"):
        rectangle = _clip_rectangle(row, width, height)
        class_name = str(row["Kusur sınıfı"])
        severity = float(DEFECT_SEVERITY_BY_NAME_TR.get(class_name, 1.0))
        confidence = _clamp(_safe_float(row["Güven (%)"]) / 100.0, 0.0, 1.0)
        confidence_effect = 0.5 + 0.5 * confidence
        area_percent = 0.0
        if rectangle is not None:
            left, top, right, bottom = rectangle
            area_percent = ((right - left) * (bottom - top) / image_area) * 100.0
            rectangles.append(rectangle)
            rectangles_by_class.setdefault(class_name, []).append(rectangle)

        box_areas.append(round(area_percent, 3))
        severities.append(severity)
        raw_area_penalty += area_percent * severity * confidence_effect * 0.55
        raw_count_penalty += 1.25 * severity * confidence_effect

    detailed["Kutu alanı (%)"] = box_areas
    detailed["Önem ağırlığı"] = severities
    detailed = detailed.reindex(columns=DETAIL_COLUMNS)

    area_penalty = min(70.0, raw_area_penalty)
    count_penalty = min(25.0, raw_count_penalty)
    score = round(max(0.0, 100.0 - area_penalty - count_penalty), 1)
    grade = quality_grade(score)
    coefficient = QUALITY_PRICE_COEFFICIENTS[grade]
    normalized_price = float(reference_price) if reference_price is not None else None
    suggested_price = (
        round(normalized_price * coefficient, 2) if normalized_price is not None else None
    )
    value_loss_percent = round((1.0 - coefficient) * 100.0, 1)
    value_loss_amount = (
        round(normalized_price - suggested_price, 2)
        if normalized_price is not None and suggested_price is not None
        else None
    )

    covered_area = (_rectangle_union_area(rectangles) / image_area) * 100.0
    class_summary = _build_class_summary(detailed, rectangles_by_class, image_area)
    return QualityAssessment(
        score=score,
        grade=grade,
        grade_label=QUALITY_GRADE_LABELS[grade],
        covered_area_percent=round(covered_area, 3),
        area_penalty=round(area_penalty, 3),
        count_penalty=round(count_penalty, 3),
        price_coefficient=coefficient,
        reference_price=normalized_price,
        suggested_price=suggested_price,
        value_loss_percent=value_loss_percent,
        value_loss_amount=value_loss_amount,
        currency=currency,
        detailed_detections=detailed,
        class_summary=class_summary,
    )


def quality_grade(score: float) -> str:
    """0-100 kalite puanını A, B veya C sınıfına dönüştürür."""
    numeric_score = float(score)
    if not isfinite(numeric_score) or not 0.0 <= numeric_score <= 100.0:
        raise ValueError("Kalite puanı 0 ile 100 arasında olmalıdır.")
    if numeric_score >= QUALITY_GRADE_THRESHOLDS["A"]:
        return "A"
    if numeric_score >= QUALITY_GRADE_THRESHOLDS["B"]:
        return "B"
    return "C"


def _build_class_summary(
    detailed: pd.DataFrame,
    rectangles_by_class: dict[str, list[tuple[float, float, float, float]]],
    image_area: float,
) -> pd.DataFrame:
    if detailed.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows: list[dict[str, object]] = []
    for class_name, group in detailed.groupby("Kusur sınıfı", sort=False):
        class_area = _rectangle_union_area(rectangles_by_class.get(str(class_name), []))
        rows.append(
            {
                "Kusur sınıfı": class_name,
                "Tespit sayısı": int(len(group)),
                "Birleşik alan (%)": round((class_area / image_area) * 100.0, 3),
                "En yüksek güven (%)": round(float(group["Güven (%)"].max()), 1),
                "Önem ağırlığı": round(float(group["Önem ağırlığı"].max()), 1),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).sort_values(
        ["Tespit sayısı", "Birleşik alan (%)"], ascending=False, ignore_index=True
    )


def _clip_rectangle(
    row: dict[str, object], width: int, height: int
) -> tuple[float, float, float, float] | None:
    values = [_safe_float(row[name]) for name in ("Sol", "Üst", "Sağ", "Alt")]
    if not all(isfinite(value) for value in values):
        return None
    left, top, right, bottom = values
    left = _clamp(left, 0.0, float(width))
    right = _clamp(right, 0.0, float(width))
    top = _clamp(top, 0.0, float(height))
    bottom = _clamp(bottom, 0.0, float(height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _rectangle_union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    """Eksen hizalı dikdörtgenlerin çakışmaları tek sayan tam birleşim alanı."""
    if not rectangles:
        return 0.0
    x_values = sorted({coordinate for rectangle in rectangles for coordinate in (rectangle[0], rectangle[2])})
    total_area = 0.0
    for left, right in zip(x_values, x_values[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (top, bottom)
            for rect_left, top, rect_right, bottom in rectangles
            if rect_left < right and rect_right > left
        )
        if not intervals:
            continue
        covered_y = 0.0
        current_top, current_bottom = intervals[0]
        for top, bottom in intervals[1:]:
            if top <= current_bottom:
                current_bottom = max(current_bottom, bottom)
            else:
                covered_y += current_bottom - current_top
                current_top, current_bottom = top, bottom
        covered_y += current_bottom - current_top
        total_area += (right - left) * covered_y
    return total_area


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))
