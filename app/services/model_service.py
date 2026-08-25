"""YOLO model yükleme ve tahmin servisi."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from config import CLASS_ID_TO_NAME_TR, MODEL_WEIGHTS_PATH


@dataclass(frozen=True)
class DetectionResult:
    """Tek görüntü analizi sonucunu taşır."""

    annotated_image: np.ndarray
    detections: pd.DataFrame
    summary: str


@dataclass(frozen=True)
class Detection:
    """Modelden alınan tek kusur tespitini temsil eder."""

    class_id: int
    class_name: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


BOX_COLORS = (
    (220, 53, 69),
    (0, 123, 255),
    (40, 167, 69),
    (255, 193, 7),
    (111, 66, 193),
    (23, 162, 184),
    (232, 62, 140),
    (253, 126, 20),
    (32, 120, 96),
    (102, 16, 242),
    (13, 110, 253),
    (108, 117, 125),
)


def load_model(model_path: Path = MODEL_WEIGHTS_PATH) -> YOLO:
    """Eğitilmiş YOLO modelini yükler."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    return YOLO(str(model_path))


def predict_image(model: YOLO, image: Image.Image, confidence: float = 0.25) -> DetectionResult:
    """Pillow görüntüsünü YOLO ile analiz eder."""
    if not 0.0 < confidence <= 1.0:
        raise ValueError("Güven eşiği 0 ile 1 arasında olmalıdır.")

    rgb_image = image.convert("RGB")
    results = model.predict(source=rgb_image, conf=confidence, verbose=False)

    result = results[0]
    extracted_detections = _extract_detections(result)
    detections = _detections_to_dataframe(extracted_detections)
    annotated_image = _draw_turkish_annotations(rgb_image, extracted_detections)
    summary = build_analysis_summary(detections)
    return DetectionResult(annotated_image=annotated_image, detections=detections, summary=summary)


def build_analysis_summary(detections: pd.DataFrame) -> str:
    """Tespit tablosundan kısa Türkçe analiz özeti üretir."""
    if detections.empty:
        return "Bu görüntüde seçilen güven eşiğinin üzerinde kusur tespit edilmedi."

    defect_count = len(detections)
    unique_classes = detections["Kusur sınıfı"].nunique()
    most_confident = detections.sort_values("Güven (%)", ascending=False).iloc[0]
    return (
        f"Toplam {defect_count} kusur tespit edildi. "
        f"Tespitler {unique_classes} farklı kusur sınıfına ait. "
        f"En yüksek güvenli tespit: {most_confident['Kusur sınıfı']} "
        f"(%{most_confident['Güven (%)']:.1f})."
    )


def _extract_detections(result: object) -> list[Detection]:
    detections: list[Detection] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy_values = boxes.xyxy.cpu().numpy()
    confidence_values = boxes.conf.cpu().numpy()
    class_values = boxes.cls.cpu().numpy().astype(int)

    for class_id, confidence, box in zip(class_values, confidence_values, xyxy_values):
        detections.append(
            Detection(
                class_id=class_id,
                class_name=CLASS_ID_TO_NAME_TR.get(class_id, f"Bilinmeyen sınıf ({class_id})"),
                confidence=float(confidence),
                xmin=float(box[0]),
                ymin=float(box[1]),
                xmax=float(box[2]),
                ymax=float(box[3]),
            )
        )
    return detections


def _detections_to_dataframe(detections: list[Detection]) -> pd.DataFrame:
    columns = ["Kusur sınıfı", "Güven (%)", "Sol", "Üst", "Sağ", "Alt"]
    rows = [
        {
            "Kusur sınıfı": detection.class_name,
            "Güven (%)": round(detection.confidence * 100, 1),
            "Sol": round(detection.xmin, 1),
            "Üst": round(detection.ymin, 1),
            "Sağ": round(detection.xmax, 1),
            "Alt": round(detection.ymax, 1),
        }
        for detection in detections
    ]
    return pd.DataFrame(rows, columns=columns)


def _draw_turkish_annotations(image: Image.Image, detections: list[Detection]) -> np.ndarray:
    """Tespit kutularını ve Türkçe sınıf etiketlerini görüntünün üzerine çizer."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size
    line_width = max(2, round(min(width, height) / 220))
    font_size = max(12, min(28, round(min(width, height) / 24)))

    for detection in detections:
        color = BOX_COLORS[detection.class_id % len(BOX_COLORS)]
        left = max(0, min(width - 1, round(detection.xmin)))
        top = max(0, min(height - 1, round(detection.ymin)))
        right = max(left + 1, min(width, round(detection.xmax)))
        bottom = max(top + 1, min(height, round(detection.ymax)))
        draw.rectangle((left, top, right, bottom), outline=color, width=line_width)

        label = f"{detection.class_name} %{detection.confidence * 100:.1f}"
        font = _fit_annotation_font(label, font_size, max(1, width - 2 * line_width))
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        padding = max(3, line_width)
        label_left = min(left, max(0, width - text_width - 2 * padding))
        label_top = top - text_height - 2 * padding
        if label_top < 0:
            label_top = min(height - text_height - 2 * padding, top + line_width)

        draw.rectangle(
            (
                label_left,
                label_top,
                label_left + text_width + 2 * padding,
                label_top + text_height + 2 * padding,
            ),
            fill=color,
        )
        draw.text(
            (label_left + padding, label_top + padding - text_box[1]),
            label,
            fill=(255, 255, 255),
            font=font,
        )

    return np.asarray(annotated)[:, :, ::-1].copy()


def _fit_annotation_font(
    label: str,
    preferred_size: int,
    available_width: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = preferred_size
    while size >= 6:
        font = _load_annotation_font(size)
        text_box = font.getbbox(label)
        if text_box[2] - text_box[0] <= available_width:
            return font
        size -= 1
    return _load_annotation_font(6)


@lru_cache(maxsize=32)
def _load_annotation_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[Path] = []
    matplotlib_spec = find_spec("matplotlib")
    if matplotlib_spec and matplotlib_spec.origin:
        candidates.append(
            Path(matplotlib_spec.origin).parent
            / "mpl-data"
            / "fonts"
            / "ttf"
            / "DejaVuSans.ttf"
        )
    candidates.extend(
        [
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    for font_path in candidates:
        if font_path.exists():
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()
