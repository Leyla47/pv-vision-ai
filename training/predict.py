"""Eğitilmiş YOLO modeliyle görüntü analizi ve çıktı kaydetme aracı."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.model_service import load_model, predict_image  # noqa: E402
from config import IMAGE_EXTENSIONS, MODEL_WEIGHTS_PATH, PREDICTIONS_DIR  # noqa: E402


def collect_image_paths(source: Path) -> list[Path]:
    """Tek görüntü veya klasörden desteklenen görüntüleri toplar."""
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Desteklenmeyen görüntü uzantısı: {source.suffix}")
        return [source]

    if source.is_dir():
        image_paths = [
            path
            for path in sorted(source.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_paths:
            raise ValueError(f"Klasörde desteklenen görüntü bulunamadı: {source}")
        return image_paths

    raise FileNotFoundError(f"Görüntü veya klasör bulunamadı: {source}")


def save_prediction_outputs(image_path: Path, output_dir: Path, annotated_image, detections, summary: str) -> None:
    """Tek görüntü için görsel, CSV ve özet çıktıları kaydeder."""
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / f"{image_path.stem}_annotated.jpg"
    csv_path = output_dir / f"{image_path.stem}_detections.csv"
    summary_path = output_dir / f"{image_path.stem}_summary.txt"

    cv2.imwrite(str(annotated_path), annotated_image)
    detections.to_csv(csv_path, index=False, encoding="utf-8")
    summary_path.write_text(summary + "\n", encoding="utf-8")

    print(f"- {image_path.name}: {len(detections)} tespit")
    print(f"  Görsel: {annotated_path}")
    print(f"  CSV: {csv_path}")
    print(f"  Özet: {summary_path}")


def run_prediction(source: Path, model_path: Path, output_dir: Path, confidence: float) -> None:
    """Kaynak görüntüler üzerinde tahmin çalıştırır ve çıktıları kaydeder."""
    image_paths = collect_image_paths(source)
    model = load_model(model_path)

    print("PV Vision AI tahmin işlemi")
    print("-------------------------")
    print(f"Model: {model_path}")
    print(f"Kaynak: {source}")
    print(f"Görüntü sayısı: {len(image_paths)}")
    print(f"Güven eşiği: {confidence}")
    print(f"Çıktı klasörü: {output_dir}")

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        result = predict_image(model, image, confidence=confidence)
        save_prediction_outputs(
            image_path=image_path,
            output_dir=output_dir,
            annotated_image=result.annotated_image,
            detections=result.detections,
            summary=result.summary,
        )


def main() -> None:
    """Komut satırı giriş noktası."""
    parser = argparse.ArgumentParser(description="PV Vision AI modeliyle görüntü tahmini yapar.")
    parser.add_argument("--source", type=Path, required=True, help="Analiz edilecek görüntü veya klasör")
    parser.add_argument("--model", type=Path, default=MODEL_WEIGHTS_PATH, help="Kullanılacak best.pt yolu")
    parser.add_argument("--output", type=Path, default=PREDICTIONS_DIR, help="Tahmin çıktılarının kaydedileceği klasör")
    parser.add_argument("--conf", type=float, default=0.25, help="Güven eşiği")
    args = parser.parse_args()

    run_prediction(source=args.source, model_path=args.model, output_dir=args.output, confidence=args.conf)


if __name__ == "__main__":
    main()
