"""Hazırlanmış PVEL-AD veri setinin sınıf dağılımını raporlar."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    CLASS_ID_TO_NAME_TR,
    NUM_CLASSES,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ClassDistribution:
    """Tek kusur sınıfının split bazlı örnek dağılımı."""

    class_id: int
    class_name: str
    train_objects: int
    train_images: int
    val_objects: int
    val_images: int
    test_objects: int
    test_images: int
    training_support: str
    validation_coverage: str


def analyze_dataset(labels_root: Path) -> list[ClassDistribution]:
    """YOLO label dosyalarından nesne ve görüntü bazlı sınıf sayılarını çıkarır."""
    object_counts: dict[str, Counter[int]] = {}
    image_counts: dict[str, Counter[int]] = {}

    for split in SPLITS:
        label_dir = labels_root / split
        if not label_dir.exists():
            raise FileNotFoundError(f"Label klasörü bulunamadı: {label_dir}")

        split_objects: Counter[int] = Counter()
        split_images: Counter[int] = Counter()
        label_paths = sorted(label_dir.glob("*.txt"))
        if not label_paths:
            raise ValueError(f"Label klasörü boş: {label_dir}")

        for label_path in label_paths:
            classes_in_image: set[int] = set()
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Geçersiz YOLO satırı: {label_path}:{line_number}")
                class_id = int(parts[0])
                if class_id not in CLASS_ID_TO_NAME_TR:
                    raise ValueError(f"Geçersiz sınıf ID: {label_path}:{line_number} -> {class_id}")
                split_objects[class_id] += 1
                classes_in_image.add(class_id)
            split_images.update(classes_in_image)

        object_counts[split] = split_objects
        image_counts[split] = split_images

    return [
        ClassDistribution(
            class_id=class_id,
            class_name=CLASS_ID_TO_NAME_TR[class_id],
            train_objects=object_counts["train"][class_id],
            train_images=image_counts["train"][class_id],
            val_objects=object_counts["val"][class_id],
            val_images=image_counts["val"][class_id],
            test_objects=object_counts["test"][class_id],
            test_images=image_counts["test"][class_id],
            training_support=_training_support(object_counts["train"][class_id]),
            validation_coverage=_validation_coverage(object_counts["val"][class_id]),
        )
        for class_id in range(NUM_CLASSES)
    ]


def write_dataset_reports(
    distributions: list[ClassDistribution],
    output_dir: Path = REPORTS_DIR,
) -> tuple[Path, Path, Path]:
    """Dağılım sonuçlarını CSV, JSON ve Türkçe Markdown olarak kaydeder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dataset_class_distribution.csv"
    json_path = output_dir / "dataset_class_distribution.json"
    markdown_path = output_dir / "dataset_analysis.md"

    csv_rows = [
        {
            "Sınıf ID": item.class_id,
            "Kusur sınıfı": item.class_name,
            "Train kutu": item.train_objects,
            "Train görüntü": item.train_images,
            "Validation kutu": item.val_objects,
            "Validation görüntü": item.val_images,
            "Test kutu": item.test_objects,
            "Test görüntü": item.test_images,
            "Eğitim desteği": item.training_support,
            "Validation kapsaması": item.validation_coverage,
        }
        for item in distributions
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    payload = {
        "class_count": len(distributions),
        "totals": {
            split: sum(getattr(item, f"{split}_objects") for item in distributions)
            for split in SPLITS
        },
        "classes": [asdict(item) for item in distributions],
        "warnings": build_dataset_warnings(distributions),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    warnings = build_dataset_warnings(distributions)
    lines = [
        "# PV Vision AI Veri Seti Analizi",
        "",
        "Bu rapor her sınıf için annotation kutusu ve sınıfı içeren görüntü sayısını gösterir.",
        "",
        "## Genel Toplamlar",
        "",
        f"- Eğitim: {payload['totals']['train']} kusur kutusu",
        f"- Validation: {payload['totals']['val']} kusur kutusu",
        f"- Test: {payload['totals']['test']} kusur kutusu",
        "",
        "## Sınıf Dağılımı",
        "",
        "| Kusur sınıfı | Train kutu | Val kutu | Test kutu | Eğitim desteği | Validation kapsaması |",
        "|---|---:|---:|---:|---|---|",
    ]
    lines.extend(
        (
            f"| {item.class_name} | {item.train_objects} | {item.val_objects} | "
            f"{item.test_objects} | {item.training_support} | {item.validation_coverage} |"
        )
        for item in distributions
    )
    lines.extend(["", "## Uyarılar", ""])
    lines.extend(f"- {warning}" for warning in warnings)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, markdown_path


def build_dataset_warnings(distributions: list[ClassDistribution]) -> list[str]:
    """Model kalitesini etkileyebilecek sınıf kapsamı uyarılarını üretir."""
    warnings: list[str] = []
    for item in distributions:
        if item.train_objects < 20:
            warnings.append(
                f"{item.class_name}: eğitim setinde yalnızca {item.train_objects} kutu var; "
                "bu sınıf için güvenilir model performansı beklenmemelidir."
            )
        if item.val_objects == 0:
            warnings.append(
                f"{item.class_name}: validation örneği yok; bu sınıf validation metrikleriyle ölçülemez."
            )
        elif item.val_objects < 5:
            warnings.append(
                f"{item.class_name}: validation setinde yalnızca {item.val_objects} kutu var; "
                "sınıf metriği istatistiksel olarak güvenilir değildir."
            )
    return warnings or ["Kritik sınıf dağılımı uyarısı bulunmadı."]


def _training_support(object_count: int) -> str:
    if object_count >= 100:
        return "Yeterli"
    if object_count >= 20:
        return "Sınırlı"
    if object_count > 0:
        return "Çok sınırlı"
    return "Yok"


def _validation_coverage(object_count: int) -> str:
    if object_count >= 20:
        return "Kapsanıyor"
    if object_count >= 5:
        return "Sınırlı"
    if object_count > 0:
        return "Güvenilmez"
    return "Değerlendirilemez"


def main() -> None:
    parser = argparse.ArgumentParser(description="PVEL-AD sınıf dağılımını raporlar.")
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROCESSED_DATA_DIR / "labels",
        help="train/val/test label klasörlerini içeren kök yol",
    )
    parser.add_argument("--output", type=Path, default=REPORTS_DIR, help="Rapor çıktı klasörü")
    args = parser.parse_args()

    distributions = analyze_dataset(args.labels)
    csv_path, json_path, markdown_path = write_dataset_reports(distributions, args.output)
    warnings = build_dataset_warnings(distributions)

    print("PV Vision AI veri seti sınıf analizi")
    print("------------------------------------")
    for item in distributions:
        print(
            f"- {item.class_name}: train={item.train_objects}, "
            f"val={item.val_objects}, test={item.test_objects} "
            f"({item.training_support}; {item.validation_coverage})"
        )
    print(f"\nUyarı sayısı: {len(warnings)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Rapor: {markdown_path}")


if __name__ == "__main__":
    main()
