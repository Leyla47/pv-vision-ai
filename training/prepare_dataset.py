"""PVEL-AD veri setini YOLO nesne tespiti formatına hazırlar."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_YAML_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from training.convert_annotations import (  # noqa: E402
    build_class_mapping,
    convert_voc_xml_to_yolo,
    read_class_names,
)


@dataclass(frozen=True)
class RawDatasetPaths:
    """PVEL-AD ham veri yolları."""

    classes_file: Path
    trainval_images: Path
    trainval_annotations: Path
    test_images: Path
    test_annotations: Path


@dataclass(frozen=True)
class SplitItem:
    """Hazırlanacak tek görüntü ve annotation çifti."""

    image_path: Path
    annotation_path: Path


@dataclass
class SplitSummary:
    """Bir split için üretilen çıktı özeti."""

    image_count: int = 0
    label_count: int = 0
    object_count: int = 0


def get_raw_dataset_paths() -> RawDatasetPaths:
    """Beklenen PVEL-AD ham veri yollarını döndürür."""
    el2021_dir = RAW_DATA_DIR / "solar_cell_EL_image" / "PVELAD" / "EL2021"
    return RawDatasetPaths(
        classes_file=el2021_dir / "annotation_classes.txt",
        trainval_images=el2021_dir / "trainval" / "JPEGImages",
        trainval_annotations=el2021_dir / "trainval" / "Annotations",
        test_images=el2021_dir / "test" / "JPEGImages",
        test_annotations=RAW_DATA_DIR / "test_annotation" / "test",
    )


def validate_raw_dataset(paths: RawDatasetPaths) -> None:
    """Ham veri klasörlerinin ve dosya eşleşmelerinin hazır olduğunu kontrol eder."""
    required_paths = [
        paths.classes_file,
        paths.trainval_images,
        paths.trainval_annotations,
        paths.test_images,
        paths.test_annotations,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        formatted = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(f"Eksik ham veri yolu bulundu:\n{formatted}")

    _validate_image_annotation_pairs("trainval", paths.trainval_images, paths.trainval_annotations)
    _validate_image_annotation_pairs("test", paths.test_images, paths.test_annotations)


def build_split_items(image_dir: Path, annotation_dir: Path) -> list[SplitItem]:
    """Aynı dosya adına sahip görüntü ve XML dosyalarını eşleştirir."""
    image_paths = sorted(image_dir.glob("*.jpg"))
    return [
        SplitItem(image_path=image_path, annotation_path=annotation_dir / f"{image_path.stem}.xml")
        for image_path in image_paths
    ]


def split_train_val(items: list[SplitItem], val_ratio: float, seed: int) -> tuple[list[SplitItem], list[SplitItem]]:
    """Trainval listesini deterministik train ve validation listelerine ayırır."""
    if not 0 < val_ratio < 1:
        raise ValueError("Validation oranı 0 ile 1 arasında olmalıdır.")

    shuffled_items = items[:]
    random.Random(seed).shuffle(shuffled_items)

    val_count = round(len(shuffled_items) * val_ratio)
    val_items = sorted(shuffled_items[:val_count], key=lambda item: item.image_path.name)
    train_items = sorted(shuffled_items[val_count:], key=lambda item: item.image_path.name)
    return train_items, val_items


def prepare_processed_dataset(
    train_items: list[SplitItem],
    val_items: list[SplitItem],
    test_items: list[SplitItem],
    class_names: list[str],
    *,
    use_symlink: bool = True,
) -> dict[str, SplitSummary]:
    """YOLO klasörlerini, image linklerini, label txt dosyalarını ve dataset.yaml dosyasını üretir."""
    class_to_id = build_class_mapping(class_names)
    split_map = {
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }

    summaries: dict[str, SplitSummary] = {}
    for split_name, split_items in split_map.items():
        image_output_dir = PROCESSED_DATA_DIR / "images" / split_name
        label_output_dir = PROCESSED_DATA_DIR / "labels" / split_name
        image_output_dir.mkdir(parents=True, exist_ok=True)
        label_output_dir.mkdir(parents=True, exist_ok=True)
        _synchronize_generated_files(
            image_output_dir,
            {item.image_path.name for item in split_items},
            suffix=".jpg",
        )
        _synchronize_generated_files(
            label_output_dir,
            {f"{item.image_path.stem}.txt" for item in split_items},
            suffix=".txt",
        )

        summary = SplitSummary()
        for item in split_items:
            image_output_path = image_output_dir / item.image_path.name
            label_output_path = label_output_dir / f"{item.image_path.stem}.txt"

            _link_or_copy_image(item.image_path, image_output_path, use_symlink=use_symlink)
            object_count = convert_voc_xml_to_yolo(item.annotation_path, label_output_path, class_to_id)

            summary.image_count += 1
            summary.label_count += 1
            summary.object_count += object_count
        summaries[split_name] = summary

    write_dataset_yaml(class_names)
    return summaries


def write_dataset_yaml(class_names: list[str]) -> None:
    """Ultralytics YOLO için dataset.yaml dosyasını üretir."""
    DATASET_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset_config = {
        "path": str(PROCESSED_DATA_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    DATASET_YAML_PATH.write_text(
        yaml.safe_dump(dataset_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def validate_processed_dataset(class_count: int) -> dict[str, SplitSummary]:
    """YOLO çıktı dosyalarının temel format ve eşleşme kontrollerini yapar."""
    summaries: dict[str, SplitSummary] = {}
    split_image_stems: dict[str, set[str]] = {}
    for split_name in ("train", "val", "test"):
        image_dir = PROCESSED_DATA_DIR / "images" / split_name
        label_dir = PROCESSED_DATA_DIR / "labels" / split_name
        image_paths = sorted(image_dir.glob("*.jpg"))
        label_paths = sorted(label_dir.glob("*.txt"))

        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in label_paths}
        split_image_stems[split_name] = image_stems
        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)[:10]
            missing_images = sorted(label_stems - image_stems)[:10]
            raise ValueError(
                f"{split_name} splitinde görüntü/label eşleşmesi bozuk. "
                f"Eksik label örnekleri: {missing_labels}, eksik görüntü örnekleri: {missing_images}"
            )

        summary = SplitSummary(image_count=len(image_paths), label_count=len(label_paths))
        for label_path in label_paths:
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Geçersiz YOLO satırı: {label_path}:{line_number} -> {line}")

                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
                if class_id < 0 or class_id >= class_count:
                    raise ValueError(f"Class id aralık dışında: {label_path}:{line_number} -> {class_id}")
                if any(value < 0 or value > 1 for value in values):
                    raise ValueError(f"Koordinat 0-1 aralığı dışında: {label_path}:{line_number} -> {line}")
                summary.object_count += 1
        summaries[split_name] = summary

    _validate_disjoint_split_stems(split_image_stems)

    if not DATASET_YAML_PATH.exists():
        raise FileNotFoundError(f"dataset.yaml üretilemedi: {DATASET_YAML_PATH}")

    return summaries


def print_summary(generated: dict[str, SplitSummary], validated: dict[str, SplitSummary], class_names: list[str]) -> None:
    """Veri hazırlama sonucunu Türkçe özetler."""
    print("\nPV Vision AI veri hazırlama özeti")
    print("--------------------------------")
    print(f"Sınıf sayısı: {len(class_names)}")
    print(f"YOLO dataset.yaml: {DATASET_YAML_PATH}")
    print("\nSplit özetleri:")
    for split_name in ("train", "val", "test"):
        generated_summary = generated[split_name]
        validated_summary = validated[split_name]
        print(
            f"- {split_name}: "
            f"{validated_summary.image_count} görüntü, "
            f"{validated_summary.label_count} label, "
            f"{generated_summary.object_count} kusur kutusu"
        )


def _validate_image_annotation_pairs(split_name: str, image_dir: Path, annotation_dir: Path) -> None:
    image_stems = {path.stem for path in image_dir.glob("*.jpg")}
    annotation_stems = {path.stem for path in annotation_dir.glob("*.xml")}

    if not image_stems:
        raise ValueError(f"{split_name} görüntü klasörü boş: {image_dir}")
    if not annotation_stems:
        raise ValueError(f"{split_name} annotation klasörü boş: {annotation_dir}")
    if image_stems != annotation_stems:
        missing_annotations = sorted(image_stems - annotation_stems)[:10]
        missing_images = sorted(annotation_stems - image_stems)[:10]
        raise ValueError(
            f"{split_name} görüntü/XML eşleşmesi bozuk. "
            f"Eksik XML örnekleri: {missing_annotations}, eksik görüntü örnekleri: {missing_images}"
        )


def _link_or_copy_image(source: Path, destination: Path, *, use_symlink: bool) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Hedefte beklenmeyen dosya var, üzerine yazılmadı: {destination}")

    if use_symlink:
        destination.symlink_to(source.resolve())
    else:
        import shutil

        shutil.copy2(source, destination)


def _synchronize_generated_files(
    directory: Path,
    expected_names: set[str],
    *,
    suffix: str,
) -> None:
    """Önceki hazırlamadan kalan ve yeni split'e ait olmayan üretilmiş dosyaları kaldırır."""
    for path in directory.glob(f"*{suffix}"):
        if path.name not in expected_names and (path.is_file() or path.is_symlink()):
            path.unlink()


def _validate_disjoint_split_stems(split_stems: dict[str, set[str]]) -> None:
    """Aynı görüntünün birden fazla split'te bulunmasını engeller."""
    split_names = list(split_stems)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            overlap = split_stems[left_name] & split_stems[right_name]
            if overlap:
                examples = sorted(overlap)[:10]
                raise ValueError(
                    f"Veri sızıntısı: {left_name} ve {right_name} splitlerinde "
                    f"{len(overlap)} ortak görüntü var. Örnekler: {examples}"
                )


def main() -> None:
    """PVEL-AD veri setini data/processed altında YOLO formatına hazırlar."""
    parser = argparse.ArgumentParser(description="PVEL-AD veri setini YOLO formatına hazırlar.")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Trainval içinden ayrılacak validation oranı")
    parser.add_argument("--seed", type=int, default=42, help="Deterministik train/val ayırma tohumu")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Symlink yerine görüntü dosyalarını kopyalar",
    )
    args = parser.parse_args()

    raw_paths = get_raw_dataset_paths()
    validate_raw_dataset(raw_paths)

    class_names = read_class_names(raw_paths.classes_file)
    trainval_items = build_split_items(raw_paths.trainval_images, raw_paths.trainval_annotations)
    test_items = build_split_items(raw_paths.test_images, raw_paths.test_annotations)
    train_items, val_items = split_train_val(trainval_items, args.val_ratio, args.seed)

    generated = prepare_processed_dataset(
        train_items,
        val_items,
        test_items,
        class_names,
        use_symlink=not args.copy_images,
    )
    validated = validate_processed_dataset(len(class_names))
    print_summary(generated, validated, class_names)


if __name__ == "__main__":
    main()
