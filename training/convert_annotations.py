"""PVEL-AD Pascal VOC XML annotationlarını YOLO formatına dönüştürür."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFECT_CLASSES_TECHNICAL, NAME_TECHNICAL_TO_CLASS_ID  # noqa: E402


@dataclass(frozen=True)
class VocObject:
    """Bir Pascal VOC nesne kutusunun sade gösterimi."""

    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def read_class_names(classes_path: Path) -> list[str]:
    """Sınıf isimlerini annotation_classes.txt dosyasından sırayla okur."""
    if not classes_path.exists():
        raise FileNotFoundError(f"Sınıf dosyası bulunamadı: {classes_path}")

    class_names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not class_names:
        raise ValueError(f"Sınıf dosyası boş: {classes_path}")
    return class_names


def parse_voc_xml(xml_path: Path) -> tuple[int, int, list[VocObject]]:
    """Pascal VOC XML dosyasından görüntü boyutu ve nesne kutularını okur."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"XML içinde size alanı yok: {xml_path}")

    width = int(float(_required_text(size, "width", xml_path)))
    height = int(float(_required_text(size, "height", xml_path)))
    if width <= 0 or height <= 0:
        raise ValueError(f"Geçersiz görüntü boyutu: {xml_path} ({width}x{height})")

    objects: list[VocObject] = []
    for obj in root.findall("object"):
        class_name = _required_text(obj, "name", xml_path)
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Object içinde bndbox alanı yok: {xml_path}")

        voc_object = VocObject(
            class_name=class_name,
            xmin=float(_required_text(box, "xmin", xml_path)),
            ymin=float(_required_text(box, "ymin", xml_path)),
            xmax=float(_required_text(box, "xmax", xml_path)),
            ymax=float(_required_text(box, "ymax", xml_path)),
        )
        _validate_box(voc_object, width, height, xml_path)
        objects.append(voc_object)

    if not objects:
        raise ValueError(f"XML içinde object alanı yok: {xml_path}")

    return width, height, objects


def voc_to_yolo_line(voc_object: VocObject, class_to_id: dict[str, int], width: int, height: int) -> str:
    """Tek Pascal VOC kutusunu YOLO label satırına çevirir."""
    if voc_object.class_name not in class_to_id:
        raise ValueError(f"Bilinmeyen sınıf adı: {voc_object.class_name}")

    x_center = ((voc_object.xmin + voc_object.xmax) / 2) / width
    y_center = ((voc_object.ymin + voc_object.ymax) / 2) / height
    box_width = (voc_object.xmax - voc_object.xmin) / width
    box_height = (voc_object.ymax - voc_object.ymin) / height

    values = (x_center, y_center, box_width, box_height)
    if any(value < 0 or value > 1 for value in values):
        raise ValueError(f"YOLO koordinatı 0-1 aralığı dışında: {voc_object}")

    class_id = class_to_id[voc_object.class_name]
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def convert_voc_xml_to_yolo(xml_path: Path, output_path: Path, class_to_id: dict[str, int]) -> int:
    """Bir XML annotation dosyasını YOLO txt dosyasına dönüştürür."""
    width, height, objects = parse_voc_xml(xml_path)
    lines = [
        voc_to_yolo_line(voc_object, class_to_id, width, height)
        for voc_object in objects
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def build_class_mapping(class_names: list[str]) -> dict[str, int]:
    """Sınıf listesinden YOLO class_id eşlemesi üretir."""
    if len(class_names) != len(set(class_names)):
        raise ValueError("annotation_classes.txt içinde yinelenen sınıf adı var.")

    class_to_id = {class_name: class_id for class_id, class_name in enumerate(class_names)}

    missing_from_config = set(class_to_id) - set(NAME_TECHNICAL_TO_CLASS_ID)
    if missing_from_config:
        missing = ", ".join(sorted(missing_from_config))
        raise ValueError(f"config.py içinde Türkçe karşılığı olmayan sınıflar var: {missing}")

    if class_names != DEFECT_CLASSES_TECHNICAL:
        raise ValueError(
            "annotation_classes.txt sınıf adları veya sırası config.py ile eşleşmiyor. "
            f"Beklenen: {DEFECT_CLASSES_TECHNICAL}; bulunan: {class_names}"
        )

    return class_to_id


def _required_text(element: ET.Element, tag: str, xml_path: Path) -> str:
    child = element.find(tag)
    if child is None or child.text is None or not child.text.strip():
        raise ValueError(f"XML alanı eksik: {tag} ({xml_path})")
    return child.text.strip()


def _validate_box(voc_object: VocObject, width: int, height: int, xml_path: Path) -> None:
    if voc_object.xmin >= voc_object.xmax or voc_object.ymin >= voc_object.ymax:
        raise ValueError(f"Geçersiz bounding box: {xml_path} ({voc_object})")
    if voc_object.xmin < 0 or voc_object.ymin < 0:
        raise ValueError(f"Negatif bounding box koordinatı: {xml_path} ({voc_object})")
    if voc_object.xmax > width or voc_object.ymax > height:
        raise ValueError(f"Bounding box görüntü sınırını aşıyor: {xml_path} ({voc_object})")


def main() -> None:
    """Tek bir XML dosyasını komut satırından YOLO formatına dönüştürür."""
    parser = argparse.ArgumentParser(description="Pascal VOC XML dosyasını YOLO txt formatına dönüştürür.")
    parser.add_argument("xml_path", type=Path, help="Dönüştürülecek XML dosyası")
    parser.add_argument("output_path", type=Path, help="Üretilecek YOLO txt dosyası")
    parser.add_argument(
        "--classes",
        type=Path,
        default=PROJECT_ROOT / "data/raw/solar_cell_EL_image/PVELAD/EL2021/annotation_classes.txt",
        help="Sınıf isimlerini içeren annotation_classes.txt dosyası",
    )
    args = parser.parse_args()

    class_names = read_class_names(args.classes)
    class_to_id = build_class_mapping(class_names)
    object_count = convert_voc_xml_to_yolo(args.xml_path, args.output_path, class_to_id)
    print(f"Dönüştürüldü: {args.xml_path} -> {args.output_path} ({object_count} kutu)")


if __name__ == "__main__":
    main()
