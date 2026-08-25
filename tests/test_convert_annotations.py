"""Pascal VOC -> YOLO dönüşümünün birim testleri."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import DEFECT_CLASSES_TECHNICAL
from training.convert_annotations import (
    VocObject,
    build_class_mapping,
    convert_voc_xml_to_yolo,
    voc_to_yolo_line,
)


class ConvertAnnotationsTests(unittest.TestCase):
    def test_class_mapping_requires_the_exact_central_class_order(self) -> None:
        mapping = build_class_mapping(DEFECT_CLASSES_TECHNICAL)
        self.assertEqual(mapping["crack"], 0)
        self.assertEqual(mapping["short_circuit"], 11)

        with self.assertRaisesRegex(ValueError, "sırası"):
            build_class_mapping(list(reversed(DEFECT_CLASSES_TECHNICAL)))

    def test_voc_box_is_normalized_with_the_expected_formula(self) -> None:
        item = VocObject("crack", xmin=10, ymin=20, xmax=50, ymax=60)
        line = voc_to_yolo_line(item, {"crack": 0}, width=100, height=80)
        self.assertEqual(line, "0 0.300000 0.500000 0.400000 0.500000")

    def test_xml_is_converted_to_a_yolo_label_file(self) -> None:
        xml = """<annotation>
  <size><width>100</width><height>80</height></size>
  <object>
    <name>crack</name>
    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox>
  </object>
</annotation>
"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            xml_path = directory / "sample.xml"
            label_path = directory / "sample.txt"
            xml_path.write_text(xml, encoding="utf-8")

            count = convert_voc_xml_to_yolo(xml_path, label_path, {"crack": 0})

            self.assertEqual(count, 1)
            self.assertEqual(
                label_path.read_text(encoding="utf-8"),
                "0 0.300000 0.500000 0.400000 0.500000\n",
            )


if __name__ == "__main__":
    unittest.main()
