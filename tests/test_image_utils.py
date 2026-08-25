"""Görüntü yükleme ve dışa aktarma yardımcılarının testleri."""

from __future__ import annotations

import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.utils.image_utils import MAX_IMAGE_PIXELS, bgr_image_to_png_bytes, open_uploaded_image


class ImageUtilsTests(unittest.TestCase):
    def test_valid_image_bytes_are_opened_as_rgb(self) -> None:
        buffer = io.BytesIO()
        Image.new("L", (12, 8), color=128).save(buffer, format="PNG")

        image = open_uploaded_image(buffer.getvalue())

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (12, 8))

    def test_invalid_or_empty_image_is_rejected(self) -> None:
        for payload in (b"", b"not-an-image"):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    open_uploaded_image(payload)

    def test_image_above_pixel_limit_is_rejected_before_conversion(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10), color=(10, 20, 30)).save(buffer, format="PNG")

        with patch("app.utils.image_utils.MAX_IMAGE_PIXELS", 99):
            with self.assertRaisesRegex(ValueError, "40 megapiksel"):
                open_uploaded_image(buffer.getvalue())

        self.assertEqual(MAX_IMAGE_PIXELS, 40_000_000)

    def test_bgr_array_is_encoded_as_png_with_correct_colors(self) -> None:
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[:, :] = (0, 0, 255)

        png_bytes = bgr_image_to_png_bytes(bgr)
        decoded = Image.open(io.BytesIO(png_bytes)).convert("RGB")

        self.assertEqual(decoded.getpixel((0, 0)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
