"""Görüntü yükleme yardımcıları."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


MAX_IMAGE_PIXELS = 40_000_000


def open_uploaded_image(file_bytes: bytes) -> Image.Image:
    """Yüklenen dosya içeriğini Pillow görüntüsüne çevirir."""
    if not file_bytes:
        raise ValueError("Yüklenen görüntü dosyası boş.")

    try:
        image = Image.open(BytesIO(file_bytes))
        image.verify()
        image = Image.open(BytesIO(file_bytes))
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                "Görüntü çözünürlüğü çok yüksek. En fazla 40 megapiksel görüntü yüklenebilir."
            )
        return image.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise ValueError("Görüntü güvenli çözünürlük sınırını aşıyor.") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Yüklenen dosya geçerli bir görüntü değil.") from exc


def bgr_image_to_png_bytes(image: np.ndarray) -> bytes:
    """OpenCV BGR görüntüsünü indirilebilir PNG baytlarına dönüştürür."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG çıktısı için üç kanallı bir görüntü bekleniyor.")

    rgb_image = Image.fromarray(image[:, :, ::-1])
    buffer = BytesIO()
    rgb_image.save(buffer, format="PNG")
    return buffer.getvalue()
