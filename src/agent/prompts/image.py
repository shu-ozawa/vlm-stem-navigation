"""The image a reader sends: the observation as a data URI, with the green square for locate."""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw

BOX_FRAC = 0.20  # green square side / FOV side (+-10% around the centre): the success square (TaskRules.center_frac)


def png_data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def draw_box(png: bytes, frac: float = BOX_FRAC) -> bytes:
    """Observation with a green square outlining its central `frac` region."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size
    x0, y0 = w * (1 - frac) / 2, h * (1 - frac) / 2
    ImageDraw.Draw(img).rectangle([x0, y0, w - x0 - 1, h - y0 - 1], outline=(0, 220, 0), width=max(2, w // 170))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
