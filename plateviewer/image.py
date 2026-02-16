"""Image-level utilities: dtype conversion, labeling, encoding."""

import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plateviewer import config as cfg


def uint16_to_uint8(img, plow=cfg.PERCENTILE_LOW, phigh=cfg.PERCENTILE_HIGH):
    """Convert a uint16 image to uint8 using percentile contrast stretch."""
    img = img.astype(np.float32)
    lo = np.percentile(img, plow)
    hi = np.percentile(img, phigh)
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255)
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)


def burn_label(tile, label, font_size=48):
    """Burn a text label onto the top-center of a uint8 tile."""
    img_pil = Image.fromarray(tile, mode='L')
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(cfg.FONT_PATH, font_size) if cfg.FONT_PATH else ImageFont.load_default()
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    x = (tile.shape[1] - text_w) // 2
    y = 5
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            draw.text((x + dx, y + dy), label, fill=0, font=font)
    draw.text((x, y), label, fill=255, font=font)
    return np.array(img_pil)


def numpy_to_b64png(arr):
    """Convert a uint8 numpy array to a base64-encoded PNG data URI."""
    img = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
