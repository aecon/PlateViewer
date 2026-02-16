"""Centralized constants for CRISPR PlateViewer."""

import os

# ---------------------------------------------------------------------------
# Montage defaults
# ---------------------------------------------------------------------------
MONTAGE_N_IMAGES = 32
MONTAGE_ROWS = 4
MONTAGE_COLS = 8
MONTAGE_CROP_SIZE = 1020
MONTAGE_SPACING = 5
MONTAGE_FONT_SIZE = 72

# ---------------------------------------------------------------------------
# Single-well montage
# ---------------------------------------------------------------------------
WELL_DOWNSAMPLE = 4
WELL_FONT_SIZE = 24
WELL_SPACING = 5

# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------
CONTACT_THUMB_SIZE = 128
CONTACT_SPACING = 2
CONTACT_FIG_WIDTH = 1150

# ---------------------------------------------------------------------------
# Heatmap figure
# ---------------------------------------------------------------------------
HEATMAP_FIG_WIDTH = 900
HEATMAP_FIG_HEIGHT = 500

# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------
PERCENTILE_LOW = 1
PERCENTILE_HIGH = 99

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
FALLBACK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------
N_WORKERS = 8

# ---------------------------------------------------------------------------
# Font (cross-platform resolution)
# ---------------------------------------------------------------------------
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   # Linux
    "/System/Library/Fonts/Helvetica.ttc",                      # macOS
    r"C:\Windows\Fonts\arialbd.ttf",                            # Windows
]


def _resolve_font():
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


FONT_PATH = _resolve_font()
