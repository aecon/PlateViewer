"""Plate heatmap computation: intensity, focus. Threaded I/O with disk caching."""

import concurrent.futures
import hashlib
import os

import numpy as np
import tifffile

from plate import find_images, parse_well, well_to_row_col
from image import uint16_to_uint8
import config as cfg

# Fallback cache directory inside the code directory
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_FALLBACK_CACHE_DIR = os.path.join(_CODE_DIR, ".cache")


# ---------------------------------------------------------------------------
# Per-image worker functions
# ---------------------------------------------------------------------------

def _read_and_mean(fpath):
    """Read a TIF, subsample 2x, return (well_id, mean_intensity)."""
    well = parse_well(fpath)
    if well is None:
        return None
    img = tifffile.imread(fpath)[::2, ::2]
    return (well, float(np.mean(img)))


def _read_and_focus(fpath):
    """Read a TIF, subsample 2x, convert to uint8, return (well_id, laplacian_variance)."""
    from scipy.ndimage import laplace
    well = parse_well(fpath)
    if well is None:
        return None
    img = tifffile.imread(fpath)[::2, ::2]
    if img.dtype != np.uint8:
        img = uint16_to_uint8(img)
    lap = laplace(img.astype(np.float32))
    return (well, float(np.var(lap)))


def _power_log_log_slope(img):
    """Compute slope of radially-averaged power spectrum on a log-log scale.

    More negative values indicate blurrier images (loss of high frequencies).
    Reference: Bray et al., J Biomol Screen, 2012.
    """
    from scipy import ndimage as ndi
    f = np.fft.fftshift(np.fft.fft2(img))
    magnitude = np.abs(f) ** 2

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cx, cy)

    idx = np.arange(1, max_r)
    radial_sum = ndi.sum(magnitude, r, index=idx)
    radial_count = ndi.sum(np.ones_like(magnitude), r, index=idx)
    radial_mean = radial_sum / (radial_count + 1e-10)

    freqs = np.arange(1, len(radial_mean) + 1)
    valid = radial_mean > 0
    if valid.sum() < 10:
        return 0.0
    log_f = np.log(freqs[valid])
    log_p = np.log(radial_mean[valid])
    slope, _ = np.polyfit(log_f, log_p, 1)
    return float(slope)


def _read_and_plls(fpath):
    """Read a TIF, subsample 2x, return (well_id, power_log_log_slope)."""
    well = parse_well(fpath)
    if well is None:
        return None
    img = tifffile.imread(fpath)[::2, ::2].astype(np.float64)
    return (well, _power_log_log_slope(img))


# ---------------------------------------------------------------------------
# Aggregation and caching
# ---------------------------------------------------------------------------

def _aggregate_to_heatmap(results, plate_rows=16, plate_cols=24):
    """Take list of (well_id, value) tuples and return a plate-shaped array."""
    well_values = {}
    for item in results:
        if item is None:
            continue
        well, val = item
        well_values.setdefault(well, []).append(val)
    heatmap = np.full((plate_rows, plate_cols), np.nan)
    for well, vals in well_values.items():
        r, c = well_to_row_col(well)
        if r < plate_rows and c < plate_cols:
            heatmap[r, c] = np.mean(vals)
    return heatmap


def _cache_path(plate_folder, channel, metric):
    """Return path for a cached heatmap .npy file.

    Prefers writing inside the plate folder (keeps cache next to data).
    Falls back to <code_dir>/.cache/<hash>/ when the plate folder is read-only.
    """
    plate_name = os.path.basename(os.path.normpath(plate_folder))
    filename = f".plateviewer_{plate_name}_{channel}_{metric}.npy"

    # Prefer the plate folder itself
    plate_cache = os.path.join(plate_folder, filename)
    if os.access(plate_folder, os.W_OK):
        return plate_cache

    # If an existing cache file is already readable there, use it (read-only OK)
    if os.path.isfile(plate_cache):
        return plate_cache

    # Fall back to <code_dir>/.cache/<hash>/
    abs_path = os.path.abspath(plate_folder)
    folder_hash = hashlib.sha256(abs_path.encode()).hexdigest()[:12]
    fallback_dir = os.path.join(_FALLBACK_CACHE_DIR, folder_hash)
    os.makedirs(fallback_dir, exist_ok=True)
    return os.path.join(fallback_dir, filename)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_intensity_heatmap(plate_folder, channel, plate_rows=16, plate_cols=24):
    """Compute mean intensity per well for a given channel. Returns a plate-shaped array."""
    cache = _cache_path(plate_folder, channel, "intensity")
    if os.path.exists(cache):
        cached = np.load(cache)
        if cached.shape == (plate_rows, plate_cols):
            print(f"  Loading cached intensity heatmap: {cache}")
            return cached
    images = find_images(plate_folder, channel=channel)
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.N_WORKERS) as pool:
        results = list(pool.map(_read_and_mean, images))
    heatmap = _aggregate_to_heatmap(results, plate_rows, plate_cols)
    np.save(cache, heatmap)
    print(f"  Cached intensity heatmap: {cache}")
    return heatmap


def compute_focus_heatmap(plate_folder, channel, plate_rows=16, plate_cols=24):
    """Compute Laplacian variance (focus metric) per well. Returns a plate-shaped array."""
    cache = _cache_path(plate_folder, channel, "focus")
    if os.path.exists(cache):
        cached = np.load(cache)
        if cached.shape == (plate_rows, plate_cols):
            print(f"  Loading cached focus heatmap: {cache}")
            return cached
    images = find_images(plate_folder, channel=channel)
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.N_WORKERS) as pool:
        results = list(pool.map(_read_and_focus, images))
    heatmap = _aggregate_to_heatmap(results, plate_rows, plate_cols)
    np.save(cache, heatmap)
    print(f"  Cached focus heatmap: {cache}")
    return heatmap


def compute_plls_heatmap(plate_folder, channel, plate_rows=16, plate_cols=24):
    """Compute Power Log-Log Slope per well. Returns a plate-shaped array.

    More negative values indicate blurrier wells (high-frequency loss).
    Reference: Bray et al., J Biomol Screen, 2012.
    """
    cache = _cache_path(plate_folder, channel, "plls")
    if os.path.exists(cache):
        cached = np.load(cache)
        if cached.shape == (plate_rows, plate_cols):
            print(f"  Loading cached PLLS heatmap: {cache}")
            return cached
    images = find_images(plate_folder, channel=channel)
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.N_WORKERS) as pool:
        results = list(pool.map(_read_and_plls, images))
    heatmap = _aggregate_to_heatmap(results, plate_rows, plate_cols)
    np.save(cache, heatmap)
    print(f"  Cached PLLS heatmap: {cache}")
    return heatmap
