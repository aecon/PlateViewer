"""Plate heatmap computation: intensity, focus. Threaded I/O with disk caching."""

import concurrent.futures
import os

import numpy as np
import tifffile

from plate import find_images, parse_well, well_to_row_col

N_WORKERS = 8


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
    """Read a TIF, subsample 2x, return (well_id, laplacian_variance)."""
    from scipy.ndimage import laplace
    well = parse_well(fpath)
    if well is None:
        return None
    img = tifffile.imread(fpath)[::2, ::2].astype(np.float32)
    lap = laplace(img)
    return (well, float(np.var(lap)))


# ---------------------------------------------------------------------------
# Aggregation and caching
# ---------------------------------------------------------------------------

def _aggregate_to_heatmap(results):
    """Take list of (well_id, value) tuples and return a 16x24 array."""
    well_values = {}
    for item in results:
        if item is None:
            continue
        well, val = item
        well_values.setdefault(well, []).append(val)
    heatmap = np.full((16, 24), np.nan)
    for well, vals in well_values.items():
        r, c = well_to_row_col(well)
        heatmap[r, c] = np.mean(vals)
    return heatmap


def _cache_path(plate_folder, channel, metric):
    """Return path for a cached heatmap .npy file."""
    plate_name = os.path.basename(os.path.normpath(plate_folder))
    return os.path.join(plate_folder, f".plateviewer_{plate_name}_{channel}_{metric}.npy")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_intensity_heatmap(plate_folder, channel):
    """Compute mean intensity per well for a given channel. Returns a 16x24 array."""
    cache = _cache_path(plate_folder, channel, "intensity")
    if os.path.exists(cache):
        print(f"  Loading cached intensity heatmap: {cache}")
        return np.load(cache)
    images = find_images(plate_folder, channel=channel)
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_read_and_mean, images))
    heatmap = _aggregate_to_heatmap(results)
    np.save(cache, heatmap)
    print(f"  Cached intensity heatmap: {cache}")
    return heatmap


def compute_focus_heatmap(plate_folder, channel):
    """Compute Laplacian variance (focus metric) per well. Returns a 16x24 array."""
    cache = _cache_path(plate_folder, channel, "focus")
    if os.path.exists(cache):
        print(f"  Loading cached focus heatmap: {cache}")
        return np.load(cache)
    images = find_images(plate_folder, channel=channel)
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(_read_and_focus, images))
    heatmap = _aggregate_to_heatmap(results)
    np.save(cache, heatmap)
    print(f"  Cached focus heatmap: {cache}")
    return heatmap
