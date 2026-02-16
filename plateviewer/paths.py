"""Centralised output and cache path resolution for CRISPR PlateViewer."""

import hashlib
import os

from plateviewer import config as cfg


def _plate_data_dir(plate_folder):
    """Return the per-plate data directory under APP_DATA_DIR.

    Uses <plate_name>_<hash8> to avoid collisions between plates
    with the same name in different locations.
    """
    plate_name = os.path.basename(os.path.normpath(plate_folder))
    abs_path = os.path.abspath(plate_folder)
    folder_hash = hashlib.sha256(abs_path.encode()).hexdigest()[:8]
    return os.path.join(cfg.APP_DATA_DIR, f"{plate_name}_{folder_hash}")


def plate_cache_dir(plate_folder):
    """Return the cache directory for a plate, creating it if needed."""
    d = os.path.join(_plate_data_dir(plate_folder), "cache")
    os.makedirs(d, exist_ok=True)
    return d


def plate_output_dir(plate_folder):
    """Return the default output directory for a plate, creating it if needed."""
    d = os.path.join(_plate_data_dir(plate_folder), "output")
    os.makedirs(d, exist_ok=True)
    return d
