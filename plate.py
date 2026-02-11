"""Plate-level logic: file discovery, filename parsing, well utilities."""

import glob
import os
import re


def find_images(plate_folder, channel=None):
    """Find all TIF images in a plate folder, optionally filtered by channel."""
    all_tifs = glob.glob(os.path.join(plate_folder, "*.tif"))
    if channel:
        all_tifs = [f for f in all_tifs if channel in os.path.basename(f)]
    all_tifs.sort()
    return all_tifs


def parse_filename(filepath):
    """Extract well and field label from a filename like 'A - 01(fld 1 wv 390 - Blue).tif'.

    Returns e.g. 'A01 f1', or '' if parsing fails.
    """
    basename = os.path.basename(filepath)
    m = re.match(r'^([A-P])\s*-\s*(\d+)\(fld\s+(\d+)\s+wv\s+.+\)\.tif$', basename)
    if m:
        return f"{m.group(1)}{m.group(2)} f{m.group(3)}"
    return ""


def parse_well(filepath):
    """Extract well ID (e.g. 'A01') from a filepath, or None."""
    m = re.match(r'^([A-P])\s*-\s*(\d+)\(', os.path.basename(filepath))
    if m:
        return f"{m.group(1)}{m.group(2).zfill(2)}"
    return None


def detect_channels(plate_folder):
    """Scan filenames in a plate folder and return sorted list of channel names."""
    tifs = glob.glob(os.path.join(plate_folder, "*.tif"))
    channels = set()
    for f in tifs[:200]:
        m = re.search(r'wv\s+\d+\s*-\s*(\w+)', os.path.basename(f))
        if m:
            channels.add(m.group(1))
    return sorted(channels)


def detect_wells(plate_folder):
    """Scan filenames and return sorted list of unique well IDs (e.g. 'A01')."""
    tifs = glob.glob(os.path.join(plate_folder, "*.tif"))
    wells = set()
    for f in tifs:
        w = parse_well(f)
        if w:
            wells.add(w)
    return sorted(wells)


def well_to_row_col(well_id):
    """Convert 'A01' -> (0, 0), 'P24' -> (15, 23)."""
    row = ord(well_id[0]) - ord('A')
    col = int(well_id[1:]) - 1
    return row, col
