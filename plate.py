"""Plate-level logic: file discovery, filename parsing, well utilities."""

import glob
import math
import os
import re


PLATE_FORMATS = {
    '384': (16, 24),
    '96': (8, 12),
}


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


def detect_fields(plate_folder):
    """Scan filenames and return sorted list of field numbers (e.g. [1, 2, ..., 9])."""
    tifs = glob.glob(os.path.join(plate_folder, "*.tif"))
    fields = set()
    for f in tifs[:200]:
        m = re.search(r'fld\s+(\d+)', os.path.basename(f))
        if m:
            fields.add(int(m.group(1)))
    return sorted(fields)


def center_field(fields):
    """Given a sorted list of field numbers, return the center field number."""
    if not fields:
        return 5  # fallback default
    return fields[math.ceil(len(fields) / 2) - 1]


def well_to_row_col(well_id):
    """Convert 'A01' -> (0, 0), 'P24' -> (15, 23)."""
    row = ord(well_id[0]) - ord('A')
    col = int(well_id[1:]) - 1
    return row, col


def parse_well_spec(spec, plate_rows=16):
    """Parse a flexible well specification string into a set of well IDs.

    Supported formats (comma-separated, can be mixed):
        A05         — single well
        A-H:5       — rows A through H, column 5
        I-P:13      — rows I through P, column 13
        col:1-2     — all rows, columns 1 and 2
        A01-A09     — well range within a single row

    Parameters
    ----------
    spec : str
        Well specification string.
    plate_rows : int
        Number of rows in the plate (16 for 384-well, 8 for 96-well).

    Example: "A-H:5, I-P:13" -> {'A05','B05',...,'H05','I13','J13',...,'P13'}
    """
    wells = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue

        # Format: col:N or col:N-M (full columns)
        m = re.match(r'^col[: ](\d+)(?:-(\d+))?$', part, re.IGNORECASE)
        if m:
            c_start = int(m.group(1))
            c_end = int(m.group(2)) if m.group(2) else c_start
            for c in range(c_start, c_end + 1):
                for row_letter in [chr(ord('A') + r) for r in range(plate_rows)]:
                    wells.add(f"{row_letter}{c:02d}")
            continue

        # Format: A-H:5 (row range : column)
        m = re.match(r'^([A-P])-([A-P]):(\d+)$', part, re.IGNORECASE)
        if m:
            r_start = ord(m.group(1).upper()) - ord('A')
            r_end = ord(m.group(2).upper()) - ord('A')
            col = int(m.group(3))
            for r in range(r_start, r_end + 1):
                wells.add(f"{chr(ord('A') + r)}{col:02d}")
            continue

        # Format: A01-A09 (well range in same row)
        m = re.match(r'^([A-P])(\d+)-([A-P])(\d+)$', part, re.IGNORECASE)
        if m:
            r_start = ord(m.group(1).upper()) - ord('A')
            c_start = int(m.group(2))
            r_end = ord(m.group(3).upper()) - ord('A')
            c_end = int(m.group(4))
            for r in range(r_start, r_end + 1):
                for c in range(c_start, c_end + 1):
                    wells.add(f"{chr(ord('A') + r)}{c:02d}")
            continue

        # Format: A05 (single well)
        m = re.match(r'^([A-P])(\d+)$', part, re.IGNORECASE)
        if m:
            wells.add(f"{m.group(1).upper()}{int(m.group(2)):02d}")
            continue

    return wells


def filter_images_by_wells(image_list, well_set):
    """Filter an image list to only include images from the specified wells."""
    filtered = []
    for fpath in image_list:
        w = parse_well(fpath)
        if w and w in well_set:
            filtered.append(fpath)
    return filtered
