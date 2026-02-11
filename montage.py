#!/usr/bin/env python3
"""
Generate a random montage of images from a high-content screening plate folder.

Usage:
    conda run -n PlateViewer python montage.py /path/to/plate_folder
    conda run -n PlateViewer python montage.py /path/to/plate_folder --channel Blue --n_images 64 --rows 8 --cols 8
"""

import argparse
import concurrent.futures
import os
import sys

import numpy as np
import tifffile

from plate import find_images, parse_filename, parse_well
from image import uint16_to_uint8, burn_label


def make_montage(image_list, n_images=32, rows=4, cols=8, crop_size=512, spacing=5):
    """Randomly select images and assemble into a montage.

    Parameters
    ----------
    image_list : list of str
        Paths to TIF images.
    n_images : int
        Number of images to sample (must equal rows * cols).
    rows : int
        Number of rows in the montage grid.
    cols : int
        Number of columns in the montage grid.
    crop_size : int
        Size of the center crop from each image (pixels).
    spacing : int
        Gap between tiles (pixels).

    Returns
    -------
    montage : np.ndarray (uint8)
        The assembled montage image.
    selected_files : list of str
        Paths of the selected images (in order).
    """
    assert rows * cols == n_images, f"rows*cols ({rows*cols}) must equal n_images ({n_images})"

    if len(image_list) < n_images:
        print(f"WARNING: Only {len(image_list)} images available, need {n_images}. Using all.")
        n_images = len(image_list)
        cols = min(cols, n_images)
        rows = int(np.ceil(n_images / cols))

    indices = np.random.choice(len(image_list), size=n_images, replace=False)
    selected_files = [image_list[i] for i in indices]

    # load, convert to uint8, and center-crop images
    tiles = []
    for fpath in selected_files:
        img = tifffile.imread(fpath)
        if img.dtype != np.uint8:
            img = uint16_to_uint8(img)
        h, w = img.shape[:2]
        y0 = max(0, (h - crop_size) // 2)
        x0 = max(0, (w - crop_size) // 2)
        tile = img[y0:y0 + crop_size, x0:x0 + crop_size]
        if tile.shape[0] < crop_size or tile.shape[1] < crop_size:
            padded = np.zeros((crop_size, crop_size), dtype=np.uint8)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded
        tiles.append(tile)

    # assemble montage
    montage_h = rows * crop_size + (rows - 1) * spacing
    montage_w = cols * crop_size + (cols - 1) * spacing
    montage = np.zeros((montage_h, montage_w), dtype=np.uint8)

    for idx, tile in enumerate(tiles):
        label = parse_filename(selected_files[idx])
        if label:
            tile = burn_label(tile, label)
        r = idx // cols
        c = idx % cols
        y0 = r * (crop_size + spacing)
        x0 = c * (crop_size + spacing)
        montage[y0:y0 + crop_size, x0:x0 + crop_size] = tile

    return montage, selected_files


def make_contact_sheet(image_list, thumb_size=128, spacing=2, plate_rows=16, plate_cols=24):
    """Create a plate-layout contact sheet: one thumbnail per well in 16x24 grid.

    Uses the first field found for each well as the representative image.

    Parameters
    ----------
    image_list : list of str
        Paths to TIF images (should be pre-filtered by channel).
    thumb_size : int
        Size of each thumbnail (pixels).
    spacing : int
        Gap between thumbnails (pixels).
    plate_rows : int
        Number of rows in the plate (16 for 384-well).
    plate_cols : int
        Number of columns in the plate (24 for 384-well).

    Returns
    -------
    sheet : np.ndarray (uint8)
        The assembled contact sheet.
    well_positions : dict
        Mapping of well_id -> (x_center, y_center) in pixel coords of the sheet.
    """
    # group images by well, prefer field 5 (center tile)
    well_images = {}
    for fpath in image_list:
        w = parse_well(fpath)
        if not w:
            continue
        basename = os.path.basename(fpath)
        is_fld5 = 'fld 5' in basename
        if w not in well_images or is_fld5:
            well_images[w] = fpath

    sheet_h = plate_rows * thumb_size + (plate_rows - 1) * spacing
    sheet_w = plate_cols * thumb_size + (plate_cols - 1) * spacing
    sheet = np.zeros((sheet_h, sheet_w), dtype=np.uint8)

    from skimage.transform import resize

    def _load_thumb(args):
        well, fpath = args
        img = tifffile.imread(fpath)
        if img.dtype != np.uint8:
            img = uint16_to_uint8(img)
        return well, resize(img, (thumb_size, thumb_size), preserve_range=True, anti_aliasing=True).astype(np.uint8)

    well_positions = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = pool.map(_load_thumb, well_images.items())
        for well, thumb in results:
            row = ord(well[0]) - ord('A')
            col = int(well[1:]) - 1
            y0 = row * (thumb_size + spacing)
            x0 = col * (thumb_size + spacing)
            sheet[y0:y0 + thumb_size, x0:x0 + thumb_size] = thumb
            well_positions[well] = (x0 + thumb_size // 2, y0 + thumb_size // 2)

    return sheet, well_positions


def main():
    import os
    parser = argparse.ArgumentParser(description="Generate a random image montage from a plate folder.")
    parser.add_argument("plate_folder", help="Path to the plate folder containing TIF images.")
    parser.add_argument("-c", "--channel", default=None,
                        help="Filter by channel name substring (e.g. 'Blue', 'Green2', 'FarRed').")
    parser.add_argument("-n", "--n_images", type=int, default=32,
                        help="Number of images to sample (default: 32).")
    parser.add_argument("--rows", type=int, default=4, help="Montage rows (default: 4).")
    parser.add_argument("--cols", type=int, default=8, help="Montage columns (default: 8).")
    parser.add_argument("--crop_size", type=int, default=1020, help="Center crop size in pixels (default: 1020).")
    parser.add_argument("-o", "--output", default=None,
                        help="Output TIF path. Default: <plate_folder>_montage_<rows>x<cols>.tif")
    args = parser.parse_args()

    if args.rows * args.cols != args.n_images:
        print(f"ERROR: rows*cols ({args.rows}*{args.cols}={args.rows * args.cols}) must equal n_images ({args.n_images})")
        sys.exit(1)

    images = find_images(args.plate_folder, channel=args.channel)
    if not images:
        print(f"ERROR: No TIF images found in {args.plate_folder}" +
              (f" matching channel '{args.channel}'" if args.channel else ""))
        sys.exit(1)

    print(f"Found {len(images)} images" +
          (f" (channel: {args.channel})" if args.channel else ""))

    montage, selected = make_montage(
        images, n_images=args.n_images, rows=args.rows, cols=args.cols, crop_size=args.crop_size,
    )

    if args.output:
        out_path = args.output
    else:
        plate_name = os.path.basename(os.path.normpath(args.plate_folder))
        channel_tag = f"_{args.channel}" if args.channel else ""
        out_path = os.path.join(
            args.plate_folder,
            f"{plate_name}{channel_tag}_montage_{args.rows}x{args.cols}.tif"
        )

    tifffile.imwrite(out_path, montage)
    print(f"Montage saved: {out_path}")
    print(f"  Shape: {montage.shape}, dtype: {montage.dtype}")


if __name__ == "__main__":
    main()
