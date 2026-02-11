#!/usr/bin/env python3
"""
Generate a random montage of images from a high-content screening plate folder.

Randomly selects N images (default 32) from a plate folder and assembles them
into a montage TIF (4 rows x 8 columns). Images are converted to uint8 on load
to save memory. Output is uint8 TIF.

Usage:
    conda run -n PlateViewer python montage.py /path/to/plate_folder
    conda run -n PlateViewer python montage.py /path/to/plate_folder --channel Blue --n_images 64 --rows 8 --cols 8
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile


def uint16_to_uint8(img, plow=1, phigh=99):
    """Convert a uint16 image to uint8 using percentile contrast stretch."""
    img = img.astype(np.float32)
    lo = np.percentile(img, plow)
    hi = np.percentile(img, phigh)
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255)
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)


def parse_filename(filepath):
    """Extract well and field info from a filename like 'A - 01(fld 1 wv 390 - Blue).tif'."""
    basename = os.path.basename(filepath)
    m = re.match(r'^([A-P])\s*-\s*(\d+)\(fld\s+(\d+)\s+wv\s+.+\)\.tif$', basename)
    if m:
        row, col, fld = m.group(1), m.group(2), m.group(3)
        return f"{row}{col} f{fld}"
    return ""


def burn_label(tile, label, font_size=48):
    """Burn a text label onto the top-center of a uint8 tile."""
    img_pil = Image.fromarray(tile, mode='L')
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    x = (tile.shape[1] - text_w) // 2
    y = 5
    # draw dark outline then white text for contrast
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            draw.text((x + dx, y + dy), label, fill=0, font=font)
    draw.text((x, y), label, fill=255, font=font)
    return np.array(img_pil)


def find_images(plate_folder, channel=None):
    """Find all TIF images in a plate folder, optionally filtered by channel.

    Excludes any files inside 'output*' subdirectories.
    """
    all_tifs = glob.glob(os.path.join(plate_folder, "*.tif"))
    if channel:
        all_tifs = [f for f in all_tifs if channel in os.path.basename(f)]
    all_tifs.sort()
    return all_tifs


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
        # adjust grid to fit
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
        # center crop
        y0 = max(0, (h - crop_size) // 2)
        x0 = max(0, (w - crop_size) // 2)
        tile = img[y0:y0 + crop_size, x0:x0 + crop_size]
        # pad if image is smaller than crop_size
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


def main():
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
        images,
        n_images=args.n_images,
        rows=args.rows,
        cols=args.cols,
        crop_size=args.crop_size,
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
