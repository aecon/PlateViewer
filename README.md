# PlateViewer

Quality control visualization tools for arrayed high-content screening experiments (384-well plates).

## Setup

```bash
conda create -n PlateViewer python=3.11 -y
conda run -n PlateViewer pip install numpy scikit-image tifffile dash plotly pillow scipy
```

## Web Interface

```bash
conda run -n PlateViewer python app.py
```

The browser opens automatically. Use `--port 8051` to change the port.

1. Enter the plate folder path and click **Load**
2. Select a channel from the dropdown
3. Use the tabs:
   - **Random Montage** — 4x8 grid of randomly sampled images with well/field labels
   - **Intensity Heatmap** — mean pixel intensity per well (16x24 plate layout)
   - **Focus Heatmap** — Laplacian variance per well to detect out-of-focus fields

Heatmap results are cached as `.npy` files in the plate folder. Delete them to force recomputation.

## CLI Montage Tool

```bash
conda run -n PlateViewer python montage.py /path/to/plate_folder --channel Blue
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-c, --channel` | all | Filter by channel name (e.g. `Blue`, `Green2`, `FarRed`) |
| `-n, --n_images` | 32 | Number of images to sample |
| `--rows` | 4 | Montage grid rows |
| `--cols` | 8 | Montage grid columns |
| `--crop_size` | 1020 | Center crop size in pixels |
| `-o, --output` | auto | Output TIF path |

## Project Structure

```
plate.py        — Plate-level logic: file discovery, filename parsing, well utilities
image.py        — Image utilities: uint8 conversion, label burning, PNG encoding
montage.py      — Montage assembly + CLI entry point
heatmaps.py     — Plate heatmap computation (threaded I/O, disk caching)
app.py          — Dash web UI (thin, imports from above modules)
```

## Expected File Naming

Images must follow the GE InCell naming convention:

```
ROW - COL(fld FIELD wv WAVELENGTH - CHANNEL).tif
```

Example: `A - 01(fld 1 wv 390 - Blue).tif`
