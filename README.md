# PlateViewer

Quality control visualization tools for arrayed high-content screening experiments (384-well and 96-well plates).

PlateViewer helps microscopists and screening scientists quickly assess image quality across entire plates. It flags out-of-focus wells, reveals intensity patterns such as edge effects or dispensing artifacts, and lets you visually inspect any well or field — all from a browser-based interface.

## Requirements

- Python 3.11+
- Images must be TIFF files (`.tif`) following the naming convention described in [Expected File Naming](#expected-file-naming)

## Installation

### Steps:
1. Clone this repository to your local computer:
   * Navigate to the folder where you want to clone (download) this repository.
   * Clone the repo:
```bash
git clone https://github.com/aecon/PlateViewer.git
```

2. Install the package:

```bash
cd PlateViewer
conda create -n PlateViewer python=3.11
conda activate PlateViewer
pip install .
```

On Debian/Ubuntu, the **Browse** button requires `python3-tk`, which is not available through pip. In this case install `python3-tk` as below, and then follow step 2 to re-install PlateViewer.

```bash
sudo apt install python3-tk
```

### for development:
For development (editable install — changes take effect immediately):

```bash
pip install -e .
```

### reproducing the exact tested environment:
To reproduce the exact tested environment, use the pinned `requirements.txt` instead:

```bash
pip install -r requirements.txt
```


## Usage

### Dash (default)

```bash
conda activate PlateViewer
plateviewer
```

The browser opens automatically. Use `--port 8051` to change the port.

### Streamlit (optional alternative UI)

```bash
conda activate PlateViewer  
pip install streamlit
streamlit run plateviewer/app_streamlit.py
```

Both interfaces expose identical functionality. Dash is the default; Streamlit is an optional alternative that requires no extra configuration beyond `pip install streamlit`.

#### Workflow:

1. Enter the plate folder path (or click **Browse**, or drag-and-drop) and click **Load**
2. Select a channel and plate format (384-well or 96-well)
3. Optionally enter control well specifications
4. Use the tabs:
   - **Random Montage**: 4x8 grid of randomly sampled images with well/field labels
   - **Control Montage**: 4x8 grid sampled from user-specified control wells
   - **Single Well**: all fields for a chosen well in an auto-sized grid (full images, 4x downsampled)
   - **Plate Thumbnails**: plate-layout overview with one thumbnail per well (auto-detected center field, hover for well ID)
   - **Intensity Heatmap**: mean pixel intensity per well (plate layout)
   - **Focus Heatmap**: two heatmaps — Variance of Laplacian (VoL) and Power Log-Log Slope (PLLS) — with an interpretation guide explaining how to read them together.
5. Click **Save All Plots** to export all plots as PNGs. By default they are saved to `~/PlateViewer_output/<plate_name>/output/`; use the output folder field to choose a different location.

_Note: Heatmap results are cached as `.npy` files under `~/PlateViewer_output/<plate_name>/cache/`. Delete them to force recomputation._


#### Expected File Naming: Images must follow the naming convention below!  
```
PREFIX_ROW - COL(fld FIELD wv WAVELENGTH - CHANNEL).tif
```
Where `PREFIX_` can also be empty. Examples:  
*`A - 01(fld 1 wv 390 - Blue).tif`  
*`Plate1_A - 01(fld 1 wv 390 - Blue).tif`  


#### Control well syntax:

| Format | Meaning |
|--------|---------|
| `A-H:5` | Rows A through H, column 5 |
| `I-P:13` | Rows I through P, column 13 |
| `col:1-2` | All rows, columns 1 and 2 |
| `A01` | Single well |
| `A01-A09` | Well range within a row |

Combine with commas: `A-H:5, I-P:13`


## Functionality

#### Random Montage

Inspect a subset of randomly selected images from the entire plate, corresponding to the chosen channel.

![Random Montage](docs/images/random_montage.png)

#### Single Well

Inspect all fields of a specific well. Enter a well ID (e.g. A05) to see every acquired field arranged in a grid, useful for investigating outliers spotted on heatmaps.

![Single Well](docs/images/single_well.png)

#### Plate Thumbnails

Bird's-eye view of the entire plate. Each cell shows a thumbnail of the center field for that well. Hover over any thumbnail to see the well ID.

![Plate Thumbnails](docs/images/plate_thumbnails.png)

#### Intensity Heatmap

![Intensity Heatmap](docs/images/intensity_heatmap.png)

Mean pixel intensity per well. Reveals systematic patterns such as edge effects or dispensing artifacts.

#### Focus Heatmaps

Two complementary metrics are shown:

- **Variance of Laplacian (VoL)** — measures edge/texture content. Higher values indicate sharper images, but VoL also increases with cell confluency and noise.
- **Power Log-Log Slope (PLLS)** (Bray et al., 2012) — summarizes how quickly spectral power falls off with spatial frequency. More negative values indicate blur; values near zero suggest sharp or noise-dominated images.

![Focus Heatmaps](docs/images/focus_heatmaps.png)


## License

[MIT License](LICENSE)  
Copyright (c) 2026 E. Athena Economides

If you use this software in your research, please cite it as shown in the "Cite this repository" button, and the [CITATION.cff](https://github.com/aecon/PlateViewer/blob/main/CITATION.cff) file.

PlateViewer was developed during Athena's Postdoc at:  
  *Prof. Adriano Aguzzi Lab*  
  *Institute of Neuropathology*  
  *University of Zurich & University Hospital Zurich*  
  *Schmelzbergstrasse 12*  
  *CH-8091 Zurich*  
  *Switzerland*  
