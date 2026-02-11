#!/usr/bin/env python3
"""
CRISPR PlateViewer — Dash web interface for high-content screening QC.

Usage:
    conda run -n PlateViewer python app.py
    conda run -n PlateViewer python app.py --port 8051
"""

import argparse
import base64
import concurrent.futures
import io
import os
import re
import glob

import numpy as np
from PIL import Image
import tifffile
import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go

from montage import find_images, make_montage, uint16_to_uint8, parse_filename

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_channels(plate_folder):
    """Scan filenames in a plate folder and return the list of channel names."""
    tifs = glob.glob(os.path.join(plate_folder, "*.tif"))
    channels = set()
    for f in tifs[:200]:  # sample first 200 files for speed
        m = re.search(r'wv\s+\d+\s*-\s*(\w+)', os.path.basename(f))
        if m:
            channels.add(m.group(1))
    return sorted(channels)


def detect_wells(plate_folder):
    """Scan filenames and return sorted list of unique well IDs (e.g. 'A01')."""
    tifs = glob.glob(os.path.join(plate_folder, "*.tif"))
    wells = set()
    for f in tifs:
        m = re.match(r'^([A-P])\s*-\s*(\d+)\(', os.path.basename(f))
        if m:
            wells.add(f"{m.group(1)}{m.group(2).zfill(2)}")
    return sorted(wells)


def numpy_to_b64png(arr):
    """Convert a uint8 numpy array to a base64-encoded PNG string for Dash."""
    img = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def well_to_row_col(well_id):
    """Convert 'A01' -> (0, 0), 'P24' -> (15, 23)."""
    row = ord(well_id[0]) - ord('A')
    col = int(well_id[1:]) - 1
    return row, col


N_WORKERS = 8  # threads for parallel image reading


def _parse_well_from_path(fpath):
    """Extract well ID from filepath, or None."""
    m = re.match(r'^([A-P])\s*-\s*(\d+)\(', os.path.basename(fpath))
    if m:
        return f"{m.group(1)}{m.group(2).zfill(2)}"
    return None


def _read_and_mean(fpath):
    """Read a TIF, subsample 2x, return (well_id, mean_intensity)."""
    well = _parse_well_from_path(fpath)
    if well is None:
        return None
    img = tifffile.imread(fpath)[::2, ::2]
    return (well, float(np.mean(img)))


def _read_and_focus(fpath):
    """Read a TIF, subsample 2x, return (well_id, laplacian_variance)."""
    from scipy.ndimage import laplace
    well = _parse_well_from_path(fpath)
    if well is None:
        return None
    img = tifffile.imread(fpath)[::2, ::2].astype(np.float32)
    lap = laplace(img)
    return (well, float(np.var(lap)))


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


def make_plate_heatmap_figure(heatmap, title=""):
    """Create a Plotly heatmap figure shaped as a 384-well plate."""
    row_labels = [chr(ord('A') + i) for i in range(16)]
    col_labels = [str(i + 1) for i in range(24)]
    fig = go.Figure(data=go.Heatmap(
        z=heatmap,
        x=col_labels,
        y=row_labels,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Well: %{y}%{x}<br>Value: %{z:.1f}<extra></extra>',
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(autorange='reversed', scaleanchor='x', dtick=1),
        xaxis=dict(side='top', dtick=1),
        width=900,
        height=500,
        margin=dict(l=40, r=40, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="PlateViewer")

app.layout = html.Div([
    html.H1("PlateViewer", style={'textAlign': 'center'}),

    # -- Plate folder input --
    html.Div([
        html.Label("Plate folder:"),
        dcc.Input(id='plate-folder', type='text', placeholder='/path/to/plate/folder',
                  style={'width': '600px', 'marginRight': '10px'}),
        html.Button("Load", id='btn-load', n_clicks=0),
    ], style={'margin': '10px'}),

    html.Div(id='folder-status', style={'margin': '10px', 'color': '#666'}),

    # -- Channel selector --
    html.Div([
        html.Label("Channel:"),
        dcc.Dropdown(id='channel-dropdown', style={'width': '200px', 'display': 'inline-block'}),
    ], style={'margin': '10px'}),

    # -- Tabs for different views --
    dcc.Tabs(id='tabs', value='tab-montage', children=[
        dcc.Tab(label='Random Montage', value='tab-montage'),
        dcc.Tab(label='Intensity Heatmap', value='tab-intensity'),
        dcc.Tab(label='Focus Heatmap', value='tab-focus'),
    ]),

    html.Div(id='tab-content', style={'margin': '10px'}),

    # -- Hidden store for state --
    dcc.Store(id='plate-folder-store'),
], style={'fontFamily': 'sans-serif', 'maxWidth': '1200px', 'margin': '0 auto'})


# -- Load plate folder --
@callback(
    Output('channel-dropdown', 'options'),
    Output('channel-dropdown', 'value'),
    Output('folder-status', 'children'),
    Output('plate-folder-store', 'data'),
    Input('btn-load', 'n_clicks'),
    State('plate-folder', 'value'),
    prevent_initial_call=True,
)
def load_plate(n_clicks, folder):
    if not folder or not os.path.isdir(folder):
        return [], None, "Invalid folder path.", None
    channels = detect_channels(folder)
    if not channels:
        return [], None, "No TIF images found in folder.", None
    tifs = glob.glob(os.path.join(folder, "*.tif"))
    options = [{'label': ch, 'value': ch} for ch in channels]
    return options, channels[0], f"Loaded: {len(tifs)} images, {len(channels)} channels.", folder


# -- Render active tab --
@callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    prevent_initial_call=True,
)
def render_tab(tab, channel, folder):
    if not folder or not channel:
        return html.Div("Load a plate folder first.")

    if tab == 'tab-montage':
        return html.Div([
            html.Button("Generate Montage", id='btn-montage', n_clicks=0,
                        style={'margin': '10px 0'}),
            dcc.Loading(html.Div(id='montage-output')),
        ])
    elif tab == 'tab-intensity':
        return html.Div([
            html.Button("Compute Intensity Heatmap", id='btn-intensity', n_clicks=0,
                        style={'margin': '10px 0'}),
            dcc.Loading(html.Div(id='intensity-output')),
        ])
    elif tab == 'tab-focus':
        return html.Div([
            html.Button("Compute Focus Heatmap", id='btn-focus', n_clicks=0,
                        style={'margin': '10px 0'}),
            dcc.Loading(html.Div(id='focus-output')),
        ])
    return html.Div()


# -- Generate montage --
@callback(
    Output('montage-output', 'children'),
    Input('btn-montage', 'n_clicks'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    prevent_initial_call=True,
)
def generate_montage(n_clicks, channel, folder):
    if not folder or not channel:
        return html.Div("Load a plate folder first.")
    images = find_images(folder, channel=channel)
    if len(images) < 32:
        return html.Div(f"Not enough images ({len(images)} found, need 32).")
    montage, selected = make_montage(images, n_images=32, rows=4, cols=8, crop_size=1020)
    b64 = numpy_to_b64png(montage)
    return html.Div([
        html.P(f"Showing 32 random images from {len(images)} ({channel} channel)"),
        html.Img(src=b64, style={'width': '100%', 'imageRendering': 'auto'}),
    ])


# -- Intensity heatmap --
@callback(
    Output('intensity-output', 'children'),
    Input('btn-intensity', 'n_clicks'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    prevent_initial_call=True,
)
def generate_intensity_heatmap(n_clicks, channel, folder):
    if not folder or not channel:
        return html.Div("Load a plate folder first.")
    heatmap = compute_intensity_heatmap(folder, channel)
    fig = make_plate_heatmap_figure(heatmap, title=f"Median Intensity — {channel}")
    return dcc.Graph(figure=fig)


# -- Focus heatmap --
@callback(
    Output('focus-output', 'children'),
    Input('btn-focus', 'n_clicks'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    prevent_initial_call=True,
)
def generate_focus_heatmap(n_clicks, channel, folder):
    if not folder or not channel:
        return html.Div("Load a plate folder first.")
    heatmap = compute_focus_heatmap(folder, channel)
    fig = make_plate_heatmap_figure(heatmap, title=f"Focus (Laplacian variance) — {channel}")
    return dcc.Graph(figure=fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlateViewer web interface")
    parser.add_argument("--port", type=int, default=8050, help="Port (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    args = parser.parse_args()
    app.run(debug=args.debug, port=args.port)
