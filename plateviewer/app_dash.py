#!/usr/bin/env python3
"""
CRISPR PlateViewer — Dash web interface for high-content screening QC.

Usage:
    plateviewer
    plateviewer --port 8051
    python -m plateviewer.app_dash
"""

import argparse
import glob
import os
import re

import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go

from PIL import Image

from plateviewer import config as cfg
from plateviewer.paths import plate_output_dir
from plateviewer.plate import (find_images, detect_channels, detect_fields, center_field,
                               parse_well_spec, filter_images_by_wells, sort_by_field,
                               PLATE_FORMATS)
from plateviewer.image import numpy_to_b64png
from plateviewer.montage import make_montage, make_well_montage, make_contact_sheet
from plateviewer.heatmaps import (compute_intensity_heatmap, compute_focus_heatmap,
                                  compute_plls_heatmap)

# ---------------------------------------------------------------------------
# Plotly figure helper
# ---------------------------------------------------------------------------

def make_plate_heatmap_figure(heatmap, title="", plate_rows=16, plate_cols=24):
    """Create a Plotly heatmap figure shaped as a plate."""
    row_labels = [chr(ord('A') + i) for i in range(plate_rows)]
    col_labels = [str(i + 1) for i in range(plate_cols)]
    fig = go.Figure(data=go.Heatmap(
        z=heatmap,
        x=col_labels,
        y=row_labels,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Well: %{y}%{x}<br>Value: %{z:.1f}<extra></extra>',
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(autorange='reversed', scaleanchor='x', dtick=1,
                   constrain='domain'),
        xaxis=dict(side='top', dtick=1, constrain='domain'),
        width=cfg.HEATMAP_FIG_WIDTH,
        height=cfg.HEATMAP_FIG_HEIGHT,
        margin=dict(l=40, r=40, t=60, b=20),
        plot_bgcolor='white',
    )
    return fig


def make_contact_sheet_figure(sheet, well_positions, plate_rows, plate_cols,
                               channel, n_fields, preferred_field):
    """Create a Plotly figure overlaying well hover targets on a contact sheet image."""
    b64 = numpy_to_b64png(sheet)
    h, w = sheet.shape
    thumb_size = cfg.CONTACT_THUMB_SIZE
    spacing = cfg.CONTACT_SPACING
    step = thumb_size + spacing

    col_tick_vals = [i * step + thumb_size // 2 for i in range(plate_cols)]
    col_tick_labels = [str(i + 1) for i in range(plate_cols)]
    row_tick_vals = [i * step + thumb_size // 2 for i in range(plate_rows)]
    row_tick_labels = [chr(ord('A') + i) for i in range(plate_rows)]

    fig = go.Figure()
    fig.add_layout_image(
        source=b64, xref="x", yref="y",
        x=0, y=0, sizex=w, sizey=h,
        sizing="stretch", layer="below",
    )
    xs = [pos[0] for pos in well_positions.values()]
    ys = [pos[1] for pos in well_positions.values()]
    labels = list(well_positions.keys())
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers',
        marker=dict(size=20, opacity=0),
        text=labels,
        hovertemplate='Well: %{text}<extra></extra>',
    ))
    fig.update_xaxes(
        range=[0, w], side='top',
        tickvals=col_tick_vals, ticktext=col_tick_labels,
        tickfont=dict(size=11), showgrid=False, zeroline=False,
        constrain='domain',
    )
    fig.update_yaxes(
        range=[h, 0],
        tickvals=row_tick_vals, ticktext=row_tick_labels,
        tickfont=dict(size=11), showgrid=False, zeroline=False,
        scaleanchor='x', constrain='domain',
    )
    fig_width = cfg.CONTACT_FIG_WIDTH
    fig_height = int(fig_width * h / w) + 60
    fig.update_layout(
        width=fig_width, height=fig_height,
        margin=dict(l=20, r=10, t=50, b=10),
        title=f"Plate Thumbnails — field {preferred_field} per well ({n_fields} fields detected, {channel} channel)",
        showlegend=False,
        plot_bgcolor='black',
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
        html.Button("Browse", id='btn-browse', n_clicks=0, style={'marginRight': '5px'}),
        html.Button("Load", id='btn-load', n_clicks=0),
    ], style={'margin': '10px'}),

    html.Div(id='folder-status', style={'margin': '10px', 'color': '#666'}),

    # -- Channel selector + Plate format --
    html.Div([
        html.Label("Channel:"),
        dcc.Dropdown(id='channel-dropdown', style={'width': '200px', 'display': 'inline-block'}),
        html.Label("Plate format:", style={'marginLeft': '30px'}),
        dcc.Dropdown(
            id='plate-format-dropdown',
            options=[{'label': '384-well (16×24)', 'value': '384'},
                     {'label': '96-well (8×12)', 'value': '96'}],
            value='384',
            style={'width': '200px', 'display': 'inline-block'},
            clearable=False,
        ),
    ], style={'margin': '10px', 'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),

    # -- Control wells + Save All --
    html.Div([
        html.Label("Control wells:"),
        dcc.Input(id='control-wells-input', type='text',
                  placeholder='e.g. A-H:5, I-P:13',
                  style={'width': '300px'}),
        html.Span("Syntax: A-H:5, col:1-2, A05",
                  style={'color': '#888', 'fontSize': '12px'}),
        html.Button("Save All Plots", id='btn-save-all', n_clicks=0,
                    style={'marginLeft': 'auto'}),
        dcc.Loading(html.Span(id='save-all-status', style={'color': '#666'}),
                    type='dot'),
    ], style={'margin': '10px', 'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),

    # -- Output folder --
    html.Div([
        html.Label("Output folder:"),
        dcc.Input(id='output-folder', type='text',
                  placeholder='Auto-set on Load (or choose manually)',
                  style={'width': '600px', 'marginRight': '10px'}),
        html.Button("Browse", id='btn-browse-output', n_clicks=0),
    ], style={'margin': '10px'}),

    # -- Tabs --
    dcc.Tabs(id='tabs', value='tab-montage', children=[
        dcc.Tab(label='Random Montage', value='tab-montage'),
        dcc.Tab(label='Control Montage', value='tab-controls'),
        dcc.Tab(label='Single Well', value='tab-well'),
        dcc.Tab(label='Plate Thumbnails', value='tab-contact'),
        dcc.Tab(label='Intensity Heatmap', value='tab-intensity'),
        dcc.Tab(label='Focus Heatmap', value='tab-focus'),
    ]),

    # -- Tab panels (all rendered, visibility toggled) --
    html.Div(id='panel-montage', style={'margin': '10px'}, children=[
        html.Div([
            html.Button("Generate Montage", id='btn-montage', n_clicks=0),
            html.Button("Save", id='btn-save-montage', n_clicks=0, style={'marginLeft': '10px'}),
            html.Span(id='save-montage-status', style={'color': '#666', 'marginLeft': '10px'}),
        ], style={'margin': '10px 0'}),
        dcc.Loading(html.Div(id='montage-output')),
    ]),
    html.Div(id='panel-controls', style={'margin': '10px', 'display': 'none'}, children=[
        html.Div([
            html.Button("Generate Control Montage", id='btn-controls', n_clicks=0),
            html.Button("Save", id='btn-save-controls', n_clicks=0, style={'marginLeft': '10px'}),
            html.Span(id='save-controls-status', style={'color': '#666', 'marginLeft': '10px'}),
        ], style={'margin': '10px 0'}),
        dcc.Loading(html.Div(id='controls-output')),
    ]),
    html.Div(id='panel-well', style={'margin': '10px', 'display': 'none'}, children=[
        html.Div([
            html.Label("Well ID: "),
            dcc.Input(id='well-id-input', type='text', placeholder='e.g. A05',
                      style={'width': '100px', 'marginRight': '10px'}),
            html.Button("Generate", id='btn-well', n_clicks=0),
            html.Button("Save", id='btn-save-well', n_clicks=0, style={'marginLeft': '10px'}),
            html.Span(id='save-well-status', style={'color': '#666', 'marginLeft': '10px'}),
        ], style={'margin': '10px 0'}),
        dcc.Loading(html.Div(id='well-output')),
    ]),
    html.Div(id='panel-contact', style={'margin': '10px', 'display': 'none'}, children=[
        html.Button("Generate Plate Thumbnails", id='btn-contact', n_clicks=0,
                    style={'margin': '10px 0'}),
        dcc.Loading(html.Div(id='contact-output')),
    ]),
    html.Div(id='panel-intensity', style={'margin': '10px', 'display': 'none'}, children=[
        html.Button("Compute Intensity Heatmap", id='btn-intensity', n_clicks=0,
                    style={'margin': '10px 0'}),
        dcc.Loading(html.Div(id='intensity-output')),
    ]),
    html.Div(id='panel-focus', style={'margin': '10px', 'display': 'none'}, children=[
        html.Div([
            html.P([
                html.B("Variance of Laplacian (VoL): "),
                "Measures how much high-frequency structure (edges/texture) is present. Higher VoL values usually correspond to sharper, more structured images, however VoL increases also with cell confluency and with increasing noise.",
            ]),
            html.P([
                html.B("Power Log-Log Slope (PLLS):"), " (Bray et al., 2012) "
                "Summarizes how quickly spectral power falls off with spatial frequency. Lower values (i.e. more negative slopes) indicate blur (loss of high frequencies), while larger values (i.e. slopes closer to zero) suggest a flatter, noise-dominated spectrum.",
            ]),
            html.P([
                html.B("Reading both together: "),
                "Use VoL as 'how much structure/edge content exists' and PLLS as 'is high-frequency content preserved vs suppressed': true defocus tends to decrease PLLS (often with reduced VoL), whereas noisy/artefactual images can show high VoL and PLLS.",
            ]),
        ], style={'fontSize': '13px', 'color': '#444', 'lineHeight': '1.5',
                  'maxWidth': '900px', 'padding': '10px', 'background': '#f8f8f8',
                  'borderRadius': '6px', 'border': '1px solid #ddd', 'margin': '10px 0'}),
        html.Button("Compute Focus Heatmap", id='btn-focus', n_clicks=0,
                    style={'margin': '10px 0'}),
        dcc.Loading(html.Div(id='focus-output')),
    ]),

    # -- Hidden store --
    dcc.Store(id='plate-folder-store'),
], style={'fontFamily': 'sans-serif', 'maxWidth': '1200px', 'margin': '0 auto'})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _browse_folder_subprocess(title="Select folder", initial=None):
    """Open a native folder-picker dialog in a subprocess.

    Dash callbacks run in Flask worker threads. On macOS, tkinter requires the
    main thread, so calling it directly from a callback raises NSException.
    Spawning a subprocess gives tkinter its own main thread.
    """
    import subprocess
    import sys
    code = "\n".join([
        "import tkinter as tk",
        "from tkinter import filedialog",
        "root = tk.Tk()",
        "root.withdraw()",
        "root.attributes('-topmost', True)",
        f"result = filedialog.askdirectory(title={repr(title)}, initialdir={repr(initial or '')})",
        "print(result if result else '', end='')",
        "root.destroy()",
    ])
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    folder = proc.stdout.strip()
    return folder if folder else None


def _resolve_output_dir(output_folder, plate_folder):
    """Return the output directory to use, creating it if needed.

    Uses output_folder if provided, otherwise falls back to
    the centralised plate output dir under ~/PlateViewer_output/.
    """
    out_dir = output_folder.strip() if output_folder else ""
    if not out_dir:
        out_dir = plate_output_dir(plate_folder)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output('plate-folder', 'value'),
    Input('btn-browse', 'n_clicks'),
    State('plate-folder', 'value'),
    prevent_initial_call=True,
)
def browse_folder(n_clicks, current_folder):
    initial = None
    if current_folder:
        parent = os.path.dirname(os.path.normpath(current_folder))
        if os.path.isdir(parent):
            initial = parent
    folder = _browse_folder_subprocess("Select plate folder", initial=initial)
    return folder if folder else dash.no_update


@callback(
    Output('output-folder', 'value', allow_duplicate=True),
    Input('btn-browse-output', 'n_clicks'),
    prevent_initial_call=True,
)
def browse_output_folder(n_clicks):
    initial = cfg.APP_DATA_DIR if os.path.isdir(cfg.APP_DATA_DIR) else None
    folder = _browse_folder_subprocess("Select output folder", initial=initial)
    return folder if folder else dash.no_update


@callback(
    Output('channel-dropdown', 'options'),
    Output('channel-dropdown', 'value'),
    Output('folder-status', 'children'),
    Output('plate-folder-store', 'data'),
    Output('output-folder', 'value'),
    Output('plate-folder', 'value', allow_duplicate=True),
    Input('btn-load', 'n_clicks'),
    State('plate-folder', 'value'),
    prevent_initial_call=True,
)
def load_plate(n_clicks, folder):
    if folder:
        from urllib.parse import unquote
        folder = unquote(folder.strip().removeprefix('file://'))
    if not folder or not os.path.isdir(folder):
        return [], None, "Invalid folder path.", None, "", dash.no_update
    channels = detect_channels(folder)
    if not channels:
        return [], None, "No TIF images found in folder.", None, "", dash.no_update
    tifs = glob.glob(os.path.join(folder, "*.tif"))
    options = [{'label': ch, 'value': ch} for ch in channels]
    status = f"Loaded: {len(tifs)} images, {len(channels)} channels."

    out_folder = plate_output_dir(folder)

    return options, channels[0], status, folder, out_folder, folder


@callback(
    Output('panel-montage', 'style'),
    Output('panel-controls', 'style'),
    Output('panel-well', 'style'),
    Output('panel-contact', 'style'),
    Output('panel-intensity', 'style'),
    Output('panel-focus', 'style'),
    Input('tabs', 'value'),
)
def toggle_tab_visibility(tab):
    hidden = {'margin': '10px', 'display': 'none'}
    visible = {'margin': '10px', 'display': 'block'}
    tab_map = {'tab-montage': 0, 'tab-controls': 1, 'tab-well': 2,
               'tab-contact': 3, 'tab-intensity': 4, 'tab-focus': 5}
    styles = [hidden] * 6
    styles[tab_map.get(tab, 0)] = visible
    return styles


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
    if len(images) < cfg.MONTAGE_N_IMAGES:
        return html.Div(f"Not enough images ({len(images)} found, need {cfg.MONTAGE_N_IMAGES}).")
    montage, selected = make_montage(images, crop_size=cfg.MONTAGE_CROP_SIZE)
    b64 = numpy_to_b64png(montage)
    return html.Div([
        html.P(f"Showing {cfg.MONTAGE_N_IMAGES} random images from {len(images)} ({channel} channel)"),
        html.Img(src=b64, style={'width': '100%', 'imageRendering': 'auto'}),
    ])


@callback(
    Output('intensity-output', 'children'),
    Input('btn-intensity', 'n_clicks'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    State('plate-format-dropdown', 'value'),
    prevent_initial_call=True,
)
def generate_intensity_heatmap(n_clicks, channel, folder, plate_fmt):
    if not folder or not channel:
        return html.Div("Load a plate folder first.")
    plate_rows, plate_cols = PLATE_FORMATS[plate_fmt]
    heatmap = compute_intensity_heatmap(folder, channel, plate_rows, plate_cols)
    fig = make_plate_heatmap_figure(heatmap, title=f"Mean Intensity — {channel}",
                                    plate_rows=plate_rows, plate_cols=plate_cols)
    return dcc.Graph(figure=fig)


@callback(
    Output('focus-output', 'children'),
    Input('btn-focus', 'n_clicks'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    State('plate-format-dropdown', 'value'),
    prevent_initial_call=True,
)
def generate_focus_heatmap(n_clicks, channel, folder, plate_fmt):
    if not folder or not channel:
        return html.Div("Load a plate folder first.")
    plate_rows, plate_cols = PLATE_FORMATS[plate_fmt]
    heatmap_lap = compute_focus_heatmap(folder, channel, plate_rows, plate_cols)
    fig_lap = make_plate_heatmap_figure(heatmap_lap, title=f"Focus (Laplacian variance) — {channel}",
                                        plate_rows=plate_rows, plate_cols=plate_cols)
    heatmap_plls = compute_plls_heatmap(folder, channel, plate_rows, plate_cols)
    fig_plls = make_plate_heatmap_figure(heatmap_plls, title=f"Focus (Power Log-Log Slope) — {channel}",
                                         plate_rows=plate_rows, plate_cols=plate_cols)
    return html.Div([
        dcc.Graph(figure=fig_lap),
        dcc.Graph(figure=fig_plls),
    ])


@callback(
    Output('controls-output', 'children'),
    Input('btn-controls', 'n_clicks'),
    State('control-wells-input', 'value'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    State('plate-format-dropdown', 'value'),
    prevent_initial_call=True,
)
def generate_controls_montage(n_clicks, well_spec, channel, folder, plate_fmt):
    if not folder or not channel:
        return html.Div("Load a plate folder first.")
    if not well_spec:
        return html.Div("Enter control well specification.")
    plate_rows, plate_cols = PLATE_FORMATS[plate_fmt]
    well_set = parse_well_spec(well_spec, plate_rows=plate_rows)
    if not well_set:
        return html.Div(f"Could not parse well specification: {well_spec}")
    images = find_images(folder, channel=channel)
    filtered = filter_images_by_wells(images, well_set)
    if not filtered:
        return html.Div(f"No images found for wells: {sorted(well_set)}")
    if len(filtered) < cfg.MONTAGE_N_IMAGES:
        return html.Div(f"Not enough control images ({len(filtered)} found, need {cfg.MONTAGE_N_IMAGES}).")
    montage, selected = make_montage(filtered, crop_size=cfg.MONTAGE_CROP_SIZE)
    b64 = numpy_to_b64png(montage)
    return html.Div([
        html.P(f"Showing {cfg.MONTAGE_N_IMAGES} random images from {len(filtered)} control images, "
               f"{len(well_set)} wells ({channel} channel)"),
        html.Img(src=b64, style={'width': '100%', 'imageRendering': 'auto'}),
    ])


@callback(
    Output('well-output', 'children'),
    Input('btn-well', 'n_clicks'),
    State('well-id-input', 'value'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    prevent_initial_call=True,
)
def generate_well_montage(n_clicks, well_id, channel, folder):
    if not folder or not channel:
        return html.Div("Load a plate folder first.")
    if not well_id or not re.match(r'^[A-P]\d+$', well_id.strip(), re.IGNORECASE):
        return html.Div("Enter a valid well ID (e.g. A05).")
    well_id = f"{well_id[0].upper()}{int(well_id[1:]):02d}"
    images = find_images(folder, channel=channel)
    filtered = filter_images_by_wells(images, {well_id})
    if not filtered:
        return html.Div(f"No images found for well {well_id}.")
    sort_by_field(filtered)
    montage = make_well_montage(filtered)
    b64 = numpy_to_b64png(montage)
    return html.Div([
        html.P(f"Well {well_id} — {len(filtered)} fields ({channel} channel)"),
        html.Img(src=b64, style={'width': '100%', 'imageRendering': 'auto'}),
    ])


# -- Save individual montage callbacks --

@callback(
    Output('save-montage-status', 'children'),
    Input('btn-save-montage', 'n_clicks'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    State('output-folder', 'value'),
    prevent_initial_call=True,
)
def save_montage(n_clicks, channel, folder, output_folder):
    if not folder or not channel:
        return "Load a plate folder first."
    images = find_images(folder, channel=channel)
    if len(images) < cfg.MONTAGE_N_IMAGES:
        return f"Not enough images ({len(images)})."
    try:
        out_dir = _resolve_output_dir(output_folder, folder)
        montage, _ = make_montage(images, crop_size=cfg.MONTAGE_CROP_SIZE)
        path = os.path.join(out_dir, f"{channel}_montage.png")
        Image.fromarray(montage).save(path)
        return f"Saved: {path}"
    except PermissionError:
        return "Cannot save: output folder is not writable."


@callback(
    Output('save-controls-status', 'children'),
    Input('btn-save-controls', 'n_clicks'),
    State('control-wells-input', 'value'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    State('plate-format-dropdown', 'value'),
    State('output-folder', 'value'),
    prevent_initial_call=True,
)
def save_controls_montage(n_clicks, well_spec, channel, folder, plate_fmt, output_folder):
    if not folder or not channel:
        return "Load a plate folder first."
    if not well_spec:
        return "Enter control well specification."
    plate_rows, _ = PLATE_FORMATS[plate_fmt]
    well_set = parse_well_spec(well_spec, plate_rows=plate_rows)
    if not well_set:
        return "Could not parse well specification."
    images = find_images(folder, channel=channel)
    filtered = filter_images_by_wells(images, well_set)
    if len(filtered) < cfg.MONTAGE_N_IMAGES:
        return f"Not enough control images ({len(filtered)})."
    try:
        out_dir = _resolve_output_dir(output_folder, folder)
        montage, _ = make_montage(filtered, crop_size=cfg.MONTAGE_CROP_SIZE)
        path = os.path.join(out_dir, f"{channel}_controls.png")
        Image.fromarray(montage).save(path)
        return f"Saved: {path}"
    except PermissionError:
        return "Cannot save: output folder is not writable."


@callback(
    Output('save-well-status', 'children'),
    Input('btn-save-well', 'n_clicks'),
    State('well-id-input', 'value'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    State('output-folder', 'value'),
    prevent_initial_call=True,
)
def save_well_montage(n_clicks, well_id, channel, folder, output_folder):
    if not folder or not channel:
        return "Load a plate folder first."
    if not well_id or not re.match(r'^[A-P]\d+$', well_id.strip(), re.IGNORECASE):
        return "Enter a valid well ID."
    well_id = f"{well_id[0].upper()}{int(well_id[1:]):02d}"
    images = find_images(folder, channel=channel)
    filtered = filter_images_by_wells(images, {well_id})
    if not filtered:
        return f"No images for well {well_id}."
    sort_by_field(filtered)
    try:
        out_dir = _resolve_output_dir(output_folder, folder)
        montage = make_well_montage(filtered)
        path = os.path.join(out_dir, f"{channel}_well_{well_id}.png")
        Image.fromarray(montage).save(path)
        return f"Saved: {path}"
    except PermissionError:
        return "Cannot save: output folder is not writable."


@callback(
    Output('contact-output', 'children'),
    Input('btn-contact', 'n_clicks'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    State('plate-format-dropdown', 'value'),
    prevent_initial_call=True,
)
def generate_contact_sheet(n_clicks, channel, folder, plate_fmt):
    if not folder or not channel:
        return html.Div("Load a plate folder first.")
    images = find_images(folder, channel=channel)
    if not images:
        return html.Div("No images found.")

    plate_rows, plate_cols = PLATE_FORMATS[plate_fmt]
    fields = detect_fields(folder)
    pref_field = center_field(fields)
    n_fields = len(fields) if fields else '?'

    sheet, well_positions = make_contact_sheet(
        images, plate_rows=plate_rows, plate_cols=plate_cols,
        preferred_field=pref_field,
    )
    fig = make_contact_sheet_figure(sheet, well_positions, plate_rows, plate_cols,
                                    channel, n_fields, pref_field)
    return dcc.Graph(figure=fig, style={'width': '100%'}, config={'responsive': True})


@callback(
    Output('save-all-status', 'children'),
    Input('btn-save-all', 'n_clicks'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    State('plate-format-dropdown', 'value'),
    State('control-wells-input', 'value'),
    State('output-folder', 'value'),
    prevent_initial_call=True,
)
def save_all_plots(n_clicks, channel, folder, plate_fmt, well_spec, output_folder):
    if not folder or not channel:
        return "Load a plate folder first."

    plate_rows, plate_cols = PLATE_FORMATS[plate_fmt]
    try:
        out_dir = _resolve_output_dir(output_folder, folder)
    except PermissionError:
        return "Cannot save: output folder is not writable."
    saved = []
    print(f"[Save All] Saving plots to {out_dir}/ ...")

    # 1. Random montage
    print("[Save All] 1/6 Generating random montage...")
    images = find_images(folder, channel=channel)
    if len(images) >= cfg.MONTAGE_N_IMAGES:
        montage, _ = make_montage(images, crop_size=cfg.MONTAGE_CROP_SIZE)
        path = os.path.join(out_dir, f"{channel}_montage.png")
        Image.fromarray(montage).save(path)
        saved.append("montage")

    # 2. Control montage (only if well spec provided)
    print("[Save All] 2/6 Generating control montage...")
    if well_spec:
        well_set = parse_well_spec(well_spec, plate_rows=plate_rows)
        if well_set:
            filtered = filter_images_by_wells(images, well_set)
            if len(filtered) >= cfg.MONTAGE_N_IMAGES:
                ctrl_montage, _ = make_montage(filtered, crop_size=cfg.MONTAGE_CROP_SIZE)
                path = os.path.join(out_dir, f"{channel}_controls.png")
                Image.fromarray(ctrl_montage).save(path)
                saved.append("controls")

    # 3. Plate thumbnails
    print("[Save All] 3/6 Generating plate thumbnails...")
    fields = detect_fields(folder)
    pref_field = center_field(fields)
    n_fields = len(fields) if fields else '?'
    sheet, well_positions = make_contact_sheet(
        images, plate_rows=plate_rows, plate_cols=plate_cols,
        preferred_field=pref_field,
    )
    fig_contact = make_contact_sheet_figure(sheet, well_positions, plate_rows, plate_cols,
                                            channel, n_fields, pref_field)
    path = os.path.join(out_dir, f"{channel}_thumbnails.png")
    fig_contact.write_image(path, scale=cfg.EXPORT_SCALE)
    saved.append("thumbnails")

    # 4. Intensity heatmap
    print("[Save All] 4/6 Computing intensity heatmap...")
    heatmap_int = compute_intensity_heatmap(folder, channel, plate_rows, plate_cols)
    fig_int = make_plate_heatmap_figure(heatmap_int, title=f"Mean Intensity — {channel}",
                                        plate_rows=plate_rows, plate_cols=plate_cols)
    path = os.path.join(out_dir, f"{channel}_intensity.png")
    fig_int.write_image(path, scale=cfg.EXPORT_SCALE)
    saved.append("intensity")

    # 5. Focus heatmap (Laplacian variance)
    print("[Save All] 5/6 Computing focus heatmap (Laplacian variance)...")
    heatmap_foc = compute_focus_heatmap(folder, channel, plate_rows, plate_cols)
    fig_foc = make_plate_heatmap_figure(heatmap_foc, title=f"Focus (Laplacian variance) — {channel}",
                                        plate_rows=plate_rows, plate_cols=plate_cols)
    path = os.path.join(out_dir, f"{channel}_focus.png")
    fig_foc.write_image(path, scale=cfg.EXPORT_SCALE)
    saved.append("focus")

    # 6. Focus heatmap (PLLS)
    print("[Save All] 6/6 Computing focus heatmap (PLLS)...")
    heatmap_plls = compute_plls_heatmap(folder, channel, plate_rows, plate_cols)
    fig_plls = make_plate_heatmap_figure(heatmap_plls, title=f"Focus (Power Log-Log Slope) — {channel}",
                                         plate_rows=plate_rows, plate_cols=plate_cols)
    path = os.path.join(out_dir, f"{channel}_plls.png")
    fig_plls.write_image(path, scale=cfg.EXPORT_SCALE)
    saved.append("plls")

    print(f"[Save All] Done — saved {len(saved)} plots.")
    return f"Saved {len(saved)} plots to {out_dir}/: {', '.join(saved)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import threading
    import webbrowser
    parser = argparse.ArgumentParser(description="PlateViewer web interface")
    parser.add_argument("--port", type=int, default=8050, help="Port (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    args = parser.parse_args()
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
