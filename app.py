#!/usr/bin/env python3
"""
CRISPR PlateViewer — Dash web interface for high-content screening QC.

Usage:
    conda run -n PlateViewer python app.py
    conda run -n PlateViewer python app.py --port 8051
"""

import argparse
import glob
import math
import os

import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go

from PIL import Image

from plate import (find_images, detect_channels, detect_fields, center_field,
                   parse_well_spec, filter_images_by_wells, PLATE_FORMATS)
from image import numpy_to_b64png
from montage import make_montage, make_contact_sheet
from heatmaps import compute_intensity_heatmap, compute_focus_heatmap

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
        html.Span(id='save-all-status', style={'color': '#666'}),
    ], style={'margin': '10px', 'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),

    # -- Tabs --
    dcc.Tabs(id='tabs', value='tab-montage', children=[
        dcc.Tab(label='Random Montage', value='tab-montage'),
        dcc.Tab(label='Control Montage', value='tab-controls'),
        dcc.Tab(label='Plate Thumbnails', value='tab-contact'),
        dcc.Tab(label='Intensity Heatmap', value='tab-intensity'),
        dcc.Tab(label='Focus Heatmap', value='tab-focus'),
    ]),

    # -- Tab panels (all rendered, visibility toggled) --
    html.Div(id='panel-montage', style={'margin': '10px'}, children=[
        html.Button("Generate Montage", id='btn-montage', n_clicks=0,
                    style={'margin': '10px 0'}),
        dcc.Loading(html.Div(id='montage-output')),
    ]),
    html.Div(id='panel-controls', style={'margin': '10px', 'display': 'none'}, children=[
        html.Button("Generate Control Montage", id='btn-controls', n_clicks=0,
                    style={'margin': '10px 0'}),
        dcc.Loading(html.Div(id='controls-output')),
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
        html.Button("Compute Focus Heatmap", id='btn-focus', n_clicks=0,
                    style={'margin': '10px 0'}),
        dcc.Loading(html.Div(id='focus-output')),
    ]),

    # -- Hidden store --
    dcc.Store(id='plate-folder-store'),
], style={'fontFamily': 'sans-serif', 'maxWidth': '1200px', 'margin': '0 auto'})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output('plate-folder', 'value'),
    Input('btn-browse', 'n_clicks'),
    prevent_initial_call=True,
)
def browse_folder(n_clicks):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory(title="Select plate folder")
    root.destroy()
    return folder if folder else dash.no_update


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
    if folder:
        folder = folder.strip().removeprefix('file://')
    if not folder or not os.path.isdir(folder):
        return [], None, "Invalid folder path.", None
    channels = detect_channels(folder)
    if not channels:
        return [], None, "No TIF images found in folder.", None
    tifs = glob.glob(os.path.join(folder, "*.tif"))
    options = [{'label': ch, 'value': ch} for ch in channels]
    return options, channels[0], f"Loaded: {len(tifs)} images, {len(channels)} channels.", folder


@callback(
    Output('panel-montage', 'style'),
    Output('panel-controls', 'style'),
    Output('panel-contact', 'style'),
    Output('panel-intensity', 'style'),
    Output('panel-focus', 'style'),
    Input('tabs', 'value'),
)
def toggle_tab_visibility(tab):
    hidden = {'margin': '10px', 'display': 'none'}
    visible = {'margin': '10px', 'display': 'block'}
    tab_map = {'tab-montage': 0, 'tab-controls': 1, 'tab-contact': 2,
               'tab-intensity': 3, 'tab-focus': 4}
    styles = [hidden] * 5
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
    if len(images) < 32:
        return html.Div(f"Not enough images ({len(images)} found, need 32).")
    montage, selected = make_montage(images, n_images=32, rows=4, cols=8, crop_size=1020)
    b64 = numpy_to_b64png(montage)
    return html.Div([
        html.P(f"Showing 32 random images from {len(images)} ({channel} channel)"),
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
    heatmap = compute_focus_heatmap(folder, channel, plate_rows, plate_cols)
    fig = make_plate_heatmap_figure(heatmap, title=f"Focus (Laplacian variance) — {channel}",
                                    plate_rows=plate_rows, plate_cols=plate_cols)
    return dcc.Graph(figure=fig)


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
    if len(filtered) < 32:
        return html.Div(f"Not enough control images ({len(filtered)} found, need 32).")
    montage, selected = make_montage(filtered, n_images=32, rows=4, cols=8, crop_size=1020)
    b64 = numpy_to_b64png(montage)
    return html.Div([
        html.P(f"Showing 32 random images from {len(filtered)} control images, "
               f"{len(well_set)} wells ({channel} channel)"),
        html.Img(src=b64, style={'width': '100%', 'imageRendering': 'auto'}),
    ])


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

    thumb_size = 128
    spacing = 2
    sheet, well_positions = make_contact_sheet(
        images, thumb_size=thumb_size, spacing=spacing,
        plate_rows=plate_rows, plate_cols=plate_cols,
        preferred_field=pref_field,
    )
    b64 = numpy_to_b64png(sheet)
    h, w = sheet.shape

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
    fig_width = 1150
    fig_height = int(fig_width * h / w) + 60
    n_fields = len(fields) if fields else '?'
    fig.update_layout(
        width=fig_width, height=fig_height,
        margin=dict(l=20, r=10, t=50, b=10),
        title=f"Plate Thumbnails — field {pref_field} per well ({n_fields} fields detected, {channel} channel)",
        showlegend=False,
        plot_bgcolor='black',
    )
    return dcc.Graph(figure=fig, style={'width': '100%'}, config={'responsive': True})


@callback(
    Output('save-all-status', 'children'),
    Input('btn-save-all', 'n_clicks'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    State('plate-format-dropdown', 'value'),
    State('control-wells-input', 'value'),
    prevent_initial_call=True,
)
def save_all_plots(n_clicks, channel, folder, plate_fmt, well_spec):
    if not folder or not channel:
        return "Load a plate folder first."

    plate_rows, plate_cols = PLATE_FORMATS[plate_fmt]
    out_dir = os.path.join(folder, "PlateViewer")
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    # 1. Random montage
    images = find_images(folder, channel=channel)
    if len(images) >= 32:
        montage, _ = make_montage(images, n_images=32, rows=4, cols=8, crop_size=1020)
        path = os.path.join(out_dir, f"{channel}_montage.png")
        Image.fromarray(montage).save(path)
        saved.append("montage")

    # 2. Control montage (only if well spec provided)
    if well_spec:
        well_set = parse_well_spec(well_spec, plate_rows=plate_rows)
        if well_set:
            filtered = filter_images_by_wells(images, well_set)
            if len(filtered) >= 32:
                ctrl_montage, _ = make_montage(filtered, n_images=32, rows=4, cols=8, crop_size=1020)
                path = os.path.join(out_dir, f"{channel}_controls.png")
                Image.fromarray(ctrl_montage).save(path)
                saved.append("controls")

    # 3. Plate thumbnails
    fields = detect_fields(folder)
    pref_field = center_field(fields)
    thumb_size, spacing = 128, 2
    sheet, well_positions = make_contact_sheet(
        images, thumb_size=thumb_size, spacing=spacing,
        plate_rows=plate_rows, plate_cols=plate_cols,
        preferred_field=pref_field,
    )
    # Build the same Plotly figure as the contact sheet tab
    b64 = numpy_to_b64png(sheet)
    h, w = sheet.shape
    step = thumb_size + spacing
    col_tick_vals = [i * step + thumb_size // 2 for i in range(plate_cols)]
    col_tick_labels = [str(i + 1) for i in range(plate_cols)]
    row_tick_vals = [i * step + thumb_size // 2 for i in range(plate_rows)]
    row_tick_labels = [chr(ord('A') + i) for i in range(plate_rows)]
    fig_contact = go.Figure()
    fig_contact.add_layout_image(
        source=b64, xref="x", yref="y",
        x=0, y=0, sizex=w, sizey=h,
        sizing="stretch", layer="below",
    )
    fig_contact.update_xaxes(
        range=[0, w], side='top',
        tickvals=col_tick_vals, ticktext=col_tick_labels,
        tickfont=dict(size=11), showgrid=False, zeroline=False,
    )
    fig_contact.update_yaxes(
        range=[h, 0],
        tickvals=row_tick_vals, ticktext=row_tick_labels,
        tickfont=dict(size=11), showgrid=False, zeroline=False,
        scaleanchor='x',
    )
    n_fields = len(fields) if fields else '?'
    fig_contact.update_layout(
        width=1150, height=int(1150 * h / w) + 60,
        margin=dict(l=20, r=10, t=50, b=10),
        title=f"Plate Thumbnails — field {pref_field} per well ({n_fields} fields, {channel})",
        showlegend=False, plot_bgcolor='black',
    )
    path = os.path.join(out_dir, f"{channel}_thumbnails.png")
    fig_contact.write_image(path, scale=2)
    saved.append("thumbnails")

    # 4. Intensity heatmap
    heatmap_int = compute_intensity_heatmap(folder, channel, plate_rows, plate_cols)
    fig_int = make_plate_heatmap_figure(heatmap_int, title=f"Mean Intensity — {channel}",
                                        plate_rows=plate_rows, plate_cols=plate_cols)
    path = os.path.join(out_dir, f"{channel}_intensity.png")
    fig_int.write_image(path, scale=2)
    saved.append("intensity")

    # 5. Focus heatmap
    heatmap_foc = compute_focus_heatmap(folder, channel, plate_rows, plate_cols)
    fig_foc = make_plate_heatmap_figure(heatmap_foc, title=f"Focus (Laplacian variance) — {channel}",
                                        plate_rows=plate_rows, plate_cols=plate_cols)
    path = os.path.join(out_dir, f"{channel}_focus.png")
    fig_foc.write_image(path, scale=2)
    saved.append("focus")

    return f"Saved {len(saved)} plots to {out_dir}/: {', '.join(saved)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import webbrowser, threading
    parser = argparse.ArgumentParser(description="PlateViewer web interface")
    parser.add_argument("--port", type=int, default=8050, help="Port (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    args = parser.parse_args()
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    app.run(debug=args.debug, port=args.port)
