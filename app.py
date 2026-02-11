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

from plate import find_images, detect_channels, parse_well_spec, filter_images_by_wells
from image import numpy_to_b64png
from montage import make_montage, make_contact_sheet
from heatmaps import compute_intensity_heatmap, compute_focus_heatmap

# ---------------------------------------------------------------------------
# Plotly figure helper
# ---------------------------------------------------------------------------

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
        html.Button("Browse", id='btn-browse', n_clicks=0, style={'marginRight': '5px'}),
        html.Button("Load", id='btn-load', n_clicks=0),
    ], style={'margin': '10px'}),

    html.Div(id='folder-status', style={'margin': '10px', 'color': '#666'}),

    # -- Channel selector --
    html.Div([
        html.Label("Channel:"),
        dcc.Dropdown(id='channel-dropdown', style={'width': '200px', 'display': 'inline-block'}),
    ], style={'margin': '10px'}),

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
        html.Div([
            html.Label("Control wells: "),
            dcc.Input(id='control-wells-input', type='text',
                      placeholder='e.g. A-H:5, I-P:13',
                      style={'width': '400px', 'marginRight': '10px'}),
            html.Button("Generate", id='btn-controls', n_clicks=0),
        ], style={'margin': '10px 0'}),
        html.P("Syntax: A-H:5 (rows A-H col 5), col:1-2 (full columns), A05 (single well)",
               style={'color': '#888', 'fontSize': '12px'}),
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
    prevent_initial_call=True,
)
def generate_intensity_heatmap(n_clicks, channel, folder):
    if not folder or not channel:
        return html.Div("Load a plate folder first.")
    heatmap = compute_intensity_heatmap(folder, channel)
    fig = make_plate_heatmap_figure(heatmap, title=f"Mean Intensity — {channel}")
    return dcc.Graph(figure=fig)


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


@callback(
    Output('controls-output', 'children'),
    Input('btn-controls', 'n_clicks'),
    State('control-wells-input', 'value'),
    State('channel-dropdown', 'value'),
    State('plate-folder-store', 'data'),
    prevent_initial_call=True,
)
def generate_controls_montage(n_clicks, well_spec, channel, folder):
    if not folder or not channel:
        return html.Div("Load a plate folder first.")
    if not well_spec:
        return html.Div("Enter control well specification.")
    well_set = parse_well_spec(well_spec)
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
    prevent_initial_call=True,
)
def generate_contact_sheet(n_clicks, channel, folder):
    if not folder or not channel:
        return html.Div("Load a plate folder first.")
    images = find_images(folder, channel=channel)
    if not images:
        return html.Div("No images found.")
    sheet, well_positions = make_contact_sheet(images, thumb_size=128, spacing=2)
    b64 = numpy_to_b64png(sheet)
    h, w = sheet.shape

    fig = go.Figure()
    # display the contact sheet as a background image
    fig.add_layout_image(
        source=b64, xref="x", yref="y",
        x=0, y=0, sizex=w, sizey=h,
        sizing="stretch", layer="below",
    )
    # invisible scatter markers for hover
    xs = [pos[0] for pos in well_positions.values()]
    ys = [pos[1] for pos in well_positions.values()]
    labels = list(well_positions.keys())
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers',
        marker=dict(size=20, opacity=0),
        text=labels,
        hovertemplate='Well: %{text}<extra></extra>',
    ))
    fig.update_xaxes(range=[0, w], visible=False, scaleanchor='y')
    fig.update_yaxes(range=[h, 0], visible=False)
    fig.update_layout(
        width=w, height=h,
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Plate Thumbnails — field 5 per well ({channel} channel)",
    )
    return dcc.Graph(figure=fig)


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
