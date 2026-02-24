#!/usr/bin/env python3
"""
CRISPR PlateViewer — Streamlit web interface for high-content screening QC.

Usage:
    streamlit run plateviewer/app_streamlit.py
    streamlit run plateviewer/app_streamlit.py --server.port 8051

Requires: pip install streamlit
"""

import glob
import os
import re

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from plateviewer import config as cfg
from plateviewer.heatmaps import (compute_focus_heatmap, compute_intensity_heatmap,
                                  compute_plls_heatmap)
from plateviewer.image import numpy_to_b64png
from plateviewer.montage import make_contact_sheet, make_montage, make_well_montage
from plateviewer.paths import plate_output_dir
from plateviewer.plate import (PLATE_FORMATS, center_field, detect_channels, detect_fields,
                               filter_images_by_wells, find_images, parse_well_spec,
                               sort_by_field)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PlateViewer", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_DEFAULTS = {
    'plate_folder_text': '',
    'output_folder_text': '',
    'channels': [],
    'load_status': '',
    'montage_img': None,
    'montage_info': None,
    'controls_img': None,
    'controls_info': None,
    'well_img': None,
    'well_info': None,
    'contact_fig': None,
    'intensity_fig': None,
    'focus_fig_lap': None,
    'focus_fig_plls': None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _browse_folder(title="Select folder", initial=None):
    """Open a native folder-picker dialog.

    Spawns a fresh subprocess so tkinter runs on that process's main thread,
    avoiding the NSException crash that occurs when tkinter is called from a
    Streamlit background thread on macOS.
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
    out_dir = output_folder.strip() if output_folder else ""
    if not out_dir:
        out_dir = plate_output_dir(plate_folder)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Plotly figure builders (identical logic to app.py)
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
        yaxis=dict(autorange='reversed', scaleanchor='x', dtick=1, constrain='domain'),
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
        title=(f"Plate Thumbnails — field {preferred_field} per well "
               f"({n_fields} fields detected, {channel} channel)"),
        showlegend=False,
        plot_bgcolor='black',
    )
    return fig


# ---------------------------------------------------------------------------
# Top-of-page controls
# ---------------------------------------------------------------------------
st.title("PlateViewer")

# Flush any folder picked by Browse into the widget key *before* the widget renders.
# This is necessary because Streamlit forbids writing to session_state[key] after
# the widget with that key has already been rendered in the same script run.
if st.session_state.get('_plate_folder_pick'):
    st.session_state['plate_folder_text'] = st.session_state['_plate_folder_pick']
    del st.session_state['_plate_folder_pick']
if st.session_state.get('_output_folder_pick'):
    st.session_state['output_folder_text'] = st.session_state['_output_folder_pick']
    del st.session_state['_output_folder_pick']

# -- Plate folder --
c1, c2, c3 = st.columns([6, 1, 1])
with c1:
    st.text_input("Plate folder", key='plate_folder_text',
                  placeholder="/path/to/plate/folder")
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Browse", key="btn_browse"):
        cur = st.session_state['plate_folder_text']
        initial = None
        if cur:
            parent = os.path.dirname(os.path.normpath(cur))
            if os.path.isdir(parent):
                initial = parent
        picked = _browse_folder("Select plate folder", initial=initial)
        if picked:
            st.session_state['_plate_folder_pick'] = picked
            st.rerun()
with c3:
    st.markdown("<br>", unsafe_allow_html=True)
    load_clicked = st.button("Load", key="btn_load")

if load_clicked:
    folder_raw = st.session_state['plate_folder_text'].strip()
    if folder_raw:
        from urllib.parse import unquote
        folder_raw = unquote(folder_raw.removeprefix('file://'))
        # Stage the cleaned path — flushed into the widget key on the next rerun
        st.session_state['_plate_folder_pick'] = folder_raw
    if folder_raw and os.path.isdir(folder_raw):
        channels = detect_channels(folder_raw)
        if channels:
            tifs = glob.glob(os.path.join(folder_raw, "*.tif"))
            st.session_state['channels'] = channels
            st.session_state['load_status'] = (
                f"Loaded: {len(tifs)} images, {len(channels)} channels.")
            # Stage the output path — flushed into the widget key on the next rerun
            st.session_state['_output_folder_pick'] = plate_output_dir(folder_raw)
        else:
            st.session_state['channels'] = []
            st.session_state['load_status'] = "No TIF images found in folder."
    else:
        st.session_state['channels'] = []
        st.session_state['load_status'] = "Invalid folder path."

if st.session_state['load_status']:
    st.caption(st.session_state['load_status'])

# -- Channel + plate format --
c1, c2 = st.columns(2)
with c1:
    channels = st.session_state['channels']
    selected_channel = st.selectbox("Channel", options=channels,
                                    disabled=(len(channels) == 0))
with c2:
    plate_fmt = st.selectbox(
        "Plate format", options=['384', '96'],
        format_func=lambda x: '384-well (16×24)' if x == '384' else '96-well (8×12)',
    )

plate_rows, plate_cols = PLATE_FORMATS[plate_fmt]

# -- Control wells + Save All --
c1, c2 = st.columns([5, 1])
with c1:
    control_wells = st.text_input(
        "Control wells", placeholder="e.g. A-H:5, I-P:13",
        help="Syntax: A-H:5 (row range + column), col:1-2 (full columns), A05 (single well)",
    )
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    save_all_clicked = st.button("Save All Plots", key="btn_save_all")

# -- Output folder --
c1, c2 = st.columns([6, 1])
with c1:
    st.text_input("Output folder", key='output_folder_text',
                  placeholder="Auto-set on Load (or choose manually)")
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Browse", key="btn_browse_output"):
        initial = cfg.APP_DATA_DIR if os.path.isdir(cfg.APP_DATA_DIR) else None
        picked = _browse_folder("Select output folder", initial=initial)
        if picked:
            st.session_state['_output_folder_pick'] = picked
            st.rerun()

st.divider()

# Convenience shorthands (valid for this entire script run)
folder = st.session_state['plate_folder_text']
channel = selected_channel
out_folder = st.session_state['output_folder_text']

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
(tab_montage, tab_controls, tab_well,
 tab_contact, tab_intensity, tab_focus) = st.tabs([
    "Random Montage", "Control Montage", "Single Well",
    "Plate Thumbnails", "Intensity Heatmap", "Focus Heatmap",
])

# ── Random Montage ──────────────────────────────────────────────────────────
with tab_montage:
    c1, c2, _ = st.columns([1, 1, 8])
    with c1:
        gen_montage = st.button("Generate Montage", key="btn_gen_montage")
    with c2:
        save_montage = st.button("Save", key="btn_save_montage")

    if gen_montage:
        if not folder or not channel:
            st.warning("Load a plate folder first.")
        else:
            images = find_images(folder, channel=channel)
            if len(images) < cfg.MONTAGE_N_IMAGES:
                st.warning(f"Not enough images ({len(images)} found, "
                           f"need {cfg.MONTAGE_N_IMAGES}).")
            else:
                with st.spinner("Generating montage…"):
                    montage, _ = make_montage(images, crop_size=cfg.MONTAGE_CROP_SIZE)
                st.session_state['montage_img'] = montage
                st.session_state['montage_info'] = (len(images), channel)

    if st.session_state['montage_img'] is not None:
        n_total, ch = st.session_state['montage_info']
        st.caption(f"Showing {cfg.MONTAGE_N_IMAGES} random images from {n_total} "
                   f"({ch} channel)")
        st.image(st.session_state['montage_img'], use_container_width=True)

    if save_montage:
        if st.session_state['montage_img'] is None:
            st.warning("Generate a montage first.")
        elif not folder:
            st.warning("Load a plate folder first.")
        else:
            try:
                out_dir = _resolve_output_dir(out_folder, folder)
                path = os.path.join(out_dir, f"{channel}_montage.png")
                Image.fromarray(st.session_state['montage_img']).save(path)
                st.success(f"Saved: {path}")
            except PermissionError:
                st.error("Cannot save: output folder is not writable.")

# ── Control Montage ─────────────────────────────────────────────────────────
with tab_controls:
    c1, c2, _ = st.columns([1, 1, 8])
    with c1:
        gen_controls = st.button("Generate Control Montage", key="btn_gen_controls")
    with c2:
        save_controls = st.button("Save", key="btn_save_controls")

    if gen_controls:
        if not folder or not channel:
            st.warning("Load a plate folder first.")
        elif not control_wells:
            st.warning("Enter control well specification in the 'Control wells' field above.")
        else:
            well_set = parse_well_spec(control_wells, plate_rows=plate_rows)
            if not well_set:
                st.warning(f"Could not parse well specification: {control_wells}")
            else:
                images = find_images(folder, channel=channel)
                filtered = filter_images_by_wells(images, well_set)
                if not filtered:
                    st.warning(f"No images found for wells: {sorted(well_set)}")
                elif len(filtered) < cfg.MONTAGE_N_IMAGES:
                    st.warning(f"Not enough control images ({len(filtered)} found, "
                               f"need {cfg.MONTAGE_N_IMAGES}).")
                else:
                    with st.spinner("Generating control montage…"):
                        montage, _ = make_montage(filtered, crop_size=cfg.MONTAGE_CROP_SIZE)
                    st.session_state['controls_img'] = montage
                    st.session_state['controls_info'] = (len(filtered), len(well_set), channel)

    if st.session_state['controls_img'] is not None:
        n_filt, n_wells, ch = st.session_state['controls_info']
        st.caption(f"Showing {cfg.MONTAGE_N_IMAGES} random images from {n_filt} "
                   f"control images, {n_wells} wells ({ch} channel)")
        st.image(st.session_state['controls_img'], use_container_width=True)

    if save_controls:
        if st.session_state['controls_img'] is None:
            st.warning("Generate a control montage first.")
        elif not folder:
            st.warning("Load a plate folder first.")
        else:
            try:
                out_dir = _resolve_output_dir(out_folder, folder)
                path = os.path.join(out_dir, f"{channel}_controls.png")
                Image.fromarray(st.session_state['controls_img']).save(path)
                st.success(f"Saved: {path}")
            except PermissionError:
                st.error("Cannot save: output folder is not writable.")

# ── Single Well ──────────────────────────────────────────────────────────────
with tab_well:
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        well_id_input = st.text_input("Well ID", placeholder="e.g. A05",
                                      key="well_id_input")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_well = st.button("Generate", key="btn_gen_well")
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        save_well = st.button("Save", key="btn_save_well")

    if gen_well:
        if not folder or not channel:
            st.warning("Load a plate folder first.")
        elif not well_id_input or not re.match(r'^[A-P]\d+$',
                                               well_id_input.strip(), re.IGNORECASE):
            st.warning("Enter a valid well ID (e.g. A05).")
        else:
            well_id = f"{well_id_input[0].upper()}{int(well_id_input[1:]):02d}"
            images = find_images(folder, channel=channel)
            filtered = filter_images_by_wells(images, {well_id})
            if not filtered:
                st.warning(f"No images found for well {well_id}.")
            else:
                sort_by_field(filtered)
                with st.spinner(f"Generating well {well_id} montage…"):
                    montage = make_well_montage(filtered)
                st.session_state['well_img'] = montage
                st.session_state['well_info'] = (well_id, len(filtered), channel)

    if st.session_state['well_img'] is not None:
        wid, n_flds, ch = st.session_state['well_info']
        st.caption(f"Well {wid} — {n_flds} fields ({ch} channel)")
        st.image(st.session_state['well_img'], use_container_width=True)

    if save_well:
        if st.session_state['well_img'] is None:
            st.warning("Generate a well montage first.")
        elif not folder:
            st.warning("Load a plate folder first.")
        else:
            wid = st.session_state['well_info'][0]
            try:
                out_dir = _resolve_output_dir(out_folder, folder)
                path = os.path.join(out_dir, f"{channel}_well_{wid}.png")
                Image.fromarray(st.session_state['well_img']).save(path)
                st.success(f"Saved: {path}")
            except PermissionError:
                st.error("Cannot save: output folder is not writable.")

# ── Plate Thumbnails ─────────────────────────────────────────────────────────
with tab_contact:
    gen_contact = st.button("Generate Plate Thumbnails", key="btn_gen_contact")

    if gen_contact:
        if not folder or not channel:
            st.warning("Load a plate folder first.")
        else:
            images = find_images(folder, channel=channel)
            if not images:
                st.warning("No images found.")
            else:
                fields = detect_fields(folder)
                pref_field = center_field(fields)
                n_fields_count = len(fields) if fields else '?'
                with st.spinner("Generating plate thumbnails…"):
                    sheet, well_positions = make_contact_sheet(
                        images, plate_rows=plate_rows, plate_cols=plate_cols,
                        preferred_field=pref_field,
                    )
                fig = make_contact_sheet_figure(
                    sheet, well_positions, plate_rows, plate_cols,
                    channel, n_fields_count, pref_field,
                )
                st.session_state['contact_fig'] = fig

    if st.session_state['contact_fig'] is not None:
        st.plotly_chart(st.session_state['contact_fig'], use_container_width=True)

# ── Intensity Heatmap ────────────────────────────────────────────────────────
with tab_intensity:
    gen_intensity = st.button("Compute Intensity Heatmap", key="btn_gen_intensity")

    if gen_intensity:
        if not folder or not channel:
            st.warning("Load a plate folder first.")
        else:
            with st.spinner("Computing intensity heatmap…"):
                heatmap = compute_intensity_heatmap(folder, channel, plate_rows, plate_cols)
            fig = make_plate_heatmap_figure(
                heatmap, title=f"Mean Intensity — {channel}",
                plate_rows=plate_rows, plate_cols=plate_cols,
            )
            st.session_state['intensity_fig'] = fig

    if st.session_state['intensity_fig'] is not None:
        st.plotly_chart(st.session_state['intensity_fig'], use_container_width=True)

# ── Focus Heatmap ────────────────────────────────────────────────────────────
with tab_focus:
    st.info(
        "**Variance of Laplacian (VoL):** Measures how much high-frequency structure "
        "(edges/texture) is present. Higher VoL values usually correspond to sharper, more "
        "structured images, however VoL increases also with cell confluency and with "
        "increasing noise.  \n"
        "**Power Log-Log Slope (PLLS):** (Bray et al., 2012) Summarizes how quickly spectral "
        "power falls off with spatial frequency. Lower values (i.e. more negative slopes) "
        "indicate blur (loss of high frequencies), while larger values (i.e. slopes closer to "
        "zero) suggest a flatter, noise-dominated spectrum.  \n"
        "**Reading both together:** Use VoL as 'how much structure/edge content exists' and "
        "PLLS as 'is high-frequency content preserved vs suppressed': true defocus tends to "
        "decrease PLLS (often with reduced VoL), whereas noisy/artefactual images can show "
        "high VoL and PLLS."
    )
    gen_focus = st.button("Compute Focus Heatmap", key="btn_gen_focus")

    if gen_focus:
        if not folder or not channel:
            st.warning("Load a plate folder first.")
        else:
            with st.spinner("Computing focus heatmaps (this may take a while)…"):
                heatmap_lap = compute_focus_heatmap(folder, channel, plate_rows, plate_cols)
                heatmap_plls = compute_plls_heatmap(folder, channel, plate_rows, plate_cols)
            fig_lap = make_plate_heatmap_figure(
                heatmap_lap, title=f"Focus (Laplacian variance) — {channel}",
                plate_rows=plate_rows, plate_cols=plate_cols,
            )
            fig_plls = make_plate_heatmap_figure(
                heatmap_plls, title=f"Focus (Power Log-Log Slope) — {channel}",
                plate_rows=plate_rows, plate_cols=plate_cols,
            )
            st.session_state['focus_fig_lap'] = fig_lap
            st.session_state['focus_fig_plls'] = fig_plls

    if st.session_state['focus_fig_lap'] is not None:
        st.plotly_chart(st.session_state['focus_fig_lap'], use_container_width=True)
        st.plotly_chart(st.session_state['focus_fig_plls'], use_container_width=True)

# ---------------------------------------------------------------------------
# Save All Plots
# ---------------------------------------------------------------------------
if save_all_clicked:
    if not folder or not channel:
        st.error("Load a plate folder first.")
    else:
        try:
            out_dir = _resolve_output_dir(out_folder, folder)
        except PermissionError:
            st.error("Cannot save: output folder is not writable.")
            st.stop()

        saved = []
        with st.spinner("Saving all plots…"):
            images = find_images(folder, channel=channel)

            # 1. Random montage
            if len(images) >= cfg.MONTAGE_N_IMAGES:
                montage, _ = make_montage(images, crop_size=cfg.MONTAGE_CROP_SIZE)
                path = os.path.join(out_dir, f"{channel}_montage.png")
                Image.fromarray(montage).save(path)
                saved.append("montage")

            # 2. Control montage (only if well spec provided)
            if control_wells:
                well_set = parse_well_spec(control_wells, plate_rows=plate_rows)
                if well_set:
                    filtered = filter_images_by_wells(images, well_set)
                    if len(filtered) >= cfg.MONTAGE_N_IMAGES:
                        ctrl_montage, _ = make_montage(filtered, crop_size=cfg.MONTAGE_CROP_SIZE)
                        path = os.path.join(out_dir, f"{channel}_controls.png")
                        Image.fromarray(ctrl_montage).save(path)
                        saved.append("controls")

            # 3. Plate thumbnails
            fields = detect_fields(folder)
            pref_field = center_field(fields)
            n_fields_count = len(fields) if fields else '?'
            sheet, well_positions = make_contact_sheet(
                images, plate_rows=plate_rows, plate_cols=plate_cols,
                preferred_field=pref_field,
            )
            fig_contact = make_contact_sheet_figure(
                sheet, well_positions, plate_rows, plate_cols,
                channel, n_fields_count, pref_field,
            )
            path = os.path.join(out_dir, f"{channel}_thumbnails.png")
            fig_contact.write_image(path, scale=cfg.EXPORT_SCALE)
            saved.append("thumbnails")

            # 4. Intensity heatmap
            heatmap_int = compute_intensity_heatmap(folder, channel, plate_rows, plate_cols)
            fig_int = make_plate_heatmap_figure(
                heatmap_int, title=f"Mean Intensity — {channel}",
                plate_rows=plate_rows, plate_cols=plate_cols,
            )
            path = os.path.join(out_dir, f"{channel}_intensity.png")
            fig_int.write_image(path, scale=cfg.EXPORT_SCALE)
            saved.append("intensity")

            # 5. Focus (Laplacian variance)
            heatmap_foc = compute_focus_heatmap(folder, channel, plate_rows, plate_cols)
            fig_foc = make_plate_heatmap_figure(
                heatmap_foc, title=f"Focus (Laplacian variance) — {channel}",
                plate_rows=plate_rows, plate_cols=plate_cols,
            )
            path = os.path.join(out_dir, f"{channel}_focus.png")
            fig_foc.write_image(path, scale=cfg.EXPORT_SCALE)
            saved.append("focus")

            # 6. Focus (Power Log-Log Slope)
            heatmap_plls_data = compute_plls_heatmap(folder, channel, plate_rows, plate_cols)
            fig_plls = make_plate_heatmap_figure(
                heatmap_plls_data, title=f"Focus (Power Log-Log Slope) — {channel}",
                plate_rows=plate_rows, plate_cols=plate_cols,
            )
            path = os.path.join(out_dir, f"{channel}_plls.png")
            fig_plls.write_image(path, scale=cfg.EXPORT_SCALE)
            saved.append("plls")

        st.success(f"Saved {len(saved)} plots to {out_dir}/: {', '.join(saved)}")
