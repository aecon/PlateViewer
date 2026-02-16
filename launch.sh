#!/bin/bash
# ---------------------------------------------------------------------------
# PlateViewer Desktop Launcher
# ---------------------------------------------------------------------------
# A .desktop file cannot activate a conda environment directly because it
# runs outside of an interactive shell (no .bashrc is sourced).  This script
# bridges the gap: it initialises conda's shell hooks, activates the
# PlateViewer environment, and starts the Dash app.  The .desktop entry on
# the Desktop simply points here.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

eval "$(conda shell.bash hook)"
conda activate PlateViewer
exec plateviewer
