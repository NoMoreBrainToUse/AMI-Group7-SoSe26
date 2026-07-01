#!/usr/bin/env bash
# Launch the Group 7 Hybrid Vision web interface.
# Run from the repository root (paths in gui.py are repo-root-relative).
#
#   ./run_gui.sh            # open on http://localhost:8501
#
# --server.fileWatcherType none avoids "inotify watch limit reached" spam on
# systems with a low fs.inotify.max_user_watches (trade-off: no code hot-reload).
set -euo pipefail
cd "$(dirname "$0")"

VENV="/home/lihengrf/repos/AMI/.venv"
STREAMLIT="$VENV/bin/streamlit"

if [ ! -x "$STREAMLIT" ]; then
  echo "ERROR: streamlit not found at $STREAMLIT" >&2
  echo "       Create the env first:  python3 -m venv $VENV && $VENV/bin/pip install -r requirements-gui.txt" >&2
  exit 1
fi

exec "$STREAMLIT" run gui.py --server.fileWatcherType none "$@"
