#!/usr/bin/env python3
"""Launch the Hybrid Vision web interface.

  python run_gui.py                # http://localhost:8501
  python run_gui.py --port 9000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "gui"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid Vision GUI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("server:app", host=args.host, port=args.port,
                app_dir=str(Path(__file__).resolve().parent / "gui"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
