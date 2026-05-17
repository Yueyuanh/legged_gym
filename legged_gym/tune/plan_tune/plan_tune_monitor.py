#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Re-open a Plan Tune dashboard for an existing output directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_SCRIPT = SCRIPT_DIR / "plan_tune_dashboard.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Plan Tune monitor dashboard")
    parser.add_argument("--output-dir", required=True, type=str, help="Existing plan_tune run directory")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if not (output_dir / "dashboard_state").is_dir():
        raise SystemExit(f"dashboard_state not found: {output_dir / 'dashboard_state'}")
    cmd = [
        sys.executable,
        str(DASHBOARD_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print("Starting Plan Tune dashboard:")
    print("  " + " ".join(cmd))
    print(f"Dashboard URL: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
