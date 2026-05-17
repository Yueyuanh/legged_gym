#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Launch the auto-tune frontend independently from the training backend.

The training process only needs to keep writing dashboard_state/*.json.
This monitor can be restarted at any time to pick up frontend changes
without interrupting training.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARD_SCRIPT = SCRIPT_DIR / "auto_tune_dashboard.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Launch auto-tune monitoring services")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="logs/auto_tune/<run> directory")
    parser.add_argument("--dashboard-port", type=int, default=None)
    parser.add_argument("--tensorboard-port", type=int, default=None)
    parser.add_argument("--public-host", type=str, default="localhost",
                        help="Host shown in links, for example localhost or server IP")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Do not launch the dashboard frontend")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Do not launch TensorBoard")
    return parser.parse_args()


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    state_dir = output_dir / "dashboard_state"
    session = read_json(state_dir / "session.json", {})

    if not state_dir.is_dir():
        raise SystemExit(f"dashboard_state not found: {state_dir}")

    tb_port = args.tensorboard_port or session.get("tensorboard_port", 1230)
    dashboard_port = args.dashboard_port or session.get("dashboard_port", 8050)
    log_root = session.get("log_root")
    tb_url = f"http://{args.public_host}:{tb_port}"

    procs = []

    if not args.no_tensorboard:
        if not log_root:
            raise SystemExit(
                "session.json does not contain log_root; please pass a run created by auto_tune_rewards.py"
            )
        tb_cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", os.path.abspath(log_root),
            "--bind_all",
            "--port", str(tb_port),
        ]
        print("Starting TensorBoard:")
        print("  " + " ".join(tb_cmd))
        procs.append(subprocess.Popen(tb_cmd))

    if not args.no_dashboard:
        dash_cmd = [
            sys.executable,
            str(DASHBOARD_SCRIPT),
            "--state-dir",
            str(state_dir),
            "--output-dir",
            str(output_dir),
            "--port",
            str(dashboard_port),
            "--tb-url",
            tb_url,
        ]
        print("Starting dashboard:")
        print("  " + " ".join(dash_cmd))
        procs.append(subprocess.Popen(dash_cmd))

    print(f"Dashboard URL: http://{args.public_host}:{dashboard_port}")
    print(f"TensorBoard URL: {tb_url}")
    print("Press Ctrl+C to stop the monitor processes.")

    try:
        while True:
            for proc in procs:
                rc = proc.poll()
                if rc is not None:
                    raise SystemExit(f"monitor subprocess exited with code {rc}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in procs:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()
