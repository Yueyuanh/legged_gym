#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Plan Tune WebUI backend.

Serves a standalone HTML frontend plus JSON APIs for creating experiment
sequences, starting the runner, and reading live dashboard state.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
RUNNER_SCRIPT = SCRIPT_DIR / "plan_tune_runner.py"
HTML_PATH = SCRIPT_DIR / "plan_tune_dashboard.html"
DEFAULT_TASK = "go2"
HELPER_JSON_PREFIX = "PLAN_TUNE_HELPER_JSON:"

RUNNERS: dict[str, subprocess.Popen] = {}
INITIAL_OUTPUT_DIR = ""
DISABLE_GPU_MEMORY_CHECK = False


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def safe_name(value: str, fallback: str = "experiment") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return cleaned[:80] or fallback


def parse_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty numeric value")
    return float(text)


def default_output_dir(task: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str((PROJECT_ROOT / "logs" / "plan_tune" / f"{safe_name(task)}_{stamp}").resolve())


def resolve_sequence_path(path_text: str) -> Path:
    raw_path = Path(path_text or "").expanduser()
    if raw_path.is_dir():
        direct = raw_path / "experiment_sequence.json"
        if direct.is_file():
            return direct
        candidates = sorted(
            raw_path.glob("**/experiment_sequence.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return raw_path


def ensure_project_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def list_registered_tasks_direct() -> list[str]:
    ensure_project_path()
    from legged_gym.envs import task_registry

    return sorted(task_registry.task_classes.keys())


def load_reward_scales_direct(task: str) -> dict[str, float]:
    ensure_project_path()
    from legged_gym.envs import task_registry

    env_cfg, _ = task_registry.get_cfgs(task)
    scales_obj = env_cfg.rewards.scales
    scales: dict[str, float] = {}
    for name in dir(scales_obj):
        if name.startswith("_"):
            continue
        value = getattr(scales_obj, name)
        if callable(value):
            continue
        if isinstance(value, (int, float)):
            scales[name] = float(value)
    return dict(sorted(scales.items()))


def run_registry_helper(action: str, task: str | None = None) -> dict[str, Any]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--registry-helper", action]
    if task:
        cmd.extend(["--task", task])
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    payload_line = ""
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(HELPER_JSON_PREFIX):
            payload_line = line[len(HELPER_JSON_PREFIX) :]
            break
    if proc.returncode != 0 or not payload_line:
        detail = (proc.stderr.strip() or proc.stdout.strip() or f"helper exited {proc.returncode}")[-2000:]
        raise RuntimeError(f"registry helper failed: {detail}")
    payload = json.loads(payload_line)
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("error") or "registry helper failed"))
    return payload


def list_registered_tasks() -> list[str]:
    return list(run_registry_helper("tasks").get("tasks") or [])


def load_reward_scales(task: str) -> dict[str, float]:
    payload = run_registry_helper("reward_scales", task=task)
    return {str(k): float(v) for k, v in (payload.get("reward_scales") or {}).items()}


def registry_helper_main(action: str, task: str | None) -> int:
    try:
        if action == "tasks":
            payload = {"ok": True, "tasks": list_registered_tasks_direct()}
        elif action == "reward_scales":
            payload = {"ok": True, "task": task or DEFAULT_TASK, "reward_scales": load_reward_scales_direct(task or DEFAULT_TASK)}
        else:
            raise ValueError(f"unknown helper action: {action}")
    except Exception as exc:
        payload = {"ok": False, "error": str(exc)}
    print(HELPER_JSON_PREFIX + json.dumps(payload, ensure_ascii=False, default=str), flush=True)
    return 0 if payload.get("ok") else 1


def sequence_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    task = str(payload.get("task") or DEFAULT_TASK)
    group_name = str(payload.get("experiment_name") or f"{task}_PlanTune")
    base_scales = {str(k): parse_number(v) for k, v in (payload.get("base_scales") or {}).items()}
    experiments = payload.get("experiments") or []
    normalized = []
    for idx, experiment in enumerate(experiments):
        overrides = {str(k): parse_number(v) for k, v in (experiment.get("reward_scales") or {}).items()}
        full = dict(base_scales)
        full.update(overrides)
        normalized.append(
            {
                "id": int(experiment.get("id", idx)),
                "name": str(experiment.get("name") or f"experiment_{idx:03d}"),
                "iterations": int(experiment.get("iterations") or 500),
                "reward_scales": overrides,
                "full_reward_scales": full,
                "range_group": experiment.get("range_group"),
                "range_meta": experiment.get("range_meta"),
            }
        )
    return {
        "task": task,
        "experiment_name": group_name,
        "num_envs": int(payload.get("num_envs") or 4096),
        "headless": bool(payload.get("headless", True)),
        "tensorboard_port": int(payload.get("tensorboard_port") or 1230),
        "inter_experiment_delay": 0.0 if DISABLE_GPU_MEMORY_CHECK else float(payload.get("inter_experiment_delay") or 20.0),
        "oom_cooldown": 0.0 if DISABLE_GPU_MEMORY_CHECK else float(payload.get("oom_cooldown") or 75.0),
        "min_free_gpu_mb": 0 if DISABLE_GPU_MEMORY_CHECK else int(payload.get("min_free_gpu_mb") or 900),
        "top_video_limit": int(payload.get("top_video_limit") or 20),
        "video_record_delay": float(payload.get("video_record_delay") or 3.0),
        "timeout": 7200,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiments": normalized,
    }


def snapshot(output_dir: str) -> dict[str, Any]:
    if not output_dir:
        return {"session": {}, "experiments": [], "live": {}, "live_curve": {}, "video_jobs": []}
    state_dir = Path(output_dir) / "dashboard_state"
    jobs_dir = state_dir / "video_jobs"
    jobs = []
    if jobs_dir.exists():
        for path in sorted(jobs_dir.glob("*.json")):
            payload = read_json(path, None)
            if isinstance(payload, dict):
                jobs.append(payload)
    return {
        "session": read_json(state_dir / "session.json", {}),
        "experiments": read_json(state_dir / "experiments.json", []),
        "live": read_json(state_dir / "live_progress.json", {}),
        "live_curve": read_json(state_dir / "live_curve.json", {}),
        "video_jobs": jobs,
    }


def generate_sequence(payload: dict[str, Any]) -> dict[str, Any]:
    sequence = sequence_from_payload(payload)
    output_dir = Path(payload.get("output_dir") or default_output_dir(sequence["task"])).resolve()
    sequence_path = output_dir / "experiment_sequence.json"
    write_json(sequence_path, sequence)
    state_dir = output_dir / "dashboard_state"
    write_json(
        state_dir / "session.json",
        {
            "status": "sequence_generated",
            "task": sequence["task"],
            "experiment_name": sequence["experiment_name"],
            "output_dir": str(output_dir),
            "sequence_path": str(sequence_path),
            "total_experiments": len(sequence["experiments"]),
            "completed_experiments": 0,
            "updated_at": time.time(),
        },
    )
    write_json(state_dir / "experiments.json", [])
    return {"output_dir": str(output_dir), "sequence_path": str(sequence_path), "sequence": sequence}


def start_training(payload: dict[str, Any]) -> dict[str, Any]:
    generated = generate_sequence(payload)
    output_dir = generated["output_dir"]
    existing = RUNNERS.get(output_dir)
    if existing and existing.poll() is None:
        return {"started": False, "pid": existing.pid, **generated}
    cmd = [
        sys.executable,
        str(RUNNER_SCRIPT),
        "--sequence",
        generated["sequence_path"],
        "--output-dir",
        output_dir,
        "--tensorboard-port",
        str(payload.get("tensorboard_port") or 1230),
        "--inter-experiment-delay",
        str(0.0 if DISABLE_GPU_MEMORY_CHECK else (payload.get("inter_experiment_delay") or 20.0)),
        "--oom-cooldown",
        str(0.0 if DISABLE_GPU_MEMORY_CHECK else (payload.get("oom_cooldown") or 75.0)),
        "--min-free-gpu-mb",
        str(0 if DISABLE_GPU_MEMORY_CHECK else (payload.get("min_free_gpu_mb") or 900)),
        "--top-video-limit",
        str(payload.get("top_video_limit") or 20),
        "--video-record-delay",
        str(payload.get("video_record_delay") or 3.0),
    ]
    if DISABLE_GPU_MEMORY_CHECK:
        cmd.append("--no-gpu-memory-check")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
    )
    RUNNERS[output_dir] = proc
    return {"started": True, "pid": proc.pid, **generated}


def load_sequence(path_text: str) -> dict[str, Any]:
    path = resolve_sequence_path(path_text)
    sequence = read_json(path, {})
    if not isinstance(sequence, dict) or not sequence.get("experiments"):
        raise ValueError(f"not a valid experiment_sequence.json: {path}")
    task = str(sequence.get("task") or DEFAULT_TASK)
    base_scales = load_reward_scales(task)
    experiments = []
    for idx, experiment in enumerate(sequence.get("experiments") or []):
        experiments.append(
            {
                "id": int(experiment.get("id", idx)),
                "name": str(experiment.get("name") or f"experiment_{idx:03d}"),
                "iterations": int(experiment.get("iterations") or 500),
                "reward_scales": {
                    str(k): parse_number(v)
                    for k, v in (experiment.get("reward_scales") or {}).items()
                },
                "range_group": experiment.get("range_group"),
                "range_meta": experiment.get("range_meta"),
            }
        )
    return {
        "path": str(path),
        "output_dir": str(path.parent.resolve()),
        "task": task,
        "experiment_name": str(sequence.get("experiment_name") or f"{task}_PlanTune"),
        "num_envs": int(sequence.get("num_envs") or 4096),
        "headless": bool(sequence.get("headless", True)),
        "tensorboard_port": int(sequence.get("tensorboard_port") or 1230),
        "base_scales": base_scales,
        "experiments": experiments,
    }


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def file_response(handler: BaseHTTPRequestHandler, path: Path, content_type: str | None = None) -> None:
    data = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def error_response(handler: BaseHTTPRequestHandler, exc: Exception, status: int = 400) -> None:
    json_response(handler, {"ok": False, "error": str(exc)}, status)


class PlanTuneHandler(BaseHTTPRequestHandler):
    server_version = "PlanTuneWebUI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        message = fmt % args
        if "GET /api/snapshot" in message:
            return
        print(f"[plan_tune_dashboard] {self.address_string()} - {message}")

    def read_body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                file_response(self, HTML_PATH, "text/html; charset=utf-8")
                return
            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            if parsed.path == "/api/config":
                json_response(
                    self,
                    {
                        "ok": True,
                        "runs_dir": str((PROJECT_ROOT / "logs" / "plan_tune").resolve()),
                        "initial_output_dir": INITIAL_OUTPUT_DIR,
                    },
                )
                return
            if parsed.path == "/api/tasks":
                json_response(self, {"ok": True, "tasks": list_registered_tasks(), "default_task": DEFAULT_TASK})
                return
            if parsed.path == "/api/reward_scales":
                task = parse_qs(parsed.query).get("task", [DEFAULT_TASK])[0]
                json_response(self, {"ok": True, "task": task, "reward_scales": load_reward_scales(task)})
                return
            if parsed.path == "/api/load_sequence":
                path = parse_qs(parsed.query).get("path", [str(PROJECT_ROOT / "logs" / "plan_tune")])[0]
                json_response(self, {"ok": True, **load_sequence(path)})
                return
            if parsed.path == "/api/snapshot":
                output_dir = parse_qs(parsed.query).get("output_dir", [""])[0]
                json_response(self, {"ok": True, **snapshot(output_dir)})
                return
            if parsed.path.startswith("/videos/"):
                self.serve_video(parsed.path[len("/videos/") :])
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            error_response(self, exc)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_body_json()
            if parsed.path == "/api/generate":
                json_response(self, {"ok": True, **generate_sequence(payload)})
                return
            if parsed.path == "/api/start":
                json_response(self, {"ok": True, **start_training(payload)})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            error_response(self, exc)

    def serve_video(self, rel_path: str) -> None:
        query = parse_qs(urlparse(self.path).query)
        output_dir = query.get("output_dir", [""])[0]
        video_root = (Path(output_dir) / "videos").resolve()
        candidate = (video_root / unquote(rel_path)).resolve()
        try:
            candidate.relative_to(video_root)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid video path")
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Video not found")
            return
        file_response(self, candidate, mimetypes.guess_type(candidate.name)[0] or "video/mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan Tune HTML dashboard")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--output-dir", type=str, default=None, help="Existing or preferred Plan Tune output directory")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab automatically")
    parser.add_argument("--no-gpu-memory-check", action="store_true", help="Disable inter-experiment GPU memory checks and cooldown waiting")
    parser.add_argument("--registry-helper", choices=["tasks", "reward_scales"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    global INITIAL_OUTPUT_DIR, DISABLE_GPU_MEMORY_CHECK
    args = parse_args()
    if args.registry_helper:
        raise SystemExit(registry_helper_main(args.registry_helper, args.task))
    INITIAL_OUTPUT_DIR = str(Path(args.output_dir).resolve()) if args.output_dir else ""
    DISABLE_GPU_MEMORY_CHECK = bool(args.no_gpu_memory_check)
    server = ThreadingHTTPServer((args.host, args.port), PlanTuneHandler)
    shown_host = "localhost" if args.host in {"0.0.0.0", ""} else args.host
    url = f"http://{shown_host}:{args.port}"
    print(f"Plan Tune dashboard ready: {url}")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Plan Tune dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
