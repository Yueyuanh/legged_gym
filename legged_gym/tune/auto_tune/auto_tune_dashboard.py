#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Dash dashboard for auto_tune_rewards.py.

This frontend reads dashboard_state/*.json written by the training backend and
renders a live overview of trial progress, reward curves, and recorded videos.

Usage:
    python legged_gym/scripts/auto_tune_dashboard.py \
        --state-dir /path/to/dashboard_state \
        --output-dir /path/to/logs/auto_tune \
        --port 8050 --tb-url http://localhost:1230
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import dash
from dash import Input, Output, State, dash_table, dcc, html
import flask
import plotly.graph_objs as go


REFRESH_MS = 5000
WEB_VIDEO_CACHE_DIRNAME = ".dashboard_web_videos"

COLORS = {
    "bg": "#09111a",
    "bg_panel": "#0f1d2b",
    "bg_panel_alt": "#132538",
    "line": "#1e3852",
    "line_soft": "#274769",
    "text": "#edf4fb",
    "muted": "#92a8bf",
    "accent": "#4dd0e1",
    "accent_soft": "#163746",
    "good": "#8bd38b",
    "good_soft": "#173223",
    "warn": "#f7c46b",
    "warn_soft": "#392b14",
    "bad": "#ff8f8f",
    "bad_soft": "#411e24",
    "best": "#ffe08a",
    "live": "#7fe7f2",
}

TABLE_COLUMNS = [
    {"name": "Trial", "id": "trial_id", "type": "numeric"},
    {"name": "Status", "id": "status"},
    {"name": "Strategy", "id": "strategy"},
    {"name": "Progress", "id": "progress_text"},
    {"name": "Iter", "id": "iteration", "type": "numeric"},
    {"name": "Reward", "id": "best_reward", "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "Score", "id": "score", "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "Step", "id": "step", "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "StallX", "id": "stallx", "type": "numeric", "format": {"specifier": ".2f"}},
    {"name": "Time", "id": "time_label"},
]

CURVE_MODE_OPTIONS = [
    {"label": "Rewards", "value": "reward"},
    {"label": "Score", "value": "score"},
]

REWARD_CONTRIBUTION_MODE_OPTIONS = [
    {"label": "Reward Values", "value": "values"},
    {"label": "Contribution Ratio", "value": "ratio"},
]

REWARD_CATEGORY_COLORS = {
    "tracking": COLORS["accent"],
    "stability": COLORS["good"],
    "gait": COLORS["warn"],
    "regularization": "#b89cff",
    "safety": COLORS["bad"],
    "other": COLORS["muted"],
}

REWARD_CATEGORY_KEYWORDS = {
    "tracking": ("tracking", "track_vel", "low_speed"),
    "stability": ("orientation", "base_height", "base_acc", "default_joint"),
    "gait": ("feet_air", "feet_clearance", "feet_contact_number", "feet_distance", "knee_distance"),
    "regularization": ("torques", "dof_vel", "dof_acc", "action_smoothness", "stand_still", "contact_no_vel", "foot_slip"),
    "safety": ("collision", "contact_forces"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-Tune Dash Dashboard")
    parser.add_argument("--state-dir", type=str, required=True,
                        help="Directory with dashboard_state/*.json")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Auto-tune output directory (for videos)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--tb-url", type=str, default=None,
                        help="Public TensorBoard URL shown in the dashboard")
    parser.add_argument("--tb-port", type=int, default=1230,
                        help="TensorBoard port for link")
    return parser.parse_args()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_elapsed(seconds: Any) -> str:
    value = safe_float(seconds, 0.0) or 0.0
    if value >= 3600:
        return f"{value / 3600:.1f}h"
    if value >= 60:
        return f"{value / 60:.1f}min"
    return f"{value:.0f}s"


def format_metric(value: Any, digits: int = 2, fallback: str = "--") -> str:
    number = safe_float(value)
    if number is None:
        return fallback
    return f"{number:.{digits}f}"


def format_duration(value: Any, fallback: str = "--") -> str:
    number = safe_float(value)
    if number is None:
        return fallback
    return format_elapsed(number)


def format_strategy(value: Any, fallback: str = "--") -> str:
    strategy = str(value or "").strip()
    if not strategy:
        return fallback
    return strategy.replace("_", " ").upper()


def reward_term_category(name: str) -> str:
    term = str(name or "").replace("rew_", "", 1).lower()
    for category, keywords in REWARD_CATEGORY_KEYWORDS.items():
        if any(keyword in term for keyword in keywords):
            return category
    return "other"


def tail_mean_curve_value(curve: Any) -> Optional[float]:
    if not isinstance(curve, list) or not curve:
        return None
    tail_len = max(1, len(curve) // 5)
    values = []
    for point in curve[-tail_len:]:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            value = safe_float(point[1])
            if value is not None:
                values.append(value)
    if not values:
        return None
    return float(sum(values) / len(values))


def build_reward_contribution_rows(trial: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trial:
        return []
    curves = trial.get("episode_metric_curves", {}) or {}
    rows = []
    for name, curve in curves.items():
        if not str(name).startswith("rew_"):
            continue
        value = tail_mean_curve_value(curve)
        if value is None:
            continue
        rows.append({
            "name": str(name),
            "value": value,
            "category": reward_term_category(str(name)),
        })
    denominator = sum(abs(row["value"]) for row in rows)
    for row in rows:
        row["ratio"] = abs(row["value"]) / denominator if denominator > 1e-12 else 0.0
    rows.sort(key=lambda row: abs(row["value"]), reverse=True)
    return rows


def iterations_per_trial(snapshot: dict[str, Any]) -> Optional[int]:
    return safe_int((snapshot.get("session", {}) or {}).get("iterations_per_trial"))


def iteration_progress_ratio(
    iteration: Optional[int],
    total_iterations: Optional[int],
    completed: bool = False,
) -> Optional[float]:
    if completed:
        return 1.0
    if iteration is None or total_iterations is None or total_iterations <= 0:
        return None
    return max(0.0, min(1.0, float(iteration + 1) / float(total_iterations)))


def progress_bar_text(ratio: Optional[float], width: int = 10) -> str:
    if ratio is None:
        return "--"
    clamped = max(0.0, min(1.0, float(ratio)))
    filled = min(width, max(0, int(round(clamped * width))))
    return f"{'█' * filled}{'░' * (width - filled)} {clamped * 100:>3.0f}%"


def current_trial_timing(snapshot: dict[str, Any]) -> dict[str, Any]:
    session = snapshot.get("session", {}) or {}
    total_iterations = iterations_per_trial(snapshot)
    current_trial_id = active_live_trial_id(snapshot)
    current_iteration = live_iteration(snapshot)
    started_at = safe_float(session.get("current_trial_started_at"))

    progress_ratio = iteration_progress_ratio(current_iteration, total_iterations)
    elapsed_seconds = None
    if started_at is not None:
        elapsed_seconds = max(0.0, time.time() - started_at)

    total_estimated_seconds = None
    remaining_seconds = None
    if (
        elapsed_seconds is not None and
        progress_ratio is not None and
        progress_ratio > 0.01
    ):
        total_estimated_seconds = elapsed_seconds / progress_ratio
        remaining_seconds = max(0.0, total_estimated_seconds - elapsed_seconds)

    return {
        "trial_id": current_trial_id,
        "iteration": current_iteration,
        "progress_ratio": progress_ratio,
        "elapsed_seconds": elapsed_seconds,
        "remaining_seconds": remaining_seconds,
        "total_estimated_seconds": total_estimated_seconds,
        "total_iterations": total_iterations,
    }


def overall_timing(snapshot: dict[str, Any]) -> dict[str, Any]:
    session = snapshot.get("session", {}) or {}
    trials = snapshot.get("trials", []) or []
    target_total = safe_int(session.get("total_trials"), len(trials)) or len(trials)
    current = current_trial_timing(snapshot)
    running = 1 if current.get("trial_id") is not None else 0

    elapsed_trials = [
        value
        for value in (safe_float(trial.get("elapsed")) for trial in trials)
        if value is not None and value > 0
    ]
    avg_trial_seconds = None
    if elapsed_trials:
        avg_trial_seconds = float(sum(elapsed_trials) / len(elapsed_trials))
    elif current.get("total_estimated_seconds") is not None:
        avg_trial_seconds = float(current["total_estimated_seconds"])

    remaining_future_trials = max(target_total - len(trials) - running, 0)
    total_remaining_seconds = None
    if avg_trial_seconds is not None:
        total_remaining_seconds = float(avg_trial_seconds * remaining_future_trials)
        if current.get("remaining_seconds") is not None:
            total_remaining_seconds += float(current["remaining_seconds"])
        elif running:
            total_remaining_seconds += avg_trial_seconds

    return {
        "target_total": target_total,
        "finished_trials": len(trials),
        "avg_trial_seconds": avg_trial_seconds,
        "total_remaining_seconds": total_remaining_seconds,
        "current": current,
    }


def progress_strip(
    label: str,
    ratio: Optional[float],
    headline: str,
    subline: str,
    tone: str = "accent",
) -> html.Div:
    color = COLORS.get(tone, COLORS["accent"])
    background = COLORS.get(f"{tone}_soft", COLORS["accent_soft"])
    clamped = max(0.0, min(1.0, float(ratio or 0.0)))
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "8px",
            "padding": "14px 16px",
            "borderRadius": "16px",
            "border": f"1px solid {COLORS['line']}",
            "backgroundColor": COLORS["bg"],
        },
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "gap": "12px", "alignItems": "center"},
                children=[
                    html.Div(label, style={"fontSize": "12px", "letterSpacing": "0.08em", "textTransform": "uppercase", "color": COLORS["muted"]}),
                    html.Div(progress_bar_text(ratio), style={"fontFamily": "'IBM Plex Mono', monospace", "fontSize": "12px", "color": COLORS["text"]}),
                ],
            ),
            html.Div(
                style={"height": "12px", "borderRadius": "999px", "overflow": "hidden", "backgroundColor": background},
                children=[
                    html.Div(
                        style={
                            "width": f"{clamped * 100:.1f}%",
                            "height": "100%",
                            "borderRadius": "999px",
                            "background": f"linear-gradient(90deg, {color} 0%, {COLORS['good']} 100%)",
                            "transition": "width 0.35s ease",
                        }
                    )
                ],
            ),
            html.Div(headline, style={"fontSize": "15px", "fontWeight": 600, "color": COLORS["text"]}),
            html.Div(subline, style={"fontSize": "12px", "color": COLORS["muted"]}),
        ],
    )


def format_video_size(size_bytes: Optional[int]) -> str:
    if not size_bytes:
        return "--"
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return "--"


def make_signature(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def normalize_video_status(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"ready", "done", "completed"}:
        return "ready"
    if normalized in {"running", "encoding", "postprocessing"}:
        return "running"
    if normalized in {"pending", "queued"}:
        return "pending"
    if normalized in {"failed", "error"}:
        return "failed"
    return "unknown"


def display_video_status(status: str) -> str:
    return {
        "ready": "READY",
        "running": "ENCODING",
        "pending": "QUEUED",
        "failed": "FAILED",
        "missing": "MISSING",
        "unknown": "UNKNOWN",
        "none": "--",
    }.get(status, status.upper())


def status_tone(status: str) -> str:
    if status in {"SUCCESS", "READY", "running"}:
        return "good"
    if status in {"RUNNING", "ENCODING", "QUEUED", "pending"}:
        return "accent"
    if status in {"FAILED", "MISSING", "failed"}:
        return "bad"
    return "warn"


def card_style() -> dict[str, Any]:
    return {
        "background": f"linear-gradient(180deg, {COLORS['bg_panel_alt']} 0%, {COLORS['bg_panel']} 100%)",
        "border": f"1px solid {COLORS['line']}",
        "borderRadius": "18px",
        "boxShadow": "0 20px 40px rgba(0, 0, 0, 0.18)",
    }


def stat_card(label: str, value: str, subtext: str, tone: str = "accent") -> html.Div:
    color = COLORS.get(tone, COLORS["accent"])
    soft = COLORS.get(f"{tone}_soft", COLORS["accent_soft"])
    return html.Div(
        style={
            **card_style(),
            "padding": "16px 18px",
            "display": "flex",
            "flexDirection": "column",
            "gap": "8px",
            "minHeight": "110px",
        },
        children=[
            html.Div(label, style={"fontSize": "12px", "letterSpacing": "0.12em",
                                   "textTransform": "uppercase", "color": COLORS["muted"]}),
            html.Div(value, style={"fontSize": "28px", "fontWeight": 700, "color": color}),
            html.Div(subtext, style={"fontSize": "13px", "color": COLORS["text"]}),
            html.Div(style={"width": "64px", "height": "4px", "borderRadius": "999px",
                            "backgroundColor": soft}),
        ],
    )


def badge(text: str, tone: str) -> html.Span:
    color = COLORS.get(tone, COLORS["accent"])
    background = COLORS.get(f"{tone}_soft", COLORS["accent_soft"])
    return html.Span(
        text,
        style={
            "display": "inline-flex",
            "alignItems": "center",
            "padding": "5px 10px",
            "borderRadius": "999px",
            "fontSize": "12px",
            "fontWeight": 600,
            "letterSpacing": "0.03em",
            "color": color,
            "backgroundColor": background,
            "border": f"1px solid {color}33",
        },
    )


class DashboardStateCache:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self._lock = threading.Lock()
        self._json_cache: dict[Path, tuple[Optional[tuple[int, int]], Any]] = {}
        self._jobs_cache: tuple[Optional[tuple[Any, ...]], list[dict[str, Any]]] = (None, [])

    @staticmethod
    def _signature(path: Path) -> Optional[tuple[int, int]]:
        if not path.exists():
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _load_json(self, name: str, default: Any) -> Any:
        path = self.state_dir / name
        signature = self._signature(path)
        with self._lock:
            cached = self._json_cache.get(path)
            if cached and cached[0] == signature:
                return cached[1]

        data = read_json(path, default)
        with self._lock:
            self._json_cache[path] = (signature, data)
        return data

    def _load_video_jobs(self) -> list[dict[str, Any]]:
        job_dir = self.state_dir / "video_jobs"
        if not job_dir.exists():
            return []

        job_paths = sorted(job_dir.glob("*.json"))
        signature = tuple((path.name, self._signature(path)) for path in job_paths)
        with self._lock:
            cached_signature, cached_jobs = self._jobs_cache
            if cached_signature == signature:
                return cached_jobs

        jobs = []
        for path in job_paths:
            payload = read_json(path, None)
            if isinstance(payload, dict):
                jobs.append(payload)

        with self._lock:
            self._jobs_cache = (signature, jobs)
        return jobs

    def snapshot(self) -> dict[str, Any]:
        trials = self._load_json("trials.json", [])
        live = self._load_json("live_progress.json", {})
        live_curve = self._load_json("live_curve.json", {})
        session = self._load_json("session.json", {})
        video_jobs = self._load_video_jobs()

        trials = trials if isinstance(trials, list) else []
        live = live if isinstance(live, dict) else {}
        live_curve = live_curve if isinstance(live_curve, dict) else {}
        session = session if isinstance(session, dict) else {}
        video_jobs = video_jobs if isinstance(video_jobs, list) else []

        trials = sorted(
            trials,
            key=lambda trial: safe_int(trial.get("trial_id"), 10**9) or 10**9,
        )
        return {
            "trials": trials,
            "live": live,
            "live_curve": live_curve,
            "session": session,
            "video_jobs": video_jobs,
        }


@lru_cache(maxsize=256)
def probe_video_stream(path_str: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}

    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,codec_long_name,width,height,pix_fmt",
        "-of", "json",
        path_str,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return {}
    if proc.returncode != 0:
        return {}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}
    streams = payload.get("streams") or []
    return streams[0] if streams else {}


def get_video_stream(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return {}
    return probe_video_stream(str(path), mtime_ns)


def needs_web_transcode(stream: dict[str, Any]) -> bool:
    codec = stream.get("codec_name")
    pix_fmt = stream.get("pix_fmt")
    if not codec:
        return False
    return codec != "h264" or (pix_fmt and pix_fmt != "yuv420p")


def resolve_video_path(raw_path: Any, video_root: Path) -> Optional[Path]:
    if not raw_path:
        return None

    raw = Path(str(raw_path))
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append(video_root / raw.name)
    if "videos" in raw.parts:
        idx = raw.parts.index("videos")
        rel = Path(*raw.parts[idx + 1:])
        candidates.append(video_root / rel)

    root_resolved = video_root.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root_resolved)
        except (OSError, ValueError):
            continue
        if resolved.exists():
            return resolved
    return None


def relative_video_path(video_path: Path, video_root: Path) -> Optional[Path]:
    try:
        return video_path.resolve().relative_to(video_root.resolve())
    except (OSError, ValueError):
        return None


_web_transcode_lock = threading.Lock()


def web_cache_path(source_path: Path, video_root: Path, web_cache_root: Path) -> Optional[Path]:
    rel_path = relative_video_path(source_path, video_root)
    if rel_path is None:
        return None
    return (web_cache_root / rel_path).with_suffix(".mp4")


def ensure_web_video(source_path: Path, video_root: Path, web_cache_root: Path) -> Path:
    stream = get_video_stream(source_path)
    if not needs_web_transcode(stream):
        return source_path

    ffmpeg = shutil.which("ffmpeg")
    target_path = web_cache_path(source_path, video_root, web_cache_root)
    if not ffmpeg or target_path is None:
        return source_path

    try:
        source_mtime = source_path.stat().st_mtime_ns
    except OSError:
        return source_path

    with _web_transcode_lock:
        if target_path.exists():
            try:
                if target_path.stat().st_mtime_ns >= source_mtime and target_path.stat().st_size > 0:
                    return target_path
            except OSError:
                pass

        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(".tmp.mp4")
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel", "error",
            "-i", str(source_path),
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "veryfast",
            str(tmp_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError:
            return source_path
        if proc.returncode == 0 and tmp_path.exists():
            try:
                if tmp_path.stat().st_size > 0:
                    tmp_path.replace(target_path)
                    return target_path
            except OSError:
                pass
        tmp_path.unlink(missing_ok=True)
    return source_path


def make_video_url(video_path: Path, video_root: Path) -> Optional[str]:
    rel_path = relative_video_path(video_path, video_root)
    if rel_path is None:
        return None
    try:
        version = video_path.stat().st_mtime_ns
    except OSError:
        version = 0
    return f"/videos/{quote(rel_path.as_posix(), safe='/')}?v={version}"


def best_trial_id(trials: list[dict[str, Any]]) -> Optional[int]:
    candidates = []
    for trial in trials:
        if not trial.get("success"):
            continue
        score = safe_float(trial.get("score"))
        trial_id = safe_int(trial.get("trial_id"))
        if score is None or trial_id is None:
            continue
        candidates.append((score, trial_id))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def current_live_trial_id(snapshot: dict[str, Any]) -> Optional[int]:
    live_curve = snapshot.get("live_curve", {})
    live = snapshot.get("live", {})
    return safe_int(live_curve.get("trial_id", live.get("trial_id")))


def active_live_trial_id(snapshot: dict[str, Any]) -> Optional[int]:
    live_trial_id = current_live_trial_id(snapshot)
    if live_trial_id is None or live_trial_id in completed_trial_ids(snapshot):
        return None
    return live_trial_id


def live_iteration(snapshot: dict[str, Any]) -> Optional[int]:
    live_curve = snapshot.get("live_curve", {})
    live = snapshot.get("live", {})
    reward_curve = live_curve.get("reward_curve") or []
    if reward_curve:
        return safe_int(reward_curve[-1][0])
    return safe_int(live.get("iteration"))


def live_reward(snapshot: dict[str, Any]) -> Optional[float]:
    live_curve = snapshot.get("live_curve", {})
    live = snapshot.get("live", {})
    reward_curve = live_curve.get("reward_curve") or []
    if reward_curve:
        return safe_float(reward_curve[-1][1])
    return safe_float(live.get("reward"))


def completed_trial_ids(snapshot: dict[str, Any]) -> set[int]:
    ids = set()
    for trial in snapshot.get("trials", []):
        trial_id = safe_int(trial.get("trial_id"))
        if trial_id is not None:
            ids.add(trial_id)
    return ids


def aggregate_video_status(snapshot: dict[str, Any], video_root: Path) -> dict[int, str]:
    per_trial: dict[int, str] = {}
    priority = {
        "ready": 4,
        "running": 3,
        "pending": 2,
        "failed": 1,
        "none": 0,
        "unknown": 0,
    }

    def promote(trial_id: Optional[int], candidate: str):
        if trial_id is None:
            return
        current = per_trial.get(trial_id, "none")
        if priority.get(candidate, 0) >= priority.get(current, 0):
            per_trial[trial_id] = candidate

    for job in snapshot.get("video_jobs", []):
        trial_id = safe_int(job.get("trial_id"))
        status = normalize_video_status(job.get("status"))
        output_path = resolve_video_path(job.get("output_path") or job.get("filename"), video_root)
        if status == "ready" and output_path is None:
            status = "missing"
        promote(trial_id, status)

    for trial in snapshot.get("trials", []):
        trial_id = safe_int(trial.get("trial_id"))
        for raw_path in trial.get("video_paths", []) or []:
            if resolve_video_path(raw_path, video_root):
                promote(trial_id, "ready")

    return per_trial


def build_table_rows(snapshot: dict[str, Any], video_root: Path) -> list[dict[str, Any]]:
    rows = []
    video_status_by_trial = aggregate_video_status(snapshot, video_root)
    session = snapshot.get("session", {}) or {}
    total_iterations = iterations_per_trial(snapshot)

    for trial in snapshot.get("trials", []):
        trial_id = safe_int(trial.get("trial_id"))
        reward_curve = trial.get("reward_curve", []) or []
        sampler_info = trial.get("sampler_info", {}) or {}
        last_iteration = safe_int(reward_curve[-1][0]) if reward_curve else None
        progress_ratio = iteration_progress_ratio(
            last_iteration,
            total_iterations,
            completed=bool(trial.get("success")),
        )
        rows.append({
            "id": trial_id,
            "trial_id": trial_id,
            "status": "SUCCESS" if trial.get("success") else "FAILED",
            "strategy": format_strategy(sampler_info.get("strategy")),
            "progress_text": progress_bar_text(progress_ratio),
            "step": safe_float(sampler_info.get("anneal_scale")),
            "stallx": safe_float(sampler_info.get("stall_scale")),
            "iteration": last_iteration,
            "best_reward": safe_float(trial.get("trial_best_reward")),
            "score": safe_float(trial.get("score")),
            "time_label": format_elapsed(trial.get("elapsed")),
        })

    live_trial_id = current_live_trial_id(snapshot)
    live_completed = live_trial_id in completed_trial_ids(snapshot)
    if live_trial_id is not None and not live_completed:
        current_sampler_info = session.get("current_sampler_info", {}) or {}
        timing = current_trial_timing(snapshot)
        rows.append({
            "id": live_trial_id,
            "trial_id": live_trial_id,
            "status": "RUNNING",
            "strategy": format_strategy(current_sampler_info.get("strategy")),
            "progress_text": progress_bar_text(timing.get("progress_ratio")),
            "step": safe_float(current_sampler_info.get("anneal_scale")),
            "stallx": safe_float(current_sampler_info.get("stall_scale")),
            "iteration": live_iteration(snapshot),
            "best_reward": live_reward(snapshot),
            "score": live_reward(snapshot),
            "time_label": (
                f"ETA {format_duration(timing.get('remaining_seconds'))}"
                if timing.get("remaining_seconds") is not None else
                format_duration(timing.get("elapsed_seconds"))
            ),
        })

    return rows


def trial_id_from_rows(index: Optional[int], rows: list[dict[str, Any]]) -> Optional[int]:
    if index is None or index < 0 or index >= len(rows):
        return None
    return safe_int(rows[index].get("trial_id"))


def choose_selected_trial(
    table_rows: list[dict[str, Any]],
    active_cell: Optional[dict[str, Any]],
    virtual_rows: Optional[list[dict[str, Any]]],
    stored_selected_trial_id: Optional[int],
    active_cell_triggered: bool,
) -> Optional[int]:
    available = {safe_int(row.get("trial_id")) for row in table_rows}
    available.discard(None)
    current_rows = virtual_rows if virtual_rows else table_rows

    if active_cell and active_cell_triggered:
        row_id = safe_int(active_cell.get("row_id"))
        if row_id in available:
            return None if row_id == stored_selected_trial_id else row_id
        row_index = safe_int(active_cell.get("row"))
        selected = trial_id_from_rows(row_index, current_rows)
        if selected in available:
            return None if selected == stored_selected_trial_id else selected

    if stored_selected_trial_id in available:
        return stored_selected_trial_id

    return None


def find_trial(snapshot: dict[str, Any], trial_id: Optional[int]) -> Optional[dict[str, Any]]:
    if trial_id is None:
        return None
    for trial in snapshot.get("trials", []):
        if safe_int(trial.get("trial_id")) == trial_id:
            return trial
    return None


def build_header(snapshot: dict[str, Any], tb_url: Optional[str], output_dir: Path) -> html.Div:
    session = snapshot.get("session", {})
    task = session.get("task") or "Auto-Tune"
    experiment = session.get("experiment_name") or output_dir.name
    backend_status = str(session.get("status") or "unknown").upper()
    live_trial = active_live_trial_id(snapshot)
    live_iter_value = live_iteration(snapshot)

    right_column = [
        badge(backend_status, status_tone(backend_status)),
    ]
    if live_trial is not None:
        right_column.append(
            badge(f"TRIAL {live_trial} @ {live_iter_value or 0}", "accent")
        )
    if tb_url:
        right_column.append(
            html.A(
                "TensorBoard",
                href=tb_url,
                target="_blank",
                style={
                    "textDecoration": "none",
                    "color": COLORS["text"],
                    "fontSize": "13px",
                    "padding": "10px 14px",
                    "borderRadius": "999px",
                    "border": f"1px solid {COLORS['line_soft']}",
                    "backgroundColor": COLORS["bg_panel"],
                },
            )
        )

    return html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "flex-start",
            "gap": "18px",
            "flexWrap": "wrap",
        },
        children=[
            html.Div(
                children=[
                    html.Div(
                        task,
                        style={
                            "fontSize": "12px",
                            "letterSpacing": "0.16em",
                            "textTransform": "uppercase",
                            "color": COLORS["accent"],
                            "marginBottom": "10px",
                        },
                    ),
                    html.H1(
                        experiment,
                        style={
                            "margin": "0 0 8px 0",
                            "fontSize": "34px",
                            "lineHeight": "1.1",
                            "fontWeight": 700,
                            "color": COLORS["text"],
                        },
                    ),
                    html.Div(
                        str(output_dir),
                        style={"fontSize": "13px", "color": COLORS["muted"], "wordBreak": "break-all"},
                    ),
                ],
            ),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "10px",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                },
                children=right_column,
            ),
        ],
    )


def build_progress(snapshot: dict[str, Any]) -> html.Div:
    trials = snapshot.get("trials", [])
    timing = overall_timing(snapshot)
    current = timing.get("current", {})
    target_total = timing.get("target_total", len(trials))
    finished = timing.get("finished_trials", len(trials))
    successes = sum(1 for trial in trials if trial.get("success"))
    failures = max(finished - successes, 0)
    running = 1 if current.get("trial_id") is not None else 0
    pct = 0 if target_total <= 0 else min(100.0, 100.0 * finished / target_total)
    current_iteration = current.get("iteration")
    total_iterations = current.get("total_iterations")
    trial_headline = (
        f"Trial {current.get('trial_id')} · Iter {current_iteration if current_iteration is not None else '--'} / {total_iterations or '--'}"
        if current.get("trial_id") is not None else
        "No live trial running right now"
    )
    trial_subline = (
        f"Trial ETA {format_duration(current.get('remaining_seconds'))} · Elapsed {format_duration(current.get('elapsed_seconds'))}"
        if current.get("trial_id") is not None else
        "Start a new trial to see live iteration progress."
    )
    overall_subline = (
        f"Total ETA {format_duration(timing.get('total_remaining_seconds'))} · Avg trial {format_duration(timing.get('avg_trial_seconds'))}"
        if timing.get("total_remaining_seconds") is not None else
        "Waiting for enough timing signal to estimate total remaining time."
    )

    return html.Div(
        style={**card_style(), "padding": "18px 20px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "gap": "16px", "flexWrap": "wrap", "marginBottom": "12px"},
                children=[
                    html.Div(
                        f"Finished {finished} / {target_total} trials",
                        style={"fontSize": "17px", "fontWeight": 600, "color": COLORS["text"]},
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                        children=[
                            badge(f"{successes} success", "good"),
                            badge(f"{failures} failed", "bad" if failures else "good"),
                            badge(f"{running} running", "accent" if running else "warn"),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(300px, 1fr))",
                    "gap": "14px",
                },
                children=[
                    progress_strip(
                        "Campaign Progress",
                        pct / 100.0,
                        f"Finished {finished} / {target_total} trials",
                        overall_subline,
                        "good",
                    ),
                    progress_strip(
                        "Current Trial Progress",
                        current.get("progress_ratio"),
                        trial_headline,
                        trial_subline,
                        "accent" if current.get("trial_id") is not None else "warn",
                    ),
                ],
            ),
        ],
    )


def build_summary_cards(snapshot: dict[str, Any], video_root: Path) -> list[html.Div]:
    trials = snapshot.get("trials", [])
    session = snapshot.get("session", {})
    target_total = safe_int(session.get("total_trials"), len(trials)) or len(trials)
    best_id = best_trial_id(trials)
    best_trial = find_trial(snapshot, best_id)
    live_trial = active_live_trial_id(snapshot)
    live_iter_value = live_iteration(snapshot)

    jobs = snapshot.get("video_jobs", [])
    ready_jobs = sum(1 for job in jobs if normalize_video_status(job.get("status")) == "ready")
    pending_jobs = sum(1 for job in jobs if normalize_video_status(job.get("status")) in {"running", "pending"})
    failed_jobs = sum(1 for job in jobs if normalize_video_status(job.get("status")) == "failed")
    ready_from_trials = 0
    for trial in trials:
        for path in trial.get("video_paths", []) or []:
            if resolve_video_path(path, video_root):
                ready_from_trials += 1
    ready_total = max(ready_jobs, ready_from_trials)

    cards = [
        stat_card(
            "Trials",
            f"{len(trials)} / {target_total}",
            "Completed results written by the backend.",
            "accent",
        ),
        stat_card(
            "Best Score",
            format_metric(best_trial.get("score") if best_trial else None),
            f"Trial {best_id}" if best_id is not None else "Waiting for a successful trial.",
            "good" if best_trial else "warn",
        ),
        stat_card(
            "Running",
            f"Trial {live_trial}" if live_trial is not None else "--",
            f"Latest iteration {live_iter_value}" if live_trial is not None else "No live trial right now.",
            "accent" if live_trial is not None else "warn",
        ),
        stat_card(
            "Videos",
            f"{ready_total} ready",
            f"{pending_jobs} encoding, {failed_jobs} failed",
            "good" if ready_total else ("accent" if pending_jobs else "warn"),
        ),
    ]
    return cards


def build_table_style_data(selected_trial_id: Optional[int]) -> list[dict[str, Any]]:
    styles = [
        {"if": {"row_index": "odd"}, "backgroundColor": "#0c1824"},
        {"if": {"filter_query": "{status} = 'SUCCESS'"}, "borderLeft": f"3px solid {COLORS['good']}"},
        {"if": {"filter_query": "{status} = 'FAILED'"}, "borderLeft": f"3px solid {COLORS['bad']}"},
        {"if": {"filter_query": "{status} = 'RUNNING'"}, "borderLeft": f"3px solid {COLORS['accent']}"},
        {
            "if": {"column_id": "progress_text"},
            "fontFamily": "'IBM Plex Mono', monospace",
            "fontSize": "12px",
        },
        {
            "if": {"state": "active"},
            "backgroundColor": "rgba(0, 0, 0, 0)",
            "border": "1px solid transparent",
            "color": COLORS["text"],
        },
        {
            "if": {"state": "selected"},
            "backgroundColor": "rgba(0, 0, 0, 0)",
            "border": "1px solid transparent",
            "color": COLORS["text"],
        },
    ]
    if selected_trial_id is not None:
        styles.append(
            {
                "if": {"filter_query": f"{{trial_id}} = {selected_trial_id}"},
                "backgroundColor": "#173246",
                "color": COLORS["text"],
                "borderTop": f"1px solid {COLORS['accent']}",
                "borderBottom": f"1px solid {COLORS['accent']}",
            }
        )
    return styles


def add_curve_trace(
    fig: go.Figure,
    trial_id: int,
    curve: list[Any],
    name: str,
    color: str,
    width: int,
    opacity: float = 1.0,
    dash_style: str = "solid",
) -> None:
    if not curve:
        return
    x_values = []
    y_values = []
    for point in curve:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x_value = safe_int(point[0])
        y_value = safe_float(point[1])
        if x_value is None or y_value is None:
            continue
        x_values.append(x_value)
        y_values.append(y_value)
    if not x_values:
        return

    mode = "lines+markers" if len(x_values) < 3 else "lines"
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode=mode,
        name=name,
        line=dict(color=color, width=width, dash=dash_style),
        marker=dict(size=5),
        opacity=opacity,
        hovertemplate=(
            f"Trial {trial_id}"
            "<br>Iteration: %{x}"
            "<br>Reward: %{y:.2f}<extra></extra>"
        ),
    ))


def reward_figure_signature_payload(
    snapshot: dict[str, Any],
    selected_trial_id: Optional[int],
    curve_mode: str,
) -> dict[str, Any]:
    trials = snapshot.get("trials", [])
    best_id = best_trial_id(trials)
    live_trial = current_live_trial_id(snapshot)
    live_curve = snapshot.get("live_curve", {}).get("reward_curve", []) or []
    payload: dict[str, Any] = {
        "curve_mode": curve_mode,
        "selected_trial_id": selected_trial_id,
        "best_id": best_id,
    }

    if curve_mode == "score":
        payload["score_points"] = [
            {
                "trial_id": safe_int(trial.get("trial_id")),
                "score": safe_float(trial.get("score")),
                "success": bool(trial.get("success")),
            }
            for trial in trials
        ]
        return payload

    if selected_trial_id is not None:
        selected_trial = find_trial(snapshot, selected_trial_id)
        payload["selected_curve"] = selected_trial.get("reward_curve", []) if selected_trial else []
        if best_id is not None and best_id != selected_trial_id:
            best_trial = find_trial(snapshot, best_id)
            payload["best_curve"] = best_trial.get("reward_curve", []) if best_trial else []
    else:
        successful = [trial for trial in trials if trial.get("success")]
        successful.sort(key=lambda trial: safe_float(trial.get("score"), -1e18) or -1e18, reverse=True)
        payload["overview_curves"] = [
            {
                "trial_id": safe_int(trial.get("trial_id")),
                "reward_curve": trial.get("reward_curve", []) or [],
                "score": safe_float(trial.get("score")),
            }
            for trial in successful
        ]

    if live_trial is not None and live_curve and live_trial not in completed_trial_ids(snapshot):
        payload["live_trial"] = live_trial
        payload["live_curve"] = live_curve

    return payload


def build_reward_figure(snapshot: dict[str, Any], selected_trial_id: Optional[int]) -> go.Figure:
    fig = go.Figure()
    overview_mode = selected_trial_id is None
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=56, r=24, t=56, b=50),
        title=dict(
            text=f"Reward Curves · Trial {selected_trial_id}" if selected_trial_id is not None else "Reward Curves · All Trials",
            font=dict(size=16),
        ),
        xaxis=dict(title="Iteration", gridcolor=COLORS["line"], zeroline=False),
        yaxis=dict(title="Mean Reward", gridcolor=COLORS["line"], zeroline=False),
        showlegend=not overview_mode,
        legend=(
            dict(
                orientation="h",
                x=0,
                xanchor="left",
                y=1.12,
                yanchor="bottom",
                bgcolor="rgba(0, 0, 0, 0)",
                font=dict(size=11),
            )
        ) if not overview_mode else {},
        hovermode="x unified",
        font=dict(color=COLORS["text"]),
        uirevision=f"reward-curve-{selected_trial_id if selected_trial_id is not None else 'overview'}",
    )

    trials = snapshot.get("trials", [])
    live_trial = current_live_trial_id(snapshot)
    best_id = best_trial_id(trials)
    live_curve = snapshot.get("live_curve", {}).get("reward_curve", []) or []

    if selected_trial_id is not None:
        selected_trial = find_trial(snapshot, selected_trial_id)
        if selected_trial:
            add_curve_trace(
                fig,
                selected_trial_id,
                selected_trial.get("reward_curve", []) or [],
                f"Trial {selected_trial_id}",
                COLORS["accent"],
                3,
            )
            if best_id is not None and best_id != selected_trial_id:
                best_trial = find_trial(snapshot, best_id)
                if best_trial:
                    add_curve_trace(
                        fig,
                        best_id,
                        best_trial.get("reward_curve", []) or [],
                        f"Trial {best_id} (best)",
                        COLORS["best"],
                        2,
                        opacity=0.85,
                        dash_style="dot",
                    )
        elif live_trial == selected_trial_id and best_id is not None:
            best_trial = find_trial(snapshot, best_id)
            if best_trial:
                add_curve_trace(
                    fig,
                    best_id,
                    best_trial.get("reward_curve", []) or [],
                    f"Trial {best_id} (best)",
                    COLORS["best"],
                    2,
                    opacity=0.85,
                    dash_style="dot",
                )
        if live_trial == selected_trial_id and live_curve and live_trial not in completed_trial_ids(snapshot):
            add_curve_trace(
                fig,
                selected_trial_id,
                live_curve,
                f"Trial {selected_trial_id} (running)",
                COLORS["live"],
                3,
            )
    else:
        successful = [trial for trial in trials if trial.get("success")]
        successful.sort(key=lambda trial: safe_float(trial.get("score"), -1e18) or -1e18, reverse=True)
        for trial in successful:
            trial_id = safe_int(trial.get("trial_id"))
            if trial_id is None:
                continue
            is_best = trial_id == best_id
            add_curve_trace(
                fig,
                trial_id,
                trial.get("reward_curve", []) or [],
                f"Trial {trial_id}" + (" (best)" if is_best else ""),
                COLORS["best"] if is_best else COLORS["line_soft"],
                3 if is_best else 1,
                opacity=1.0 if is_best else 0.28,
            )
        if live_trial is not None and live_curve and live_trial not in completed_trial_ids(snapshot):
            add_curve_trace(
                fig,
                live_trial,
                live_curve,
                f"Trial {live_trial} (running)",
                COLORS["live"],
                3,
            )

    return fig


def build_score_figure(snapshot: dict[str, Any], selected_trial_id: Optional[int]) -> go.Figure:
    fig = go.Figure()
    overview_mode = selected_trial_id is None
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=56, r=24, t=56, b=50),
        title=dict(
            text=f"Trial Score Evolution · Trial {selected_trial_id}" if selected_trial_id is not None else "Trial Score Evolution",
            font=dict(size=16),
        ),
        xaxis=dict(title="Trial", gridcolor=COLORS["line"], zeroline=False),
        yaxis=dict(title="Final Score", gridcolor=COLORS["line"], zeroline=False),
        showlegend=not overview_mode,
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=1.12,
            yanchor="bottom",
            bgcolor="rgba(0, 0, 0, 0)",
            font=dict(size=11),
        ) if not overview_mode else {},
        hovermode="x unified",
        font=dict(color=COLORS["text"]),
        uirevision=f"score-curve-{selected_trial_id if selected_trial_id is not None else 'overview'}",
    )

    successful = []
    for trial in snapshot.get("trials", []):
        if not trial.get("success"):
            continue
        trial_id = safe_int(trial.get("trial_id"))
        score = safe_float(trial.get("score"))
        if trial_id is None or score is None:
            continue
        successful.append((trial_id, score))

    successful.sort(key=lambda item: item[0])
    if not successful:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No successful trials yet.",
            showarrow=False,
            font=dict(color=COLORS["muted"], size=14),
        )
        return fig

    best_id = best_trial_id(snapshot.get("trials", []))
    x_values = [trial_id for trial_id, _score in successful]
    y_values = [score for _trial_id, score in successful]
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines+markers",
        name="Scores",
        line=dict(color=COLORS["line_soft"], width=2),
        marker=dict(size=7, color=COLORS["accent"]),
        hovertemplate="Trial %{x}<br>Score: %{y:.2f}<extra></extra>",
    ))

    if best_id is not None:
        best_trial = find_trial(snapshot, best_id)
        best_score = safe_float(best_trial.get("score")) if best_trial else None
        if best_score is not None:
            fig.add_trace(go.Scatter(
                x=[best_id],
                y=[best_score],
                mode="markers",
                name=f"Best · Trial {best_id}",
                marker=dict(size=14, color=COLORS["best"], symbol="star"),
                hovertemplate=f"Best Trial {best_id}<br>Score: %{{y:.2f}}<extra></extra>",
            ))

    if selected_trial_id is not None:
        selected_trial = find_trial(snapshot, selected_trial_id)
        selected_score = safe_float(selected_trial.get("score")) if selected_trial else None
        if selected_score is not None and selected_trial.get("success"):
            fig.add_trace(go.Scatter(
                x=[selected_trial_id],
                y=[selected_score],
                mode="markers",
                name=f"Selected · Trial {selected_trial_id}",
                marker=dict(size=13, color=COLORS["accent"], symbol="diamond"),
                hovertemplate=f"Selected Trial {selected_trial_id}<br>Score: %{{y:.2f}}<extra></extra>",
            ))

    return fig


def reward_contribution_signature_payload(
    snapshot: dict[str, Any],
    selected_trial_id: Optional[int],
    mode: str,
) -> dict[str, Any]:
    trial = find_trial(snapshot, selected_trial_id)
    return {
        "selected_trial_id": selected_trial_id,
        "mode": mode,
        "episode_metric_curves": (trial or {}).get("episode_metric_curves", {}),
    }


def build_reward_contribution_figure(
    snapshot: dict[str, Any],
    selected_trial_id: Optional[int],
    mode: str,
) -> go.Figure:
    active_mode = "ratio" if mode == "ratio" else "values"
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=12, r=16, t=34, b=38),
        height=430,
        font=dict(color=COLORS["text"]),
        showlegend=False,
        uirevision=f"reward-contribution-{selected_trial_id}-{active_mode}",
    )

    if selected_trial_id is None:
        fig.add_annotation(
            text="Select a trial row to inspect reward contribution.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=COLORS["muted"], size=14),
        )
        return fig

    selected_trial = find_trial(snapshot, selected_trial_id)
    rows = build_reward_contribution_rows(selected_trial)
    if not rows:
        fig.add_annotation(
            text="No reward episode metrics recorded for this trial.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=COLORS["muted"], size=14),
        )
        return fig

    x_values = [
        row["value"] if active_mode == "values" else row["ratio"] * 100.0
        for row in rows
    ]
    y_values = [row["name"] for row in rows]
    colors = [REWARD_CATEGORY_COLORS.get(row["category"], COLORS["muted"]) for row in rows]
    text_values = [
        format_metric(row["value"], digits=4)
        if active_mode == "values"
        else f"{row['ratio'] * 100.0:.1f}%"
        for row in rows
    ]
    customdata = [
        [row["category"], row["value"], row["ratio"] * 100.0]
        for row in rows
    ]

    fig.add_trace(go.Bar(
        x=x_values,
        y=y_values,
        orientation="h",
        marker=dict(color=colors),
        text=text_values,
        textposition="auto",
        customdata=customdata,
        hovertemplate=(
            "Term: %{y}<br>"
            "Category: %{customdata[0]}<br>"
            "Value: %{customdata[1]:.4f}<br>"
            "Abs contribution: %{customdata[2]:.1f}%"
            "<extra></extra>"
        ),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(
        title="Signed final/tail reward value"
        if active_mode == "values"
        else "Absolute contribution (%)",
        zeroline=True,
        zerolinecolor=COLORS["line_soft"],
        gridcolor="rgba(39, 71, 105, 0.45)",
    )
    return fig


def build_trial_details(
    snapshot: dict[str, Any],
    selected_trial_id: Optional[int],
    video_status_by_trial: dict[int, str],
) -> html.Div:
    selected_trial = find_trial(snapshot, selected_trial_id)
    live_trial = current_live_trial_id(snapshot)
    session = snapshot.get("session", {}) or {}
    is_live_only = selected_trial is None and selected_trial_id is not None and selected_trial_id == live_trial

    if selected_trial is None and not is_live_only:
        return html.Div(
            style={"padding": "10px 0", "color": COLORS["muted"]},
            children="Select a trial row to inspect its score breakdown and videos.",
        )

    if is_live_only:
        sampler_info = session.get("current_sampler_info", {}) or {}
        timing = current_trial_timing(snapshot)
        info_rows = [
            ("Status", "RUNNING"),
            ("Strategy", format_strategy(sampler_info.get("strategy"))),
            ("Step", format_metric(sampler_info.get("anneal_scale"))),
            ("StallX", format_metric(sampler_info.get("stall_scale"))),
            ("Iteration", str(live_iteration(snapshot) or "--")),
            ("Reward", format_metric(live_reward(snapshot))),
            ("Video", display_video_status(video_status_by_trial.get(selected_trial_id, "none"))),
        ]
        reward_scales = session.get("current_reward_scales", {}) or {}
        progress_ratio = timing.get("progress_ratio")
        progress_headline = (
            f"Iter {timing.get('iteration') if timing.get('iteration') is not None else '--'} / {timing.get('total_iterations') or '--'}"
        )
        progress_subline = (
            f"Trial ETA {format_duration(timing.get('remaining_seconds'))} · Elapsed {format_duration(timing.get('elapsed_seconds'))}"
        )
    else:
        best_ckpt = safe_int(selected_trial.get("trial_best_model_iter"), None)
        sampler_info = selected_trial.get("sampler_info", {}) or {}
        total_iterations = iterations_per_trial(snapshot)
        reward_curve = selected_trial.get("reward_curve", []) or []
        last_iteration = safe_int(reward_curve[-1][0]) if reward_curve else None
        progress_ratio = iteration_progress_ratio(
            last_iteration,
            total_iterations,
            completed=bool(selected_trial.get("success")),
        )
        progress_headline = (
            f"Iter {last_iteration if last_iteration is not None else '--'} / {total_iterations or '--'}"
        )
        progress_subline = (
            f"Elapsed {format_elapsed(selected_trial.get('elapsed'))} · Best reward {format_metric(selected_trial.get('trial_best_reward'))}"
        )
        info_rows = [
            ("Status", "SUCCESS" if selected_trial.get("success") else "FAILED"),
            ("Strategy", format_strategy(sampler_info.get("strategy"))),
            ("Step", format_metric(sampler_info.get("anneal_scale"))),
            ("StallX", format_metric(sampler_info.get("stall_scale"))),
            ("Score", format_metric(selected_trial.get("score"))),
            ("Best Reward", format_metric(selected_trial.get("trial_best_reward"))),
            ("Best Ckpt", str(best_ckpt) if best_ckpt is not None else "--"),
            ("Elapsed", format_elapsed(selected_trial.get("elapsed"))),
            ("Video", display_video_status(video_status_by_trial.get(selected_trial_id, "none"))),
        ]
        reward_scales = selected_trial.get("reward_scales", {}) or {}

    sorted_scales = sorted(
        reward_scales.items(),
        key=lambda item: abs(safe_float(item[1], 0.0) or 0.0),
        reverse=True,
    )

    return html.Div(
        children=[
            html.Div(
                f"Trial {selected_trial_id}",
                style={"fontSize": "22px", "fontWeight": 700, "color": COLORS["text"], "marginBottom": "14px"},
            ),
            progress_strip(
                "Iteration Progress",
                progress_ratio,
                progress_headline,
                progress_subline,
                "accent" if is_live_only else ("good" if selected_trial and selected_trial.get("success") else "warn"),
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(140px, 1fr))",
                    "gap": "10px",
                    "marginTop": "16px",
                    "marginBottom": "18px",
                },
                children=[
                    html.Div(
                        style={
                            "padding": "12px 14px",
                            "borderRadius": "14px",
                            "border": f"1px solid {COLORS['line']}",
                            "backgroundColor": COLORS["bg"],
                        },
                        children=[
                            html.Div(label, style={"fontSize": "12px", "color": COLORS["muted"],
                                                   "textTransform": "uppercase", "letterSpacing": "0.08em"}),
                            html.Div(value, style={"fontSize": "16px", "fontWeight": 600,
                                                   "marginTop": "6px", "color": COLORS["text"]}),
                        ],
                    )
                    for label, value in info_rows
                ],
            ),
            html.Div(
                "Reward scales",
                style={"fontSize": "13px", "fontWeight": 600, "color": COLORS["muted"], "marginBottom": "10px"},
            ),
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "8px"},
                children=[
                    badge(f"{name}: {format_metric(value, digits=3)}", "accent")
                    for name, value in sorted_scales
                ] or [html.Div("No reward scales recorded yet.", style={"color": COLORS["muted"]})],
            ),
        ]
    )


def format_video_label(trial_id: int, filename: str, checkpoint_iter: Optional[int]) -> str:
    if checkpoint_iter is not None:
        return f"Best iteration {checkpoint_iter}"
    label = filename.replace(".mp4", "")
    label = label.replace(f"trial_{trial_id:03d}_best_iter_", "Best iteration ")
    label = label.replace(f"trial_{trial_id:03d}_iter_", "Iteration ")
    return label


def build_video_entries(
    snapshot: dict[str, Any],
    trial_id: Optional[int],
    video_root: Path,
    web_cache_root: Path,
) -> list[dict[str, Any]]:
    if trial_id is None:
        return []

    selected_trial = find_trial(snapshot, trial_id)
    jobs = [
        job for job in snapshot.get("video_jobs", [])
        if safe_int(job.get("trial_id")) == trial_id
    ]

    entries: dict[str, dict[str, Any]] = {}
    for job in jobs:
        key = str(job.get("output_path") or job.get("filename") or len(entries))
        entry = entries.setdefault(key, {
            "trial_id": trial_id,
            "output_path": job.get("output_path"),
            "filename": job.get("filename") or os.path.basename(str(job.get("output_path") or "")),
            "checkpoint_iter": safe_int(job.get("checkpoint_iter")),
            "status": "none",
            "updated_at": safe_float(job.get("updated_at")),
            "error": job.get("error"),
        })
        entry["status"] = normalize_video_status(job.get("status"))
        entry["updated_at"] = safe_float(job.get("updated_at")) or entry["updated_at"]
        entry["error"] = job.get("error") or entry["error"]

    if selected_trial:
        for raw_path in selected_trial.get("video_paths", []) or []:
            key = str(raw_path)
            entry = entries.setdefault(key, {
                "trial_id": trial_id,
                "output_path": raw_path,
                "filename": os.path.basename(str(raw_path)),
                "checkpoint_iter": safe_int(selected_trial.get("trial_best_model_iter")),
                "status": "ready",
                "updated_at": None,
                "error": None,
            })
            if entry.get("status") in {"none", "unknown"}:
                entry["status"] = "ready"

    resolved_entries = []
    for entry in entries.values():
        resolved_path = resolve_video_path(entry.get("output_path") or entry.get("filename"), video_root)
        filename = entry.get("filename") or os.path.basename(str(entry.get("output_path") or "")) or "video.mp4"
        stream = get_video_stream(resolved_path) if resolved_path else {}
        cache_path = web_cache_path(resolved_path, video_root, web_cache_root) if resolved_path else None
        cache_ready = cache_path.exists() if cache_path else False
        resolved_entries.append({
            **entry,
            "filename": filename,
            "label": format_video_label(trial_id, filename, entry.get("checkpoint_iter")),
            "resolved_path": resolved_path,
            "url": make_video_url(resolved_path, video_root) if resolved_path else None,
            "size_bytes": resolved_path.stat().st_size if resolved_path and resolved_path.exists() else None,
            "stream": stream,
            "needs_web_transcode": needs_web_transcode(stream),
            "cache_ready": cache_ready,
        })

    priority = {"ready": 4, "running": 3, "pending": 2, "failed": 1, "missing": 0, "unknown": 0, "none": 0}
    for entry in resolved_entries:
        if entry["status"] == "ready" and entry["resolved_path"] is None:
            entry["status"] = "missing"

    resolved_entries.sort(
        key=lambda entry: (
            -priority.get(entry["status"], 0),
            -(entry.get("checkpoint_iter") or -1),
            entry["filename"],
        )
    )
    return resolved_entries


def build_video_gallery(
    snapshot: dict[str, Any],
    selected_trial_id: Optional[int],
    video_root: Path,
    web_cache_root: Path,
) -> list[html.Div]:
    entries = build_video_entries(snapshot, selected_trial_id, video_root, web_cache_root)
    if selected_trial_id is None:
        return [html.Div("Select a trial row to inspect its videos.", style={"color": COLORS["muted"]})]
    if not entries:
        return [html.Div("No videos recorded for this trial yet.", style={"color": COLORS["muted"]})]

    cards = []
    for entry in entries:
        status = entry["status"]
        tone = "good" if status == "ready" else "accent" if status in {"running", "pending"} else "bad"
        meta_line = []
        stream = entry.get("stream") or {}
        codec_name = stream.get("codec_name")
        width = safe_int(stream.get("width"))
        height = safe_int(stream.get("height"))
        if codec_name:
            meta_line.append(f"codec {codec_name}")
        if width and height:
            meta_line.append(f"{width}x{height}")
        meta_line.append(format_video_size(entry.get("size_bytes")))

        body = [
            html.Div(
                style={"display": "flex", "justifyContent": "space-between",
                       "alignItems": "center", "gap": "10px", "marginBottom": "10px"},
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                f"Trial {entry['trial_id']}",
                                style={"fontSize": "11px", "letterSpacing": "0.08em", "textTransform": "uppercase", "color": COLORS["muted"], "marginBottom": "4px"},
                            ),
                            html.Div(entry["label"], style={"fontSize": "16px", "fontWeight": 600,
                                                             "color": COLORS["text"]}),
                            html.Div(" | ".join(meta_line), style={"fontSize": "12px", "color": COLORS["muted"],
                                                                   "marginTop": "4px"}),
                        ]
                    ),
                    badge(display_video_status(status), tone),
                ],
            )
        ]

        if status == "ready" and entry.get("url"):
            body.extend([
                html.Video(
                    src=entry["url"],
                    controls=True,
                    preload="metadata",
                    playsInline=True,
                    style={
                        "width": "100%",
                        "borderRadius": "14px",
                        "border": f"1px solid {COLORS['line']}",
                        "backgroundColor": "#000000",
                    },
                ),
                html.Div(
                    style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginTop": "10px"},
                    children=[
                        html.A(
                            "Open video",
                            href=entry["url"],
                            target="_blank",
                            style={"color": COLORS["accent"], "fontSize": "13px"},
                        ),
                        badge("web cache ready" if entry.get("cache_ready") else "auto web optimize", "accent")
                        if entry.get("needs_web_transcode") else badge("browser ready", "good"),
                    ],
                ),
            ])
        elif status in {"running", "pending"}:
            body.append(html.Div(
                "Video is still being rendered. Refresh will pick it up automatically.",
                style={"fontSize": "13px", "color": COLORS["muted"]},
            ))
        else:
            message = entry.get("error") or "Video file is missing or failed to render."
            body.append(html.Div(message, style={"fontSize": "13px", "color": COLORS["bad"]}))

        cards.append(
            html.Div(
                style={**card_style(), "padding": "14px", "height": "100%"},
                children=body,
            )
        )

    return cards


def video_gallery_signature_payload(
    entries: list[dict[str, Any]],
    selected_trial_id: Optional[int],
) -> dict[str, Any]:
    return {
        "selected_trial_id": selected_trial_id,
        "entries": [
            {
                "label": entry.get("label"),
                "status": entry.get("status"),
                "filename": entry.get("filename"),
                "checkpoint_iter": entry.get("checkpoint_iter"),
                "url": entry.get("url"),
                "size_bytes": entry.get("size_bytes"),
                "needs_web_transcode": entry.get("needs_web_transcode"),
                "cache_ready": entry.get("cache_ready"),
                "stream": entry.get("stream"),
                "error": entry.get("error"),
            }
            for entry in entries
        ],
    }


def curve_panel_title(curve_mode: str, selected_trial_id: Optional[int]) -> str:
    if curve_mode == "score":
        return "Score Curves"
    if selected_trial_id is None:
        return "Reward Curves · All Trials"
    return f"Reward Curves · Trial {selected_trial_id}"


def video_panel_title(selected_trial_id: Optional[int]) -> str:
    if selected_trial_id is None:
        return "Videos"
    return f"Videos · Trial {selected_trial_id}"


def create_app(state_dir: Path, output_dir: Path, tb_url: Optional[str]):
    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )
    app.title = "Auto-Tune Dashboard"

    video_root = (output_dir / "videos").resolve()
    web_cache_root = (output_dir / WEB_VIDEO_CACHE_DIRNAME).resolve()
    state_cache = DashboardStateCache(state_dir)

    @app.server.route("/videos/<path:relative_path>")
    def serve_video(relative_path):
        candidate = (video_root / relative_path).resolve()
        try:
            candidate.relative_to(video_root)
        except ValueError:
            flask.abort(404)
        if not candidate.exists():
            flask.abort(404)

        served_path = ensure_web_video(candidate, video_root, web_cache_root)
        response = flask.send_file(
            served_path,
            mimetype="video/mp4",
            conditional=True,
            etag=True,
            max_age=0,
        )
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Accept-Ranges"] = "bytes"
        return response

    app.layout = html.Div(
        style={
            "minHeight": "100vh",
            "padding": "24px",
            "color": COLORS["text"],
            "fontFamily": "'IBM Plex Sans', 'Segoe UI', sans-serif",
            "background": (
                "radial-gradient(circle at top left, rgba(77, 208, 225, 0.14), transparent 34%),"
                "linear-gradient(180deg, #08101a 0%, #0b1622 100%)"
            ),
        },
        children=[
            dcc.Interval(id="refresh", interval=REFRESH_MS),
            dcc.Store(id="reward-curves-signature"),
            dcc.Store(id="reward-contribution-signature"),
            dcc.Store(id="video-gallery-signature"),
            dcc.Store(id="selected-trial-id"),
            html.Div(id="header-shell", style={**card_style(), "padding": "22px", "marginBottom": "18px"}),
            html.Div(id="progress-shell", style={"marginBottom": "18px"}),
            html.Div(
                id="summary-shell",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(210px, 1fr))",
                    "gap": "14px",
                    "marginBottom": "18px",
                },
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(0, 1.3fr) minmax(0, 1fr)",
                    "gap": "18px",
                    "marginBottom": "18px",
                },
                children=[
                    html.Div(
                        style={**card_style(), "padding": "18px", "minWidth": 0},
                        children=[
                            html.Div("Trials", style={"fontSize": "20px", "fontWeight": 700, "marginBottom": "14px"}),
                            dash_table.DataTable(
                                id="trial-table",
                                columns=TABLE_COLUMNS,
                                data=[],
                                cell_selectable=True,
                                sort_action="native",
                                filter_action="none",
                                page_size=12,
                                style_as_list_view=True,
                                style_table={"width": "100%", "overflowX": "hidden"},
                                style_header={
                                    "backgroundColor": COLORS["bg"],
                                    "color": COLORS["text"],
                                    "fontWeight": "bold",
                                    "border": "none",
                                    "padding": "9px 6px",
                                    "fontSize": "11px",
                                },
                                style_cell={
                                    "backgroundColor": "rgba(0, 0, 0, 0)",
                                    "color": COLORS["text"],
                                    "padding": "7px 6px",
                                    "fontSize": "11px",
                                    "lineHeight": "16px",
                                    "border": "none",
                                    "textAlign": "left",
                                    "whiteSpace": "nowrap",
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": "trial_id"}, "width": "42px"},
                                    {"if": {"column_id": "status"}, "width": "64px"},
                                    {"if": {"column_id": "strategy"}, "width": "86px"},
                                    {"if": {"column_id": "progress_text"}, "width": "104px"},
                                    {"if": {"column_id": "iteration"}, "width": "48px"},
                                    {"if": {"column_id": "best_reward"}, "width": "62px"},
                                    {"if": {"column_id": "score"}, "width": "58px"},
                                    {"if": {"column_id": "step"}, "width": "50px"},
                                    {"if": {"column_id": "stallx"}, "width": "52px"},
                                    {"if": {"column_id": "time_label"}, "width": "72px"},
                                ],
                                style_data_conditional=build_table_style_data(None),
                                css=[
                                    {"selector": ".dash-cell.focused", "rule": "outline: none !important;"},
                                    {"selector": ".dash-cell.cell--selected", "rule": "background-color: #173246 !important;"},
                                    {"selector": "td.dash-cell", "rule": "cursor: pointer;"},
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style={**card_style(), "padding": "18px", "minWidth": 0},
                        children=[
                            html.Div(
                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "14px", "marginBottom": "14px", "flexWrap": "wrap"},
                                children=[
                                    html.Div(id="curve-title", children="Reward Curves · All Trials", style={"fontSize": "20px", "fontWeight": 700}),
                                    dcc.RadioItems(
                                        id="curve-mode",
                                        options=CURVE_MODE_OPTIONS,
                                        value="reward",
                                        inline=True,
                                        labelStyle={"display": "inline-flex", "alignItems": "center", "marginRight": "12px", "color": COLORS["text"], "fontSize": "13px"},
                                        inputStyle={"marginRight": "6px"},
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                id="reward-curves",
                                style={"height": "460px"},
                                config={"displayModeBar": False, "responsive": True},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))",
                    "gap": "18px",
                    "marginBottom": "18px",
                },
                children=[
                    html.Div(
                        style={**card_style(), "padding": "18px", "minWidth": 0},
                        children=[
                            html.Div(
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "gap": "14px",
                                    "marginBottom": "14px",
                                    "flexWrap": "wrap",
                                },
                                children=[
                                    html.Div("Reward Contribution", style={"fontSize": "20px", "fontWeight": 700}),
                                    dcc.RadioItems(
                                        id="reward-contribution-mode",
                                        options=REWARD_CONTRIBUTION_MODE_OPTIONS,
                                        value="values",
                                        inline=True,
                                        labelStyle={
                                            "display": "inline-flex",
                                            "alignItems": "center",
                                            "marginRight": "12px",
                                            "color": COLORS["text"],
                                            "fontSize": "13px",
                                        },
                                        inputStyle={"marginRight": "6px"},
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                id="reward-contribution",
                                style={"height": "430px"},
                                config={"displayModeBar": False, "responsive": True},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(360px, 1fr))",
                    "gap": "18px",
                },
                children=[
                    html.Div(
                        style={**card_style(), "padding": "18px", "minWidth": 0},
                        children=[
                            html.Div("Trial Details", style={"fontSize": "20px", "fontWeight": 700, "marginBottom": "14px"}),
                            html.Div(id="trial-details"),
                        ],
                    ),
                    html.Div(
                        style={**card_style(), "padding": "18px", "minWidth": 0},
                        children=[
                            html.Div(id="video-title", children="Videos", style={"fontSize": "20px", "fontWeight": 700, "marginBottom": "14px"}),
                            html.Div(
                                id="video-gallery",
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                                    "gap": "14px",
                                },
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        [
            Output("header-shell", "children"),
            Output("progress-shell", "children"),
            Output("summary-shell", "children"),
            Output("trial-table", "data"),
            Output("trial-table", "style_data_conditional"),
            Output("reward-curves", "figure"),
            Output("reward-curves-signature", "data"),
            Output("reward-contribution", "figure"),
            Output("reward-contribution-signature", "data"),
            Output("selected-trial-id", "data"),
            Output("curve-title", "children"),
            Output("trial-details", "children"),
            Output("video-title", "children"),
            Output("video-gallery", "children"),
            Output("video-gallery-signature", "data"),
        ],
        [
            Input("refresh", "n_intervals"),
            Input("trial-table", "active_cell"),
            Input("curve-mode", "value"),
            Input("reward-contribution-mode", "value"),
        ],
        [
            State("trial-table", "derived_virtual_data"),
            State("reward-curves-signature", "data"),
            State("reward-contribution-signature", "data"),
            State("video-gallery-signature", "data"),
            State("selected-trial-id", "data"),
        ],
    )
    def refresh_dashboard(
        _n,
        active_cell,
        curve_mode,
        reward_contribution_mode,
        virtual_rows,
        reward_signature,
        reward_contribution_signature,
        video_signature,
        stored_selected_trial_id,
    ):
        callback_context = dash.callback_context
        triggered_prop_id = (
            callback_context.triggered[0]["prop_id"]
            if callback_context.triggered else ""
        )
        active_cell_triggered = triggered_prop_id == "trial-table.active_cell"
        snapshot = state_cache.snapshot()
        table_rows = build_table_rows(snapshot, video_root)
        selected_trial_id = choose_selected_trial(
            table_rows,
            active_cell,
            virtual_rows,
            safe_int(stored_selected_trial_id),
            active_cell_triggered,
        )
        video_status_by_trial = aggregate_video_status(snapshot, video_root)
        reward_payload = reward_figure_signature_payload(snapshot, selected_trial_id, curve_mode)
        next_reward_signature = make_signature(reward_payload)
        reward_figure = (
            dash.no_update
            if reward_signature == next_reward_signature else
            (build_reward_figure(snapshot, selected_trial_id) if curve_mode == "reward" else build_score_figure(snapshot, selected_trial_id))
        )
        reward_signature_output = (
            dash.no_update if reward_signature == next_reward_signature else next_reward_signature
        )

        reward_contribution_payload = reward_contribution_signature_payload(
            snapshot,
            selected_trial_id,
            reward_contribution_mode,
        )
        next_reward_contribution_signature = make_signature(reward_contribution_payload)
        reward_contribution_figure = (
            dash.no_update
            if reward_contribution_signature == next_reward_contribution_signature else
            build_reward_contribution_figure(snapshot, selected_trial_id, reward_contribution_mode)
        )
        reward_contribution_signature_output = (
            dash.no_update
            if reward_contribution_signature == next_reward_contribution_signature else
            next_reward_contribution_signature
        )

        video_entries = build_video_entries(snapshot, selected_trial_id, video_root, web_cache_root)
        video_payload = video_gallery_signature_payload(video_entries, selected_trial_id)
        next_video_signature = make_signature(video_payload)
        video_gallery = (
            dash.no_update
            if video_signature == next_video_signature else
            build_video_gallery(snapshot, selected_trial_id, video_root, web_cache_root)
        )
        video_signature_output = (
            dash.no_update if video_signature == next_video_signature else next_video_signature
        )

        return (
            build_header(snapshot, tb_url, output_dir),
            build_progress(snapshot),
            build_summary_cards(snapshot, video_root),
            table_rows,
            build_table_style_data(selected_trial_id),
            reward_figure,
            reward_signature_output,
            reward_contribution_figure,
            reward_contribution_signature_output,
            selected_trial_id,
            curve_panel_title(curve_mode, selected_trial_id),
            build_trial_details(snapshot, selected_trial_id, video_status_by_trial),
            video_panel_title(selected_trial_id),
            video_gallery,
            video_signature_output,
        )

    return app


if __name__ == "__main__":
    args = parse_args()
    state_dir = Path(args.state_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not state_dir.is_dir():
        print(f"Error: state directory not found: {state_dir}", file=sys.stderr)
        sys.exit(1)

    session = read_json(state_dir / "session.json", {})
    tb_url = args.tb_url
    if not tb_url:
        tb_port = session.get("tensorboard_port", args.tb_port)
        tb_url = f"http://localhost:{tb_port}"

    app = create_app(state_dir, output_dir, tb_url)
    print(f"Dashboard starting at http://localhost:{args.port}")
    print(f"State dir: {state_dir}")
    app.run(debug=False, host="0.0.0.0", port=args.port)
