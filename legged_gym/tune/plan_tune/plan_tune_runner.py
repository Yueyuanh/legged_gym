#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Run a hand-authored reward-scale experiment sequence.

Plan Tune is intentionally close to auto_tune_rewards.py, but it does not
sample parameters. It consumes experiment_sequence.json created by the
dashboard and runs the experiments in order.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


SCRIPT_DIR = Path(__file__).resolve().parent  # .../legged_gym/tune/plan_tune/
TUNE_DIR = SCRIPT_DIR.parent  # .../legged_gym/tune/
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # legged_gym repo root
TRIAL_RUNNER = TUNE_DIR / "_run_trial.py"
RECORD_VIDEO_SCRIPT = TUNE_DIR / "_record_video.py"
TENSORBOARD_SCRIPT = None  # not available in legged_gym

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_FLOAT_RE = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
_ITER_RE = re.compile(r"Learning\s+iteration\s+(\d+)/(\d+)")
_REWARD_RE = re.compile(rf"Mean\s+reward:\s+{_FLOAT_RE}")
_EPLEN_RE = re.compile(rf"Mean\s+episode\s+length:\s+{_FLOAT_RE}")
_EPISODE_METRIC_RE = re.compile(rf"Mean\s+episode\s+([A-Za-z0-9_]+):\s+{_FLOAT_RE}")
_LOG_DIR_RE = re.compile(r"TRIAL_LOG_DIR:\s+(.*)")
_FINAL_MODEL_RE = re.compile(r"TRIAL_FINAL_MODEL:\s+(.*)")
_COMPLETE_RE = re.compile(r"TRIAL_COMPLETE:\s+(.*)")
_WARNING_RE = re.compile(r"TRIAL_WARNING:\s+(.*)")
_MODEL_RE = re.compile(r"TRIAL_MODEL:\s+(\d+):(.*)")
_INFO_RE = re.compile(r"TRIAL_INFO:\s+(.*)")
_CUDA_OOM_RE = re.compile(r"(CUDA out of memory|torch\.OutOfMemoryError|out of memory)", re.IGNORECASE)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp_path.replace(path)


def parse_trial_output_fallback(stdout: str) -> dict[str, Any]:
    result = {
        "reward_curve": [],
        "ep_len_curve": [],
        "episode_metric_curves": {},
        "log_dir": None,
        "final_model": None,
        "model_checkpoints": [],
        "warnings": [],
        "info": [],
        "success": False,
    }
    current_iteration = 0
    fallback_iteration = 0
    for line in stdout.splitlines():
        clean = _ANSI_RE.sub("", line)
        iter_match = _ITER_RE.search(clean)
        if iter_match:
            current_iteration = int(iter_match.group(1))
            continue
        log_match = _LOG_DIR_RE.search(clean)
        if log_match:
            result["log_dir"] = log_match.group(1).strip()
            continue
        model_match = _MODEL_RE.search(clean)
        if model_match:
            result["model_checkpoints"].append((int(model_match.group(1)), model_match.group(2).strip()))
            continue
        final_match = _FINAL_MODEL_RE.search(clean)
        if final_match:
            result["final_model"] = final_match.group(1).strip()
            continue
        if _COMPLETE_RE.search(clean):
            result["success"] = True
            continue
        warning_match = _WARNING_RE.search(clean)
        if warning_match:
            result["warnings"].append(warning_match.group(1).strip())
            continue
        info_match = _INFO_RE.search(clean)
        if info_match:
            result["info"].append(info_match.group(1).strip())
            continue
        reward_match = _REWARD_RE.search(clean)
        if reward_match:
            result["reward_curve"].append((current_iteration, float(reward_match.group(1))))
            continue
        eplen_match = _EPLEN_RE.search(clean)
        if eplen_match and result["reward_curve"]:
            result["ep_len_curve"].append((current_iteration, float(eplen_match.group(1))))
            fallback_iteration += 1
            current_iteration = max(current_iteration, fallback_iteration)
            continue
        metric_match = _EPISODE_METRIC_RE.search(clean)
        if metric_match:
            result["episode_metric_curves"].setdefault(metric_match.group(1).strip(), []).append(
                (current_iteration, float(metric_match.group(2)))
            )
    return result


def compute_trial_score_fallback(metrics: dict[str, Any], policy_dt: float = 0.01) -> dict[str, Any]:
    del policy_dt
    curve = metrics.get("reward_curve") or []
    if not curve:
        return {"score": -999.0, "base_reward_score": -999.0}
    values = [float(value) for _iteration, value in curve]
    tail_len = max(1, len(values) // 5)
    tail_mean = sum(values[-tail_len:]) / tail_len
    overall_mean = sum(values) / len(values)
    trend = tail_mean - (sum(values[:tail_len]) / tail_len)
    score = overall_mean * 0.3 + tail_mean * 0.5 + max(0.0, trend) * 0.2
    return {"score": float(score), "base_reward_score": float(score)}


def estimate_policy_dt_fallback(task_name: str, default_dt: float = 0.01) -> float:
    del task_name
    return default_dt


def load_plan_tune_helpers() -> tuple[Any, Any, Any]:
    # Keep the runner parent process CPU-only. Importing auto_tune_rewards or
    # task_registry loads Isaac Gym/PyTorch CUDA and leaves a long-lived GPU
    # context in the parent, which distorts memory checks and can starve trials.
    return parse_trial_output_fallback, compute_trial_score_fallback, estimate_policy_dt_fallback


def safe_name(value: str, fallback: str = "experiment") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return cleaned[:80] or fallback


def subprocess_env(extra_vars: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = {**os.environ}
    python_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")
    if os.path.isdir(python_lib):
        existing = env.get("LD_LIBRARY_PATH", "")
        if python_lib not in existing:
            env["LD_LIBRARY_PATH"] = f"{python_lib}:{existing}" if existing else python_lib
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if extra_vars:
        env.update(extra_vars)
    return env


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_session(state_dir: Path, **updates: Any) -> None:
    session = read_json(state_dir / "session.json", {})
    session.update(updates)
    session["updated_at"] = time.time()
    write_json_atomic(state_dir / "session.json", session)


def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}min"
    return f"{seconds:.0f}s"


def format_eta(iteration: int, total: int, elapsed: float) -> str:
    if iteration <= 0 or total <= 0:
        return "--"
    remaining = max(0.0, elapsed / iteration * (total - iteration))
    return format_duration(remaining)


def launch_tensorboard(log_root: Path, port: int) -> tuple[Optional[subprocess.Popen], str]:
    log_root.mkdir(parents=True, exist_ok=True)
    url = f"http://localhost:{port}"
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(log_root),
        "--bind_all",
        "--port", str(port),
    ]
    try:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"TensorBoard: {url}  logdir={log_root}", flush=True)
        return proc, url
    except Exception as exc:
        print(f"[WARN] Failed to launch TensorBoard: {exc}", flush=True)
        return None, url


def first_visible_cuda_device() -> Optional[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("NVIDIA_VISIBLE_DEVICES")
    if not visible or visible.strip().lower() in {"all", "none", "void"}:
        return None
    first = visible.split(",")[0].strip()
    return first or None


def nvidia_smi_free_memory_mb() -> Optional[dict[str, Any]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    rows = []
    for line in proc.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            rows.append(
                {
                    "index": parts[0],
                    "uuid": parts[1],
                    "pci_bus_id": parts[2],
                    "free_mb": int(parts[3]),
                    "total_mb": int(parts[4]),
                }
            )
        except ValueError:
            continue
    if not rows:
        return None

    visible = first_visible_cuda_device()
    selected = rows[0]
    source = "nvidia-smi first GPU"
    if visible:
        for row in rows:
            if visible == row["index"] or visible == row["uuid"] or visible == row["pci_bus_id"]:
                selected = row
                source = f"CUDA_VISIBLE_DEVICES={visible}"
                break
            if visible.startswith("GPU-") and row["uuid"].startswith(visible):
                selected = row
                source = f"CUDA_VISIBLE_DEVICES={visible}"
                break
        else:
            source = f"CUDA_VISIBLE_DEVICES={visible} not found; fallback first GPU"
    selected["source"] = source
    return selected


def gpu_free_memory_mb() -> Optional[dict[str, Any]]:
    return nvidia_smi_free_memory_mb()


def wait_between_experiments(
    state_dir: Path,
    seconds: float,
    min_free_mb: int,
    reason: str,
) -> None:
    seconds = max(0.0, float(seconds))
    min_free_mb = max(0, int(min_free_mb))
    if seconds <= 0 and min_free_mb <= 0:
        return
    print("\n" + "-" * 72, flush=True)
    print(f"Inter-experiment cooldown: {format_duration(seconds)} ({reason})", flush=True)
    if min_free_mb <= 0:
        print("  | GPU memory check disabled; waiting by time only.", flush=True)
    start = time.time()
    deadline = start + seconds
    if min_free_mb <= 0:
        while True:
            remaining = max(0.0, deadline - time.time())
            write_session(
                state_dir,
                status="cooldown",
                cooldown_reason=reason,
                cooldown_remaining=remaining,
                cooldown_min_free_mb=0,
                cooldown_gpu_memory="disabled",
            )
            print(f"  | waiting {format_duration(remaining)}", flush=True)
            if time.time() >= deadline:
                break
            time.sleep(2.0)
        write_session(
            state_dir,
            status="running",
            cooldown_reason=None,
            cooldown_remaining=0,
            cooldown_gpu_memory=None,
        )
        print("-" * 72, flush=True)
        return

    while True:
        free_info = gpu_free_memory_mb()
        free_text = "GPU memory: unavailable"
        enough_memory = min_free_mb <= 0
        if free_info is not None:
            free_mb = int(free_info["free_mb"])
            total_mb = int(free_info["total_mb"])
            free_text = (
                f"GPU {free_info['index']} memory free: {free_mb}/{total_mb} MiB"
                f" ({free_info['source']})"
            )
            enough_memory = min_free_mb <= 0 or free_mb >= min_free_mb
        remaining = max(0.0, deadline - time.time())
        write_session(
            state_dir,
            status="cooldown",
            cooldown_reason=reason,
            cooldown_remaining=remaining,
            cooldown_min_free_mb=min_free_mb,
            cooldown_gpu_memory=free_text,
        )
        print(f"  | {free_text} | waiting {format_duration(remaining)}", flush=True)
        if time.time() >= deadline and enough_memory:
            break
        if time.time() >= deadline and free_info is None:
            break
        time.sleep(2.0)
    write_session(
        state_dir,
        status="running",
        cooldown_reason=None,
        cooldown_remaining=0,
        cooldown_gpu_memory=None,
    )
    print("-" * 72, flush=True)


def write_live_curve(state_dir: Path, experiment_id: int, iteration: int, reward: float) -> None:
    write_json_atomic(
        state_dir / "live_progress.json",
        {
            "experiment_id": experiment_id,
            "iteration": iteration,
            "reward": reward,
            "timestamp": time.time(),
        },
    )
    curve_path = state_dir / "live_curve.json"
    live_curve = read_json(curve_path, {"experiment_id": experiment_id, "reward_curve": []})
    if live_curve.get("experiment_id") != experiment_id:
        live_curve = {"experiment_id": experiment_id, "reward_curve": []}
    curve = live_curve.setdefault("reward_curve", [])
    point = [int(iteration), float(reward)]
    if curve and curve[-1][0] == point[0]:
        curve[-1] = point
    else:
        curve.append(point)
    write_json_atomic(curve_path, live_curve)


def reset_live_curve(state_dir: Path, experiment_id: int) -> None:
    write_json_atomic(
        state_dir / "live_curve.json",
        {"experiment_id": experiment_id, "reward_curve": []},
    )
    write_json_atomic(state_dir / "live_progress.json", {})


def video_job_path(state_dir: Path, output_path: Path) -> Path:
    return state_dir / "video_jobs" / f"{output_path.stem}.json"


def write_video_job(
    state_dir: Path,
    experiment_id: int,
    experiment_name: str,
    output_path: Path,
    status: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_iter: Optional[int] = None,
    error: Optional[str] = None,
) -> Path:
    payload = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "output_path": str(output_path),
        "filename": output_path.name,
        "status": status,
        "checkpoint_path": checkpoint_path,
        "checkpoint_iter": checkpoint_iter,
        "error": error,
        "updated_at": time.time(),
    }
    if status in {"pending", "running"}:
        payload["started_at"] = time.time()
    if status in {"ready", "failed"}:
        payload["finished_at"] = time.time()
    path = video_job_path(state_dir, output_path)
    existing = read_json(path, {})
    if isinstance(existing, dict):
        existing.update(payload)
        payload = existing
    write_json_atomic(path, payload)
    return path


def select_best_checkpoint(metrics: dict[str, Any]) -> Optional[tuple[int, str, int, float]]:
    checkpoints = metrics.get("model_checkpoints", []) or []
    curve = metrics.get("reward_curve", []) or []
    usable = []
    for iter_num, path in checkpoints:
        try:
            iter_value = int(iter_num)
        except (TypeError, ValueError):
            continue
        if path and Path(path).is_file():
            usable.append((iter_value, str(path)))
    if not usable:
        final_model = metrics.get("final_model")
        if final_model and Path(final_model).is_file():
            return -1, str(final_model), -1, float("nan")
        return None
    if curve:
        best_reward_iter, best_reward = max(curve, key=lambda item: float(item[1]))
        ckpt_iter, ckpt_path = min(
            usable,
            key=lambda item: (abs(item[0] - int(best_reward_iter)), -item[0]),
        )
        return ckpt_iter, ckpt_path, int(best_reward_iter), float(best_reward)
    ckpt_iter, ckpt_path = usable[-1]
    return ckpt_iter, ckpt_path, ckpt_iter, float("nan")


def preserve_best_model(metrics: dict[str, Any], trial_best_dir: Path) -> Optional[Path]:
    selected = select_best_checkpoint(metrics)
    if selected is None:
        metrics["trial_best_model"] = None
        return None
    ckpt_iter, ckpt_path, reward_iter, reward_value = selected
    src = Path(ckpt_path)
    if not src.is_file():
        metrics["trial_best_model"] = None
        return None
    experiment_id = int(metrics.get("experiment_id", metrics.get("trial_id", -1)))
    name = safe_name(metrics.get("experiment_name", f"experiment_{experiment_id:03d}"))
    dest = trial_best_dir / f"experiment_{experiment_id:03d}_{name}_best_iter_{ckpt_iter}{src.suffix or '.pt'}"
    trial_best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    meta = {
        "experiment_id": experiment_id,
        "experiment_name": metrics.get("experiment_name"),
        "score": metrics.get("score"),
        "checkpoint_iter": ckpt_iter,
        "source_model": str(src),
        "saved_model": str(dest),
        "best_reward_iter": reward_iter,
        "best_reward": reward_value,
        "reward_scales": metrics.get("reward_scales", {}),
        "changed_reward_scales": metrics.get("changed_reward_scales", {}),
    }
    write_json_atomic(trial_best_dir / f"experiment_{experiment_id:03d}_{name}_best_meta.json", meta)
    metrics["trial_best_model"] = str(dest)
    metrics["trial_best_model_iter"] = ckpt_iter
    metrics["trial_best_reward_iter"] = reward_iter
    metrics["trial_best_reward"] = reward_value
    return dest


def start_video_recording(
    metrics: dict[str, Any],
    sequence: dict[str, Any],
    output_dir: Path,
    state_dir: Path,
    wait: bool = False,
) -> None:
    model_path = metrics.get("trial_best_model")
    if not model_path or not Path(model_path).is_file():
        metrics["video_paths"] = []
        return
    experiment_id = int(metrics.get("experiment_id", -1))
    name = safe_name(metrics.get("experiment_name", f"experiment_{experiment_id:03d}"))
    ckpt_iter = metrics.get("trial_best_model_iter", "unknown")
    video_out = output_dir / "videos" / f"experiment_{experiment_id:03d}_{name}_best_iter_{ckpt_iter}.mp4"
    video_out.parent.mkdir(parents=True, exist_ok=True)
    status_path = write_video_job(
        state_dir,
        experiment_id=experiment_id,
        experiment_name=str(metrics.get("experiment_name", "")),
        output_path=video_out,
        status="pending",
        checkpoint_path=str(model_path),
        checkpoint_iter=ckpt_iter if isinstance(ckpt_iter, int) else None,
    )
    metrics["video_paths"] = [str(video_out)]
    cmd = [
        sys.executable,
        str(RECORD_VIDEO_SCRIPT),
        "--task",
        str(sequence["task"]),
        "--model",
        str(model_path),
        "--output",
        str(video_out),
        "--status-file",
        str(status_path),
        "--trial-id",
        str(experiment_id),
        "--headless",
    ]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if wait:
        proc.wait()


def video_output_path(metrics: dict[str, Any], output_dir: Path) -> Path:
    experiment_id = int(metrics.get("experiment_id", -1))
    name = safe_name(metrics.get("experiment_name", f"experiment_{experiment_id:03d}"))
    ckpt_iter = metrics.get("trial_best_model_iter", "unknown")
    return output_dir / "videos" / f"experiment_{experiment_id:03d}_{name}_best_iter_{ckpt_iter}.mp4"


def mark_video_skipped(metrics: dict[str, Any], output_dir: Path, state_dir: Path, reason: str) -> None:
    experiment_id = int(metrics.get("experiment_id", -1))
    ckpt_iter = metrics.get("trial_best_model_iter", "unknown")
    video_out = video_output_path(metrics, output_dir)
    metrics["video_paths"] = []
    write_video_job(
        state_dir,
        experiment_id=experiment_id,
        experiment_name=str(metrics.get("experiment_name", "")),
        output_path=video_out,
        status="skipped",
        checkpoint_path=str(metrics.get("trial_best_model") or ""),
        checkpoint_iter=ckpt_iter if isinstance(ckpt_iter, int) else None,
        error=reason,
    )


def delete_video_file(metrics: dict[str, Any], output_dir: Path) -> None:
    paths = [Path(path) for path in (metrics.get("video_paths") or [])]
    paths.append(video_output_path(metrics, output_dir))
    for path in paths:
        try:
            if path.is_file():
                path.unlink()
        except OSError as exc:
            print(f"  [WARN] Failed to delete dropped Top video {path}: {exc}", flush=True)


def has_recorded_video(metrics: dict[str, Any], output_dir: Path) -> bool:
    for path in metrics.get("video_paths") or []:
        if Path(path).is_file():
            return True
    return video_output_path(metrics, output_dir).is_file()


def update_rolling_top_videos(
    results: list[dict[str, Any]],
    sequence: dict[str, Any],
    output_dir: Path,
    state_dir: Path,
) -> None:
    limit = int(sequence.get("top_video_limit", 20) or 20)
    successful = ranked_successful(results)
    top_ids = {int(item.get("experiment_id", -1)) for item in successful[:limit]}
    for item in results:
        if not item.get("success"):
            continue
        if int(item.get("experiment_id", -1)) in top_ids:
            continue
        if has_recorded_video(item, output_dir):
            delete_video_file(item, output_dir)
            print(
                f"  [VIDEO DROP] #{item.get('experiment_id')} {item.get('experiment_name')} dropped outside top {limit}; video removed.",
                flush=True,
            )
        mark_video_skipped(item, output_dir, state_dir, f"dropped outside rolling top {limit} score ranking")

    for rank, item in enumerate(successful[:limit], start=1):
        if has_recorded_video(item, output_dir):
            if not item.get("video_paths"):
                item["video_paths"] = [str(video_output_path(item, output_dir))]
            continue
        experiment_id = int(item.get("experiment_id", -1))
        print(
            f"  [VIDEO TOP{rank:02d}] Recording rolling top video for "
            f"#{experiment_id} {item.get('experiment_name')} score={item.get('score')}",
            flush=True,
        )
        write_session(
            state_dir,
            status="recording_videos",
            video_top_limit=limit,
            video_total_selected=len(top_ids),
            current_video_experiment=experiment_id,
            current_video_experiment_name=item.get("experiment_name"),
            current_video_rank=rank,
        )
        start_video_recording(item, sequence, output_dir, state_dir, wait=True)
        time.sleep(float(sequence.get("video_record_delay", 3.0) or 3.0))

    write_session(
        state_dir,
        current_video_experiment=None,
        current_video_experiment_name=None,
        current_video_rank=None,
    )


def run_one_experiment(
    experiment: dict[str, Any],
    sequence: dict[str, Any],
    output_dir: Path,
    state_dir: Path,
    policy_dt: float,
    parse_trial_output_func: Any,
    compute_trial_score_func: Any,
) -> dict[str, Any]:
    experiment_id = int(experiment.get("id", 0))
    experiment_name = str(experiment.get("name") or f"experiment_{experiment_id:03d}")
    iterations = int(experiment.get("iterations") or sequence.get("iterations") or 500)
    changed_scales = {
        str(k): float(v)
        for k, v in (experiment.get("reward_scales") or {}).items()
    }
    full_scales = {
        str(k): float(v)
        for k, v in (experiment.get("full_reward_scales") or changed_scales).items()
    }
    config_dir = output_dir / "experiment_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"experiment_{experiment_id:03d}.json"
    write_json_atomic(config_path, full_scales)
    stdout_dir = output_dir / "stdout_logs"
    stdout_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = stdout_dir / f"experiment_{experiment_id:03d}_{safe_name(experiment_name)}.log"

    run_name = f"plan_{experiment_id:03d}_{safe_name(experiment_name)}"
    cmd = [
        sys.executable,
        str(TRIAL_RUNNER),
        "--task",
        str(sequence["task"]),
        "--max_iterations",
        str(iterations),
        "--experiment_name",
        str(sequence.get("experiment_name") or f"{sequence['task']}_PlanTune"),
        "--run_name",
        run_name,
        "--num_envs",
        str(sequence.get("num_envs", 4096)),
    ]
    if bool(sequence.get("headless", True)):
        cmd.append("--headless")

    write_session(
        state_dir,
        status="running",
        current_experiment=experiment_id,
        current_experiment_name=experiment_name,
        current_iterations=iterations,
        current_config_path=str(config_path),
        current_experiment_started_at=time.time(),
    )
    reset_live_curve(state_dir, experiment_id)

    total_experiments = int(sequence.get("total_experiments") or len(sequence.get("experiments", [])) or 1)
    print("\n" + "=" * 72, flush=True)
    print(f"Experiment {experiment_id + 1:03d}/{total_experiments:03d}: {experiment_name}", flush=True)
    print(f"Task: {sequence['task']}  |  Iterations: {iterations}  |  Envs: {sequence.get('num_envs', 4096)}", flush=True)
    print(f"Run name: {run_name}", flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"Stdout log: {stdout_path}", flush=True)
    print("Changed reward scales:", flush=True)
    for key, value in changed_scales.items():
        print(f"  {key}: {value}", flush=True)
    if not changed_scales:
        print("  baseline (no changed scales)", flush=True)
    print("Command: " + " ".join(cmd), flush=True)
    print("=" * 72, flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=subprocess_env({"REWARD_CONFIG_PATH": str(config_path)}),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    start = time.time()
    deadline = start + int(sequence.get("timeout", 7200))
    current_iter = 0
    timed_out = False

    last_reward: Optional[float] = None
    last_ep_len: Optional[float] = None
    assert proc.stdout is not None
    with stdout_path.open("w", encoding="utf-8") as stdout_file:
        while True:
            if time.time() > deadline:
                proc.kill()
                timed_out = True
                break
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
                continue
            stdout_file.write(line)
            stdout_file.flush()
            stdout_lines.append(line)
            clean = _ANSI_RE.sub("", line.rstrip())
            iter_match = re.search(r"Learning\s+iteration\s+(\d+)/(\d+)", clean)
            if iter_match:
                current_iter = int(iter_match.group(1))
                total_iter = int(iter_match.group(2))
                elapsed_now = time.time() - start
                print(
                    f"\n  --- {experiment_name} | Iter {current_iter}/{total_iter}"
                    f" | Elapsed {format_duration(elapsed_now)} | ETA {format_eta(current_iter, total_iter, elapsed_now)} ---",
                    flush=True,
                )
                continue
            reward_match = _REWARD_RE.search(clean)
            if reward_match:
                last_reward = float(reward_match.group(1))
                write_live_curve(state_dir, experiment_id, current_iter, last_reward)
                print(f"  | Mean reward: {last_reward:.3f}", flush=True)
                continue
            eplen_match = _EPLEN_RE.search(clean)
            if eplen_match:
                last_ep_len = float(eplen_match.group(1))
                print(f"  | Mean episode length: {last_ep_len:.2f}", flush=True)
                continue
            if any(key in clean for key in ("TRIAL_LOG_DIR", "TRIAL_MODEL", "TRIAL_FINAL_MODEL", "TRIAL_COMPLETE", "TRIAL_WARNING", "TRIAL_ERROR")):
                print(f"  | {clean}", flush=True)

    proc.wait()
    elapsed = time.time() - start
    combined = "".join(stdout_lines)
    metrics = parse_trial_output_func(combined)
    cuda_oom = bool(_CUDA_OOM_RE.search(combined))
    metrics.update(
        {
            "trial_id": experiment_id,
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "iterations": iterations,
            "elapsed": elapsed,
            "returncode": proc.returncode,
            "reward_scales": full_scales,
            "changed_reward_scales": changed_scales,
            "config_path": str(config_path),
            "stdout_log": str(stdout_path),
            "cuda_oom": cuda_oom,
        }
    )
    if timed_out:
        metrics["success"] = False
        metrics.setdefault("warnings", []).append("timeout")

    if metrics.get("success"):
        score_breakdown = compute_trial_score_func(metrics, policy_dt=policy_dt)
        metrics["score_breakdown"] = score_breakdown
        metrics["score"] = float(score_breakdown["score"])
        preserve_best_model(metrics, output_dir / "trial_best_models")
        metrics["video_paths"] = []
        print(
            f"  [OK] {experiment_name} finished in {format_duration(elapsed)}, "
            f"score={metrics['score']:.3f}, best_reward={metrics.get('trial_best_reward', last_reward)}",
            flush=True,
        )
    else:
        metrics["score"] = -999.0
        metrics["score_breakdown"] = {}
        metrics["video_paths"] = []
        if cuda_oom:
            metrics.setdefault("warnings", []).append("cuda_out_of_memory")
            print("  [WARN] CUDA OOM detected; next experiment will use extended cooldown.", flush=True)
        print(f"  [FAIL] {experiment_name} rc={proc.returncode} elapsed={format_duration(elapsed)}", flush=True)

    return metrics


def dashboard_entry(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": result.get("experiment_id"),
        "experiment_name": result.get("experiment_name"),
        "iterations": result.get("iterations"),
        "success": result.get("success", False),
        "score": result.get("score"),
        "score_breakdown": result.get("score_breakdown", {}),
        "reward_scales": result.get("reward_scales", {}),
        "changed_reward_scales": result.get("changed_reward_scales", {}),
        "trial_best_model": result.get("trial_best_model"),
        "trial_best_model_iter": result.get("trial_best_model_iter"),
        "trial_best_reward_iter": result.get("trial_best_reward_iter"),
        "trial_best_reward": result.get("trial_best_reward"),
        "video_paths": result.get("video_paths", []),
        "stdout_log": result.get("stdout_log"),
        "reward_curve": [list(point) for point in result.get("reward_curve", [])],
        "ep_len_curve": [list(point) for point in result.get("ep_len_curve", [])],
        "elapsed": result.get("elapsed", 0),
        "warnings": result.get("warnings", []),
        "cuda_oom": result.get("cuda_oom", False),
    }


def ranked_successful(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [item for item in results if item.get("success")]
    successful.sort(
        key=lambda item: (
            float(item.get("score", -float("inf"))),
            -int(item.get("experiment_id", 10**9)),
        ),
        reverse=True,
    )
    return successful


def generate_charts(results: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping Plan Tune charts: {exc}", flush=True)
        return

    successful = ranked_successful(results)
    if results:
        fig, ax = plt.subplots(figsize=(10, 5))
        for result in results:
            curve = result.get("reward_curve") or []
            if not curve:
                continue
            xs, ys = zip(*curve)
            label = f"{result.get('experiment_id')} {result.get('experiment_name')}"
            ax.plot(xs, ys, linewidth=1.2, alpha=0.8, label=label)
        ax.set_title("Plan Tune Reward Curves")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean Reward")
        ax.grid(True, alpha=0.3)
        if len(results) <= 12:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / "reward_curves.png", dpi=150)
        plt.close(fig)

    if successful:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [f"{item.get('experiment_id')} {safe_name(item.get('experiment_name', ''))}" for item in successful]
        scores = [float(item.get("score", 0.0)) for item in successful]
        ax.bar(labels, scores, color="#4dd0e1")
        ax.set_title("Plan Tune Score Ranking")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "score_ranking.png", dpi=150)
        plt.close(fig)


def generate_report(results: list[dict[str, Any]], sequence: dict[str, Any], output_dir: Path) -> None:
    successful = ranked_successful(results)
    best = successful[0] if successful else {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Plan Tune Training Report",
        "",
        f"- Task: `{sequence.get('task')}`",
        f"- Experiment group: `{sequence.get('experiment_name')}`",
        f"- Date: {now}",
        f"- Experiments: {len(results)} total, {len(successful)} successful",
        f"- Best experiment: `{best.get('experiment_name', 'N/A')}` (id={best.get('experiment_id', 'N/A')}, score={best.get('score', 'N/A')})",
        "",
        "## Ranked Results",
        "",
        "| Rank | ID | Name | Iterations | Score | Best Reward | Best Model | Video | Stdout |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for rank, item in enumerate(successful, 1):
        videos = ", ".join(item.get("video_paths") or [])
        lines.append(
            f"| {rank} | {item.get('experiment_id')} | {item.get('experiment_name')} | "
            f"{item.get('iterations')} | {float(item.get('score', 0.0)):.4f} | "
            f"{item.get('trial_best_reward', 'N/A')} | `{item.get('trial_best_model', '')}` | "
            f"`{videos}` | `{item.get('stdout_log', '')}` |"
        )

    failed = [item for item in results if not item.get("success")]
    if failed:
        lines.extend(["", "## Failed Experiments", ""])
        for item in failed:
            warnings = "; ".join(str(w) for w in item.get("warnings", []))
            lines.append(f"- `{item.get('experiment_name')}` id={item.get('experiment_id')}: {warnings or 'failed'}")

    lines.extend(["", "## Reward Scale Changes", ""])
    for item in results:
        lines.append(f"### {item.get('experiment_id')} - {item.get('experiment_name')}")
        changed = item.get("changed_reward_scales") or {}
        if not changed:
            lines.append("- No reward scale changes from task defaults.")
        else:
            for key, value in sorted(changed.items()):
                lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    markdown = "\n".join(lines).rstrip() + "\n"
    (output_dir / "training_report.md").write_text(markdown, encoding="utf-8")

    text_lines = [
        "Plan Tune Training Report",
        f"Task: {sequence.get('task')}",
        f"Experiment group: {sequence.get('experiment_name')}",
        f"Date: {now}",
        f"Experiments: {len(results)} total, {len(successful)} successful",
        "",
        "Ranked Results:",
    ]
    for rank, item in enumerate(successful, 1):
        text_lines.append(
            f"{rank}. id={item.get('experiment_id')} name={item.get('experiment_name')} "
            f"iterations={item.get('iterations')} score={item.get('score')} "
            f"best_reward={item.get('trial_best_reward')} model={item.get('trial_best_model')} "
            f"videos={item.get('video_paths')} stdout_log={item.get('stdout_log')}"
        )
    (output_dir / "training_report.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    generate_charts(results, output_dir)


def run_sequence(sequence_path: Path, output_dir: Optional[Path] = None) -> Path:
    sequence_path = sequence_path.resolve()
    sequence = read_json(sequence_path, {})
    if not isinstance(sequence, dict) or not sequence.get("experiments"):
        raise ValueError(f"Invalid experiment sequence: {sequence_path}")
    output_dir = (output_dir or sequence_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "dashboard_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "video_jobs").mkdir(parents=True, exist_ok=True)

    canonical_sequence = output_dir / "experiment_sequence.json"
    if canonical_sequence.resolve() != sequence_path:
        shutil.copy2(sequence_path, canonical_sequence)

    tensorboard_port = int(sequence.get("tensorboard_port") or getattr(run_sequence, "tensorboard_port", 1230))
    experiment_name = str(sequence.get("experiment_name") or f"{sequence.get('task')}_PlanTune")
    log_root = PROJECT_ROOT / "logs" / experiment_name
    tensorboard_proc, tensorboard_url = launch_tensorboard(log_root, tensorboard_port)

    sequence["total_experiments"] = len(sequence.get("experiments", []))
    sequence["tensorboard_port"] = tensorboard_port
    sequence["tensorboard_url"] = tensorboard_url
    sequence.setdefault("inter_experiment_delay", getattr(run_sequence, "inter_experiment_delay", 20.0))
    sequence.setdefault("oom_cooldown", getattr(run_sequence, "oom_cooldown", 75.0))
    sequence.setdefault("min_free_gpu_mb", getattr(run_sequence, "min_free_gpu_mb", 900))
    sequence.setdefault("top_video_limit", getattr(run_sequence, "top_video_limit", 20))
    sequence.setdefault("video_record_delay", getattr(run_sequence, "video_record_delay", 3.0))

    parse_trial_output_func, compute_trial_score_func, estimate_policy_dt_func = load_plan_tune_helpers()
    policy_dt = estimate_policy_dt_func(str(sequence.get("task")))
    results: list[dict[str, Any]] = []
    write_json_atomic(state_dir / "experiments.json", [])
    write_session(
        state_dir,
        status="running",
        task=sequence.get("task"),
        experiment_name=sequence.get("experiment_name"),
        output_dir=str(output_dir),
        sequence_path=str(canonical_sequence),
        total_experiments=len(sequence.get("experiments", [])),
        completed_experiments=0,
        current_experiment=None,
        current_experiment_name=None,
        score_policy_dt=policy_dt,
        log_root=str(log_root),
        tensorboard_port=tensorboard_port,
        tensorboard_url=tensorboard_url,
        started_at=time.time(),
    )

    try:
        experiments = sequence.get("experiments", [])
        for index, experiment in enumerate(experiments):
            result = run_one_experiment(
                experiment,
                sequence,
                output_dir,
                state_dir,
                policy_dt,
                parse_trial_output_func,
                compute_trial_score_func,
            )
            results.append(result)
            write_json_atomic(state_dir / "experiments.json", [dashboard_entry(item) for item in results])
            write_json_atomic(output_dir / "plan_tune_state.json", {"sequence": sequence, "results": results})
            best = ranked_successful(results)[0] if ranked_successful(results) else {}
            write_session(
                state_dir,
                completed_experiments=len(results),
                current_experiment=None,
                current_experiment_name=None,
                current_experiment_started_at=None,
                best_experiment=best.get("experiment_id"),
                best_experiment_name=best.get("experiment_name"),
                best_score=best.get("score"),
            )
            update_rolling_top_videos(results, sequence, output_dir, state_dir)
            write_json_atomic(state_dir / "experiments.json", [dashboard_entry(item) for item in results])
            write_json_atomic(output_dir / "plan_tune_state.json", {"sequence": sequence, "results": results})
            if index < len(experiments) - 1:
                delay = float(sequence.get("inter_experiment_delay", 20.0))
                reason = "standard resource release"
                if result.get("cuda_oom"):
                    delay = max(delay, float(sequence.get("oom_cooldown", 75.0)))
                    reason = "CUDA OOM recovery"
                wait_between_experiments(
                    state_dir,
                    seconds=delay,
                    min_free_mb=int(sequence.get("min_free_gpu_mb", 900)),
                    reason=reason,
                )
        generate_report(results, sequence, output_dir)
        write_session(
            state_dir,
            status="completed",
            completed_experiments=len(results),
            report_path=str(output_dir / "training_report.md"),
        )
    except KeyboardInterrupt:
        write_session(state_dir, status="cancelled")
        raise
    except Exception as exc:
        write_session(state_dir, status="failed", error=str(exc))
        raise
    # TensorBoard is intentionally left running so users can review metrics
    # after the plan sequence completes.
    del tensorboard_proc
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Plan Tune experiment sequence")
    parser.add_argument("--sequence", required=True, type=str, help="Path to experiment_sequence.json")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory; defaults to sequence parent")
    parser.add_argument("--tensorboard-port", type=int, default=1230)
    parser.add_argument("--inter-experiment-delay", type=float, default=20.0, help="Seconds to wait after each experiment")
    parser.add_argument("--oom-cooldown", type=float, default=75.0, help="Minimum cooldown seconds after CUDA OOM")
    parser.add_argument("--min-free-gpu-mb", type=int, default=900, help="Wait until GPU free memory reaches this value when nvidia-smi is available")
    parser.add_argument("--no-gpu-memory-check", action="store_true", help="Disable inter-experiment GPU memory checks and cooldown waiting")
    parser.add_argument("--top-video-limit", type=int, default=20, help="Record videos only for the top N successful experiments by score")
    parser.add_argument("--video-record-delay", type=float, default=3.0, help="Seconds to wait between top-score video recordings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sequence.tensorboard_port = args.tensorboard_port
    run_sequence.inter_experiment_delay = 0.0 if args.no_gpu_memory_check else args.inter_experiment_delay
    run_sequence.oom_cooldown = 0.0 if args.no_gpu_memory_check else args.oom_cooldown
    run_sequence.min_free_gpu_mb = 0 if args.no_gpu_memory_check else args.min_free_gpu_mb
    run_sequence.top_video_limit = args.top_video_limit
    run_sequence.video_record_delay = args.video_record_delay
    output_dir = run_sequence(
        Path(args.sequence),
        Path(args.output_dir).resolve() if args.output_dir else None,
    )
    print(f"Plan Tune complete: {output_dir}")


if __name__ == "__main__":
    main()
