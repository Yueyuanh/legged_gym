#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Auto-tune reward scales for legged_gym-gym PPO training.

Iteratively samples reward scale combinations, runs short training sessions,
keeps only the best-performing model, and generates a tuning report with
reward evolution charts.

Usage:
    python legged_gym/tune/auto_tune/auto_tune_rewards.py --task go2 --trials 20 --headless

    # Resume a previous tuning run:
    python legged_gym/tune/auto_tune/auto_tune_rewards.py --task go2 --resume /path/to/logs/auto_tune/run
"""

import os
import sys
import json
import subprocess
import shutil
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

def _maybe_launch_tensorboard(args, project_root, enable):
    """Launch TensorBoard if enable is True."""
    if not enable:
        return
    log_root = os.path.join(project_root, "logs", getattr(args, "experiment_name", "tune"))
    log_root = os.path.abspath(log_root)
    os.makedirs(log_root, exist_ok=True)
    try:
        import subprocess as _sp
        cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", log_root,
            "--bind_all",
            "--port", str(getattr(args, "tensorboard_port", 6006)),
        ]
        _sp.Popen(cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        print(f"  -> TensorBoard ready: http://localhost:{args.tensorboard_port}")
    except Exception as e:
        print(f"  [WARN] Failed to launch TensorBoard: {e}")


# Lazy import for launch_tensorboard (used when --tensorboard is set)

import numpy as np

# Matplotlib setup (non-interactive backend for headless servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent  # .../legged_gym/tune/auto_tune/
TUNE_DIR = SCRIPT_DIR.parent  # .../legged_gym/tune/
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # legged_gym repo root
TRIAL_RUNNER = TUNE_DIR / "_run_trial.py"
RECORD_VIDEO_SCRIPT = TUNE_DIR / "_record_video.py"
DASHBOARD_SCRIPT = SCRIPT_DIR / "auto_tune_dashboard.py"
MONITOR_SCRIPT = SCRIPT_DIR / "auto_tune_monitor.py"
MAX_TOP_TRIAL_ARTIFACTS = 10
BEST_CONFIG_LOG_FILENAME = "best_config.log"


def _subprocess_env(extra_vars: dict = None) -> dict:
    """Build environment dict for subprocess, ensuring LD_LIBRARY_PATH
    includes the active Python's lib directory (required by Isaac Gym)."""
    env = {**os.environ}
    # Find libpython from the currently running Python executable.
    # This is the one the subprocess will use, so its lib dir must be on
    # LD_LIBRARY_PATH for Isaac Gym's native bindings to load.
    python_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")
    if os.path.isdir(python_lib):
        existing = env.get("LD_LIBRARY_PATH", "")
        if python_lib not in existing:
            env["LD_LIBRARY_PATH"] = f"{python_lib}:{existing}" if existing else python_lib
    env["PYTHONUNBUFFERED"] = "1"
    if extra_vars:
        env.update(extra_vars)
    return env


def _write_json_atomic(path: Path, data: Any) -> None:
    """Atomically write JSON so the dashboard never reads half-written state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _trial_config_filename(trial_id: int) -> str:
    return f"trial_{trial_id:03d}.json"


def _trial_config_path(output_dir: Path, trial_id: int) -> Path:
    return output_dir / "trial_configs" / _trial_config_filename(trial_id)


def _cleanup_legacy_trial_configs(output_dir: Path, trial_configs_dir: Path) -> int:
    """Remove legacy duplicate root-level trial config files."""
    removed = 0
    for legacy_path in output_dir.glob("trial_*_config.json"):
        if not legacy_path.is_file():
            continue
        match = re.match(r"trial_(\d+)_config\.json$", legacy_path.name)
        if not match:
            continue
        canonical_path = trial_configs_dir / f"trial_{int(match.group(1)):03d}.json"
        if not canonical_path.exists():
            shutil.move(str(legacy_path), str(canonical_path))
        else:
            legacy_path.unlink()
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# Default search space
# ---------------------------------------------------------------------------
# Each entry: (low, high, is_log_uniform, description)
# log_uniform=True means we sample in log space (good for wide ranges).
DEFAULT_SEARCH_SPACE = {
    # --- core tracking ---
    "tracking_lin_vel":     (0.5, 5.0, True,  "linear velocity tracking"),
    "tracking_ang_vel":     (0.1, 3.0, True,  "angular velocity tracking (yaw)"),
    # --- posture ---
    "orientation":          (0.1, 1.0, True,  "flat base orientation"),
    "base_height":          (0.1, 1.0, True,  "target height tracking"),
    # --- gait ---
    "feet_air_time":        (0.5, 3.0, True,  "swing-phase air time"),
}

# Parameters kept at their original values (less impactful or highly coupled)
FIXED_SCALES = {
    "torques": -0.00001,
    "dof_vel": 0.,
    "dof_acc": -2.5e-7,
    "collision": -1.,
    "action_rate": -0.01,
    "stand_still": 0.,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "feet_stumble": 0.,
    "termination": 0.,
}

# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------
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
_ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

SPEED_PRIORITY_TERMS = {
    "tracking_lin_vel": {
        "episode_key": "rew_tracking_lin_vel",
        "max_raw": 1.0,
        "weight": 110.0,
        "floor": 0.3,
    },
    "tracking_ang_vel": {
        "episode_key": "rew_tracking_ang_vel",
        "max_raw": 1.0,
        "weight": 45.0,
        "floor": 0.1,
    },
}


def parse_trial_output(stdout: str) -> Dict[str, Any]:
    """Extract training metrics and paths from trial subprocess stdout."""
    result = {
        "reward_curve": [],       # list of (iteration, mean_reward)
        "ep_len_curve": [],       # list of (iteration, mean_ep_length)
        "episode_metric_curves": {},  # dict[name] -> list of (iteration, value)
        "log_dir": None,
        "final_model": None,
        "model_checkpoints": [],  # list of (iter_num, path)
        "warnings": [],
        "info": [],
        "success": False,
    }
    fallback_iteration = 0
    current_iteration = 0
    for line in stdout.splitlines():
        clean_line = _ANSI_RE.sub("", line)
        iter_m = _ITER_RE.search(clean_line)
        if iter_m:
            current_iteration = int(iter_m.group(1))
            continue

        # Trial runner markers
        log_m = _LOG_DIR_RE.search(clean_line)
        if log_m:
            result["log_dir"] = log_m.group(1).strip()
            continue
        model_m = _MODEL_RE.search(clean_line)
        if model_m:
            result["model_checkpoints"].append(
                (int(model_m.group(1)), model_m.group(2).strip())
            )
            continue
        final_m = _FINAL_MODEL_RE.search(clean_line)
        if final_m:
            result["final_model"] = final_m.group(1).strip()
            continue
        if _COMPLETE_RE.search(clean_line):
            result["success"] = True
            continue
        warn_m = _WARNING_RE.search(clean_line)
        if warn_m:
            result["warnings"].append(warn_m.group(1).strip())
            continue
        info_m = _INFO_RE.search(clean_line)
        if info_m:
            result["info"].append(info_m.group(1).strip())
            continue

        # PPO runner log lines — reward and ep-length are on separate lines
        rew_m = _REWARD_RE.search(clean_line)
        if rew_m:
            reward = float(rew_m.group(1))
            result["reward_curve"].append((current_iteration, reward))
            # episode length usually follows reward on the next line
            continue
        eplen_m = _EPLEN_RE.search(clean_line)
        if eplen_m and result["reward_curve"]:
            ep_len = float(eplen_m.group(1))
            result["ep_len_curve"].append((current_iteration, ep_len))
            fallback_iteration += 1
            current_iteration = max(current_iteration, fallback_iteration)
            continue

        ep_metric_m = _EPISODE_METRIC_RE.search(clean_line)
        if ep_metric_m:
            metric_name = ep_metric_m.group(1).strip()
            metric_value = float(ep_metric_m.group(2))
            result["episode_metric_curves"].setdefault(metric_name, []).append(
                (current_iteration, metric_value)
            )

    return result


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------
def _curve_stats(curve: List[Tuple[int, float]]) -> Optional[Dict[str, float]]:
    if not curve:
        return None
    values = [float(value) for _, value in curve]
    if not values:
        return None
    tail_len = max(1, len(values) // 5)
    tail_mean = float(np.mean(values[-tail_len:]))
    initial_mean = float(np.mean(values[:tail_len]))
    overall_mean = float(np.mean(values))
    return {
        "tail_mean": tail_mean,
        "initial_mean": initial_mean,
        "overall_mean": overall_mean,
        "trend": tail_mean - initial_mean,
        "latest": values[-1],
    }


def _normalized_speed_metric_curve(
    metrics: Dict[str, Any],
    reward_name: str,
    policy_dt: float,
) -> List[Tuple[int, float]]:
    term_cfg = SPEED_PRIORITY_TERMS[reward_name]
    episode_key = term_cfg["episode_key"]
    curve = metrics.get("episode_metric_curves", {}).get(episode_key, [])
    if not curve:
        return []

    scale = abs(float(metrics.get("reward_scales", {}).get(reward_name, 0.0)))
    if scale <= 1e-8 or policy_dt <= 0:
        return []

    max_per_second = float(term_cfg["max_raw"]) / policy_dt
    if max_per_second <= 0:
        return []

    normalized_curve = []
    for iteration, value in curve:
        quality = (float(value) / scale) / max_per_second
        quality = float(np.clip(quality, -2.0, 1.0))
        normalized_curve.append((int(iteration), float(quality)))
    return normalized_curve


def estimate_policy_dt(task_name: str, default_dt: float = 0.01) -> float:
    """Estimate policy/control timestep for normalizing episode reward terms."""
    try:
        project_root_str = str(PROJECT_ROOT)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        from legged_gym.envs import task_registry

        env_cfg, _ = task_registry.get_cfgs(name=task_name)
        sim_dt = float(getattr(env_cfg.sim, "dt", default_dt))
        decimation = float(getattr(env_cfg.control, "decimation", 1.0))
        policy_dt = sim_dt * decimation
        if policy_dt > 0:
            return policy_dt
    except Exception as exc:
        print(
            f"[WARN] Failed to infer policy dt for task '{task_name}': {exc}. "
            f"Falling back to {default_dt:.4f}s."
        )
    return default_dt


def compute_trial_score(
    metrics: Dict[str, Any],
    policy_dt: float = 0.01,
) -> Dict[str, Any]:
    """
    Velocity-prioritized composite score for ranking trials.

    Prioritizes speed-tracking reward terms: `tracking_lin_vel` is the primary
    objective, `tracking_ang_vel` is the supporting objective, and total reward
    is secondary.
    """
    curve = metrics.get("reward_curve", [])
    if len(curve) < 10:
        return {
            "score": -999.0,
            "base_reward_score": -999.0,
            "tracking_lin_vel_quality": 0.0,
            "tracking_ang_vel_quality": 0.0,
            "gate_penalty": 0.0,
        }

    reward_stats = _curve_stats(curve)
    assert reward_stats is not None
    base_reward_score = (
        reward_stats["overall_mean"] * 0.3
        + reward_stats["tail_mean"] * 0.5
        + max(0.0, reward_stats["trend"]) * 0.2
    )

    term_stats: Dict[str, Dict[str, float]] = {}
    score = base_reward_score * 0.2
    gate_penalty = 0.0

    for reward_name, cfg in SPEED_PRIORITY_TERMS.items():
        normalized_curve = _normalized_speed_metric_curve(metrics, reward_name, policy_dt)
        stats = _curve_stats(normalized_curve) or {
            "tail_mean": 0.0,
            "initial_mean": 0.0,
            "overall_mean": 0.0,
            "trend": 0.0,
            "latest": 0.0,
        }
        term_stats[reward_name] = stats

        tail_mean = stats["tail_mean"]
        trend = max(0.0, stats["trend"])
        score += cfg["weight"] * tail_mean
        score += 0.2 * cfg["weight"] * trend

        floor = float(cfg["floor"])
        if tail_mean < floor:
            gate_penalty += (floor - tail_mean) * cfg["weight"] * 1.5

    score -= gate_penalty

    return {
        "score": float(score),
        "base_reward_score": float(base_reward_score),
        "tracking_lin_vel_quality": float(term_stats["tracking_lin_vel"]["tail_mean"]),
        "tracking_ang_vel_quality": float(term_stats["tracking_ang_vel"]["tail_mean"]),
        "tracking_lin_vel_trend": float(max(0.0, term_stats["tracking_lin_vel"]["trend"])),
        "tracking_ang_vel_trend": float(max(0.0, term_stats["tracking_ang_vel"]["trend"])),
        "gate_penalty": float(gate_penalty),
    }


# ---------------------------------------------------------------------------
# Best-config bootstrap helpers
# ---------------------------------------------------------------------------
def _coerce_reward_scale_map(payload: Any) -> Dict[str, float]:
    """Normalize different JSON payload shapes into a flat reward-scale dict."""
    if isinstance(payload, dict):
        if isinstance(payload.get("reward_scales"), dict):
            payload = payload["reward_scales"]
        elif isinstance(payload.get("best_meta"), dict):
            nested = payload.get("best_meta", {}).get("reward_scales")
            if isinstance(nested, dict):
                payload = nested
    if not isinstance(payload, dict):
        return {}

    valid_names = set(DEFAULT_SEARCH_SPACE) | set(FIXED_SCALES)
    reward_scales: Dict[str, float] = {}
    for name, value in payload.items():
        if name not in valid_names:
            continue
        try:
            reward_scales[name] = float(value)
        except (TypeError, ValueError):
            continue
    return reward_scales


def resolve_best_config_input(config_input: str) -> Path:
    """Resolve a best-config input that can be either a JSON file or a tune output dir."""
    raw_path = Path(config_input).expanduser()
    if raw_path.is_file():
        return raw_path.resolve()
    if raw_path.is_dir():
        candidates = [
            raw_path / "best_reward_scales.json",
            raw_path / "best_config.json",
            raw_path / "tuner_state.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"No best-config JSON found under directory: {raw_path}. "
            f"Tried: {[str(path.name) for path in candidates]}"
        )
    raise FileNotFoundError(f"Best-config path not found: {config_input}")


def load_best_config_seed(config_input: Optional[str]) -> Tuple[Optional[Path], Dict[str, float], Dict[str, float]]:
    """
    Load a previous best config for bootstrapping.

    Returns:
      - resolved source path
      - tunable subset (DEFAULT_SEARCH_SPACE keys only)
      - full reward scale map (including fixed scales when present)
    """
    if not config_input:
        return None, {}, {}

    resolved = resolve_best_config_input(config_input)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read best config from {resolved}: {exc}") from exc

    reward_scales = _coerce_reward_scale_map(payload)
    if not reward_scales:
        raise ValueError(f"No reward scales found in {resolved}")

    tunable = {
        name: float(value)
        for name, value in reward_scales.items()
        if name in DEFAULT_SEARCH_SPACE
    }
    if not tunable:
        raise ValueError(f"No tunable reward scales found in {resolved}")
    return resolved, tunable, reward_scales


def _best_config_log_path(output_dir: Path) -> Path:
    return output_dir / BEST_CONFIG_LOG_FILENAME


def emit_best_config_log(
    output_dir: Path,
    best_meta: Dict[str, Any],
    reason: str,
) -> None:
    """Append a grep-friendly best-config record and mirror it to stdout."""
    reward_scales = best_meta.get("reward_scales", {}) or {}
    if not reward_scales:
        return

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reason": reason,
        "trial_id": best_meta.get("trial_id"),
        "score": best_meta.get("score"),
        "reward_scales": reward_scales,
    }
    log_line = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    with open(_best_config_log_path(output_dir), "a", encoding="utf-8") as f:
        f.write(log_line + "\n")
    print(f"BEST_CONFIG: {log_line}")


# ---------------------------------------------------------------------------
# Search space sampling
# ---------------------------------------------------------------------------
SAMPLER_WARMUP_TRIALS = 4
SAMPLER_MIN_SUCCESSFUL = 2
SAMPLER_ELITE_FRACTION = 0.35
SAMPLER_MIN_ELITES = 2
SAMPLER_INIT_SIGMA_FRACTION = 0.22
SAMPLER_MIN_SIGMA_FRACTION = 0.04
SAMPLER_MAX_SIGMA_FRACTION = 0.45
SAMPLER_BASE_EXPLORE_PROB = 0.18
SAMPLER_BASE_MUTATION_PROB = 0.12
SAMPLER_STALL_PATIENCE = 4
SAMPLER_STALL_EXPAND = 1.35
SAMPLER_DIRECTION_GAIN = 0.35
SAMPLER_SCORE_SUPPORT_TRIALS = 6
SAMPLER_MIN_ANNEAL_SCALE = 0.38
SAMPLER_MAX_STALL_SCALE = 2.30
SAMPLER_MIN_EXPLORE_PROB = 0.05
SAMPLER_MIN_MUTATION_PROB = 0.04
SAMPLER_SEED_SIGMA_FRACTION = 0.28
SAMPLER_SEED_MUTATION_PROB = 0.16


def _sample_random_scale(
    low: float,
    high: float,
    log_uniform: bool,
    rng: np.random.Generator,
) -> float:
    """Draw one scalar directly from the original search space."""
    if log_uniform:
        if low > 0 and high > 0:
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))
        if low < 0 and high < 0:
            mag_low, mag_high = abs(high), abs(low)
            return -float(np.exp(rng.uniform(np.log(mag_low), np.log(mag_high))))
    return float(rng.uniform(low, high))


def _search_coord_bounds(
    low: float,
    high: float,
    log_uniform: bool,
) -> Tuple[float, float]:
    """Map a parameter range into the optimization coordinate system."""
    if log_uniform and low > 0 and high > 0:
        return float(np.log(low)), float(np.log(high))
    if log_uniform and low < 0 and high < 0:
        return float(np.log(abs(high))), float(np.log(abs(low)))
    return float(low), float(high)


def _to_search_coord(
    value: float,
    low: float,
    high: float,
    log_uniform: bool,
) -> float:
    """Transform a reward scale into the sampler's search coordinate."""
    if log_uniform and low > 0 and high > 0:
        return float(np.log(np.clip(value, low, high)))
    if log_uniform and low < 0 and high < 0:
        clipped = float(np.clip(value, low, high))
        return float(np.log(np.clip(abs(clipped), abs(high), abs(low))))
    return float(np.clip(value, low, high))


def _from_search_coord(
    coord: float,
    low: float,
    high: float,
    log_uniform: bool,
) -> float:
    """Inverse-transform a search coordinate back into a real reward scale."""
    coord_low, coord_high = _search_coord_bounds(low, high, log_uniform)
    coord = float(np.clip(coord, coord_low, coord_high))
    if log_uniform and low > 0 and high > 0:
        return float(np.clip(np.exp(coord), low, high))
    if log_uniform and low < 0 and high < 0:
        value = -float(np.exp(coord))
        return float(np.clip(value, low, high))
    return float(np.clip(coord, low, high))


def _trial_search_vector(
    space: dict,
    reward_scales: Dict[str, float],
) -> Dict[str, float]:
    """Project one trial's tunable scales into the sampler coordinate system."""
    vector: Dict[str, float] = {}
    for name, (low, high, log_uniform, _desc) in space.items():
        if name not in reward_scales:
            continue
        vector[name] = _to_search_coord(
            float(reward_scales[name]),
            low,
            high,
            log_uniform,
        )
    return vector


def _best_improvement_history(
    space: dict,
    all_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Track each time a new best score was discovered over chronological trials."""
    history: List[Dict[str, Any]] = []
    best_score = -float("inf")
    ordered = sorted(all_results, key=lambda r: int(r.get("trial_id", 10**9)))
    for result in ordered:
        if not result.get("success"):
            continue
        score = float(result.get("score", -float("inf")))
        if score <= best_score:
            continue
        reward_scales = result.get("reward_scales", {})
        history.append(
            {
                "trial_id": int(result.get("trial_id", -1)),
                "score": score,
                "point": _trial_search_vector(space, reward_scales),
            }
        )
        best_score = score
    return history


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _annealing_schedule(
    successful: List[Dict[str, Any]],
    best_score: float,
    stall_trials: int,
) -> Dict[str, float]:
    """
    Compute the sampler annealing schedule.

    Higher best scores shrink the local step size; longer stagnation expands it.
    The shrink factor only becomes aggressive once we have enough successful
    trials, so an early lucky run will not freeze the search prematurely.
    """
    if not successful:
        return {
            "score_confidence": 0.0,
            "anneal_scale": 1.0,
            "stall_scale": 1.0,
            "score_median": 0.0,
            "score_spread": 1.0,
        }

    scores = np.array(
        [float(result.get("score", 0.0)) for result in successful],
        dtype=float,
    )
    score_median = float(np.median(scores))
    score_q25 = float(np.percentile(scores, 25))
    score_q75 = float(np.percentile(scores, 75))
    score_spread = max(
        float(np.std(scores)),
        score_q75 - score_q25,
        abs(score_median) * 0.08,
        1.0,
    )

    score_gap = max(0.0, float(best_score) - score_median)
    raw_confidence = _sigmoid(score_gap / score_spread - 0.35)
    support = float(np.clip(len(successful) / SAMPLER_SCORE_SUPPORT_TRIALS, 0.0, 1.0))
    score_confidence = raw_confidence * (0.35 + 0.65 * support)
    anneal_scale = float(
        np.clip(1.0 - 0.62 * score_confidence, SAMPLER_MIN_ANNEAL_SCALE, 1.0)
    )

    stall_ratio = float(stall_trials) / max(1.0, float(SAMPLER_STALL_PATIENCE))
    stall_scale = float(
        np.clip(
            1.0 + stall_ratio * (SAMPLER_STALL_EXPAND - 1.0),
            1.0,
            SAMPLER_MAX_STALL_SCALE,
        )
    )

    return {
        "score_confidence": float(score_confidence),
        "anneal_scale": anneal_scale,
        "stall_scale": stall_scale,
        "score_median": score_median,
        "score_spread": score_spread,
    }


def build_sampler_state(
    space: dict,
    all_results: List[Dict[str, Any]],
    prior_best_scales: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Build an adaptive sampling state from historical trial results.

    Strategy:
      1. Warm up with a few random trials.
      2. Rank successful trials by score and compute an elite center.
      3. Bias the next sample toward the current best and the recent
         improvement direction.
      4. Expand exploration automatically when the best score stalls.
    """
    successful = _rank_successful_trials(all_results)
    improvement_history = _best_improvement_history(space, all_results)
    best_result = successful[0] if successful else None
    has_prior_seed = bool(prior_best_scales)
    best_trial_id: Any = int(best_result.get("trial_id", -1)) if best_result else ("seed" if has_prior_seed else None)
    best_score = float(best_result.get("score", -float("inf"))) if best_result else -float("inf")
    completed_trials = len(all_results)
    successful_trials = len(successful)

    state: Dict[str, Any] = {
        "mode": "warmup",
        "completed_trials": completed_trials,
        "successful_trials": successful_trials,
        "elite_count": 0,
        "best_trial_id": best_trial_id,
        "best_score": best_score,
        "stall_trials": completed_trials,
        "explore_prob": 1.0,
        "mutation_prob": 1.0,
        "center": {},
        "best_point": {},
        "direction": {},
        "sigma": {},
        "seeded_from_prior": has_prior_seed,
        "score_confidence": 0.0,
        "anneal_scale": 1.0,
        "stall_scale": 1.0,
        "score_median": 0.0,
        "score_spread": 1.0,
    }

    if best_result:
        state["best_point"] = _trial_search_vector(space, best_result.get("reward_scales", {}))
    elif prior_best_scales:
        state["best_point"] = _trial_search_vector(space, prior_best_scales)

    if completed_trials < SAMPLER_WARMUP_TRIALS or successful_trials < SAMPLER_MIN_SUCCESSFUL:
        return state

    elite_count = max(
        SAMPLER_MIN_ELITES,
        int(np.ceil(successful_trials * SAMPLER_ELITE_FRACTION)),
    )
    elite_count = min(elite_count, successful_trials)
    elites = successful[:elite_count]
    elite_scores = np.array([float(r.get("score", 0.0)) for r in elites], dtype=float)
    rank_weights = np.linspace(elite_count, 1, elite_count, dtype=float)
    if elite_scores.size and float(elite_scores.max() - elite_scores.min()) > 1e-8:
        score_weights = 1.0 + (elite_scores - elite_scores.min()) / (
            elite_scores.max() - elite_scores.min()
        )
        weights = rank_weights * score_weights
    else:
        weights = rank_weights
    weights = weights / weights.sum()

    prev_best_point = improvement_history[-2]["point"] if len(improvement_history) >= 2 else None
    last_improvement_trial = improvement_history[-1]["trial_id"] if improvement_history else -1
    stall_trials = max(0, completed_trials - 1 - last_improvement_trial)
    anneal = _annealing_schedule(successful, best_score, stall_trials)

    center: Dict[str, float] = {}
    direction: Dict[str, float] = {}
    sigma: Dict[str, float] = {}
    best_point = state["best_point"]

    for name, (low, high, log_uniform, _desc) in space.items():
        coord_low, coord_high = _search_coord_bounds(low, high, log_uniform)
        coord_width = max(1e-6, coord_high - coord_low)
        elite_coords = np.array(
            [
                _to_search_coord(
                    float(r.get("reward_scales", {}).get(name, low)),
                    low,
                    high,
                    log_uniform,
                )
                for r in elites
            ],
            dtype=float,
        )
        elite_mean = float(np.dot(weights, elite_coords))
        elite_var = float(np.dot(weights, (elite_coords - elite_mean) ** 2))
        elite_std = float(np.sqrt(max(0.0, elite_var)))

        best_coord = float(best_point.get(name, elite_mean))
        if prev_best_point and name in prev_best_point:
            raw_direction = best_coord - float(prev_best_point[name])
        else:
            raw_direction = best_coord - elite_mean
        raw_direction = float(np.clip(raw_direction, -0.5 * coord_width, 0.5 * coord_width))

        target = 0.65 * best_coord + 0.35 * elite_mean + SAMPLER_DIRECTION_GAIN * raw_direction
        center[name] = float(np.clip(target, coord_low, coord_high))
        direction[name] = raw_direction

        sigma_floor = coord_width * SAMPLER_MIN_SIGMA_FRACTION
        sigma_cap = coord_width * SAMPLER_MAX_SIGMA_FRACTION * anneal["stall_scale"]
        sigma_base = max(elite_std, coord_width * SAMPLER_INIT_SIGMA_FRACTION, sigma_floor)
        sigma_value = sigma_floor + (sigma_base - sigma_floor) * anneal["anneal_scale"]
        sigma_value *= anneal["stall_scale"]
        sigma[name] = float(np.clip(sigma_value, sigma_floor, sigma_cap))

    explore_prob = float(
        np.clip(
            SAMPLER_BASE_EXPLORE_PROB
            * (0.45 + 0.85 * anneal["anneal_scale"])
            * anneal["stall_scale"],
            SAMPLER_MIN_EXPLORE_PROB,
            0.45,
        )
    )
    mutation_prob = float(
        np.clip(
            SAMPLER_BASE_MUTATION_PROB
            * (0.60 + 0.70 * anneal["anneal_scale"])
            * anneal["stall_scale"],
            SAMPLER_MIN_MUTATION_PROB,
            0.35,
        )
    )

    state.update(
        {
            "mode": "adaptive",
            "elite_count": elite_count,
            "stall_trials": stall_trials,
            "explore_prob": explore_prob,
            "mutation_prob": mutation_prob,
            "center": center,
            "direction": direction,
            "sigma": sigma,
            "score_confidence": anneal["score_confidence"],
            "anneal_scale": anneal["anneal_scale"],
            "stall_scale": anneal["stall_scale"],
            "score_median": anneal["score_median"],
            "score_spread": anneal["score_spread"],
        }
    )
    return state


def sample_reward_scales(
    space: dict,
    rng: np.random.Generator,
    all_results: Optional[List[Dict[str, Any]]] = None,
    prior_best_scales: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Draw one candidate reward-scale set.

    Before enough signal exists we sample randomly. After warmup, we shift to
    an elite-guided search centered on high-score trials while keeping a small
    amount of exploration to avoid getting stuck.
    """
    all_results = all_results or []
    sampler_state = build_sampler_state(space, all_results, prior_best_scales=prior_best_scales)
    trial: Dict[str, float] = {}

    if sampler_state["mode"] != "adaptive":
        seeded_from_prior = bool(prior_best_scales) and bool(sampler_state.get("best_point"))
        completed_trials = int(sampler_state.get("completed_trials", 0))
        strategy = "random"
        if seeded_from_prior:
            seed_point = sampler_state.get("best_point", {}) or {}
            if completed_trials == 0:
                strategy = "seed_exact"
                for name, (low, high, log_uniform, _desc) in space.items():
                    seed_value = prior_best_scales.get(name) if prior_best_scales else None
                    if seed_value is None:
                        trial[name] = _sample_random_scale(low, high, log_uniform, rng)
                    else:
                        coord = _to_search_coord(float(seed_value), low, high, log_uniform)
                        trial[name] = _from_search_coord(coord, low, high, log_uniform)
            else:
                strategy = "seed_local"
                for name, (low, high, log_uniform, _desc) in space.items():
                    if rng.random() < SAMPLER_SEED_MUTATION_PROB:
                        trial[name] = _sample_random_scale(low, high, log_uniform, rng)
                        continue
                    coord_low, coord_high = _search_coord_bounds(low, high, log_uniform)
                    coord_width = max(1e-6, coord_high - coord_low)
                    anchor = float(seed_point.get(name, (coord_low + coord_high) * 0.5))
                    sigma = max(coord_width * SAMPLER_SEED_SIGMA_FRACTION, coord_width * SAMPLER_MIN_SIGMA_FRACTION)
                    sampled_coord = float(np.clip(rng.normal(anchor, sigma), coord_low, coord_high))
                    trial[name] = _from_search_coord(sampled_coord, low, high, log_uniform)
        else:
            for name, (low, high, log_uniform, _desc) in space.items():
                trial[name] = _sample_random_scale(low, high, log_uniform, rng)
        sampler_info = {
            "mode": "warmup",
            "strategy": strategy,
            "best_trial_id": sampler_state.get("best_trial_id"),
            "best_score": sampler_state.get("best_score"),
            "successful_trials": sampler_state.get("successful_trials", 0),
            "completed_trials": sampler_state.get("completed_trials", 0),
            "elite_count": 0,
            "stall_trials": sampler_state.get("stall_trials", 0),
            "explore_prob": 1.0,
            "mutation_prob": 1.0 if not seeded_from_prior else SAMPLER_SEED_MUTATION_PROB,
            "anneal_scale": 1.0,
            "stall_scale": 1.0,
            "score_confidence": 0.0,
            "seeded_from_prior": seeded_from_prior,
        }
        return trial, sampler_info

    strategy_roll = rng.random()
    explore_prob = float(sampler_state["explore_prob"])
    mutation_prob = float(sampler_state["mutation_prob"])
    anneal_scale = float(sampler_state.get("anneal_scale", 1.0))
    if strategy_roll < explore_prob:
        strategy = "global_explore"
        anchor = {}
        sigma_scale = 1.0
    else:
        remain_roll = (strategy_roll - explore_prob) / max(1e-8, 1.0 - explore_prob)
        if remain_roll < 0.55:
            strategy = "local_best"
            anchor = sampler_state["best_point"]
            sigma_scale = max(0.45, 0.35 + 0.55 * anneal_scale)
        else:
            strategy = "elite_center"
            anchor = sampler_state["center"]
            sigma_scale = max(0.65, 0.50 + 0.65 * anneal_scale)

    for name, (low, high, log_uniform, _desc) in space.items():
        if strategy == "global_explore" or rng.random() < mutation_prob:
            trial[name] = _sample_random_scale(low, high, log_uniform, rng)
            continue

        coord_low, coord_high = _search_coord_bounds(low, high, log_uniform)
        base_coord = float(anchor.get(name, (coord_low + coord_high) * 0.5))
        direction = float(sampler_state["direction"].get(name, 0.0))
        sigma = float(sampler_state["sigma"].get(name, (coord_high - coord_low) * 0.2))
        direction_gain = 0.25 if strategy == "local_best" else 0.45
        sampled_coord = rng.normal(base_coord + direction_gain * direction, sigma * sigma_scale)
        sampled_coord = float(np.clip(sampled_coord, coord_low, coord_high))
        trial[name] = _from_search_coord(sampled_coord, low, high, log_uniform)

    sampler_info = {
        "mode": "adaptive",
        "strategy": strategy,
        "best_trial_id": sampler_state.get("best_trial_id"),
        "best_score": sampler_state.get("best_score"),
        "successful_trials": sampler_state.get("successful_trials", 0),
        "completed_trials": sampler_state.get("completed_trials", 0),
        "elite_count": sampler_state.get("elite_count", 0),
        "stall_trials": sampler_state.get("stall_trials", 0),
        "explore_prob": explore_prob,
        "mutation_prob": mutation_prob,
        "anneal_scale": anneal_scale,
        "stall_scale": float(sampler_state.get("stall_scale", 1.0)),
        "score_confidence": float(sampler_state.get("score_confidence", 0.0)),
        "seeded_from_prior": bool(sampler_state.get("seeded_from_prior")),
    }
    return trial, sampler_info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_eta_from_iters(it: int, total: int, elapsed_s: float) -> str:
    """Format remaining time from per-iteration timing."""
    if it <= 0:
        return ""
    per_iter = elapsed_s / it
    remaining_s = per_iter * (total - it)
    if remaining_s >= 3600:
        return f"{remaining_s / 3600:.2f}h"
    elif remaining_s >= 60:
        return f"{remaining_s / 60:.1f}min"
    else:
        return f"{remaining_s:.0f}s"


def _format_sampler_summary(sampler_info: Dict[str, Any]) -> str:
    mode = sampler_info.get("mode", "unknown")
    strategy = sampler_info.get("strategy", "unknown")
    best_trial_id = sampler_info.get("best_trial_id")
    best_score = sampler_info.get("best_score", -float("inf"))
    successful = sampler_info.get("successful_trials", 0)
    elite_count = sampler_info.get("elite_count", 0)
    stall_trials = sampler_info.get("stall_trials", 0)

    if mode != "adaptive":
        warmup_note = "seed-bootstrap" if sampler_info.get("seeded_from_prior") else "waiting_for_signal"
        return (
            f"Sampling: warmup/{strategy}"
            f" | successful={successful}"
            f" | {warmup_note}"
        )

    best_score_str = "N/A" if not np.isfinite(best_score) else f"{best_score:.3f}"
    return (
        f"Sampling: adaptive/{strategy}"
        f" | best=#{best_trial_id}"
        f" score={best_score_str}"
        f" | elites={elite_count}"
        f" | successful={successful}"
        f" | stall={stall_trials}"
        f" | step={sampler_info.get('anneal_scale', 1.0):.2f}"
        f" | stallx={sampler_info.get('stall_scale', 1.0):.2f}"
        f" | explore={sampler_info.get('explore_prob', 0.0):.2f}"
    )


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------
def run_trial(
    trial_id: int,
    reward_scales: Dict[str, float],
    args: argparse.Namespace,
    config_path: Path,
    live_progress_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute a single tuning trial via subprocess."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(reward_scales, f, indent=2)

    cmd = [
        sys.executable, str(TRIAL_RUNNER),
        "--task", args.task,
        "--max_iterations", str(args.iterations),
        "--experiment_name", args.experiment_name,
        "--run_name", f"trial_{trial_id:03d}",
        "--num_envs", str(args.num_envs),
    ]
    if args.headless:
        cmd.append("--headless")

    # Pass reward config via environment variable to avoid CLI argument conflicts
    trial_env = _subprocess_env({
        "REWARD_CONFIG_PATH": str(config_path),
    })

    print(f"\n{'=' * 70}")
    trial_pct = (trial_id) / max(1, args.trials) * 100
    print(f"Trial {trial_id + 1:03d}/{args.trials:03d}  ({trial_pct:.0f}% trials done)")
    print(f"Iterations: {args.iterations}  |  Envs: {args.num_envs}")
    print(f"Reward scales:")
    for k, v in reward_scales.items():
        print(f"  {k}: {v:.4f}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    # Rough time estimate (~4s per iter based on typical Isaac Gym perf)
    est_sec = max(1, args.iterations * 4)
    if est_sec >= 3600:
        print(f"  >> Estimated: ~{est_sec/3600:.1f}h per trial")
    else:
        print(f"  >> Estimated: ~{est_sec//60}min per trial")
    print(f"  >> Live output below (Ctrl+C to abort trial)")
    print(f"  >> '-' per line separates each PPO iteration block")

    # Stream output line by line, reward breakdown shown in real-time
    t_start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr for linear reading
        text=True,
        bufsize=1,                 # line-buffered
        cwd=str(PROJECT_ROOT),
        env=trial_env,
    )

    stdout_lines = []
    timed_out = False
    deadline = time.time() + args.timeout
    iter_count = 0  # track iteration number for display
    _ansi_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def _strip_ansi(text: str) -> str:
        return _ansi_re.sub('', text)

    try:
        while True:
            if time.time() > deadline:
                proc.kill()
                timed_out = True
                break
            try:
                line = proc.stdout.readline()
            except Exception:
                line = ""
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
                continue
            stdout_lines.append(line)
            line_r = line.rstrip()

            # --- Detect new PPO iteration from the ANSI-formatted header ---
            if "Learning iteration" in line_r:
                clean = _strip_ansi(line_r)
                m = re.search(r'Learning iteration\s+(\d+)/(\d+)', clean)
                if m:
                    it = int(m.group(1))
                    total = int(m.group(2))
                    # Only print progress header at the start of iteration
                    if it == 0 or it > iter_count:
                        iter_count = it
                        elapsed = time.time() - t_start
                        eta_str = _format_eta_from_iters(it, total, elapsed)
                        print(f"")
                        print(f"  --- Trial {trial_id + 1:03d}/{args.trials:03d}"
                              f" | Iter {it}/{total}"
                              f" | Elapsed: {elapsed / 60:.1f}min"
                              f"{' | ~' + eta_str if eta_str else ''} ---")
                else:
                    print(f"  | {_strip_ansi(clean)}")
                continue

            # --- Show informational PPO runner output lines ---
            if any(kw in line_r for kw in (
                "TRIAL_", "Traceback", "Error:",
            )):
                print(f"  | {line_r}")
            elif any(kw in line_r for kw in (
                "Mean episode rew_",
                "Mean reward:", "Mean episode length:",
                "Value function loss:", "Surrogate loss:",
                "Mean action noise std:",
                "Computation:", "Total timesteps:",
                "Iteration time:", "Total time:", "ETA:",
            )):
                # Reformat ETA to show hours when applicable
                eta_m = re.search(r'ETA:\s+([\d.]+)s', line_r)
                if eta_m:
                    eta_s = float(eta_m.group(1))
                    if eta_s >= 3600:
                        line_r = line_r.replace(
                            f"{eta_s:.1f}s", f"{eta_s/3600:.2f}h"
                        )
                    elif eta_s >= 60:
                        line_r = line_r.replace(
                            f"{eta_s:.1f}s", f"{eta_s/60:.1f}min"
                        )
                print(f"  | {line_r}")

            # --- Write live progress for dashboard ---
            if live_progress_path and "Mean reward:" in line_r:
                rew_m = _REWARD_RE.search(_strip_ansi(line_r))
                if rew_m:
                    _write_dashboard_live(
                        live_progress_path.parent,
                        {
                            "trial_id": trial_id,
                            "iteration": iter_count,
                            "reward": float(rew_m.group(1)),
                            "timestamp": time.time(),
                        },
                    )
    except KeyboardInterrupt:
        proc.kill()
        print(f"\n  [CANCELLED] Trial aborted by user")
        return {"success": False, "reward_curve": [], "ep_len_curve": [],
                "log_dir": None, "final_model": None, "warnings": ["cancelled"],
                "info": []}

    elapsed = time.time() - t_start

    if timed_out:
        print(f"  [FAILED] Trial {trial_id} timed out after {args.timeout}s")
        return {"success": False, "reward_curve": [], "ep_len_curve": [],
                "log_dir": None, "final_model": None, "warnings": ["timeout"],
                "info": []}

    proc.wait()
    combined = "".join(stdout_lines)

    metrics = parse_trial_output(combined)
    metrics["elapsed"] = elapsed
    metrics["returncode"] = proc.returncode
    metrics["reward_scales"] = reward_scales
    metrics["trial_id"] = trial_id

    # Format elapsed time for display
    def _fmt_time(sec: float) -> str:
        if sec >= 3600:
            return f"{sec / 3600:.2f}h"
        elif sec >= 60:
            return f"{sec / 60:.1f}min"
        return f"{sec:.0f}s"

    if metrics["success"]:
        score_breakdown = compute_trial_score(
            metrics,
            policy_dt=getattr(args, "score_policy_dt", 0.01),
        )
        score = float(score_breakdown["score"])
        metrics["score"] = score
        metrics["score_breakdown"] = score_breakdown
        elapsed_str = _fmt_time(elapsed)
        print(f"  [OK] Trial {trial_id + 1:03d}/{args.trials:03d}"
              f" finished in {elapsed_str}, score={score:.3f}")
        print(
            "      "
            f"lin_track={score_breakdown['tracking_lin_vel_quality']:.3f}  "
            f"ang_track={score_breakdown['tracking_ang_vel_quality']:.3f}  "
            f"reward_base={score_breakdown['base_reward_score']:.3f}"
        )
    else:
        metrics["score"] = -999.0
        metrics["score_breakdown"] = {}
        elapsed_str = _fmt_time(elapsed)
        print(f"  [FAIL] Trial {trial_id + 1:03d}/{args.trials:03d}"
              f" (rc={proc.returncode}, elapsed={elapsed_str})")
        lines = combined.splitlines()
        tail = lines[-40:] if len(lines) > 40 else lines
        print(f"  [LAST OUTPUT]")
        for line in tail:
            print(f"    | {line}")

    return metrics


# ---------------------------------------------------------------------------
# Best-model management
# ---------------------------------------------------------------------------
def _select_trial_best_checkpoint(
    trial_metrics: Dict[str, Any],
) -> Optional[Tuple[int, str, int, float]]:
    """Pick the saved checkpoint closest to the best observed reward iteration."""
    checkpoints = trial_metrics.get("model_checkpoints", [])
    curve = trial_metrics.get("reward_curve", [])
    if not checkpoints:
        final_model = trial_metrics.get("final_model")
        if final_model:
            return (-1, final_model, -1, float("nan"))
        return None

    usable = [
        (int(iter_num), path)
        for iter_num, path in checkpoints
        if path and os.path.isfile(path)
    ]
    if not usable:
        return None

    if curve:
        best_reward_iter, best_reward = max(curve, key=lambda p: p[1])
        ckpt_iter, ckpt_path = min(
            usable, key=lambda p: (abs(p[0] - best_reward_iter), -p[0])
        )
        return ckpt_iter, ckpt_path, int(best_reward_iter), float(best_reward)

    ckpt_iter, ckpt_path = usable[-1]
    return ckpt_iter, ckpt_path, ckpt_iter, float("nan")


def preserve_trial_best_model(
    trial_metrics: Dict[str, Any],
    trial_best_dir: Path,
) -> Optional[Path]:
    """Copy the best saved policy for this trial before log cleanup happens."""
    selected = _select_trial_best_checkpoint(trial_metrics)
    if selected is None:
        trial_metrics["trial_best_model"] = None
        return None

    ckpt_iter, ckpt_path, reward_iter, reward_value = selected
    src = Path(ckpt_path)
    if not src.is_file():
        trial_metrics["trial_best_model"] = None
        return None

    trial_best_dir.mkdir(parents=True, exist_ok=True)
    trial_id = int(trial_metrics.get("trial_id", -1))
    suffix = src.suffix or ".pt"
    dest = trial_best_dir / f"trial_{trial_id:03d}_best_iter_{ckpt_iter}{suffix}"
    shutil.copy2(src, dest)

    meta = {
        "trial_id": trial_id,
        "score": trial_metrics.get("score"),
        "score_breakdown": trial_metrics.get("score_breakdown", {}),
        "checkpoint_iter": ckpt_iter,
        "source_model": str(src),
        "saved_model": str(dest),
        "best_reward_iter": reward_iter,
        "best_reward": reward_value,
        "reward_scales": trial_metrics.get("reward_scales", {}),
    }
    meta_path = trial_best_dir / f"trial_{trial_id:03d}_best_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    trial_metrics["trial_best_model"] = str(dest)
    trial_metrics["trial_best_model_iter"] = ckpt_iter
    trial_metrics["trial_best_reward_iter"] = reward_iter
    trial_metrics["trial_best_reward"] = reward_value
    print(
        f"  -> Preserved trial best model: {dest} "
        f"(ckpt iter {ckpt_iter}, best reward iter {reward_iter})"
    )
    return dest


def update_best_model(
    trial_metrics: Dict[str, Any],
    best_dir: Path,
    best_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Copy model to best/ if it outperforms, clean up other trial models."""
    score = trial_metrics.get("score", -999)
    if score <= best_meta.get("score", -float("inf")):
        # Not better: remove this trial's model to save disk
        log_dir = trial_metrics.get("log_dir")
        if log_dir and os.path.isdir(log_dir):
            shutil.rmtree(log_dir, ignore_errors=True)
        return best_meta  # unchanged

    # New best — remove previous best model
    prev_best = best_meta.get("log_dir")
    if prev_best and os.path.isdir(prev_best):
        shutil.rmtree(prev_best, ignore_errors=True)

    # Copy new best to best_dir
    best_dir.mkdir(parents=True, exist_ok=True)
    new_log_dir = trial_metrics.get("log_dir")
    if new_log_dir and os.path.isdir(new_log_dir):
        dest = best_dir / os.path.basename(new_log_dir)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(new_log_dir, str(dest))
        final_model = trial_metrics.get("final_model")
        if final_model:
            final_name = os.path.basename(final_model)
            copied_final = dest / final_name
            if copied_final.exists():
                trial_metrics["final_model"] = str(copied_final)
        # Remove original to save space
        shutil.rmtree(new_log_dir, ignore_errors=True)
        trial_metrics["log_dir"] = str(dest)  # update to new location

    return _best_meta_from_result(trial_metrics)


def _best_meta_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "score": result.get("score", -float("inf")),
        "trial_id": result.get("trial_id"),
        "reward_scales": result.get("reward_scales", {}),
        "score_breakdown": result.get("score_breakdown", {}),
        "log_dir": result.get("log_dir"),
        "final_model": result.get("final_model"),
        "trial_best_model": result.get("trial_best_model"),
        "trial_best_model_iter": result.get("trial_best_model_iter"),
        "reward_curve": result.get("reward_curve", []),
        "ep_len_curve": result.get("ep_len_curve", []),
    }


def rescore_completed_trials(
    all_results: List[Dict[str, Any]],
    policy_dt: float,
) -> Dict[str, Any]:
    """Recompute scores for existing results when scoring logic changes."""
    best_meta: Dict[str, Any] = {"score": -float("inf")}
    for result in all_results:
        if result.get("success"):
            score_breakdown = compute_trial_score(result, policy_dt=policy_dt)
            result["score"] = float(score_breakdown["score"])
            result["score_breakdown"] = score_breakdown
            if result["score"] > best_meta.get("score", -float("inf")):
                best_meta = _best_meta_from_result(result)
        else:
            result.setdefault("score", -999.0)
            result.setdefault("score_breakdown", {})
    return best_meta


def _rank_successful_trials(all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return successful trials sorted by descending score."""
    ranked = [r for r in all_results if r.get("success")]
    ranked.sort(
        key=lambda r: (
            float(r.get("score", -float("inf"))),
            -int(r.get("trial_id", 10**9)),
        ),
        reverse=True,
    )
    return ranked


def _trial_best_meta_path(trial_best_dir: Path, trial_id: int) -> Path:
    return trial_best_dir / f"trial_{trial_id:03d}_best_meta.json"


def _video_output_path(output_dir: Path, trial_id: int, checkpoint_iter: Any) -> Path:
    return output_dir / "videos" / f"trial_{trial_id:03d}_best_iter_{checkpoint_iter}.mp4"


def _remove_path_if_exists(path: Optional[Path]) -> None:
    if path is None:
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _prune_trial_artifacts(
    result: Dict[str, Any],
    trial_best_dir: Path,
    output_dir: Path,
    dash_state_dir: Path,
) -> None:
    """Delete per-trial stored checkpoint/video artifacts for a pruned trial."""
    model_path = result.get("trial_best_model")
    if model_path:
        _remove_path_if_exists(Path(model_path))

    trial_id = int(result.get("trial_id", -1))
    _remove_path_if_exists(_trial_best_meta_path(trial_best_dir, trial_id))

    video_candidates = list(result.get("video_paths", []) or [])
    if not video_candidates:
        checkpoint_iter = result.get("trial_best_model_iter")
        if checkpoint_iter is not None:
            video_candidates.append(
                str(_video_output_path(output_dir, trial_id, checkpoint_iter))
            )

    for video_path in video_candidates:
        video_out = Path(video_path)
        _remove_path_if_exists(video_out)
        _remove_path_if_exists(_video_job_path(dash_state_dir, video_out))

    result["trial_best_model"] = None
    result["video_paths"] = []


def _ensure_trial_video(
    result: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    dash_state_dir: Path,
) -> bool:
    """Ensure a kept top-ranked trial has a video recording job."""
    model_path = result.get("trial_best_model")
    if not model_path or not os.path.isfile(model_path):
        result["video_paths"] = []
        return False

    trial_id = int(result.get("trial_id", -1))
    checkpoint_iter = result.get("trial_best_model_iter", "unknown")
    video_out = _video_output_path(output_dir, trial_id, checkpoint_iter)
    video_out.parent.mkdir(parents=True, exist_ok=True)
    result["video_paths"] = [str(video_out)]

    status_path = _video_job_path(dash_state_dir, video_out)
    existing_job = _read_dashboard_json(status_path, {})
    status = existing_job.get("status")
    checkpoint_path = existing_job.get("checkpoint_path")

    if (
        status in {"pending", "running"}
        and checkpoint_path == str(model_path)
    ):
        return False
    if (
        status == "ready"
        and checkpoint_path == str(model_path)
        and video_out.is_file()
    ):
        return False

    _write_video_job(
        dash_state_dir,
        trial_id=trial_id,
        output_path=video_out,
        status="pending",
        checkpoint_path=str(model_path),
        checkpoint_iter=checkpoint_iter,
    )
    subprocess.Popen(
        [
            sys.executable,
            str(RECORD_VIDEO_SCRIPT),
            "--task",
            args.task,
            "--model",
            str(model_path),
            "--output",
            str(video_out),
            "--status-file",
            str(status_path),
            "--trial-id",
            str(trial_id),
            "--headless",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def _sync_top_trial_artifacts(
    all_results: List[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
    trial_best_dir: Path,
    dash_state_dir: Path,
    keep_limit: int = MAX_TOP_TRIAL_ARTIFACTS,
) -> List[Dict[str, Any]]:
    """Keep only the top-N trial checkpoints and videos on disk."""
    ranked = _rank_successful_trials(all_results)
    keep_ids = {int(r.get("trial_id", -1)) for r in ranked[:keep_limit]}
    started_videos = 0

    for result in all_results:
        if not result.get("success"):
            result["video_paths"] = []
            continue

        trial_id = int(result.get("trial_id", -1))
        if trial_id in keep_ids and result.get("trial_best_model"):
            started_videos += int(
                _ensure_trial_video(result, args, output_dir, dash_state_dir)
            )
        else:
            _prune_trial_artifacts(result, trial_best_dir, output_dir, dash_state_dir)

    if started_videos:
        print(f"  -> Recording {started_videos} top-{keep_limit} video(s) in background...")
    return ranked[:keep_limit]


# ---------------------------------------------------------------------------
# Report & charts
# ---------------------------------------------------------------------------
def generate_report(
    all_results: List[Dict[str, Any]],
    best_meta: Dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
):
    """Generate tuning report: text summary and matplotlib charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = [r for r in all_results if r.get("success")]

    # --- Text report ---
    report_path = output_dir / "tuning_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  Reward Scale Auto-Tuning Report\n")
        f.write(f"  Task: {args.task}\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Trials: {len(all_results)} total, {len(successful)} successful\n")
        f.write(f"  Iterations per trial: {args.iterations}\n")
        f.write("=" * 70 + "\n\n")

        # Best configuration
        f.write("--- Best Configuration ---\n")
        f.write(f"  Trial ID: {best_meta.get('trial_id', 'N/A')}\n")
        f.write(f"  Score: {best_meta.get('score', 'N/A'):.4f}\n")
        f.write(f"  Model: {best_meta.get('final_model', 'N/A')}\n\n")
        breakdown = best_meta.get("score_breakdown", {}) or {}
        if breakdown:
            f.write("  Speed-priority metrics:\n")
            f.write(f"    tracking_lin_vel quality: {breakdown.get('tracking_lin_vel_quality', 0.0):.4f}\n")
            f.write(f"    tracking_ang_vel quality:  {breakdown.get('tracking_ang_vel_quality', 0.0):.4f}\n")
            f.write(f"    base reward score:      {breakdown.get('base_reward_score', 0.0):.4f}\n")
            f.write(f"    gate penalty:           {breakdown.get('gate_penalty', 0.0):.4f}\n\n")
        f.write("  Reward scales:\n")
        for k, v in best_meta.get("reward_scales", {}).items():
            f.write(f"    {k}: {v:.6f}\n")
        f.write("\n")

        # All trials ranked
        f.write("--- All Trials (ranked by score) ---\n")
        ranked = _rank_successful_trials(successful)
        f.write(
            f"{'Rank':>5s} {'ID':>5s} {'Score':>10s} "
            f"{'LinQ':>7s} {'AngQ':>7s} {'BestCkpt':>9s}  {'Key params'}\n"
        )
        f.write("-" * 110 + "\n")
        for rank, r in enumerate(ranked, 1):
            s = r.get("score", -999)
            rid = r.get("trial_id", "?")
            best_ckpt = r.get("trial_best_model_iter", "N/A")
            breakdown = r.get("score_breakdown", {}) or {}
            rs = r.get("reward_scales", {})
            key_str = "  ".join(
                f"{k}={rs.get(k, 0):.3f}"
                for k in ["tracking_lin_vel", "tracking_ang_vel", "orientation"]
                if k in rs
            )
            f.write(
                f"{rank:5d} {rid:5d} {s:10.4f} "
                f"{breakdown.get('tracking_lin_vel_quality', 0.0):7.3f} "
                f"{breakdown.get('tracking_ang_vel_quality', 0.0):7.3f} "
                f"{str(best_ckpt):>9s}  {key_str}\n"
            )
        f.write("\n")

        f.write(f"--- Top {MAX_TOP_TRIAL_ARTIFACTS} Stored Trial Policies ---\n")
        kept_ranked = [r for r in ranked if r.get("trial_best_model")]
        for r in kept_ranked[:MAX_TOP_TRIAL_ARTIFACTS]:
            f.write(
                f"  Trial {r.get('trial_id', '?')}: "
                f"{r.get('trial_best_model', 'N/A')}\n"
            )
        f.write("\n")

        # Original config comparison
        f.write("--- Comparison with Original Config ---\n")
        orig = {
            "tracking_lin_vel": 1.0, "tracking_ang_vel": 0.5,
            "orientation": 0., "base_height": 0.,
            "feet_air_time": 1.0,
        }
        f.write(f"{'Param':>25s}  {'Original':>10s}  {'Best':>10s}  {'Change':>10s}\n")
        f.write("-" * 70 + "\n")
        best_scales = best_meta.get("reward_scales", {})
        for k in sorted(orig.keys()):
            v_orig = orig[k]
            v_best = best_scales.get(k, v_orig)
            change = (v_best - v_orig) / (abs(v_orig) + 1e-8) * 100
            f.write(f"{k:>25s}  {v_orig:10.4f}  {v_best:10.4f}  {change:+8.1f}%\n")

    print(f"\nText report saved to: {report_path}")

    # --- Reward curve chart ---
    _plot_reward_curves(all_results, best_meta, output_dir)
    # --- Parameter impact chart ---
    _plot_parameter_impact(successful, output_dir)
    # --- Best trial detailed chart ---
    _plot_best_detail(best_meta, output_dir)

    print(f"Charts saved to: {output_dir}")


def _plot_reward_curves(
    all_results: List[Dict[str, Any]],
    best_meta: Dict[str, Any],
    output_dir: Path,
):
    """Plot reward learning curves for all trials, highlighting the best."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Reward curves
    best_id = best_meta.get("trial_id", -1)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(1, len(all_results))))

    for i, r in enumerate(all_results):
        curve = r.get("reward_curve", [])
        if not curve:
            continue
        iters, rewards = zip(*curve)
        alpha = 0.12 if r["trial_id"] != best_id else 1.0
        lw = 0.6 if r["trial_id"] != best_id else 2.5
        label = f"Trial {r['trial_id']}" if r["trial_id"] == best_id else None
        ax1.plot(iters, rewards, color=colors[i % len(colors)],
                 alpha=alpha, linewidth=lw, label=label)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean Episode Reward")
    ax1.set_title("Reward Learning Curves (all trials)")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Episode length curves
    for i, r in enumerate(all_results):
        curve = r.get("ep_len_curve", [])
        if not curve:
            continue
        iters, lengths = zip(*curve)
        alpha = 0.12 if r["trial_id"] != best_id else 1.0
        lw = 0.6 if r["trial_id"] != best_id else 2.5
        ax2.plot(iters, lengths, color=colors[i % len(colors)],
                 alpha=alpha, linewidth=lw)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Mean Episode Length (s)")
    ax2.set_title("Episode Length Evolution")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "reward_curves.png", dpi=150)
    plt.close(fig)


def _plot_parameter_impact(
    successful: List[Dict[str, Any]],
    output_dir: Path,
):
    """Show how each parameter correlates with final score."""
    if len(successful) < 3:
        return

    scores = np.array([r.get("score", 0) for r in successful])
    # Get all parameter names from the first trial
    param_names = list(successful[0].get("reward_scales", {}).keys())
    if not param_names:
        return

    n_params = len(param_names)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else ([axes] if n_cols == 1 else axes)

    for idx, pname in enumerate(param_names):
        ax = axes[idx]
        values = np.array([
            r.get("reward_scales", {}).get(pname, np.nan)
            for r in successful
        ])
        valid = ~np.isnan(values)
        if valid.sum() < 3:
            ax.set_visible(False)
            continue
        ax.scatter(values[valid], scores[valid], alpha=0.6, s=30)
        # Fit a trend line
        try:
            z = np.polyfit(values[valid], scores[valid], 1)
            x_line = np.linspace(values[valid].min(), values[valid].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.6, linewidth=1)
        except Exception:
            pass
        ax.set_xlabel(pname, fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_title(pname, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Parameter Impact on Trial Score", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_impact.png", dpi=150)
    plt.close(fig)


def _plot_best_detail(best_meta: Dict[str, Any], output_dir: Path):
    """Detailed reward breakdown for the best trial."""
    curve = best_meta.get("reward_curve", [])
    ep_curve = best_meta.get("ep_len_curve", [])
    if not curve:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    iters, rewards = zip(*curve)
    ax1.plot(iters, rewards, "b-", linewidth=1.2)
    # Moving average
    window = max(1, len(rewards) // 20)
    if len(rewards) > window:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(iters[window - 1 :], ma, "r-", linewidth=2, label=f"MA({window})")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean Episode Reward")
    ax1.set_title(f"Best Trial #{best_meta.get('trial_id')} — Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if ep_curve:
        iters2, lengths = zip(*ep_curve)
        ax2.plot(iters2, lengths, "g-", linewidth=1.2)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Mean Episode Length (s)")
        ax2.set_title(f"Best Trial #{best_meta.get('trial_id')} — Episode Length")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "best_trial_detail.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Final full training
# ---------------------------------------------------------------------------
def run_full_training(
    best_meta: Dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
):
    """Run a full-length training with the best reward scales."""
    config_path = output_dir / "best_config.json"
    with open(config_path, "w") as f:
        json.dump(best_meta["reward_scales"], f, indent=2)

    cmd = [
        sys.executable, str(TRIAL_RUNNER),
        "--task", args.task,
        "--max_iterations", str(args.full_iterations),
        "--experiment_name", args.experiment_name,
        "--run_name", "best_full",
        "--num_envs", str(args.num_envs),
    ]
    if args.headless:
        cmd.append("--headless")

    # Pass reward config via environment variable
    full_env = _subprocess_env({
        "REWARD_CONFIG_PATH": str(config_path),
    })

    print(f"\n{'=' * 70}")
    print("Starting full training with best config...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log will be in: {output_dir}/best_full/")
    print(f"{'=' * 70}")

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=full_env,
    )

    if proc.returncode == 0:
        print("\nFull training completed successfully!")
    else:
        print(f"\nFull training exited with code {proc.returncode}")


# ---------------------------------------------------------------------------
# Dashboard state helpers
# ---------------------------------------------------------------------------
def _write_dashboard_trials(state_dir: Path, trials: list):
    """Write the trials summary for the Dash dashboard to consume."""
    _write_json_atomic(state_dir / "trials.json", trials)


def _read_dashboard_json(path: Path, default: Any) -> Any:
    """Read dashboard JSON state, returning default while another process writes."""
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _reset_dashboard_live_curve(state_dir: Path, trial_id: int):
    """Start a fresh live curve for the currently-running trial."""
    _write_json_atomic(
        state_dir / "live_curve.json",
        {"trial_id": trial_id, "reward_curve": []},
    )


def _write_dashboard_live(state_dir: Path, data: dict):
    """Write live progress of the currently-running trial."""
    _write_json_atomic(state_dir / "live_progress.json", data)
    curve_path = state_dir / "live_curve.json"
    live_curve = _read_dashboard_json(
        curve_path, {"trial_id": data.get("trial_id"), "reward_curve": []}
    )
    if live_curve.get("trial_id") != data.get("trial_id"):
        live_curve = {"trial_id": data.get("trial_id"), "reward_curve": []}

    point = [int(data.get("iteration", 0)), float(data.get("reward", 0.0))]
    curve = live_curve.setdefault("reward_curve", [])
    if curve and curve[-1][0] == point[0]:
        curve[-1] = point
    else:
        curve.append(point)

    _write_json_atomic(curve_path, live_curve)


def _write_session_manifest(
    state_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Persist monitor metadata so the frontend can be restarted independently."""
    session = _read_dashboard_json(state_dir / "session.json", {})
    log_root = os.path.abspath(os.path.join(PROJECT_ROOT, "logs", args.experiment_name))
    session.update(
        {
            "task": args.task,
            "experiment_name": args.experiment_name,
            "output_dir": str(output_dir),
            "state_dir": str(state_dir),
            "log_root": log_root,
            "tensorboard_port": args.tensorboard_port,
            "dashboard_port": args.dashboard_port,
            "total_trials": args.trials,
            "iterations_per_trial": args.iterations,
            "best_config_path": str(output_dir / "best_reward_scales.json"),
            "best_config_log_path": str(_best_config_log_path(output_dir)),
            "updated_at": time.time(),
        }
    )
    session.setdefault("status", "initializing")
    _write_json_atomic(state_dir / "session.json", session)


def _update_session_manifest(state_dir: Path, **updates: Any) -> None:
    session_path = state_dir / "session.json"
    session = _read_dashboard_json(session_path, {})
    session.update(updates)
    session["updated_at"] = time.time()
    _write_json_atomic(session_path, session)


def _video_job_path(state_dir: Path, output_path: Path) -> Path:
    return state_dir / "video_jobs" / f"{output_path.stem}.json"


def _write_video_job(
    state_dir: Path,
    trial_id: int,
    output_path: Path,
    status: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_iter: Optional[int] = None,
    error: Optional[str] = None,
) -> Path:
    job = {
        "trial_id": trial_id,
        "output_path": str(output_path),
        "filename": output_path.name,
        "status": status,
        "checkpoint_path": checkpoint_path,
        "checkpoint_iter": checkpoint_iter,
        "error": error,
        "updated_at": time.time(),
    }
    if status in {"pending", "running"}:
        job["started_at"] = time.time()
    if status in {"ready", "failed"}:
        job["finished_at"] = time.time()
    path = _video_job_path(state_dir, output_path)
    existing = _read_dashboard_json(path, {})
    if existing:
        existing.update(job)
        job = existing
    _write_json_atomic(path, job)
    return path


def _trial_to_dashboard_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "trial_id": result.get("trial_id"),
        "success": result.get("success", False),
        "score": result.get("score"),
        "score_breakdown": result.get("score_breakdown", {}),
        "sampler_info": result.get("sampler_info", {}),
        "reward_scales": result.get("reward_scales", {}),
        "video_paths": result.get("video_paths", []),
        "trial_best_model": result.get("trial_best_model"),
        "trial_best_model_iter": result.get("trial_best_model_iter"),
        "trial_best_reward_iter": result.get("trial_best_reward_iter"),
        "trial_best_reward": result.get("trial_best_reward"),
        "reward_curve": [list(p) for p in result.get("reward_curve", [])],
        "ep_len_curve": [list(p) for p in result.get("ep_len_curve", [])],
        "episode_metric_curves": {
            str(name): [list(point) for point in (curve or [])]
            for name, curve in (result.get("episode_metric_curves", {}) or {}).items()
        },
        "elapsed": result.get("elapsed", 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Auto-tune reward scales for legged_gym-gym PPO training"
    )
    parser.add_argument("--task", type=str, default="Pikachu_V025")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of tuning trials (default: 20)")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Training iterations per trial (default: 500)")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--headless", action="store_true",
                        help="Run Isaac Gym in headless mode")
    parser.add_argument("--experiment-name", type=str,
                        default="Pikachu_V025_AutoTune")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for tuning outputs (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for parameter sampling")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Timeout per trial in seconds (default: 7200)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a previous tuning output directory")
    parser.add_argument("--no-full-training", action="store_true",
                        help="Skip the final full training with best config")
    parser.add_argument("--full-iterations", type=int, default=3001,
                        help="Iterations for final full training (default: 3001)")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Launch TensorBoard dashboard pointing to experiment logs")
    parser.add_argument("--tensorboard-port", type=int, default=1230,
                        help="Port for TensorBoard (default: 1230)")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch live Dash visualization dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8050,
                        help="Port for Dash dashboard (default: 8050)")
    parser.add_argument("--init-best-config", type=str, default=None,
                        help="Previous best config JSON or output dir used to bootstrap exploration")
    args = parser.parse_args()
    explicit_init_best_config = bool(args.init_best_config)
    args.score_policy_dt = estimate_policy_dt(args.task)

    # --- Setup output directory ---
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif args.resume:
        output_dir = Path(args.resume).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (PROJECT_ROOT / "logs" / "auto_tune" / f"{args.task}_{ts}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Velocity-priority scoring policy dt: {args.score_policy_dt:.4f}s")
    print(
        "Monitor frontend independently with:\n"
        f"  {sys.executable} {MONITOR_SCRIPT} --output-dir {output_dir}"
    )

    # --- Launch TensorBoard ---
    _maybe_launch_tensorboard(args, PROJECT_ROOT, args.tensorboard)

    # --- Dashboard state directory ---
    dash_state_dir = output_dir / "dashboard_state"
    dash_state_dir.mkdir(parents=True, exist_ok=True)
    if not (dash_state_dir / "trials.json").exists():
        _write_dashboard_trials(dash_state_dir, [])  # init empty
    _write_session_manifest(dash_state_dir, output_dir, args)

    # Auto-launch TensorBoard when dashboard is enabled
    if args.dashboard and not args.tensorboard:
        _maybe_launch_tensorboard(args, PROJECT_ROOT, True)

    # --- Launch Dash dashboard ---
    dashboard_proc = None
    if args.dashboard:
        try:
            dash_log = open(output_dir / "dashboard.log", "w")
            dashboard_proc = subprocess.Popen(
                [sys.executable, str(DASHBOARD_SCRIPT),
                 "--state-dir", str(dash_state_dir),
                 "--output-dir", str(output_dir),
                 "--port", str(args.dashboard_port),
                 "--tb-port", str(args.tensorboard_port)],
                stdout=dash_log,
                stderr=dash_log,
            )
            print(f"  -> Dashboard starting at http://localhost:{args.dashboard_port}")
            print(f"  -> Dashboard log: {output_dir / 'dashboard.log'}")
        except Exception as e:
            print(f"  [WARN] Failed to launch dashboard: {e}")

    # --- Resume or start fresh ---
    all_results: List[Dict[str, Any]] = []
    best_meta: Dict[str, Any] = {"score": -float("inf")}
    init_best_config_path: Optional[Path] = None
    init_best_tunable_scales: Dict[str, float] = {}
    init_best_full_scales: Dict[str, float] = {}
    sampler_state: Dict[str, Any] = build_sampler_state(
        DEFAULT_SEARCH_SPACE,
        all_results,
        prior_best_scales=init_best_tunable_scales,
    )
    start_trial = 0
    best_dir = output_dir / "best_model"
    trial_best_dir = output_dir / "trial_best_models"
    trial_configs_dir = output_dir / "trial_configs"
    trial_configs_dir.mkdir(parents=True, exist_ok=True)
    removed_legacy_configs = _cleanup_legacy_trial_configs(output_dir, trial_configs_dir)
    if removed_legacy_configs:
        print(f"Removed {removed_legacy_configs} duplicate legacy trial config file(s).")

    if args.resume:
        resume_dir = Path(args.resume).resolve()
        state_path = resume_dir / "tuner_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            all_results = state.get("all_results", [])
            best_meta = state.get("best_meta", {"score": -float("inf")})
            saved_init_path = state.get("init_best_config_path")
            if saved_init_path and not explicit_init_best_config:
                args.init_best_config = saved_init_path
            saved_init_scales = state.get("init_best_tunable_scales", {})
            if isinstance(saved_init_scales, dict):
                restored_tunable: Dict[str, float] = {}
                for name, value in saved_init_scales.items():
                    if name not in DEFAULT_SEARCH_SPACE:
                        continue
                    try:
                        restored_tunable[name] = float(value)
                    except (TypeError, ValueError):
                        continue
                init_best_tunable_scales = restored_tunable
            saved_init_full = state.get("init_best_full_scales", {})
            if isinstance(saved_init_full, dict):
                restored_full: Dict[str, float] = {}
                valid_names = set(DEFAULT_SEARCH_SPACE) | set(FIXED_SCALES)
                for name, value in saved_init_full.items():
                    if name not in valid_names:
                        continue
                    try:
                        restored_full[str(name)] = float(value)
                    except (TypeError, ValueError):
                        continue
                init_best_full_scales = restored_full
            sampler_state = state.get("sampler_state", sampler_state)
            start_trial = len(all_results)
            print(f"Resumed from {args.resume}: {start_trial} trials already done")
        else:
            print(f"Warning: --resume specified but {state_path} not found. Starting fresh.")

    if args.init_best_config and (explicit_init_best_config or not init_best_tunable_scales):
        try:
            (
                init_best_config_path,
                init_best_tunable_scales,
                loaded_full_scales,
            ) = load_best_config_seed(args.init_best_config)
            if loaded_full_scales:
                init_best_full_scales = loaded_full_scales
            print(f"Bootstrap best config source: {init_best_config_path}")
            print(
                "INIT_BEST_CONFIG: "
                + json.dumps(
                    {
                        "path": str(init_best_config_path),
                        "reward_scales": init_best_full_scales,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(2)
    elif init_best_tunable_scales:
        init_best_config_path = Path(args.init_best_config).expanduser() if args.init_best_config else None
        print("Bootstrap best config restored from tuner_state.json")

    if all_results:
        best_meta = rescore_completed_trials(all_results, args.score_policy_dt)
        sampler_state = build_sampler_state(
            DEFAULT_SEARCH_SPACE,
            all_results,
            prior_best_scales=init_best_tunable_scales,
        )
        print(
            f"Re-scored {len(all_results)} existing trial(s) using velocity-priority logic. "
            f"Current best trial: {best_meta.get('trial_id')}, "
            f"score={best_meta.get('score', float('nan')):.3f}"
        )

    _update_session_manifest(
        dash_state_dir,
        status="running",
        resumed=bool(args.resume),
        completed_trials=start_trial,
        current_trial=None,
        best_trial=best_meta.get("trial_id"),
        best_score=best_meta.get("score"),
        init_best_config_path=str(init_best_config_path) if init_best_config_path else None,
        current_sampler_info=None,
        current_reward_scales=None,
    )

    # --- Save state helper ---
    def save_state():
        nonlocal sampler_state
        sampler_state = build_sampler_state(
            DEFAULT_SEARCH_SPACE,
            all_results,
            prior_best_scales=init_best_tunable_scales,
        )
        # Convert numpy values for JSON serialization
        state = {
            "all_results": [],
            "best_meta": {
                "score": best_meta.get("score", -float("inf")),
                "trial_id": best_meta.get("trial_id"),
                "reward_scales": best_meta.get("reward_scales"),
                "score_breakdown": best_meta.get("score_breakdown", {}),
                "log_dir": best_meta.get("log_dir"),
                "final_model": best_meta.get("final_model"),
                "trial_best_model": best_meta.get("trial_best_model"),
                "trial_best_model_iter": best_meta.get("trial_best_model_iter"),
                "reward_curve": best_meta.get("reward_curve", []),
                "ep_len_curve": best_meta.get("ep_len_curve", []),
            },
            "sampler_state": sampler_state,
            "init_best_config_path": str(init_best_config_path) if init_best_config_path else None,
            "init_best_tunable_scales": init_best_tunable_scales,
            "init_best_full_scales": init_best_full_scales,
        }
        for r in all_results:
            clean = dict(r)
            state["all_results"].append(clean)
        _write_json_atomic(output_dir / "tuner_state.json", state)

    # --- Main tuning loop ---
    rng = np.random.default_rng(args.seed + start_trial)

    # Track overall tuning progress
    overall_t_start = time.time()
    # Running average of trial durations (sliding window)
    trial_durations = []

    if all_results:
        _sync_top_trial_artifacts(
            all_results,
            args,
            output_dir,
            trial_best_dir,
            dash_state_dir,
        )
        _write_dashboard_trials(
            dash_state_dir,
            [_trial_to_dashboard_entry(r) for r in all_results],
        )
        save_state()

    for trial_id in range(start_trial, args.trials):
        _update_session_manifest(
            dash_state_dir,
            status="running",
            current_trial=trial_id,
            completed_trials=len(all_results),
            best_trial=best_meta.get("trial_id"),
            best_score=best_meta.get("score"),
        )
        reward_scales, sampler_info = sample_reward_scales(
            DEFAULT_SEARCH_SPACE,
            rng,
            all_results,
            prior_best_scales=init_best_tunable_scales,
        )
        # Merge fixed scales
        reward_scales.update(FIXED_SCALES)

        # Show overall progress summary before each trial
        elapsed_total = time.time() - overall_t_start
        done_so_far = trial_id - start_trial
        remaining = args.trials - trial_id
        if trial_durations:
            avg_trial = sum(trial_durations) / len(trial_durations)
            eta_total_s = avg_trial * remaining + 5 * remaining  # +5s for pause
            if eta_total_s >= 3600:
                eta_total_str = f"{eta_total_s / 3600:.2f}h"
            elif eta_total_s >= 60:
                eta_total_str = f"{eta_total_s / 60:.1f}min"
            else:
                eta_total_str = f"{eta_total_s:.0f}s"
            print(f"\n>>> Overall progress: {done_so_far + 1}/{args.trials - start_trial} done"
                  f" | Elapsed: {elapsed_total / 60:.1f}min"
                  f" | ETA completion: ~{eta_total_str}")
        else:
            print(f"\n>>> Starting trial {trial_id + 1:03d}/{args.trials:03d}"
                  f" | {args.trials - trial_id} remaining")
        print(f">>> {_format_sampler_summary(sampler_info)}")

        t_trial_start = time.time()
        live_progress_path = dash_state_dir / "live_progress.json"
        _reset_dashboard_live_curve(dash_state_dir, trial_id)
        config_path = _trial_config_path(output_dir, trial_id)
        _update_session_manifest(
            dash_state_dir,
            current_trial=trial_id,
            current_sampler_info=sampler_info,
            current_reward_scales=reward_scales,
            current_trial_config_path=str(config_path),
            current_trial_started_at=time.time(),
        )
        metrics = run_trial(
            trial_id,
            reward_scales,
            args,
            config_path,
            live_progress_path=live_progress_path,
        )
        trial_durations.append(time.time() - t_trial_start)
        metrics["sampler_info"] = sampler_info
        all_results.append(metrics)

        prev_best_score = float(best_meta.get("score", -float("inf")))
        if metrics.get("success"):
            preserve_trial_best_model(metrics, trial_best_dir)
            best_meta = update_best_model(metrics, best_dir, best_meta)
            if float(best_meta.get("score", -float("inf"))) > prev_best_score:
                emit_best_config_log(output_dir, best_meta, reason=f"trial_{trial_id:03d}_new_best")

        _sync_top_trial_artifacts(
            all_results,
            args,
            output_dir,
            trial_best_dir,
            dash_state_dir,
        )

        # --- Update dashboard state ---
        dash_trials = [_trial_to_dashboard_entry(r) for r in all_results]
        _write_dashboard_trials(dash_state_dir, dash_trials)
        _update_session_manifest(
            dash_state_dir,
            completed_trials=len(all_results),
            current_trial=None,
            best_trial=best_meta.get("trial_id"),
            best_score=best_meta.get("score"),
            current_sampler_info=None,
            current_reward_scales=None,
            current_trial_started_at=None,
        )

        save_state()

        # Brief pause between trials to let GPU cool down
        if trial_id < args.trials - 1:
            time.sleep(5)

    # --- Generate report ---
    print(f"\n{'=' * 70}")
    print(f"Tuning complete. {len([r for r in all_results if r.get('success')])}/{len(all_results)} trials successful.")
    print(f"Best trial: #{best_meta.get('trial_id')} with score {best_meta.get('score', 'N/A'):.4f}")
    print(f"{'=' * 70}")

    generate_report(all_results, best_meta, output_dir, args)
    _update_session_manifest(
        dash_state_dir,
        status="report_generated",
        completed_trials=len(all_results),
        current_trial=None,
        best_trial=best_meta.get("trial_id"),
        best_score=best_meta.get("score"),
    )

    # --- Save best config separately ---
    best_config_path = output_dir / "best_reward_scales.json"
    with open(best_config_path, "w") as f:
        json.dump(best_meta.get("reward_scales", {}), f, indent=2)
    print(f"Best config saved to: {best_config_path}")
    emit_best_config_log(output_dir, best_meta, reason="final_best")

    # --- Optional full training ---
    if not args.no_full_training and best_meta.get("reward_scales"):
        _update_session_manifest(dash_state_dir, status="full_training")
        run_full_training(best_meta, output_dir, args)
    elif best_meta.get("reward_scales"):
        print("\nSkipping full training (--no-full-training).")
        print(f"To run manually, use the best config at: {best_config_path}")

    # Cleanup dashboard subprocess
    _update_session_manifest(
        dash_state_dir,
        status="completed",
        completed_trials=len(all_results),
        current_trial=None,
        best_trial=best_meta.get("trial_id"),
        best_score=best_meta.get("score"),
        current_trial_started_at=None,
    )
    if dashboard_proc is not None:
        dashboard_proc.terminate()
        print("  -> Dashboard stopped.")


if __name__ == "__main__":
    main()
