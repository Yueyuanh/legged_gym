#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Standalone subprocess that loads a trained model checkpoint and records a
short MP4 video of the robot walking.

Usage:
    python legged_gym/scripts/_record_video.py \
        --task Pikachu_V025_No_Yaw \
        --model /path/to/model_100.pt \
        --output /path/to/output.mp4 \
        --headless

Designed to be called as a subprocess by auto_tune_rewards.py.
"""

import sys
import os
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

# ---- Isaac Gym import order: import legged_gym.envs BEFORE torch ----
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from legged_gym.envs import task_registry  # noqa: F401 — triggers isaacgym
# Now safe to import torch and other modules:
import numpy as np
import cv2
import torch
from isaacgym import gymapi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Pikachu_V025_No_Yaw")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--status-file", type=str, default=None)
    parser.add_argument("--trial-id", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--record-steps", type=int, default=250,
                        help="Number of simulation steps to record (250 @ 50fps = 5s)")
    return parser.parse_args()


def write_status(args, status: str, error: str = None, extra: dict = None):
    if not args.status_file:
        return
    status_path = Path(args.status_file)
    payload = {}
    if status_path.exists():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    payload.update({
        "trial_id": args.trial_id,
        "output_path": args.output,
        "filename": Path(args.output).name,
        "checkpoint_path": args.model,
        "status": status,
        "error": error,
        "updated_at": time.time(),
    })
    if extra:
        payload.update(extra)
    if status in {"pending", "running"}:
        payload.setdefault("started_at", time.time())
    if status in {"ready", "failed"}:
        payload["finished_at"] = time.time()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(status_path)


def make_gym_args(task: str, headless: bool, num_envs: int):
    """Build a minimal Isaac Gym args namespace using get_args()
    while sanitizing sys.argv to avoid conflicts with our custom args."""
    saved_argv = sys.argv
    sys.argv = [__file__, "--task", task, "--num_envs", str(num_envs)]
    if headless:
        sys.argv.append("--headless")
    try:
        from legged_gym.utils import get_args
        return get_args()
    finally:
        sys.argv = saved_argv


def finalize_web_video(temp_output_path: Path, output_path: Path) -> dict:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        temp_output_path.replace(output_path)
        return {
            "video_codec": "mpeg4",
            "video_pix_fmt": None,
            "browser_compatible": False,
            "transcode_warning": "ffmpeg not available; saved legacy mp4v output",
        }

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-i", str(temp_output_path),
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-preset", "veryfast",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0 and output_path.exists():
        try:
            if output_path.stat().st_size > 0:
                temp_output_path.unlink(missing_ok=True)
                return {
                    "video_codec": "h264",
                    "video_pix_fmt": "yuv420p",
                    "browser_compatible": True,
                    "transcode_warning": None,
                }
        except OSError:
            pass

    temp_output_path.replace(output_path)
    warning = proc.stderr.strip() or proc.stdout.strip() or "ffmpeg transcode failed"
    return {
        "video_codec": "mpeg4",
        "video_pix_fmt": None,
        "browser_compatible": False,
        "transcode_warning": warning,
    }


def record(args):
    write_status(args, "running")
    model_path = Path(args.model)
    output_path = Path(args.output)
    temp_output_path = output_path.with_suffix(".raw.mp4")
    if not model_path.exists():
        write_status(args, "failed", f"model not found: {model_path}")
        print(f"RECORDING_ERROR: model not found: {model_path}")
        sys.exit(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path.unlink(missing_ok=True)

    # Build a proper Isaac Gym args namespace
    gym_args = make_gym_args(args.task, args.headless, args.num_envs)

    # Load configs
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_friction = False

    # Create environment
    env, _ = task_registry.make_env(name=args.task, args=gym_args, env_cfg=env_cfg)

    # Load policy from checkpoint
    train_cfg.runner.resume = True
    try:
        ppo_runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=gym_args, train_cfg=train_cfg
        )
    except ValueError:
        train_cfg.runner.resume = False
        ppo_runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=gym_args, train_cfg=train_cfg
        )
    # Override with the specific checkpoint
    checkpoint = torch.load(str(model_path), map_location=env.device)
    ppo_runner.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"])
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.get_observations()

    # ---- Set up camera ----
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 1920
    camera_properties.height = 1080
    h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
    camera_offset = gymapi.Vec3(0.8, -0.8, 0.5)
    camera_rotation = gymapi.Quat.from_axis_angle(
        gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135)
    )
    actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
    env.gym.attach_camera_to_body(
        h1, env.envs[0], body_handle,
        gymapi.Transform(camera_offset, camera_rotation),
        gymapi.FOLLOW_POSITION,
    )

    # ---- Video writer ----
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(temp_output_path), fourcc, 50.0, (1920, 1080))
    if not video.isOpened():
        write_status(args, "failed", f"failed to open video writer: {temp_output_path}")
        print(f"RECORDING_ERROR: failed to open video writer: {temp_output_path}")
        sys.exit(1)

    # ---- Run simulation and record ----
    commands = env.commands
    commands[:, 0] = 0.3  # walk forward
    commands[:, 1] = 0.0
    commands[:, 2] = 0.0
    commands[:, 3] = 0.0

    for step in range(args.record_steps):
        actions = policy(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())

        # Reset if fallen
        if dones.any():
            obs = env.get_observations()

        # Render and capture
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
        img = np.reshape(img, (1080, 1920, 4))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        video.write(img[..., :3])

    video.release()
    video_meta = finalize_web_video(temp_output_path, output_path)
    write_status(args, "ready", extra=video_meta)
    print(f"RECORDING_DONE: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    try:
        record(args)
    except Exception as exc:
        write_status(args, "failed", str(exc))
        raise
