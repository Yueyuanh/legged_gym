"""
Keyboard-controlled play script with optional projectile support.

Usage:
    # Basic keyboard control (no projectile):
    python legged_gym/scripts/play_keyboard.py --task=go2

    # With projectile enabled:
    python legged_gym/scripts/play_keyboard.py --task=go2 --projectile

    # Other tasks:
    python legged_gym/scripts/play_keyboard.py --task=anymal_c_flat

Controls:
    W/S         — forward / backward
    A/D         — left / right (strafe)
    Q/E         — turn left / right (heading)
    Space       — fire a projectile from camera (if --projectile)
    R           — reset all projectiles to hidden positions (if --projectile)
    Esc         — quit

Features:
    - Uses pynput for global keyboard listening (no pygame window needed).
    - Keyboard works regardless of which window has focus.
    - Projectile support via Go2Env subclass — does not modify framework code.
    - ProjectileManager is created during create_sim() so tensor shapes are correct.

Dependencies:
    pip install pynput
"""

import os
import sys

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.envs.go2.go2_env import Go2Env
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.keyboard_commander import KeyboardCommander

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ---- override for play mode ----
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # disable random command resampling so keyboard takes full control
    env_cfg.commands.resampling_time = 9999.0
    env_cfg.commands.heading_command = True

    # enable projectile if requested — re-register "go2" with Go2Env which
    # creates projectiles during create_sim() so tensor shapes are correct.
    use_projectile = getattr(args, "projectile", False)
    if use_projectile:
        if not hasattr(env_cfg, "projectile"):
            print("WARNING: projectile config not found in env_cfg, using defaults")
        else:
            env_cfg.projectile.enable = True
        task_registry.register(args.task, Go2Env, env_cfg, train_cfg)

    # ---- create environment ----
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # ---- keyboard commander ----
    kb = KeyboardCommander(
        lin_vel_x_range=env.command_ranges["lin_vel_x"],
        lin_vel_y_range=env.command_ranges["lin_vel_y"],
        heading_step=0.05,
    )
    kb.start()

    # ---- projectile manager (created inside Go2Env) ----
    pm = env.projectile_manager if use_projectile else None

    # ---- load policy ----
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to:", path)

    # ---- play loop ----
    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 100
    stop_rew_log = env.max_episode_length + 1
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    total_steps = 10 * int(env.max_episode_length)
    print(f"\nKeyboard play started. Controls: WASD=move, QE=turn, Esc=quit")
    if pm:
        print(f"Projectiles enabled. Space=fire, R=reset all\n")

    for i in range(total_steps):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # ---- keyboard commands (override env's resampled commands) ----
        cmd = kb.get_commands()
        if cmd.quit:
            break
        env.commands[:, 0] = cmd.lin_vel_x
        env.commands[:, 1] = cmd.lin_vel_y
        env.commands[:, 3] = cmd.heading_target

        # ---- projectile actions ----
        if pm is not None and cmd.action:
            pm.fire_projectile_from_camera(
                env.viewer,
                speed=env_cfg.projectile.fire_speed,
                add_random_spin=env_cfg.projectile.add_random_spin,
            )
        if pm is not None and cmd.reset:
            pm.reset_all_projectiles()

        # ---- optional frame recording ----
        if RECORD_FRAMES and i % 2 == 0:
            filename = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                "logs",
                train_cfg.runner.experiment_name,
                "exported",
                "frames",
                f"{img_idx}.png",
            )
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img_idx += 1

        # ---- optional camera movement ----
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # ---- logging ----
        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item()
                    * env.cfg.control.action_scale,
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

    # ---- cleanup ----
    kb.stop()
    print("Keyboard play finished.")


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    # extract --projectile from argv before get_args() (not a standard legged_gym arg)
    _use_projectile = False
    if "--projectile" in sys.argv:
        _use_projectile = True
        sys.argv.remove("--projectile")

    args = get_args()
    args.projectile = _use_projectile
    play(args)
