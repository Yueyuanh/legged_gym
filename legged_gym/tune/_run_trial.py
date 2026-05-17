# SPDX-License-Identifier: BSD-3-Clause
"""
Single-trial training runner for auto_tune_rewards.py.

Called as a subprocess. Reward scale overrides are passed via the
environment variable REWARD_CONFIG_PATH pointing to a JSON file.
All other arguments use the standard train.py CLI (gymutil.parse_arguments).

IMPORTANT: Import order is critical for Isaac Gym — isaacgym must be loaded
BEFORE torch. This file ensures legged_gym.envs (which triggers isaacgym) is
imported before legged_gym.utils (which imports torch).
"""

import sys
import json
import os


def main():
    # ---- Path setup: ensure THIS project's legged_gym package is used ----
    # The user may have multiple copies of the project installed.
    # Resolve the repo root (parent of legged_gym/) and add to sys.path.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))  # goes up from tune/
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Read reward config path from environment variable (set by auto_tune_rewards.py)
    reward_config_path = os.environ.get("REWARD_CONFIG_PATH", "")
    if not reward_config_path or not os.path.exists(reward_config_path):
        print(f"TRIAL_ERROR: REWARD_CONFIG_PATH not set or file not found: "
              f"{reward_config_path}")
        sys.exit(1)

    # ---- Isaac Gym import order workaround ----
    # Import legged_gym.envs FIRST — it imports isaacgym before torch.
    # Without this, legged_gym.utils.helpers imports torch first, triggering
    # "PyTorch was imported before isaacgym modules" error.
    from legged_gym.envs import task_registry  # noqa: F401 — triggers isaacgym
    # Now safe to import modules that import torch:
    from legged_gym.utils import get_args
    from legged_gym.utils.helpers import update_cfg_from_args

    args = get_args()

    print(f"TRIAL_INFO: task={args.task}, repo={repo_root}, "
          f"max_iters={args.max_iterations}, headless={args.headless}, "
          f"num_envs={args.num_envs}")

    # Retrieve configs
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)

    # Apply reward scale overrides BEFORE making env (scales are read in _parse_cfg)
    _apply_reward_overrides(env_cfg, reward_config_path)

    # Apply CLI overrides to configs
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

    # Keep a handful of checkpoints per trial so auto_tune_rewards.py can
    # preserve the best saved policy for every reward-scale sample.
    train_cfg.runner.save_interval = max(1, train_cfg.runner.max_iterations // 5)
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs

    # Create env and runner
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )

    if ppo_runner.log_dir is not None:
        print(f"TRIAL_LOG_DIR: {ppo_runner.log_dir}")

    # Run training
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )

    # Signal completion and list all checkpoint models
    if ppo_runner.log_dir is not None:
        model_files = sorted(
            [f for f in os.listdir(ppo_runner.log_dir)
             if f.startswith("model_") and f.endswith(".pt")],
            key=lambda x: int(x.replace("model_", "").replace(".pt", "")),
        )
        for mf in model_files:
            iter_num = mf.replace("model_", "").replace(".pt", "")
            model_path = os.path.join(ppo_runner.log_dir, mf)
            print(f"TRIAL_MODEL: {iter_num}:{model_path}")
        if model_files:
            final_model = os.path.join(ppo_runner.log_dir, model_files[-1])
            print(f"TRIAL_FINAL_MODEL: {final_model}")
        print(f"TRIAL_COMPLETE: {ppo_runner.log_dir}")


def _apply_reward_overrides(env_cfg, overrides_path: str):
    with open(overrides_path, "r") as f:
        overrides = json.load(f)
    applied = 0
    for key, value in overrides.items():
        if not hasattr(env_cfg.rewards.scales, key):
            print(f"TRIAL_WARNING: unknown reward scale '{key}', skipping")
            continue
        setattr(env_cfg.rewards.scales, key, float(value))
        applied += 1
    print(f"TRIAL_INFO: applied {applied} reward scale overrides "
          f"from {overrides_path}")


if __name__ == "__main__":
    main()
