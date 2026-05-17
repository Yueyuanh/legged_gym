# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is **deprecated**. It has been migrated to [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and receives limited updates. It is a legged robot locomotion training framework using NVIDIA Isaac Gym with PPO (from `rsl_rl`).

## Commands

```bash
# Train a policy
python legged_gym/scripts/train.py --task=anymal_c_flat

# Train on CPU
python legged_gym/scripts/train.py --task=anymal_c_flat --sim_device=cpu --rl_device=cpu

# Train headless (no rendering)
python legged_gym/scripts/train.py --task=anymal_c_flat --headless

# Resume training from last checkpoint
python legged_gym/scripts/train.py --task=anymal_c_flat --resume

# Play a trained policy
python legged_gym/scripts/play.py --task=anymal_c_flat

# Run the test (creates env, steps with zero actions)
python legged_gym/tests/test_env.py --task=anymal_c_flat

# Override config via CLI
python legged_gym/scripts/train.py --task=anymal_c_flat --num_envs=2048 --seed=42 --max_iterations=500 --experiment_name=my_exp --run_name=v1
```

**Registered tasks**: `anymal_c_rough`, `anymal_c_flat`, `anymal_b`, `a1`, `cassie`, `go2`

## Architecture

### Class hierarchy

```
BaseConfig              (legged_gym/envs/base/base_config.py)
├── LeggedRobotCfg      (legged_gym/envs/base/legged_robot_config.py)  — env parameters
├── LeggedRobotCfgPPO   — training/PPO parameters
    └── (robot-specific configs inherit and override these)

BaseTask               (legged_gym/envs/base/base_task.py)  — Isaac Gym lifecycle (create_sim, render)
└── LeggedRobot         (legged_gym/envs/base/legged_robot.py)  — locomotion task, rewards, curriculum
    ├── Anymal          (legged_gym/envs/anymal_c/anymal.py)  — adds actuator network (LSTM-based SEA model)
    └── Cassie          (legged_gym/envs/cassie/cassie.py)
```

### Key design patterns

- **Config auto-instantiation**: `BaseConfig.__init__` recursively instantiates nested classes into objects. You define config as nested classes; the framework auto-instantiates them. Do not manually instantiate nested config classes.
- **Reward functions**: Each non-zero scale in `cfg.rewards.scales` is matched to a method named `_reward_<name>` via `getattr`. Set a scale to zero to disable a reward.
- **Task registration**: `task_registry.register(name, EnvClass, EnvCfg, TrainCfg)` in `legged_gym/envs/__init__.py`. New tasks must be registered there (or externally).
- **Callbacks**: `LeggedRobot` exposes `_process_rigid_shape_props`, `_process_dof_props`, `_process_rigid_body_props`, and `_post_physics_step_callback` for per-env randomization and custom logic.
- **Control modes**: Configurable via `cfg.control.control_type` — `'P'` (position PD), `'V'` (velocity PD), `'T'` (direct torque).

### Key files

| File | Role |
|---|---|
| `legged_gym/envs/base/legged_robot.py` | Core locomotion env: simulation, rewards, curricula, observations |
| `legged_gym/envs/base/legged_robot_config.py` | All config defaults (`LeggedRobotCfg`, `LeggedRobotCfgPPO`) |
| `legged_gym/envs/base/base_config.py` | Config base class with recursive instantiation |
| `legged_gym/utils/task_registry.py` | Singleton registry: `make_env()`, `make_alg_runner()` |
| `legged_gym/utils/helpers.py` | CLI args, config overrides, checkpoint loading, JIT export |
| `legged_gym/utils/terrain.py` | Procedural terrain generation (curriculum, randomized, selected) |
| `legged_gym/envs/__init__.py` | All task registrations |
| `legged_gym/__init__.py` | Defines `LEGGED_GYM_ROOT_DIR` and `LEGGED_GYM_ENVS_DIR` |

### Observation structure (rough terrain, 235 dims)

`[lin_vel(3), ang_vel(3), projected_gravity(3), commands(3), dof_pos(12), dof_vel(12), last_actions(12), height_samples(187)]`

Flat terrain tasks use 48 dims (no height samples). Observation scaling and noise are configured under `cfg.normalization` and `cfg.noise`.

### Observation noise configuration

Noise is added as `uniform(-1,1) * noise_scale_vec`, where `noise_scale_vec` is computed in `_get_noise_scale_vec()`. Noise scaling per observation group is in `cfg.noise.noise_scales`. The global `cfg.noise.noise_level` multiplies all noise scales. Noise is disabled during play via `env_cfg.noise.add_noise = False`.

### Dependencies

- **isaacgym** (Isaac Gym Preview 3 — Preview 2 will not work)
- **rsl_rl** (PPO implementation, v1.0.2)
- PyTorch 1.10 with CUDA 11.3
- Python 3.6–3.8
- matplotlib
