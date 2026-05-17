# Implementation Plan — legged_gym 新功能移植

> 基于 `task.md` 中的原则与 TODO，参考 `pikachu_humanoid_gym` 和 `pikachu_gym` 的实现。

---

## 约束原则

1. **仅在 `legged_gym/` 内新增文件**，不修改任何框架原有代码（`envs/base/`, `scripts/train.py`, `scripts/play.py`, `envs/__init__.py` 等均不动）
2. 每个功能独立文件夹，代码简洁高效
3. 每个功能提供一个 md 使用文档（仅 tune 相关功能需要，keyboard/projectile 用法写在代码注释中）
4. 测试在 conda `rl` 环境中进行

---

## TODO 1: Auto-tune / Plan-tune 复现

### 目标
将 `pikachu_humanoid_gym` 的 auto_tune（自动搜索最优 reward scales）和 plan_tune（手动序列实验管理）移植到 legged_gym。

### 文件清单

```
legged_gym/legged_gym/tune/
├── TUNE.md                          # 总使用文档（auto_tune + plan_tune 说明与使用方法）
├── _run_trial.py                    # 单次 trial 子进程执行器（auto/plan 共用）
├── _record_video.py                 # 异步视频录制子进程（auto/plan 共用）
├── auto_tune/
│   ├── AUTO_TUNE.md                 # Auto-tune 使用文档
│   ├── auto_tune_rewards.py         # Auto-tune 主入口（采样、评分、搜索、报告）
│   ├── auto_tune_dashboard.py       # Dash 实时看板（复用原 UI）
│   └── auto_tune_monitor.py         # 离线监控启动器
└── plan_tune/
    ├── PLAN_TUNE.md                 # Plan-tune 使用文档
    ├── plan_tune_dashboard.py       # HTTP 服务后端（纯 stdlib，零额外依赖）
    ├── plan_tune_dashboard.html     # Web 前端（单文件 HTML/CSS/JS，复用原 UI）
    ├── plan_tune_runner.py          # 序列执行器
    └── plan_tune_monitor.py         # 离线监控启动器
```

### 各文件适配说明

#### 共用文件（`tune/` 根目录）

| 文件 | 来源 | 适配要点 |
|------|------|---------|
| `_run_trial.py` | `pikachu_humanoid_gym/humanoid/scripts/_run_trial.py` | import `humanoid` → `legged_gym`；使用 legged_gym 的 `task_registry`；`get_args()` 对齐 legged_gym CLI |
| `_record_video.py` | `pikachu_humanoid_gym/humanoid/scripts/_record_video.py` | import 路径替换，其余不变 |

#### `auto_tune/` 文件夹

| 文件 | 来源 | 适配要点 |
|------|------|---------|
| `auto_tune_rewards.py` | `humanoid/scripts/auto_tune_rewards.py` | `DEFAULT_SEARCH_SPACE` 中 reward 名称对齐 `LeggedRobotCfg.rewards.scales` 字段；子进程路径指向 `tune/_run_trial.py`；输出目录 `tune_output/<task>_<timestamp>/` |
| `auto_tune_dashboard.py` | `humanoid/scripts/auto_tune_dashboard.py` | **UI 完全复用**（Dash + Dash-Table + Plotly，深色主题）；状态/视频路径对齐；端口 8050 |
| `auto_tune_monitor.py` | `humanoid/scripts/auto_tune_monitor.py` | import 路径替换，其余不变 |

#### `plan_tune/` 文件夹

| 文件 | 来源 | 适配要点 |
|------|------|---------|
| `plan_tune_dashboard.py` | `humanoid/scripts/plan_tune/plan_tune_dashboard.py` | API `/api/reward_scales` 调用 legged_gym task_registry 获取默认 scales；端口 8060 |
| `plan_tune_dashboard.html` | `humanoid/scripts/plan_tune/plan_tune_dashboard.html` | **直接复制**（约 1839 行单文件 HTML） |
| `plan_tune_runner.py` | `humanoid/scripts/plan_tune/plan_tune_runner.py` | 子进程路径指向 `tune/_run_trial.py`；输出目录 `tune/plan_tune/runs/<task>_<timestamp>/` |
| `plan_tune_monitor.py` | `humanoid/scripts/plan_tune/plan_tune_monitor.py` | import 路径替换，其余不变 |

### 依赖
- Auto-tune dashboard: `dash`, `dash-table`, `flask`, `plotly`
- 报告图表: `matplotlib`（可选）
- 视频录制: `ffmpeg`（可选）
- Plan-tune: **零额外 Python 依赖**（仅 stdlib：`http.server`, `json`, `subprocess`）

---

## TODO 2: 键盘控制 Play（无感形式）

### 目标
替换 pygame 窗口方案，使用 `pynput` 实现全局键盘监听——无需窗口，无需焦点切换，完全"无感"。

### 方案选择

| 方案 | 是否需要窗口 | 是否需要焦点 | 跨平台 |
|------|-------------|-------------|--------|
| pygame（原方案） | 是（100x100 窗口） | 是 | 是 |
| **pynput（选择）** | **否** | **否（全局监听）** | **是** |
| curses | 否（终端内） | 是 | 仅 Unix |
| evdev | 否 | 否 | 仅 Linux |

**选择 pynput**：操作系统级全局 hook，后台 daemon 线程，零 UI，与 Isaac Gym viewer 完美共存。

### 文件清单

```
legged_gym/legged_gym/utils/
├── keyboard_commander.py            # 键盘命令控制器（pynput 全局监听）

legged_gym/legged_gym/scripts/
├── play_keyboard.py                 # 键盘控制 play 脚本（用法以注释形式写在文件头部）
```

不单独创建 md 文档，使用方法以详细注释写在 `play_keyboard.py` 文件头部。

### 核心设计

#### `keyboard_commander.py`

```
class KeyboardCommander:
    """
    按键映射（WASD + QE）：
        W/S   — 前/后 (lin_vel_x)
        A/D   — 左/右 (lin_vel_y)
        Q/E   — 左转/右转 (heading)
        Space — 急停 / 发射弹射物
        R     — 重置弹射物

    实现：
        - pynput.keyboard.Listener，daemon thread
        - 维护共享 command_dict = {lin_vel_x, lin_vel_y, heading_target}
        - get_commands() 在主线程中读取当前值
        - start() / stop() 控制生命周期
    """
```

#### `play_keyboard.py`

与 `play.py` 独立，额外做三件事：
1. 创建 `KeyboardCommander` 并启动
2. 设置 `env_cfg.commands.resampling_time = 9999`（禁用随机命令重采样，让键盘接管）
3. 循环中写入 `env.commands[:, 0:3] = kb.get_commands()`

### 依赖
- `pynput`（`pip install pynput`）

---

## TODO 3: Projectile 弹射物功能

### 目标
将 `pikachu_gym` 的弹射物系统独立移植，集成到 go2 play 中。

### 文件清单

```
legged_gym/legged_gym/utils/
├── projectile_manager.py            # 弹射物管理器（直接复制，零修改）

legged_gym/legged_gym/envs/go2/
├── go2_config.py                    # 【唯一改动文件】：新增 projectile 配置类

legged_gym/legged_gym/scripts/
├── play_keyboard.py                 # 集成弹射物（空格发射，R 重置，用法见文件头部注释）
```

不单独创建 md 文档，使用方法以注释形式写在 `play_keyboard.py` 和 `projectile_manager.py` 文件头部。

### 核心设计

#### `projectile_manager.py`
- **直接复制** `pikachu_gym/legged_gym/utils/projectile_manager.py`
- 纯 isaacgym + numpy + torch 依赖，零业务耦合，无需任何修改

#### `go2_config.py` 新增内容
```python
class Go2RoughCfg(LeggedRobotCfg):
    # ... 现有配置保持不变 ...

    class projectile:          # 新增
        enable = False         # 默认关闭，play 时手动开启
        num_projectiles = 10
        box_size = 0.2
        density = 10.0
        fire_speed = 25.0
        add_random_spin = True
```

#### `play_keyboard.py` 集成方式
```python
# 创建 env 后，外部挂载 projectile manager（不改动框架代码）
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

from legged_gym.utils.projectile_manager import ProjectileManager
pm = ProjectileManager(env.gym, env.sim,
                       num_projectiles=env_cfg.projectile.num_projectiles,
                       box_size=env_cfg.projectile.box_size,
                       density=env_cfg.projectile.density)

# 重新获取 root states tensor 并绑定
actor_root_state = env.gym.acquire_actor_root_state_tensor(env.sim)
all_root_states = gymtorch.wrap_tensor(actor_root_state)
pm.bind_state_tensors(all_root_states, env.device)

# 循环中：空格 = pm.fire_projectile_from_camera(env.viewer, ...)
#        R = pm.reset_all_projectiles()
```

### 关键：不修改框架代码的原理
- ProjectileManager 以 `group=-1` 创建弹射物（全局碰撞），放入独立 `proj_env`
- 通过 `gym.acquire_actor_root_state_tensor()` + `gymtorch.wrap_tensor()` 外部重新获取 GPU tensor 并绑定
- 发射/重置在 play 循环中直接调用，无需框架感知

---

## 实施顺序

```
Phase 1: TODO 3 (Projectile) — 最简单，独立性强
    ├── 1.1 复制 projectile_manager.py 到 utils/
    ├── 1.2 在 go2_config.py 新增 projectile 配置类
    └── 1.3 测试：conda rl 中验证弹射物发射/碰撞

Phase 2: TODO 2 (Keyboard Control) — 中等复杂度
    ├── 2.1 实现 keyboard_commander.py（pynput 全局监听）
    ├── 2.2 创建 play_keyboard.py（整合 projectile + keyboard，头部写清用法注释）
    └── 2.3 测试：WASD 控制机器人，Q/E 转向，空格发射弹射物

Phase 3: TODO 1 (Auto-tune / Plan-tune) — 最大工作量
    ├── 3.1 适配 _run_trial.py 到 legged_gym
    ├── 3.2 移植 auto_tune/ 文件夹（rewards + dashboard + monitor）
    ├── 3.3 移植 plan_tune/ 文件夹（dashboard + html + runner + monitor）
    ├── 3.4 写 TUNE.md + AUTO_TUNE.md + PLAN_TUNE.md
    └── 3.5 测试：headless 小规模 auto_tune 验证 dashboard；启动 plan_tune Web UI 验证交互
```

---

## 测试验证计划

| 功能 | 测试命令 | 验收标准 |
|------|---------|---------|
| Projectile | `python legged_gym/scripts/play_keyboard.py --task go2` | 空格发射方块与机器人碰撞，R 重置 |
| Keyboard | 同上 | WASD 控制前后左右，Q/E 转向，无 pygame 窗口 |
| Auto-tune | `python legged_gym/tune/auto_tune/auto_tune_rewards.py --task go2 --trials 5 --iterations 200 --num-envs 2048 --headless --dashboard` | Dashboard :8050 可访问，trial 正常完成 |
| Plan-tune | `python legged_gym/tune/plan_tune/plan_tune_dashboard.py` | Web UI :8060 可加载 reward scales、生成序列、启动训练 |

所有测试在 `conda activate rl` 环境中执行。
