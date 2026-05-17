# Tune — Hyperparameter Tuning for legged_gym

本目录包含两个独立的 reward scale 调优工具，均通过子进程执行 PPO 训练 trial。

## 目录结构

```
tune/
├── TUNE.md                    # 本文件
├── _run_trial.py              # 单次 trial 子进程执行器（auto/plan 共用）
├── _record_video.py           # 异步视频录制子进程
├── auto_tune/                 # 自动搜索最优 reward scales
│   ├── AUTO_TUNE.md
│   ├── auto_tune_rewards.py
│   ├── auto_tune_dashboard.py
│   └── auto_tune_monitor.py
└── plan_tune/                 # 手动序列实验管理
    ├── PLAN_TUNE.md
    ├── plan_tune_dashboard.py
    ├── plan_tune_dashboard.html
    ├── plan_tune_runner.py
    └── plan_tune_monitor.py
```

## 安装依赖

```bash
conda activate rl

# auto_tune dashboard
pip install dash dash-table flask plotly

# 可选
pip install matplotlib  # 报告图表
pip install pynput      # 键盘控制（play_keyboard.py）
```

## 两个工具的区别

| 特性 | Auto-tune | Plan-tune |
|------|-----------|-----------|
| 策略 | 自适应采样，自动搜索 | 用户手动定义实验序列 |
| UI | Dash Web 前端（:8050） | 纯 HTML/CSS/JS（:8060） |
| 适用场景 | 自动探索未知 reward 空间 | 对比已知候选配置 |
| 依赖 | dash, plotly, flask | 仅 Python stdlib |

## 测试

所有工具在 conda `rl` 环境中测试：

```bash
conda activate rl
python legged_gym/tune/auto_tune/auto_tune_rewards.py --task go2 --trials 5 --iterations 200 --num-envs 2048 --headless --dashboard
```

## 输出目录

- Auto-tune: `logs/auto_tune/<task>_<timestamp>/`
- Plan-tune: `logs/plan_tune/<task>_<timestamp>/`
