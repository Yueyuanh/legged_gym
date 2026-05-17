# Plan-tune — 手动实验序列管理

通过 Web UI 手动定义 reward scale 实验序列，按序执行并对比结果。

**零额外 Python 依赖**（仅需 Python stdlib）。

## 快速开始

```bash
conda activate rl

# 启动 Web UI
python legged_gym/tune/plan_tune/plan_tune_dashboard.py

# 自定义端口
python legged_gym/tune/plan_tune/plan_tune_dashboard.py --port 8070

# 指定已有输出目录
python legged_gym/tune/plan_tune/plan_tune_dashboard.py --output-dir logs/plan_tune/go2_xxx
```

启动后打开 `http://localhost:8060`。

## Web UI 操作流程

1. **选择任务**：从下拉菜单选择 `go2` 等注册任务
2. **填写实验组名**：用于生成输出目录
3. **设置参数**：num_envs、headless 等
4. **点击 "Load Reward Scales"**：从任务配置中加载当前默认 scales
5. **编辑 "Planned" 列**：为每个 reward 项设置目标值
6. **点击 "Add Experiment"**：添加单次实验到列表
7. **点击 "Generate Experiment Sequence"**：保存序列为 JSON
8. **点击 "Start Training"**：按序执行所有实验

## Reward Range 批量生成

使用 "Reward Range" 模态框可批量生成实验：
- 选择要变化的 reward 项
- 设定范围、步长
- 自动生成嵌套循环的实验组合

## 离线监控

```bash
python legged_gym/tune/plan_tune/plan_tune_monitor.py \
    --output-dir logs/plan_tune/go2_YYYYMMDD_HHMMSS
```

## 直接运行序列（无需 Web UI）

```bash
python legged_gym/tune/plan_tune/plan_tune_runner.py \
    --sequence logs/plan_tune/go2_xxx/experiment_sequence.json
```

## 实验序列 JSON 格式

```json
{
  "task": "go2",
  "experiment_name": "go2_PlanTune",
  "num_envs": 4096,
  "headless": true,
  "experiments": [
    {
      "id": 0,
      "name": "baseline",
      "iterations": 500,
      "reward_scales": {"tracking_lin_vel": 1.0}
    },
    {
      "id": 1,
      "name": "high_tracking",
      "iterations": 500,
      "reward_scales": {"tracking_lin_vel": 2.0}
    }
  ]
}
```

## 输出结构

```
logs/plan_tune/<task>_<timestamp>/
├── dashboard_state/          # Web UI 实时状态
├── experiment_sequence.json  # 实验序列定义
├── trial_best_models/        # 各实验最优 checkpoint
├── videos/                   # 视频录制（可选）
├── training_report.md        # Markdown 报告
└── training_report.txt       # 纯文本报告
```

## API 端点

| 路径 | 用途 |
|------|------|
| `GET /` | Web UI 首页 |
| `GET /api/config` | 获取运行目录和默认输出 |
| `GET /api/tasks` | 列出注册的任务 |
| `GET /api/reward_scales?task=X` | 获取任务默认 reward scales |
| `GET /api/load_sequence?path=X` | 加载已有实验序列 |
| `GET /api/snapshot?output_dir=X` | 获取 dashboard 实时状态 |
| `POST /api/generate` | 生成实验序列 JSON |
| `POST /api/start` | 启动训练 |
| `GET /videos/*` | 视频文件服务 |
