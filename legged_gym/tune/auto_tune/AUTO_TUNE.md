# Auto-tune — 自动 Reward Scale 搜索

自动采样 reward scale 组合，运行短 PPO 训练，保留最优模型。

## 快速开始

```bash
conda activate rl

# 小规模测试（5 次 trial，200 迭代）
python legged_gym/tune/auto_tune/auto_tune_rewards.py \
    --task go2 \
    --trials 5 \
    --iterations 200 \
    --num-envs 2048 \
    --headless \
    --dashboard

# 正式搜索（20 次 trial，500 迭代）
python legged_gym/tune/auto_tune/auto_tune_rewards.py \
    --task go2 \
    --trials 20 \
    --iterations 500 \
    --num-envs 4096 \
    --headless \
    --dashboard \
    --tensorboard

# 断点续跑
python legged_gym/tune/auto_tune/auto_tune_rewards.py \
    --task go2 \
    --resume logs/auto_tune/go2_YYYYMMDD_HHMMSS \
    --headless --dashboard

# 离线监控
python legged_gym/tune/auto_tune/auto_tune_monitor.py \
    --output-dir logs/auto_tune/go2_YYYYMMDD_HHMMSS
```

## 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务名（go2, anymal_c_flat 等） | go2 |
| `--trials` | trial 总次数 | 20 |
| `--iterations` | 每次 trial 的迭代数 | 500 |
| `--num-envs` | 并行环境数 | 4096 |
| `--headless` | 无渲染模式 | False |
| `--dashboard` | 启动 Dash 实时看板 | False |
| `--tensorboard` | 启动 TensorBoard | False |
| `--resume` | 断点续跑目录 | None |
| `--init-best-config` | 初始最优 config JSON | None |

## Dashboard 访问

启动后打开 `http://localhost:8050`，查看：
- 所有 trial 的 reward 曲线
- 分数演化趋势
- 各 reward 项贡献分解

## 搜索空间

自动搜索以下 reward scale：

| Reward | 范围 | 说明 |
|--------|------|------|
| `tracking_lin_vel` | 0.5 - 5.0 | 线速度跟踪 |
| `tracking_ang_vel` | 0.1 - 3.0 | 角速度跟踪 |
| `orientation` | 0.1 - 1.0 | 姿态平整度 |
| `base_height` | 0.1 - 1.0 | 目标高度跟踪 |
| `feet_air_time` | 0.5 - 3.0 | 足端腾空时间 |

其余 reward 保持默认值不变。

## 评分策略

综合评分 = 基础总 reward × 0.2 + 速度跟踪加权分 − 门限惩罚

- `tracking_lin_vel` 权重 110，门限 0.3
- `tracking_ang_vel` 权重 45，门限 0.1

优先保障速度跟踪性能。

## 输出结构

```
logs/auto_tune/go2_YYYYMMDD_HHMMSS/
├── dashboard_state/         # Dash 看板实时状态
├── trial_configs/           # 每次 trial 的 reward config
├── trial_best_models/       # 每次 trial 的最优 checkpoint
├── best_model/              # 全局最优模型
├── best_config.json         # 全局最优配置
├── tuning_report.txt        # 文字报告
├── reward_curves.png        # 图表
└── dashboard.log            # Dash 日志
```
