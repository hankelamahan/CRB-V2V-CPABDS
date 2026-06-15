# LSTM 增强模块集成说明

## 概述

本模块为 CRB-V2V-CPABDS 系统添加了两个基于 LSTM 的异常检测增强：

| 模块 | 文件 | 功能 |
|------|------|------|
| 模块一：物理轨迹 LSTM | `LSTM_enhance/physics_lstm.py` | 基于历史运动状态预测下一帧位置，与 IMM 加权融合得出物理一致性分数 |
| 模块二：信誉序列 LSTM | `LSTM_enhance/reputation_lstm.py` | 基于历史信誉序列预测下一个信誉值，检测持证作恶行为 |

两个模块与现有组件的关系：

- **模块一** 是 IMM（交互多模型滤波器）的下游增强。IMM 提供基于卡尔曼滤波的基础残差，LSTM 识别 IMM 难以捕捉的细粒度或未知模式攻击，最终通过加权融合两者的一致性分数，得出统一的物理一致性分数。
- **模块二** 与 DIVA 模块是上下游协同关系。DIVA 专注于当前时刻的多维感知数据验证，信誉 LSTM 关注长期行为趋势，在 RSU 信誉更新环节提供持证作恶预警。

---

## 文件结构

```
CRB-V2V-CPABDS/
├── LSTM_enhance/
│   ├── physics_lstm.py               # 物理轨迹 LSTM 模型 + 预测器 + 训练函数
│   ├── reputation_lstm.py            # 信誉序列 LSTM 模型 + 预测器 + 训练函数
│   └── TODO.txt
├── physical_consistency/
│   ├── imm_core.py                   # KalmanFilter + IMM 实现
│   ├── imm_manager.py                # 多车辆 IMM 实例管理
│   ├── intermediate_fusion_manager.py # 分数融合 + 信誉更新（已接入物理 LSTM）
│   ├── main.py                       # 仿真主循环（已接入 PhysicsPredictor）
│   ├── data.py                       # 仿真数据生成
│   ├── utils.py                      # physical_score / trajectory_score / rsu_score
│   └── visualizer.py                 # 仿真结果可视化
├── RSU/
│   └── reputation_center_server.py   # FastAPI 服务端（已集成信誉 LSTM）
├── intermediate_fusion_dataset.py    # OpenCOOD 数据集类（已集成物理 LSTM）
├── physics_lstm.pth                  # 物理模型权重（需训练生成）
└── reputation_lstm.pth               # 信誉模型权重（需训练生成）
```

---

## 模块一：物理轨迹 LSTM

### 核心原理

IMM 残差反映的是当前帧与卡尔曼预测的偏差，擅长检测突变型攻击（位置跳变），但对低速累积漂移或周期性欺骗的敏感度有限。LSTM 通过学习正常车辆的长期轨迹模式，能捕捉这类 IMM 难以覆盖的异常。

最终物理一致性分数由两者加权合成：

```
physical_score = (1 - w) × imm_physical_score + w × (1 - lstm_anomaly_score)
```

其中 `w = lstm_imm_weight`，默认值 `0.4`（可在初始化时配置）。

### 集成位置 1：仿真流水线

`physical_consistency/intermediate_fusion_manager.py` — `IntermediateFusionManager`

`physical_consistency/main.py` 主循环负责提取状态向量并调用 LSTM，再将异常分数传入 `IntermediateFusionManager.compute_all_scores()`。

**`IntermediateFusionManager` 初始化：**

```python
from intermediate_fusion_manager import IntermediateFusionManager

fusion_manager = IntermediateFusionManager(lstm_imm_weight=0.4)
# lstm_imm_weight=0.0 退化为纯 IMM，向后兼容
```

**每帧调用：**

```python
physics_predictor.update_history(vid, [pos[0], pos[1], vel[0], vel[1]])
lstm_anomaly = physics_predictor.compute_anomaly_score(vid, [pos[0], pos[1]])

scores = fusion_manager.compute_all_scores(
    residual=imm_residual,
    vid=vid,
    vel=vel,
    lstm_anomaly_score=lstm_anomaly   # 新增参数，默认 0.0
)
# scores 包含：physical, imm_physical, lstm_anomaly, trajectory, rsu, fused
```

`compute_all_scores` 的 `lstm_anomaly_score` 参数默认为 `0.0`，不传时行为与改造前完全一致。

### 集成位置 2：OpenCOOD 数据集

`intermediate_fusion_dataset.py` — `IntermediateFusionDataset`

每帧处理每辆 CAV 时：

1. 从 `lidar_pose` 和 `ego_speed` 提取运动状态 `[x, y, vx, vy]`
2. 调用 `physics_predictor.update_history(cav_id, state)` 更新历史缓冲
3. 调用 `physics_predictor.compute_anomaly_score(cav_id, [x, y])` 计算异常分数（0~1）
4. 将异常分数作为 `phy_score` 通过 `reputation_adapter.report_fused_boxes(...)` 上报给 RSU

### YAML 配置（OpenCOOD）

在实验配置文件中添加：

```yaml
physics_lstm:
  enabled: true
  model_path: "physics_lstm.pth"   # 相对于 intermediate_fusion_dataset.py，或绝对路径
  seq_len: 10                       # 历史序列长度
  input_dim: 4                      # 输入维度：[x, y, vx, vy]
  max_position_error: 10.0          # 异常分数归一化上限（米）
```

若不配置 `physics_lstm` 字段，默认 `enabled=true`，模型路径为同目录下的 `physics_lstm.pth`。

### 训练

```python
from LSTM_enhance.physics_lstm import train_physics_lstm
import numpy as np

# list of np.ndarray，每个 shape (T, 4)，列为 [x, y, vx, vy]
trajectories = [...]

train_physics_lstm(
    trajectory_list=trajectories,
    seq_len=10,
    input_dim=4,
    epochs=50,
    save_path="physics_lstm.pth"
)
```

---

## 模块二：信誉序列 LSTM

### 核心原理

DIVA 在每个时刻对车辆上报的感知数据做多维验证（位置、速度、传感器数据一致性等），验证结果驱动 RSU 的信誉分增减。信誉 LSTM 在此基础上关注更长时间跨度的信誉序列趋势，专门针对"持证作恶"模式：车辆长期维持高信誉，待积累足够信任后突然执行欺骗行为。这类攻击在 DIVA 的单帧视角下难以提前识别。

### 集成位置

`RSU/reputation_center_server.py` — `update_reputation()` 函数

### 工作流程

每次信誉更新时：

1. 按正常逻辑计算 `new_rep`（±delta + clamp）
2. 调用 `predictor.update_history(vehicle_id, new_rep)` 更新历史
3. 调用 `predictor.check_anomaly(vehicle_id, new_rep)` 检测是否异常
4. 若异常（预测高但实际骤降），额外扣减 `0.05` 并重新 clamp
5. 写入数据库

### 早期预警 API

```
GET /early_warning/{vehicle_id}
```

返回：

```json
{
  "vehicle_id": "car_001",
  "early_warning_score": 0.73
}
```

预警分数基于近 5 步信誉波动性计算（0~1），越高表示行为越可疑。可作为独立监控指标，也可用于触发额外的感知数据核查。

### 模型路径

服务器启动时自动在 `RSU/` 目录下查找 `reputation_lstm.pth`。如需修改，编辑：

```python
_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reputation_lstm.pth")
```

### 训练

```python
from LSTM_enhance.reputation_lstm import train_reputation_lstm

# list of list，每个子列表为一辆车的历史信誉序列
reputation_sequences = [
    [0.5, 0.55, 0.6, 0.65, ...],
    [0.7, 0.68, 0.72, ...],
]

train_reputation_lstm(
    reputation_sequences=reputation_sequences,
    seq_len=10,
    epochs=50,
    save_path="RSU/reputation_lstm.pth"
)
```

---

## 准备工作

### 依赖安装

```bash
pip install torch numpy
```

### 训练模型

两个模型均需在正常数据上预训练。未训练时使用随机权重，异常分数为噪声。

**使用内置模拟数据快速验证：**

```bash
# 生成物理轨迹模型权重 → physics_lstm.pth
python LSTM_enhance/physics_lstm.py

# 生成信誉序列模型权重 → reputation_lstm.pth
python LSTM_enhance/reputation_lstm.py
```

### 权重文件放置位置

| 权重文件 | 放置位置 |
|----------|----------|
| `physics_lstm.pth` | 项目根目录（`intermediate_fusion_dataset.py` 和 `physical_consistency/main.py` 均从此处加载） |
| `reputation_lstm.pth` | `RSU/` 目录下 |

---

## 异常检测逻辑

### 物理轨迹异常分数

```
lstm_anomaly_score = min(1.0, ||actual_pos - lstm_predicted_pos|| / max_position_error)
```

- `0.0`：轨迹与预测完全吻合（正常）
- `1.0`：位置误差达到或超过 `max_position_error` 米（高度异常）
- 历史帧数不足 `seq_len` 时返回 `0.0`（冷启动保护）

### 信誉异常检测

```python
is_anomaly = abs(actual_rep - predicted_rep) > threshold
# 持证作恶加强检测：predicted > 0.6 且 actual < 0.4 时，阈值降为 threshold × 0.8
```

默认 `threshold=0.15`，检测到异常时额外扣减 `0.05` 信誉。

---

## 可视化仪表盘

### 文件

`LSTM_enhance/lstm_dashboard.py` — 独立可视化脚本，不修改任何现有文件。

### 运行方式

```bash
# 在项目根目录执行，推荐使用 opencood 环境以获得真实 LSTM 推理结果
python LSTM_enhance/lstm_dashboard.py
```

脚本会自动运行两次仿真（IMM+LSTM 和 IMM-only 基线），然后生成全部图表。在没有 `torch` 的环境下会自动降级为纯 numpy 距离启发式，仍可完整生成所有图表。

输出目录：`LSTM_enhance/dashboard_output/`

### 输出图表说明

| 文件 | 图表类型 | 内容说明 |
|------|----------|----------|
| `01_imm_vs_lstm.png` | 双子图时序折线 | 上方为 IMM 原始物理分数，下方为 IMM+LSTM 融合后分数；攻击车辆加粗高亮，紫色三角标注 LSTM 异常触发点（>0.5） |
| `02_reputation_heatmap.png` | 热力图 | 全部车辆×全部时间步的信誉值矩阵，RdYlGn 配色（红=低，绿=高）；攻击车辆行加红色边框，虚线分隔正常/攻击区域 |
| `03_score_decomposition.png` | 堆叠面积图 | 左图为信誉最低的攻击车辆，右图为信誉最高的正常车辆；堆叠层分别显示物理/轨迹/RSU 各项加权贡献，紫色背景区间标注 LSTM 异常触发时段 |
| `04_trajectory_reputation.png` | 轨迹地图 | 每辆车的运动轨迹，线段颜色随实时信誉值动态渐变（红低绿高）；攻击车辆终点用 × 标记，正常车辆用圆点 |
| `05_radar_profile.png` | 极坐标雷达图 | 攻击车辆均值（红）vs 正常车辆均值（绿）的六维分数侧写，维度包括：IMM物理、融合物理、轨迹、RSU、融合总分、信誉 |
| `06_roc_and_metrics.png` | ROC 曲线 + 柱状图 | 左图为 IMM-only 与 IMM+LSTM 的 ROC 曲线及 AUC 对比；右图为阈值 0.6 下的准确率/精确率/召回率/F1 并列柱状图 |

### 环境兼容性说明

脚本对 matplotlib 版本做了兼容处理：

- `legend.labelcolor` 参数在 matplotlib 3.5 以下不存在，脚本通过 `try/except` 跳过，不影响其他样式
- `scatter` 的颜色传参方式兼容旧版 API，使用 `color=` 而非 `c=[[rgba]]`

---

## 数据流总览

```
仿真流水线（physical_consistency/main.py）
    │
    ├── DataGenerator.step()  →  msg: {pos, vel}
    │
    ├── PhysicsPredictor.update_history(vid, [x, y, vx, vy])
    ├── PhysicsPredictor.compute_anomaly_score()  →  lstm_anomaly (0~1)
    │
    ├── IMMManager.step(vid, [x, y])  →  imm_residual
    │
    └── IntermediateFusionManager.compute_all_scores(
            residual, vid, vel, lstm_anomaly_score=lstm_anomaly
        )
        ├── _fuse_physical_scores(imm_phy, lstm_anomaly)
        │       = (1-w)×imm_phy + w×(1-lstm_anomaly)   →  physical
        ├── trajectory_score()                           →  trajectory
        ├── rsu_score()                                  →  rsu
        └── fuse_scores()                               →  fused


OpenCOOD 数据集流水线（intermediate_fusion_dataset.py）
    │
    ├── 每帧每辆 CAV：
    │       ├── 提取 [x, y, vx, vy]
    │       ├── PhysicsPredictor.update_history()
    │       └── PhysicsPredictor.compute_anomaly_score()  →  physics_anomaly_score
    │
    └── reputation_adapter.report_fused_boxes(phy_score=physics_anomaly_score)
                                │
                                ▼
RSU 服务端（RSU/reputation_center_server.py）
    │
    ├── POST /report_batch
    │       └── update_reputation(vehicle_id, is_verified)
    │               ├── 计算 new_rep
    │               ├── ReputationPredictor.update_history()
    │               ├── ReputationPredictor.check_anomaly()
    │               └── 异常则额外 −0.05
    │
    └── GET /early_warning/{vehicle_id}  →  早期预警分数（0~1）
```
