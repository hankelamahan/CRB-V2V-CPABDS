# RSU-Trust：基于中心化信誉的V2V协同感知异常行为防御系统

本项目面向车联网（V2V）协同感知中的信息安全问题，提出并实现了一套基于中心化信誉服务器的异常行为检测与信任评估系统。系统通过部署路侧单元（RSU）作为信誉管理中心，结合物理一致性校验、重叠视场交叉验证与加权框融合（WBF）算法，有效防御幽灵车攻击、紧急刹车欺诈、障碍物伪造等恶意行为。

## 主要功能

- **异常行为检测**：车辆端实时检测邻居车辆上报数据的物理一致性与空间重叠一致性。
- **中心化信誉管理**：RSU 维护全局信誉表，动态更新车辆信誉值（通过 +0.05，失败 -0.1），并广播信誉变化。
- **信任加权融合**：Ego 车辆基于信誉值与检测置信度进行加权框融合（WBF），低信誉车辆数据自动降权或丢弃。
- **攻击场景模拟**：支持幽灵车攻击、紧急刹车欺诈、障碍物伪造攻击的仿真验证。

## 系统架构

[数据流图](pic/v2vflow.png)

基于OpenCOOD, 本系统由三个核心模块组成：
- **车辆端异常检测模块**：物理一致性校验 + 重叠视场交叉验证
- **RSU信誉管理模块**：SQLite 数据库 + C-V2X 通信
- **信任加权融合模块**：信誉感知 WBF 算法

## 快速开始

### 环境要求

- Ubuntu 20.04 / Windows 11
- Python 3.8+
- CoppeliaSim V4.6（或 CARLA 0.9.13）
- PyTorch 1.12+
- OpenCOOD 框架

### 安装依赖

```bash
pip install -r requirements.txt
# opencood 框架需单独从 GitHub 安装： pip install git+https://github.com/DerrickXuNu/OpenCOOD.git
```

### 编译 Cython 扩展

检测后处理依赖 `box_overlaps` 的 Cython 扩展，首次使用需编译（在 `C4-main` 目录下）：

```bash
cd C4-main
python opencood/utils/setup.py build_ext --inplace
```

> 关于 GPU：稀疏卷积骨干（spconv）默认使用 GPU（implicit-gemm）。无 NVIDIA GPU 时，可设置环境变量 `SPCONV_ALGO=native` 并将卷积算法强制为 `ConvAlgo.Native` 以走 CPU 路径（速度较慢，约 100 s/帧，仅供功能验证）。

## 数据准备

数据采用 **OPV2V 风格目录结构**（由 CARLA 离线录制导出）：

```
dataset_root/
└── episode_0000/
    ├── 146/                 # 各 CAV（车辆）以 id 命名
    │   ├── 000000.pcd       # 该车 LiDAR 点云
    │   ├── 000000.yaml      # 位姿 + 周围车辆 GT 标注
    │   └── ...
    ├── 147/ ... 155/
    └── meta.yaml            # 场景元信息（地图、CAV 列表、攻击标签等）
```

- 干净数据集：正常协同感知录制。
- 攻击数据集：在干净录制上离线注入攻击得到的变体（见下文「攻击场景」），`meta.yaml` 的 `attack_label` 标明攻击类型，`adversary_cav_ids` 标明恶意车。

## 模型推理与评估

推理脚本会读取 **模型目录下 `config.yaml` 的 `validate_dir`** 作为待测数据集（指向某个 `episode_XXXX` 目录），运行后在模型目录输出各 IoU 阈值的 AP。

```bash
cd C4-main
# 1) 修改 opencood/model_weight/<模型>/config.yaml 的 root_dir / validate_dir 指向数据集
# 2) 运行推理
python opencood/tools/inference.py \
    --model_dir opencood/model_weight/trust_visual_pose_spoof/pointpillar_late_fusion_trust_visual_trust \
    --fusion_method late
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--fusion_method` | `late` / `early` / `intermediate` |
| `--save_vis` | 保存 Open3D 点云+检测框可视化（需 OpenGL） |
| `--save_vis_dir` + `--frame_indices` + `--headless` | 无 GPU 渲染时用 matplotlib 导出指定帧 |
| `--show_reputation_overlay` | 在图上叠加各 CAV 信誉值横幅 |
| `--semantic_overlay` | 叠加 pred/GT/ego/保留/过滤框的语义图例 |
| `--save_npy` | 导出点云与预测/GT 框（供 3D 可视化） |

评估结果默认输出 **AP@0.3 / 0.5 / 0.7**。模型对比时建议跑 **baseline（无防御） vs trust（防御）** 两版。

```

## 信任防御模块

核心实现位于 `C4-main/opencood/trust/`：

- `physical_consistency_manager.py` — 物理一致性校验（运动学/速度合理性）
- `overlap_field_voting.py` — 重叠视场交叉验证（多车投票）
- `reputation_manager.py` — 信誉值更新
- `late_trust_fusion.py` — 信任加权的后融合编排（低信誉车整车降权/丢弃）
- `trust_logger.py` — 输出 `reputation.jsonl` / `physical.jsonl` / `frame_summary.csv`

信誉相关参数在模型 `config.yaml` 的 `trust_fusion` / `reputation_update` 段配置（如 `drop_below` 阈值、`log_dir` 日志目录）。

## 攻击场景

系统支持多类协同感知攻击的离线注入与防御验证：

| 攻击 | 说明 | 防御机制 |
|------|------|---------|
| 幽灵车（ghost vehicle） | 上报位置违反运动学的虚构车 | 物理一致性校验 |
| 紧急刹车欺诈（brake fraud） | 谎报前车急刹，诱导不必要急刹 | 事件合理性 + 多车交叉验证 |
| 障碍物伪造（obstacle fabrication） | 上报不存在的障碍物 | 重叠视场交叉验证 |

攻击数据由离线注入器在干净录制上生成（支持 raycast 物理回波注入、世界固定障碍物、指定注入帧窗口等）。

## 可视化

- **BEV 检测可视化**：通过 `inference.py` 的 `--save_vis_dir/--headless/--show_reputation_overlay/--semantic_overlay` 导出（俯视点云 + 检测框 + 信誉横幅）。
- **照片级场景重建**：基于 CARLA（Town03），按录制位姿重建车辆与障碍物，渲染攻击/防御对照视频（如刹车欺诈 baseline vs trust）。

---

## DIVA 信誉算法接入

**路径**：`DIVA-main/`

DIVA（DID-based Reputation System for VANETs）是基于去中心化身份标识（DID）与 IOTA DAG 账本的 V2X 信誉算法，原论文发表于 *Computer Networks*（2024）。本项目将其作为 RSU 端信誉评分的参考基线和上游输入源。

**核心组件**

| 文件 | 功能 |
|------|------|
| `reputation_algorithm/v2v.py` | 核心信誉计算，分析 CAM/DENM 消息的安全性与非安全性维度 |
| `reputation_algorithm/vehicle_client.py` | 消费信誉分，判断单条消息可信度 |
| `reputation_algorithm/threshold_utils.py` | 均值 / 众数 / 中位数 / 90百分位等阈值策略 |

**数据集**：`DIVA-main/ETSI-V2V-Dataset-main/`，基于 ETSI 标准的真实 V2V 通信录制，含 CAM 数据集与恶意比例分别为 20% / 30% / 40% 的 DENM 数据集。

**与本系统的关系**：DIVA 输出的初始信誉向量（`dataset/initial_reputations.csv`）可作为 RSU 信誉服务器的冷启动输入，替代默认初始值 0.5；其逐帧信誉分也可作为 `reputation_client.py` 的补充数据源。

**运行**

```bash
cd DIVA-main
python reputation_algorithm/v2v.py \
    --denmdataset ETSI-V2V-Dataset-main/dataset/mtits-dataset/DENM-dataset/datasetDen.csv \
    --camdataset  ETSI-V2V-Dataset-main/dataset/mtits-dataset/CAM-dataset/datasetCam.csv \
    --reputation  dataset/initial_reputations.csv \
    --coverage    dataset/coverage.json \
    -b 0.5 -a 0.5
```

---

## 基线算法对比

**路径**：`1_baseline_comparison/`

实现五种 SOTA 信誉/误行为检测基线，用于与本系统进行定量对比。

**已实现基线**

| 算法 | 核心机制 |
|------|----------|
| DRAMBR | 贝叶斯声誉更新 + 指数衰减，基于位置/速度/时间戳/频率四维异常检测 |
| PlexeMDS | 滑动窗口（默认 20 帧）消息评分，三级信任阈值划分 |
| StaticReputation | 固定信誉值（0.5），不随观测更新，作为无信任管理下界 |
| MajorityVoting | 累积正/负票比，基于综合异常分数简单投票 |
| NoTrustFusion | 所有车辆信誉恒为 1.0，作为无防御上界 |

**运行**

```bash
cd 1_baseline_comparison
python run_baseline_comparison.py
```

输出：`1.baseline_comparison/performance_results.csv` 及三张对比图（综合对比、混淆矩阵、F1 分数）。

---

## 复杂场景压力测试

**路径**：`2_complex_scenario_results/`

在更接近真实 V2V 环境的场景下对所有基线算法进行压力评估，重点考察算法在非理想条件下的鲁棒性。

**场景特性**

| 特性 | 说明 |
|------|------|
| 动态攻击强度 | 恶意行为强度随时间正弦波动，模拟攻击者的间歇/加强策略 |
| 四类攻击者 | 持续攻击 / 间歇攻击（30% 概率）/ 协同攻击（周期突发）/ 自适应攻击（感知被检测后降低强度） |
| 混合攻击类型 | 位置欺骗、速度欺骗、时间戳篡改、混合攻击随机切换 |
| 动态拓扑 | 每时间步有 10% 概率随机缩减活跃车辆集，模拟车辆进出场景 |
| 环境噪声 | 基础噪声叠加周期性抖动与随机突刺（5% 概率） |

**规模**：50 辆车（其中 10 辆恶意）× 500 时间步，共约 25000 条观测。

**运行**

```bash
cd 2_complex_scenario_results
python run_complex_scenario.py
```

输出：`2.complex_scenario_results/complex_scenario_results.csv` 及时序 F1 曲线、各类攻击者检测热图等图表。

---

## Enhanced DRAMBR+

**路径**：`3. enhanced_drambr_plus/`

针对"持证作恶"与紧急刹车欺诈（EEBL）两类高级攻击对 DRAMBR 进行系统性升级，形成六项增强机制。

**核心增强**

**① 多维信任向量**（`advanced_trust_system.py`）

将单一信誉值扩展为三维向量：直接信任（实时物理检测，权重 40%）、间接信任（邻居车辆报告，30%）、全局信任（长期历史，30%）。检测到持证作恶特征（全局信任 > 0.6 且直接信任 < 0.3）时，动态将直接信任权重提升至 60%，使系统对首次作恶的响应从 5–10 个周期缩短至 1–2 个周期。

**② 预测性信誉模块**（`predictive_reputation.py`）

基于历史信誉序列（默认窗口 20 帧）预测下一时刻信誉值；偏差超过阈值 0.15 时触发预警，重点标记"预测高但实际低"的持证作恶特征。有 PyTorch 时使用 LSTM 预测器，否则自动回退至移动平均。

**③ 动态风险置信度分配**（`advanced_trust_system.py`）

根据当前信任分数动态调整可疑/异常阈值，避免固定阈值在不同风险车辆上的误判：信任值越低，阈值收紧以提高灵敏度。

**④ 紧急刹车欺诈专项检测**（`emergency_brake_fraud_detector.py`）

从物理一致性（加速度范围校验、速度-位置轨迹连续性）、多车交叉验证（邻车观测一致性）、历史行为（急刹频率、历史欺诈率）三个维度对 EEBL 消息进行专项判定。

**⑤ 车辆端轻量预筛选**（`enhanced_drambr_plus.py`）

车辆本地毫秒级快速评估，仅当风险分数 > 0.3 或风险增量 > 0.2 时才上报 RSU，降低 RSU 侧计算与通信负担约 30–40%。

**⑥ RSU 端三重确认**（`enhanced_drambr_plus.py`）

DBSCAN 聚类（印象分加权距离）、Isolation Forest 异常检测、GMM 概率分类三级串联，减少误报。

**性能对比（仿真）**

| 指标 | DRAMBR | Enhanced DRAMBR+ |
|------|--------|-----------------|
| 首次作恶检测周期 | 5–10 个周期 | 1–2 个周期 |
| 无 RSU 覆盖响应 | 需等待回到覆盖区 | 车辆端即时响应 |
| 紧急刹车欺诈检测 | 不支持 | 专项检测模块 |
| 预测性预警 | 不支持 | 序列预测偏差触发 |
| RSU 计算负担 | 基准 | 降低约 30–40% |

**运行**

```bash
cd "3. enhanced_drambr_plus"
python run_visualization_v3.py      # 生成完整仪表盘
python baseline_comparison_v3.py    # 与基线的定量对比
```

**依赖**：`numpy`（必需）；`scikit-learn`（RSU 三重确认，可选）；`torch`（LSTM 预测，可选，缺失时自动回退）。

---

## LSTM 异常检测增强

**路径**：`LSTM_enhance/`

在物理一致性检验和信誉更新两个环节分别引入 LSTM 序列模型，作为现有规则方法的深度学习补充。详细说明见 [`LSTM_enhance/README.md`](LSTM_enhance/README.md)。

**物理轨迹 LSTM**（`physics_lstm.py`）

基于历史运动状态 `[x, y, vx, vy]` 预测下一帧位置，计算预测误差作为轨迹异常分数（0–1）。与 IMM（交互多模型滤波器）残差加权融合：

```
physical_score = (1 - w) × imm_score + w × (1 - lstm_anomaly_score)
```

默认权重 `w = 0.4`，设为 0 时退化为纯 IMM，向后兼容。集成于 `physical_consistency/intermediate_fusion_manager.py` 和 `intermediate_fusion_dataset.py`。

**信誉序列 LSTM**（`reputation_lstm.py`）

基于历史信誉序列预测下一时刻信誉值，检测"持证作恶"模式（预测高但实际骤降）。异常时在 RSU 正常惩罚基础上额外扣减 0.05。集成于 `RSU/reputation_center_server.py`，并提供 `/early_warning/{vehicle_id}` 预警接口。

**可视化仪表盘**（`lstm_dashboard.py`）

独立运行，生成六张专项分析图至 `LSTM_enhance/dashboard_output/`：

```bash
python LSTM_enhance/lstm_dashboard.py
```

兼容有无 `torch` 的环境（缺失时自动使用 numpy 启发式替代）；兼容 matplotlib 3.5 以下版本。




