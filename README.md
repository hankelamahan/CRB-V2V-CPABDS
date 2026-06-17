# RSU-Trust：基于中心化信誉的V2V协同感知异常行为防御系统
feat. Chaokun Zhang, Yi Ji, Xiang Han, Jiaming Wang, YuanJie Ma, Chunpeng Wang, Daichen Li

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




