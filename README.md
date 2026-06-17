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

### 运行推理

推理脚本位于 `C4-LateFusionConbine/`，需在该目录下以 `PYTHONPATH=.` 运行。

**Baseline（无信任防御，纯 late fusion）：**

```bash
cd C4-LateFusionConbine
PYTHONPATH=. python opencood/tools/inference.py \
  --model_dir opencood/model_weight/pointpillar_late_fusion \
  --fusion_method late \
  --frame_index 0 --num_frames 50 --max_frames 50 \
  --save_vis_dir <输出目录> \
  --color_mode constant --headless --num_workers 2
```

**Trust（信任防御 + 信誉可视化）：**

```bash
cd C4-LateFusionConbine
PYTHONPATH=. python opencood/tools/inference.py \
  --model_dir opencood/model_weight/pointpillar_late_fusion_trust_visual_trust \
  --fusion_method late \
  --frame_index 0 --num_frames 50 --max_frames 50 \
  --save_vis_dir <输出目录> \
  --color_mode constant --headless --num_workers 2 \
  --show_reputation_overlay --show_gt_cav_ids
```

**常用参数说明：**

| 参数 | 说明 |
|------|------|
| `--model_dir` | 模型权重与 `config.yaml` 所在目录（数据集路径在该 config 的 `root_dir`/`validate_dir`） |
| `--fusion_method` | 融合方式，本系统用 `late` |
| `--frame_index` / `--num_frames` / `--max_frames` | 起始帧 / 处理帧数 / 上限 |
| `--save_vis_dir` | 渲染图输出目录；**不带此参数则不渲染**，调试信誉时更快 |
| `--show_reputation_overlay` | 顶部显示每辆车信誉横幅（颜色随信誉值渐变） |
| `--show_gt_cav_ids` | 标注各 CAV 车辆 id，并使其检测框按信誉着色（绿→黄→红） |
| `--headless` | 无显示环境下渲染（走 matplotlib 后端） |
| `--color_mode` | 点云着色模式（`constant` / `intensity`） |



