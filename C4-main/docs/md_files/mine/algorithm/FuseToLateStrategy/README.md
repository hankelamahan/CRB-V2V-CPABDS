# CRB 三算法接入 OpenCOOD Late Fusion 技术框架

## 1. 文档目的

本文档设计一套将 `CRB-V2V-CPABDS-master` 中三类安全协同感知算法接入 `OpenCOOD-main` 的 late fusion 策略的技术框架，并说明推荐的代码组织、接口适配、配置方式、测试方法和效果评估流程。

这里的“三算法”指：

1. 中心化信誉管理：由 RSU / reputation server 维护车辆信誉，车辆端本地缓存信誉。
2. 重叠视场投票：基于多车检测框重叠关系判断检测一致性，并更新车辆信誉。
3. 物理一致性校验：基于运动残差、轨迹平滑性、邻居投票等信号修正车辆信誉。

本文档面向 `late fusion` 接入。`intermediate fusion` 需要改模型特征融合模块，不作为第一阶段目标。

## 2. 总体结论

CRB 当前算法更适合先接入 OpenCOOD 的 `late fusion`，原因是 late fusion 的推理链路天然保留了每个 CAV 的独立模型输出：

```python
for cav_id, cav_content in batch_data.items():
    output_dict[cav_id] = model(cav_content)

dataset.post_process(batch_data, output_dict)
```

每个 CAV 都会先独立产生 `psm / rm` 检测输出，然后统一进入后处理。CRB 的信誉加权、重叠视场投票、低信誉过滤、物理一致性更新都应挂在这个后处理阶段。

不要直接把 `CRB-V2V-CPABDS-master/CRB-V2V-CPABDS-master/C4-main` 覆盖到 `OpenCOOD-main`。CRB 目录下的 `C4-main` 是一份 OpenCOOD 副本，工具链和主工程已有改动不完全一致。推荐做法是提取 CRB 的算法模块，然后在 `OpenCOOD-main` 中按主工程结构接入。

## 3. 目标架构

推荐架构如下：

```text
OpenCOOD-main
└── opencood
    ├── data_utils
    │   └── datasets
    │       └── late_fusion_dataset.py
    ├── tools
    │   ├── inference.py
    │   └── inference_utils.py
    └── trust
        ├── __init__.py
        ├── overlap_field_voting.py
        ├── physical_consistency_manager.py
        ├── reputation_manager.py
        ├── reputation_cache.py
        ├── late_trust_fusion.py
        └── id_mapper.py
```

模块职责：

- `overlap_field_voting.py`
  - 从 CRB 新版 `overlap_field_voting.py` 迁移。
  - 输入多车检测框、检测分数、类别、信誉分数。
  - 输出 overlap voting 的融合框及一致性结果。

- `physical_consistency_manager.py`
  - 从 CRB 新版 `physical_consistency_manager.py` 迁移后扩展。
  - 输入真实 pose / velocity / timestamp / detection track 信号。
  - 输出物理一致性分数。

- `reputation_manager.py`
  - 统一管理三类信誉来源：初始信誉、overlap voting 更新、physical consistency 更新。
  - 对外只暴露 `get_reputation(cav_id)`、`update_from_voting(...)`、`update_from_physical(...)`。

- `reputation_cache.py`
  - 从 CRB 根目录 `local_cache.py` 迁移。
  - 负责本地缓存和中心信誉服务同步。
  - 第一阶段可以只加载本地 `reputation_map.json`，不接真实服务器。

- `late_trust_fusion.py`
  - 封装 late 后处理中的信任融合逻辑。
  - 避免把全部 CRB 逻辑直接塞进 `LateFusionDataset.post_process()`。

- `id_mapper.py`
  - 解决 OpenCOOD CAV ID 和 CRB / DIVA reputation ID 的映射问题。
  - OpenCOOD 中 ego 车在 batch 里常被改名为 `"ego"`，需要保留 `original_cav_id`。

## 4. Late Fusion 接入点

OpenCOOD 主框架中 late fusion 的关键链路是：

```text
inference.py
  -> inference_utils.inference_late_fusion()
      -> for each cav: model(cav_content)
      -> dataset.post_process(batch_data, output_dict)
          -> VoxelPostprocessor.post_process()
              -> decode psm/rm
              -> project boxes to ego frame
              -> NMS
```

CRB 应接在 `dataset.post_process()` 内部，准确位置是在每个 CAV 的检测框已经解码并投影到 ego 坐标之后、最终 NMS 之前。

推荐第一阶段实现为 `trust-aware late NMS`：

```text
每个 CAV 独立检测
  -> 解码 3D box
  -> 投影到 ego 坐标系
  -> 生成 2D standup box 供 overlap voting 使用
  -> overlap voting 更新 reputation
  -> physical consistency 更新 reputation
  -> score = score * reputation
  -> reputation < drop_below 的 CAV 或 box 被过滤
  -> OpenCOOD 原始 3D rotated NMS
  -> 输出最终 pred_box3d_tensor / pred_score
```

第一阶段不建议直接把 CRB 的 2D fused box 当成最终预测框，因为 OpenCOOD 的评估和可视化需要 3D corners。更稳妥的方式是先用信誉影响 3D 检测框分数和过滤逻辑，保持 OpenCOOD 原有 3D NMS 与评估接口不变。

第二阶段再扩展为 `trust-aware 3D WBF`：

```text
2D overlap clustering
  -> 找回每个 cluster 对应的 3D member boxes
  -> 按 score * reputation 加权中心、尺寸、yaw
  -> 输出新的 3D fused boxes
```

## 5. 数据接口设计

### 5.1 LateFusionDataset 需要新增的字段

在 `get_item_test()` 中，每个 CAV 处理后建议额外保留：

```python
selected_cav_processed.update({
    'cav_id': cav_id,
    'original_cav_id': cav_id,
    'is_ego': cav_id == ego_id,
    'timestamp': timestamp_key,
    'scenario_index': scenario_index,
    'lidar_pose': selected_cav_base['params']['lidar_pose'],
    'ego_lidar_pose': ego_lidar_pose,
})
```

最关键的是 `original_cav_id`。当前 OpenCOOD late test 会把 ego 的 key 改成 `"ego"`，如果不额外保留原始 ID，信誉系统只能看到 `"ego"`，无法和外部 `reputation_map.json` 对齐。

### 5.2 collate_batch_test 需要透传元信息

`collate_batch_test()` 需要把以下字段原样放入 `output_dict[cav_id]`：

```python
'cav_id'
'original_cav_id'
'is_ego'
'timestamp'
'scenario_index'
'lidar_pose'
'ego_lidar_pose'
```

注意：不要在 `__getitem__()` 或 `collate_batch_test()` 中更新信誉状态。DataLoader 使用多进程时，这些逻辑可能运行在 worker 进程中，状态不会可靠同步。信誉更新应放在主进程调用的 `dataset.post_process()` 中。

## 6. 配置设计

建议在 late fusion yaml 或模型目录 `config.yaml` 中加入：

```yaml
trust_fusion:
  use_trust_fusion: true
  mode: trust_nms
  iou_thr: 0.5
  update_rate: 0.1
  default_reputation: 0.5
  min_reputation: 0.0
  drop_below: 0.3
  score_power: 1.0
  ego_reputation: 1.0
  reputation_map: ""
  id_map: ""
  log_reputation: true
  log_dir: ""

physical_consistency:
  use_physical_consistency: false
  update_rate: 0.05
  min_reputation: 0.0
  weights:
    physical: 0.4
    trajectory: 0.3
    rsu: 0.3
  residual_sigma: 5.0
  velocity_sigma: 1.0
  history_window: 5
```

默认策略：

- `use_trust_fusion` 建议默认 `false`，只在实验 yaml 中显式开启。
- `use_physical_consistency` 第一阶段建议默认 `false`，因为 CRB 当前版本仍使用 `residual=0.0`、`vel=[0.0, 0.0]` 这样的占位输入，不能代表真实物理一致性。
- `ego_reputation` 建议固定为 `1.0`，除非明确要评估 ego 被攻击的场景。

## 7. 三算法融合逻辑

### 7.1 中心化信誉管理

输入来源：

- 本地 `reputation_map.json`
- DIVA 输出 CSV 转换后的 JSON
- 未来 RSU 服务查询结果

运行方式：

1. 初始化时加载 `reputation_map`。
2. 每帧根据 `original_cav_id` 查询初始信誉。
3. overlap voting 和 physical consistency 产生增量更新。
4. 更新后的信誉写入内存状态。
5. 可选地按帧保存 reputation log。

推荐接口：

```python
rep = reputation_manager.get_reputation(original_cav_id)
reputation_manager.update_from_voting(original_cav_id, is_consistent)
reputation_manager.update_from_physical(original_cav_id, physical_score)
```

### 7.2 重叠视场投票

输入：

```python
detections_dict = {
    original_cav_id: {
        'boxes2d': [[x1, y1, x2, y2], ...],
        'boxes3d': Tensor[N, 8, 3],
        'scores': [score, ...],
        'labels': [1, ...],
        'reputation': rep,
    }
}
```

处理：

1. 将每个 box 的排序权重定义为 `score * reputation`。
2. 根据 2D IoU 聚类。
3. 对每个车辆判断其检测是否与融合结果一致。
4. 一致则提升信誉，不一致则降低信誉。

输出：

```python
{
    'fused_2d_boxes': ...,
    'fused_scores': ...,
    'consistency_by_cav': {
        original_cav_id: true_or_false
    }
}
```

第一阶段只用该结果更新信誉，不直接替换最终 3D 框。

### 7.3 物理一致性校验

第一阶段可先保留接口，但不开启真实影响：

```yaml
physical_consistency:
  use_physical_consistency: false
```

第二阶段接入真实物理信号：

- 从连续帧 `lidar_pose` 计算 CAV 速度。
- 从目标 track 或检测框中心计算目标运动残差。
- 对同一 `scenario_index + original_cav_id` 维护短时历史。
- 使用 CRB 的 `physical_score`、`trajectory_score`、`rsu_score` 得到综合分数。

不要继续使用：

```python
residual = 0.0
vel = [0.0, 0.0]
```

这会让物理一致性分数失真。

## 8. 代码接入步骤

### 8.1 新增 trust 包

从 CRB 中迁移：

```text
CRB-V2V-CPABDS-master/CRB-V2V-CPABDS-master/C4-main/opencood/data_utils/datasets/overlap_field_voting.py
  -> OpenCOOD-main/opencood/trust/overlap_field_voting.py

CRB-V2V-CPABDS-master/CRB-V2V-CPABDS-master/C4-main/opencood/data_utils/datasets/physical_consistency_manager.py
  -> OpenCOOD-main/opencood/trust/physical_consistency_manager.py

CRB-V2V-CPABDS-master/CRB-V2V-CPABDS-master/local_cache.py
  -> OpenCOOD-main/opencood/trust/reputation_cache.py
```

然后新增 `late_trust_fusion.py` 和 `id_mapper.py`。

### 8.2 修改 LateFusionDataset

目标文件：

```text
OpenCOOD-main/opencood/data_utils/datasets/late_fusion_dataset.py
```

需要修改：

1. `__init__()`
   - 读取 `trust_fusion` 和 `physical_consistency` 配置。
   - 初始化 `LateTrustFusion`。
   - 默认不开启 trust，避免影响基线实验。

2. `get_item_test()`
   - 保留 `original_cav_id`、`timestamp`、`scenario_index`、pose 等元信息。

3. `collate_batch_test()`
   - 透传上述元信息。

4. `post_process()`
   - 替换或包裹原有 `self.post_processor.post_process(data_dict, output_dict)`。
   - 在最终 NMS 前调用 trust fusion helper。

### 8.3 修正设备一致性

所有新建 score tensor 必须和预测框在同一 device 上：

```python
scores_tensor = torch.as_tensor(
    final_scores,
    device=pred_box3d_tensor.device,
    dtype=pred_box3d_tensor.dtype
)
```

不要使用默认 CPU 的：

```python
torch.from_numpy(np.array(final_scores))
```

### 8.4 保持训练链路不变

Late trust fusion 第一阶段只影响推理后处理，不改变模型结构和训练损失。训练仍使用普通 late detector：

```bash
cd /home/wcp/c4/OpenCOOD-main
conda run -n torch118 python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_late_fusion.yaml
```

注意：不能拿 intermediate 模型权重直接跑 late。late 策略应使用 `model.core_method: point_pillar` 这类单车检测模型，而不是 `point_pillar_intermediate`。

## 9. 推荐实现伪代码

`LateFusionDataset.post_process()` 的核心结构建议如下：

```python
def post_process(self, data_dict, output_dict):
    per_cav_detections = []

    for cav_id, cav_content in data_dict.items():
        raw_id = cav_content.get('original_cav_id', cav_id)
        boxes3d, scores, labels = decode_single_cav(
            cav_content,
            output_dict[cav_id]
        )

        if boxes3d is empty:
            continue

        projected_boxes3d = project_to_ego(boxes3d, cav_content['transformation_matrix'])
        boxes2d = corner_to_standup_box_torch(projected_boxes3d)

        per_cav_detections.append({
            'cav_id': cav_id,
            'original_cav_id': raw_id,
            'boxes3d': projected_boxes3d,
            'boxes2d': boxes2d,
            'scores': scores,
            'labels': labels,
            'metadata': extract_metadata(cav_content),
        })

    if self.trust_fusion is not None:
        per_cav_detections, trust_debug = self.trust_fusion.apply(
            per_cav_detections
        )

    pred_box3d_tensor, scores_tensor = merge_and_nms(per_cav_detections)
    gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)
    return pred_box3d_tensor, scores_tensor, gt_box_tensor
```

## 10. 测试方案

### 10.1 单元测试：overlap voting

目标：验证同一目标的多车检测能聚类，低信誉车辆会被降权。

建议新增脚本：

```text
OpenCOOD-main/tests/test_trust_overlap_voting.py
```

测试点：

- 两辆高信誉车检测框高度重叠，应生成一个 cluster。
- 一辆低信誉车的冲突框不应主导融合分数。
- 空检测输入应返回空结果，不抛异常。
- `min_reputation` 和 `drop_below` 生效。

临时 smoke test 可以用：

```bash
cd /home/wcp/c4/OpenCOOD-main
conda run -n torch118 python -m pytest tests/test_trust_overlap_voting.py
```

如果当前工程没有 pytest 测试体系，也可以先用独立脚本：

```bash
conda run -n torch118 python scripts/debug_trust_overlap_voting.py
```

### 10.2 回归测试：trust 关闭时等价于原 late fusion

配置：

```yaml
trust_fusion:
  use_trust_fusion: false
physical_consistency:
  use_physical_consistency: false
```

执行：

```bash
cd /home/wcp/c4/OpenCOOD-main
conda run -n torch118 python opencood/tools/inference.py \
  --model_dir opencood/logs/<late_model_dir> \
  --fusion_method late \
  --max_frames 20 \
  --save_vis_dir logs/late_baseline_check \
  --frame_index 0 \
  --num_frames 3 \
  --color_mode intensity \
  --headless
```

验收标准：

- AP@0.3 / AP@0.5 / AP@0.7 与原 late fusion 一致或只存在浮点级差异。
- 可视化图片正常生成。
- 没有新增 reputation log。

### 10.3 功能测试：trust 开启但使用中性信誉

配置：

```yaml
trust_fusion:
  use_trust_fusion: true
  default_reputation: 1.0
  min_reputation: 0.0
  drop_below: 0.0
physical_consistency:
  use_physical_consistency: false
```

目的：

- 验证 trust 模块接入后不会破坏 late inference。
- 因所有车信誉为 1.0，结果应基本等价于 baseline。

执行：

```bash
conda run -n torch118 python opencood/tools/inference.py \
  --model_dir opencood/logs/<late_trust_model_dir> \
  --fusion_method late \
  --max_frames 20 \
  --save_vis_dir logs/late_trust_neutral \
  --frame_index 0 \
  --num_frames 3 \
  --color_mode intensity \
  --headless
```

### 10.4 低信誉过滤测试

准备一个 `reputation_map.json`：

```json
{
  "ego_original_id": 1.0,
  "cav_bad_id": 0.1,
  "cav_good_id": 1.0
}
```

配置：

```yaml
trust_fusion:
  use_trust_fusion: true
  reputation_map: "opencood/logs/reputation_map.json"
  default_reputation: 0.5
  drop_below: 0.3
  log_reputation: true
physical_consistency:
  use_physical_consistency: false
```

验收标准：

- `cav_bad_id` 对应检测框被过滤，或检测分数显著下降。
- 输出日志中能看到每帧每车 reputation。
- 最终 AP 不应因为单个恶意 CAV 的高置信假框明显下降。

### 10.5 可视化对比测试

分别输出 baseline 和 trust 结果：

```bash
conda run -n torch118 python opencood/tools/inference.py \
  --model_dir opencood/logs/<late_baseline_model_dir> \
  --fusion_method late \
  --frame_indices 0,1,2,3,4 \
  --save_vis_dir logs/vis_late_baseline \
  --color_mode intensity \
  --headless

conda run -n torch118 python opencood/tools/inference.py \
  --model_dir opencood/logs/<late_trust_model_dir> \
  --fusion_method late \
  --frame_indices 0,1,2,3,4 \
  --save_vis_dir logs/vis_late_trust \
  --color_mode intensity \
  --headless
```

观察点：

- 低信誉车辆产生的离群框是否减少。
- 正常车辆共同确认的目标是否保留。
- GT 和 pred 的空间位置是否仍然对齐。

### 10.6 大样本评估

使用相同 late 模型，对比三组实验：

1. `late_baseline`
   - 原始 late fusion。
2. `late_trust_nms`
   - 开启 overlap voting + reputation score weighting。
3. `late_trust_physical`
   - 在真实 physical consistency 信号接入后开启。

执行：

```bash
conda run -n torch118 python opencood/tools/inference.py \
  --model_dir opencood/logs/<exp_dir> \
  --fusion_method late \
  --max_frames 0
```

记录指标：

- AP@0.3
- AP@0.5
- AP@0.7
- 平均每帧预测框数量
- 被过滤 CAV 数量
- 被过滤 box 数量
- reputation 均值 / 方差 / 最小值
- 推理耗时增加比例

## 11. 日志与可解释性

建议 trust 模块每帧写出 JSONL：

```json
{
  "frame": 12,
  "scenario_index": 0,
  "timestamp": "000078",
  "cavs": {
    "101": {
      "reputation_before": 0.5,
      "voting_consistent": true,
      "physical_score": null,
      "reputation_after": 0.6,
      "num_boxes_before": 43,
      "num_boxes_after": 43
    }
  }
}
```

推荐保存路径：

```text
opencood/logs/<exp_name>/trust_logs/reputation.jsonl
opencood/logs/<exp_name>/trust_logs/frame_summary.csv
```

这样可以定位性能提升或下降来自哪个 CAV、哪一帧、哪类过滤行为。

## 12. 验收标准

第一阶段接入完成应满足：

- `trust_fusion.use_trust_fusion=false` 时，late fusion 行为与原框架一致。
- `trust_fusion.use_trust_fusion=true` 时，推理能够跑完整 validation set。
- 低信誉 CAV 的检测框分数会被降低，低于阈值时会被过滤。
- AP@0.3 / AP@0.5 / AP@0.7 能正常输出。
- 可视化能正常生成。
- reputation log 能对每帧每车给出解释。
- 不影响 early / intermediate fusion。
- 不要求重新训练模型即可运行第一阶段 trust-aware late NMS。

第二阶段验收：

- physical consistency 使用真实连续帧 pose / velocity / residual，不再使用固定占位值。
- 支持从 DIVA / RSU 信誉文件初始化 reputation。
- 支持 3D trust-aware WBF，并保持 OpenCOOD 评估接口不变。

## 13. 风险与注意事项

- 当前 CRB 版本里的 `physical_consistency` 在 late dataset 中使用占位 `residual=0.0`、`vel=[0.0, 0.0]`，不能直接作为真实效果结论。
- CRB 的 overlap voting 输出是 2D standup box，OpenCOOD 最终需要 3D corners，第一阶段应只用其更新信誉和调整 score。
- 信誉状态必须在主进程后处理阶段维护，不应放在 DataLoader worker 中更新。
- `scores_tensor` 必须和 `pred_box3d_tensor` 在同一 device。
- 当前已有的 attentive intermediate 权重不能直接用于 late 策略，需要 late detector 权重。
- DIVA 的车辆 ID 与 OpenCOOD 场景目录 CAV ID 未必一致，必须通过 `id_mapper.py` 显式映射。

## 14. 推荐开发顺序

1. 新增 `opencood/trust` 包，迁移 CRB overlap voting、physical consistency、local cache。
2. 给 `LateFusionDataset` 增加元信息透传和 `original_cav_id`。
3. 实现 `late_trust_fusion.py`，完成 trust-aware score weighting 和 filtering。
4. 加 yaml 配置，默认关闭 trust。
5. 跑 trust off 回归测试。
6. 跑 neutral reputation 功能测试。
7. 跑低信誉过滤测试和定帧可视化对比。
8. 跑完整 validation AP 对比。
9. 接入真实 physical consistency 信号。
10. 扩展 3D WBF。

