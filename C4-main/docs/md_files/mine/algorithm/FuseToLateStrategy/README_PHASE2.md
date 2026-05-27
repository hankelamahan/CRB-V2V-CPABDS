# CRB Late Trust 第二阶段技术框架说明

## 1. 文档定位

本文档基于当前 `OpenCOOD-main` 代码实现，对 CRB trust 第二阶段在 OpenCOOD late fusion 中的技术框架做整理。它不是后续规划书，而是当前代码框架的实现说明、维护边界和实验解释文档。

当前第二阶段已经从早期方案调整为稳定的 `trust_nms` 主链路：

```text
per-CAV detection
  -> leave-one-out overlap voting
  -> track / pose / consensus physical evidence
  -> reputation update
  -> reputation score weighting
  -> low-reputation CAV filtering
  -> OpenCOOD rotated NMS
```

此前尝试过的 `trust_3d_wbf` 路径在实验中效果不稳定，当前代码已删除 3D WBF 运行接口和实现文件。第二阶段文档以当前保留的 `trust_nms` 路径为准。

## 2. 第二阶段当前结论

第二阶段的核心目标不是替换 OpenCOOD 的最终 3D 后处理，而是在最终 NMS 前引入可信度证据，让低可信 CAV 的检测框被降权或过滤。

当前实现坚持以下边界：

1. 不修改模型结构。
2. 不修改训练 loss。
3. 不把 CRB 2D voting 结果当成最终 3D 预测框。
4. 不使用 3D WBF 作为最终融合器。
5. 最终预测仍由 OpenCOOD 的 `merge_and_nms()` / rotated NMS 输出。

这样做的原因是：late fusion 阶段已经具备每个 CAV 的独立预测、分数、位姿和时间信息，足够完成信誉评估；同时保留 OpenCOOD 原后处理可以减少接口风险，保证 AP 评估和可视化逻辑稳定。

## 3. 当前代码结构

第二阶段相关代码集中在 `opencood/trust` 和 `LateFusionDataset`：

```text
OpenCOOD-main
└── opencood
    ├── data_utils
    │   └── datasets
    │       └── late_fusion_dataset.py
    └── trust
        ├── __init__.py
        ├── id_mapper.py
        ├── late_trust_fusion.py
        ├── motion_state_buffer.py
        ├── overlap_field_voting.py
        ├── physical_consistency_manager.py
        ├── reputation_cache.py
        ├── reputation_manager.py
        ├── reputation_source.py
        ├── track_association.py
        └── trust_logger.py
```

模块职责：

- `late_fusion_dataset.py`
  - 在 late fusion 后处理阶段解码每个 CAV 的预测框。
  - 将每车预测投影到 ego 坐标系。
  - 构造 `cav_detections` 和 `frame_context`。
  - 调用 `LateTrustFusion.apply()` 获取降权/过滤后的检测。
  - 最后统一调用 `merge_and_nms()`。

- `late_trust_fusion.py`
  - 第二阶段主编排器。
  - 读取每个 CAV 当前 reputation。
  - 调用 leave-one-out overlap voting。
  - 调用跨帧 track、pose motion 和跨车 consensus 物理一致性。
  - 根据 voting + physical evidence 更新 reputation。
  - 对 score 进行 reputation weighting，并过滤低信誉 CAV。
  - 写出 reputation、physical、frame summary 日志。

- `overlap_field_voting.py`
  - 用其他 CAV 的检测作为 reference，评估目标 CAV 是否与共识一致。
  - 采用 leave-one-out，避免目标 CAV 自己参与自己的信誉证明。
  - 输出 matched boxes、unmatched boxes、reference agent count 和 consistency ratio。

- `motion_state_buffer.py`
  - 维护 `(scenario_index, trust_id)` 级别的 CAV pose 历史。
  - 根据连续帧 `lidar_pose` 和 `timestamp_index` 计算 CAV 自身速度。
  - 历史不足、时间倒退、pose 缺失时返回 unknown / invalid。

- `track_association.py`
  - 对同一 CAV 的检测框进行跨帧 greedy association。
  - 使用 3D center distance 和 BEV IoU 作为匹配门限。
  - 根据上一帧 track 速度预测当前中心，得到 track residual。

- `physical_consistency_manager.py`
  - 将 track residual 转成 motion score。
  - 计算当前帧跨 CAV 的 consensus residual。
  - 融合 box 级 motion evidence 和 consensus evidence。
  - 将 `physical_scores`、`motion_score`、`consensus_motion_score` 原地写回每个 detection。

- `reputation_manager.py`
  - 管理 CAV 级长期 reputation。
  - 支持默认信誉、ego 信誉、上下界裁剪、非对称更新和单帧最大变化量限制。
  - 支持外部 reputation source 初始化。

- `reputation_source.py`
  - 支持 `none`、`json`、`diva_csv`、`rsu_http` 四类来源接口。
  - 当前重点是离线 JSON / DIVA CSV 初始化；`rsu_http` 保留接口，不作为推理硬依赖。

- `reputation_cache.py`
  - 提供 TTL + LRU reputation cache。
  - 可通过 source query 作为缓存回填来源。

- `id_mapper.py`
  - 处理 OpenCOOD CAV ID 与外部信誉 ID 的映射。
  - 解决 ego 在 batch 中被改名为 `"ego"` 后仍需恢复原始车辆 ID 的问题。

- `trust_logger.py`
  - 写出 `reputation.jsonl`、`physical.jsonl`、`frame_summary.csv`。
  - 用于解释每帧 reputation 变化、物理一致性证据和过滤结果。

## 4. 总体数据流

推理阶段每个 CAV 独立前向：

```text
inference_late_fusion()
  -> for each CAV: model(cav_content)
```

随后进入 late fusion 后处理：

```text
LateFusionDataset.post_process()
  -> post_process_trust()
      -> _decode_single_cav_detection()
      -> _trust_frame_context()
      -> LateTrustFusion.apply()
          -> attach reputation
          -> build voting inputs
          -> compute leave-one-out voting details
          -> compute track evidence
          -> compute pose motion evidence
          -> compute cross-CAV consensus evidence
          -> annotate physical scores
          -> update reputation
          -> weight scores by reputation
          -> filter low-reputation CAVs
          -> write trust logs
      -> merge_and_nms()
      -> generate_gt_bbx()
```

关键点是：trust 逻辑发生在最终 NMS 之前，只改变输入 NMS 的检测框集合和分数，不改变 OpenCOOD 后处理的输出格式。

## 5. CavDetection 数据接口

`LateFusionDataset._decode_single_cav_detection()` 会为每个 CAV 构造一个 dict：

```python
{
    "cav_id": cav_id,
    "original_cav_id": original_cav_id,
    "trust_id": trust_id,
    "is_ego": is_ego,
    "scenario_index": scenario_index,
    "timestamp": timestamp,
    "timestamp_index": timestamp_index,
    "boxes3d": projected_boxes3d,  # Tensor[N, 8, 3]
    "boxes2d": projected_boxes2d,  # Tensor[N, 4]
    "scores": scores,              # Tensor[N]
    "labels": labels,              # Tensor[N]
    "lidar_pose": lidar_pose,
    "ego_lidar_pose": ego_lidar_pose,
}
```

其中：

- `boxes3d` 已投影到 ego 坐标系，是最终 NMS 使用的 3D corners。
- `boxes2d` 是由 3D corners 转成的 BEV standup box，供 overlap voting 和 track association 使用。
- `trust_id` 是 reputation 系统使用的稳定 ID，来源于 `ReputationManager.external_id()`。
- `original_cav_id` 用于解决 OpenCOOD 把 ego key 改写为 `"ego"` 后的 ID 对齐问题。

## 6. Leave-One-Out Voting

Voting 的作用是判断某个 CAV 的检测是否被其他 CAV 支持。当前实现使用 leave-one-out：

```text
评估目标 CAV A：
  reference = 所有 CAV 的检测 - A 自己的检测
  consensus = fuse(reference)
  consistency[A] = compare(A, consensus)
```

这样可以避免 CRB 原始逻辑中的自证问题：

```text
错误模式：
  A 的检测参与 fused reference
  再用 A 的检测去匹配这个 reference
  A 等于间接证明了自己
```

当前 voting 输出不直接作为最终预测框，只作为 reputation evidence：

```python
{
    "consistent": True,
    "reason": "matched",
    "reference_agent_count": 2,
    "matched_boxes": 38,
    "unmatched_boxes": 9,
    "consistency_ratio": 1.0,
}
```

当 reference CAV 数量不足时：

```python
{
    "consistent": None,
    "reason": "insufficient_reference_agents",
}
```

此时不更新 reputation，避免在缺少参考证据时误奖惩。

## 7. 物理一致性证据

第二阶段的主要增强是引入真实物理一致性，不再使用固定占位 residual。当前物理一致性包含三类证据。

### 7.1 Track Residual

粒度：每个 CAV 的每个 box。

来源：`TrackAssociation.update()`。

逻辑：

```text
同一 CAV 内跨帧关联 box
  -> 用上一帧 center 和 velocity 预测当前 center
  -> residual = 当前 center 到 predicted center 的距离
  -> residual 越小，motion_score 越高
```

如果某个 box 是新 track、历史不足或没有上一帧速度，则 residual 为 `None`，该 box 的 track evidence 不参与更新。

### 7.2 Pose Motion Score

粒度：每个 CAV。

来源：`MotionStateBuffer.update_pose()`。

逻辑：

```text
连续帧 lidar_pose.xy 位移 / dt
  -> velocity_xy
  -> speed
  -> pose_motion_score
```

该分数是 CAV 级证据，会参与该 CAV 下每个 box 的 motion score 计算。

需要注意：当前 `score_velocity()` 采用速度越大分数越低的形式，因此它更像是异常速度惩罚项，而不是普通车辆运动质量评分。`velocity_sigma` 过小会明显压低非 ego CAV 的物理分数，导致 reputation 下降和预测框过滤。

### 7.3 Consensus Residual

粒度：每个 CAV 的每个 box。

来源：`PhysicalConsistencyManager.compute_consensus_scores()`。

逻辑：

```text
当前 CAV 的 box center
  -> 找其他 CAV 同 label boxes
  -> 计算最近 center distance
  -> residual 越小，consensus_motion_score 越高
```

当没有其他 CAV 可作为 reference 时，返回：

```python
{
    "residual": None,
    "score": None,
    "reason": "no_reference_agent",
}
```

### 7.4 Box 级到 CAV 级聚合

`annotate_detections()` 会原地修改 `cav_detections`：

```python
det["physical_scores"] = per_box_scores
det["physical_score"] = mean(valid per_box_scores)
det["motion_score"] = mean(valid box_motion_score)
det["consensus_motion_score"] = mean(valid consensus_score)
```

其中：

- `physical_scores` 是每个 box 的物理一致性分数。
- `physical_score` 是该 CAV 当前帧所有有效 box 的平均物理分数。
- `motion_score` 由 track residual score 和 pose motion score 聚合得到。
- `consensus_motion_score` 是跨 CAV center residual 的平均分数。

这些字段后续会被 reputation 更新和日志系统读取。

## 8. Reputation 更新逻辑

`LateTrustFusion._update_reputations()` 是 reputation 更新入口。

当 `physical_consistency.use_physical_consistency: true` 时，更新流程是：

```text
voting_consistent
  -> voting_score: True=1.0, False=0.0, None=None

det.motion_score
det.consensus_motion_score
  -> PhysicalConsistencyManager.combine_evidence()
  -> evidence_score

evidence_score
  -> ReputationManager.update_from_evidence()
  -> reputation_after
```

融合权重来自配置：

```yaml
physical_consistency:
  weights:
    voting: 0.4
    motion: 0.3
    consensus_motion: 0.3
```

`combine_evidence()` 只融合非 `None` 的证据，并对实际参与的权重重新归一化。

`update_from_evidence()` 使用非对称更新：

```text
evidence_score >= good_thr:
  reputation += positive_rate * (evidence_score - reputation)

evidence_score <= bad_thr:
  reputation -= negative_rate * (reputation - evidence_score)

bad_thr < evidence_score < good_thr:
  reputation 不变

evidence_score is None:
  若 unknown_rate <= 0，则 reputation 不变
```

并通过 `max_per_frame_delta` 限制单帧最大变化，避免 reputation 瞬间剧烈波动。

当 `use_physical_consistency: false` 时，系统退化为只使用 leave-one-out voting：

```text
consistent=True  -> reputation += update_rate
consistent=False -> reputation -= update_rate
consistent=None  -> 不更新
```

ego CAV 始终使用 `ego_reputation`，不被在线更新惩罚。

## 9. Score Weighting 与过滤

信誉更新后，`LateTrustFusion._prepare_output_detections()` 会处理每个 CAV：

```python
keep = det["is_ego"] or reputation >= drop_below
weight = reputation ** score_power
weighted_scores = scores * weight
```

含义：

- ego 永远保留。
- 非 ego CAV 的 reputation 低于 `drop_below` 时，整车检测被过滤。
- 被保留 CAV 的所有 box score 会乘以 reputation 权重。
- 最终 NMS 会基于降权后的 score 做筛选。

因此 `drop_below` 是非常敏感的配置。阈值过高时，非 ego CAV 在冷启动或物理分数偏低时容易被整车过滤，AP 会明显下降。

当前推荐配置为：

```yaml
trust_fusion:
  drop_below: 0.5
  score_power: 1.0
```

## 10. 最终融合：保留 OpenCOOD Rotated NMS

当前第二阶段最终融合不使用 WBF，而是调用：

```python
pred_box_tensor, pred_score = merge_and_nms(
    trusted_detections,
    self.post_processor.params["nms_thresh"])
```

`merge_and_nms()` 执行：

```text
concat kept CAV boxes and scores
  -> remove_large_pred_bbx()
  -> remove_bbx_abnormal_z()
  -> nms_rotated()
  -> get_mask_for_boxes_within_range_torch()
  -> pred_box3d_tensor, pred_score
```

这保证最终输出仍是 OpenCOOD 评估函数需要的：

```text
pred_box3d_tensor: Tensor[M, 8, 3]
pred_score: Tensor[M]
gt_box_tensor: Tensor[K, 8, 3]
```

## 11. 外部 Reputation Source

当前 `reputation_source.py` 支持以下 source：

```yaml
trust_fusion:
  reputation_source:
    type: none      # none | json | diva_csv | rsu_http
    path: ""
```

### 11.1 JSON

支持两类 JSON：

```json
{
  "4288": 0.75,
  "4297": 0.42
}
```

或：

```json
{
  "reputations": {
    "4288": 0.75,
    "4297": 0.42
  }
}
```

也支持 list records：

```json
[
  {"vehicle_id": "4288", "reputation": 0.75},
  {"vehicle_id": "4297", "score": 0.42}
]
```

### 11.2 DIVA CSV

配置示例：

```yaml
trust_fusion:
  reputation_source:
    type: diva_csv
    path: "path/to/diva_reputation.csv"
    vehicle_id_column: vehicle_id
    reputation_column: reputation
```

CSV 会被转换为：

```text
external_id -> reputation
```

### 11.3 RSU HTTP

`rsu_http` 当前只保留接口，不在离线推理和单元测试中引入网络依赖。真实 RSU 同步应在后续独立实现，不应阻塞当前 trust_nms 推理链路。

## 12. 当前配置说明

`opencood/model_weight/pointpillar_late_fusion_trust/config.yaml` 中当前关键配置为：

```yaml
trust_fusion:
  use_trust_fusion: true
  mode: trust_nms
  consistency_mode: leave_one_out
  min_reference_agents: 1
  min_matched_boxes: 1
  iou_thr: 0.5
  skip_box_thr: 1.0e-4
  default_reputation: 0.5
  min_reputation: 0.0
  max_reputation: 1.0
  drop_below: 0.5
  score_power: 1.0
  ego_reputation: 1.0
  reputation_map: ""
  reputation_source:
    type: none
    path: ""
  id_map: ""
  log_reputation: true
  log_dir: "logs/trust/test4_Physcial"

reputation_update:
  positive_rate: 0.10
  negative_rate: 0.10
  unknown_rate: 0.0
  good_thr: 0.7
  bad_thr: 0.4
  max_per_frame_delta: 0.10

physical_consistency:
  use_physical_consistency: true
  frame_interval: 0.05
  residual_sigma: 5.0
  velocity_sigma: 10.0
  max_valid_speed: 40.0
  history_window: 5
  weights:
    voting: 0.4
    motion: 0.3
    consensus_motion: 0.3

track_association:
  max_center_distance: 4.0
  min_bev_iou: 0.1
  max_age: 3
```

关键解释：

- `mode` 当前只保留 `trust_nms`。如果配置成其他值，`LateTrustFusion` 会回落为 `trust_nms`。
- `drop_below` 控制非 ego CAV 是否整车过滤。
- `score_power` 控制 reputation 对检测分数的影响强度。
- `positive_rate` / `negative_rate` 控制 reputation 上升和下降速度。
- `velocity_sigma` 控制 CAV 速度惩罚敏感度，过小会导致物理分数偏低。
- `weights` 控制 voting、track/pose motion、cross-CAV consensus 三类证据的融合比例。

## 13. 日志输出

当 `log_reputation: true` 且 `log_dir` 非空时，系统写出三类日志。

### 13.1 reputation.jsonl

每帧一条，记录 CAV 级信誉变化：

```json
{
  "frame": 12,
  "mode": "trust_nms",
  "physical_enabled": true,
  "cavs": {
    "4288": {
      "reputation_before": 0.5,
      "voting_consistent": false,
      "physical_score": 0.32,
      "motion_score": 0.41,
      "consensus_motion_score": 0.24,
      "evidence_score": 0.21,
      "reputation_after": 0.47,
      "num_boxes_before": 43,
      "num_boxes_after": 0
    }
  }
}
```

### 13.2 physical.jsonl

每个 box 一条，记录物理一致性证据：

```json
{
  "frame": 12,
  "trust_id": "4288",
  "box_index": 7,
  "track_id": "4288-17",
  "residual": 6.2,
  "motion_score": 0.22,
  "pose_motion_score": 0.81,
  "consensus_residual": 5.1,
  "consensus_motion_score": 0.34,
  "physical_score": 0.28,
  "used_for_update": true,
  "reason": "matched"
}
```

### 13.3 frame_summary.csv

每帧一行，记录整体变化：

```text
frame,scenario_index,timestamp,num_cavs,num_boxes_before,num_boxes_after,num_filtered_cavs,mode
```

该文件适合快速分析：

- 每帧输入框数量和输出框数量。
- 被过滤 CAV 数量。
- trust 开启后是否出现大量预测框被删除。

当前没有 `wbf_clusters.jsonl`，因为 3D WBF 已从代码中移除。

## 14. 测试覆盖

当前 trust 相关测试包括：

```text
tests/test_trust_late_fusion.py
tests/test_trust_physical_consistency.py
tests/test_trust_reputation_source.py
tests/test_trust_track_association.py
```

建议回归命令：

```bash
python -m pytest \
  tests/test_trust_late_fusion.py \
  tests/test_trust_physical_consistency.py \
  tests/test_trust_reputation_source.py \
  tests/test_trust_track_association.py
```

覆盖重点：

- 低信誉 CAV 过滤。
- 中性信誉不改变基线输出过多。
- leave-one-out 不自证。
- reference 不足时返回 unknown。
- 物理 residual 转 score。
- track association 的新 track、匹配、残差计算。
- DIVA CSV / JSON reputation source 解析。
- `id_map` 与 external ID 对齐。

## 15. 实验解释边界

当前第二阶段实验应围绕以下三组进行：

```text
late_baseline:
  use_trust_fusion: false

late_trust_nms_loo:
  use_trust_fusion: true
  mode: trust_nms
  physical_consistency.use_physical_consistency: false

late_trust_nms_physical:
  use_trust_fusion: true
  mode: trust_nms
  physical_consistency.use_physical_consistency: true
```

不再把 `late_trust_3d_wbf` 作为当前主实验组。

分析 AP 变化时需要同时查看：

- AP@0.3 / AP@0.5 / AP@0.7。
- 每帧 `num_boxes_before` 和 `num_boxes_after`。
- `num_filtered_cavs`。
- 每个 CAV 的 `reputation_before` / `reputation_after`。
- `evidence_score` 是否长期低于 `bad_thr`。
- `motion_score` 是否因为 `velocity_sigma` 过小被压低。
- `consensus_motion_score` 是否因为跨车 center residual 偏大而持续偏低。

如果开启物理一致性后 AP 明显下降，优先检查：

1. `drop_below` 是否过高，导致非 ego CAV 冷启动后被整车过滤。
2. `velocity_sigma` 是否过小，导致正常车辆速度也被强惩罚。
3. `frame_interval` 是否与实际连续帧间隔一致。
4. `residual_sigma` 是否过小，导致轻微 track 误差被放大。
5. `weights` 是否让不稳定的物理证据压过 voting evidence。

## 16. 已移除的 WBF 路径

早期第二阶段方案曾设计 `trust_3d_wbf`：

```text
score / reputation / physical_score weighted 3D WBF
  -> cluster members
  -> weighted center / size / yaw
  -> fused 3D corners
```

但当前实验结论是：该路径在现有数据和实现下不如调优后的 `trust_nms` 稳定，尤其在高 IoU AP 上存在明显风险。因此当前代码做了以下收敛：

- 删除 `opencood/trust/wbf_3d_fusion.py`。
- 删除 `tests/test_trust_wbf_3d.py`。
- 删除 `LateTrustFusion` 中的 WBF 分支。
- 删除 YAML 中的 `wbf_3d`、`physical_power`、`reputation_power` 配置。
- `LateFusionDataset.post_process_trust()` 始终回到 `merge_and_nms()`。

维护原则：

- 不要在当前第二阶段文档中继续把 WBF 写成主线目标。
- 不要把 WBF 配置重新加入默认配置。
- 如果未来重新研究 WBF，应作为独立实验分支，不影响 `trust_nms` 稳定路径。

## 17. 当前验收标准

当前第二阶段代码应满足：

- `use_trust_fusion: false` 时保持原 late fusion 行为。
- `mode: trust_nms` 时启用 reputation weighting 和低信誉过滤。
- leave-one-out voting 避免 CAV 自证。
- physical consistency 使用连续帧 pose、track residual 和跨车 consensus residual。
- reputation 可以从 JSON / DIVA CSV 初始化。
- ego CAV 不被在线 reputation 更新惩罚。
- 最终输出保持 OpenCOOD 标准 `pred_box3d_tensor` 和 `pred_score`。
- trust 日志能解释每个 CAV 每帧的信誉变化原因。
- 单元测试覆盖 trust late fusion、physical consistency、reputation source 和 track association。

## 18. 后续维护建议

后续优化应优先围绕当前稳定链路进行：

1. 标定 `frame_interval`、`velocity_sigma`、`residual_sigma`，避免物理证据过度惩罚。
2. 对 `drop_below` 做 ablation，找到不过滤正常协作 CAV 的阈值。
3. 用日志统计不同场景下 `motion_score` 和 `consensus_motion_score` 的分布。
4. 对 `positive_rate` / `negative_rate` 做非对称调参，避免 reputation 快速涨满或快速跌穿阈值。
5. 保持 WBF 从主线移除，除非有新的实验结果证明它稳定优于 `trust_nms`。

