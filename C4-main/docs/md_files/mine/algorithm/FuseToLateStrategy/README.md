# CRB Trust 接入 OpenCOOD Late Fusion 技术框架

## 1. 文档目的

本文档总结当前 `OpenCOOD-main` 中 CRB trust 逻辑接入 late fusion 的实际代码框架，说明数据流、模块职责、配置含义、测试方法和维护边界。

当前版本保留并强化 `trust_nms` 路径：

```text
per-CAV detection
  -> leave-one-out overlap voting
  -> physical consistency evidence
  -> reputation update
  -> score weighting / low-reputation filtering
  -> OpenCOOD rotated NMS
```

此前实验中的 `trust_3d_wbf` 效果不稳定，当前代码已移除 3D WBF 实现和运行接口。后续技术描述以 `trust_nms` 为准。

## 2. 当前结论

CRB trust 更适合接在 OpenCOOD late fusion 的后处理阶段，而不是训练网络或 intermediate feature fusion 中。late fusion 推理时每个 CAV 都会独立 forward：

```python
for cav_id, cav_content in batch_data.items():
    output_dict[cav_id] = model(cav_content)
```

随后进入 `LateFusionDataset.post_process()`。此时可以拿到每个 CAV 的独立检测框、分数、类别、位姿和时间信息，正好用于信誉评估和物理一致性检查。

当前实现坚持三条原则：

1. 不改模型结构和 loss。
2. 不把 CRB 的 2D voting fused box 当成最终 3D 预测框。
3. 最终预测仍走 OpenCOOD 原有 3D rotated NMS，保证评估和可视化接口稳定。

## 3. 代码架构

当前 trust 代码集中在：

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
  - 解码每个 CAV 的模型输出。
  - 将 3D boxes 投影到 ego 坐标系。
  - 构造 `cav_detections`。
  - 调用 `LateTrustFusion.apply()`。
  - 调用 `merge_and_nms()` 输出最终检测结果。

- `late_trust_fusion.py`
  - trust-aware late fusion 主编排器。
  - 读取/写回 reputation。
  - 调用 overlap voting、物理一致性和信誉更新。
  - 生成 debug 和日志记录。
  - 当前只支持 `trust_nms` 语义；非 `trust_nms` mode 会回落到 `trust_nms`。

- `overlap_field_voting.py`
  - 根据其他 CAV 的检测形成 leave-one-out reference。
  - 判断目标 CAV 是否与参考共识一致。
  - 修复 CRB 原逻辑中目标 CAV 参与自证的问题。

- `physical_consistency_manager.py`
  - 计算 box 级物理一致性证据。
  - 包含 track residual、cross-CAV consensus residual、pose motion score 的融合。
  - 将 `physical_scores`、`motion_score`、`consensus_motion_score` 写回 `det`。

- `motion_state_buffer.py`
  - 维护每个 `scenario_index + trust_id` 的 CAV pose 历史。
  - 根据连续帧 `lidar_pose` 估计 CAV 自身速度。

- `track_association.py`
  - 对同一 CAV 的检测框做跨帧 greedy association。
  - 根据历史速度预测当前中心，得到 track residual。

- `reputation_manager.py`
  - 管理 CAV 级长期信誉。
  - 支持 voting 更新、综合 evidence 更新、外部 reputation source 初始化。

- `reputation_source.py`
  - 支持 JSON / DIVA CSV 形式的初始信誉来源。

- `reputation_cache.py`
  - 提供本地信誉缓存和可选服务同步接口。

- `id_mapper.py`
  - 将 OpenCOOD 内部 CAV ID 映射到外部信誉 ID。
  - 处理 ego 在 batch 中被改名为 `"ego"` 后仍需保留 `original_cav_id` 的问题。

- `trust_logger.py`
  - 写出 `reputation.jsonl`、`physical.jsonl`、`frame_summary.csv`。

## 4. Late Fusion 数据流

当前执行链路：

```text
inference.py
  -> inference_utils.inference_late_fusion()
      -> for each cav: model(cav_content)
      -> LateFusionDataset.post_process()
          -> post_process_trust()
              -> _decode_single_cav_detection()
              -> build frame_context
              -> LateTrustFusion.apply()
                  -> attach current reputation
                  -> compute leave-one-out voting details
                  -> compute track / pose / consensus physical evidence
                  -> update reputation
                  -> weight scores by reputation
                  -> filter CAVs below drop_below
              -> merge_and_nms()
              -> generate_gt_bbx()
```

核心位置是在最终 NMS 前：

```text
per-CAV projected boxes
  -> trust evidence
  -> reputation-weighted scores
  -> filtered detections
  -> OpenCOOD rotated NMS
```

这样做的好处是：trust 逻辑可以影响最终预测，但不破坏 OpenCOOD 的 3D box 格式、评估函数和可视化函数。

## 5. CavDetection 接口

`LateFusionDataset._decode_single_cav_detection()` 输出一个普通 dict：

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

`trust_id` 由 `ReputationManager.external_id()` 生成，用于对齐 OpenCOOD CAV ID、原始 CAV ID 和外部信誉 ID。

## 6. Overlap Voting

当前 voting 使用 leave-one-out：

```text
target CAV = A
reference CAVs = all CAVs except A
```

对每个目标 CAV：

1. 用其他 CAV 的 2D standup boxes 构造 reference consensus。
2. 将目标 CAV 的 boxes 与 reference fused boxes 比较 IoU。
3. 若匹配数量不足，判为不一致。
4. 若匹配后 label 一致比例大于阈值，判为一致。

输出写入 debug：

```python
{
    "voting_consistent": True,
    "voting_reason": "matched",
    "voting_reference_agent_count": 2,
    "voting_matched_boxes": 38,
    "voting_unmatched_boxes": 9,
    "voting_consistency_ratio": 1.0,
}
```

voting 只用于信誉/evidence 更新，不直接替换最终 3D boxes。

## 7. 物理一致性

当前物理一致性已经不再使用固定占位值，而是计算三类真实证据。

### 7.1 Track Residual

粒度：每个 CAV 的每个 box。

来源：`TrackAssociation.update()`。

逻辑：

```text
上一帧 track center + last_velocity * dt
  -> predicted_center
当前 box center 与 predicted_center 的距离
  -> track residual
```

残差越小，说明该目标运动越连续。

### 7.2 Consensus Residual

粒度：每个 CAV 的每个 box。

来源：`PhysicalConsistencyManager.compute_consensus_scores()`。

逻辑：

```text
当前 CAV 的 box center
  -> 找其他 CAV 的同类 boxes
  -> 计算最近 center distance
  -> consensus residual
```

残差越小，说明当前检测被其他 CAV 的空间观测支持。

### 7.3 Pose Motion Score

粒度：每个 CAV。

来源：`MotionStateBuffer.update_pose()`。

逻辑：

```text
连续帧 lidar_pose 的 xy 位移 / dt
  -> velocity_xy
  -> speed score
```

该分数会作为 CAV 级证据参与该 CAV 下每个 box 的 motion score。

### 7.4 写回字段

`annotate_detections()` 会原地修改 `cav_detections`：

```python
det["physical_scores"]          # 每个 box 的综合物理分数
det["physical_score"]           # 当前 CAV 的平均物理分数
det["motion_score"]             # 当前 CAV 的平均运动分数
det["consensus_motion_score"]   # 当前 CAV 的平均跨车一致性分数
```

其中 reputation 更新主要读取：

```python
det.get("motion_score")
det.get("consensus_motion_score")
```

## 8. Reputation 更新

每帧对每个 CAV 更新一次 reputation。ego 信誉固定为 `ego_reputation`。

未开启 physical 时：

```text
voting consistent     -> reputation += update_rate
voting inconsistent   -> reputation -= update_rate
no reference          -> reputation unchanged
```

开启 physical 时：

```python
evidence_score = combine_evidence(
    voting_score=voting_score,
    motion_score=det.get("motion_score"),
    consensus_motion_score=det.get("consensus_motion_score"),
)
```

默认权重：

```yaml
weights:
  voting: 0.4
  motion: 0.3
  consensus_motion: 0.3
```

`ReputationManager.update_from_evidence()` 使用非对称平滑更新：

```text
evidence_score >= good_thr  -> 向 evidence_score 上升
evidence_score <= bad_thr   -> 向 evidence_score 下降
中间区间                   -> 保持不变
```

并通过 `max_per_frame_delta` 限制单帧变化幅度。

## 9. 输出策略

当前最终输出策略固定为 trust-aware NMS：

```python
keep = is_ego or reputation >= drop_below
score = score * reputation ** score_power
```

之后调用：

```python
merge_and_nms(trusted_detections, nms_thresh)
```

`merge_and_nms()` 会：

1. 合并所有保留 CAV 的 projected 3D boxes。
2. 合并 reputation-weighted scores。
3. 移除异常大框和 z 异常框。
4. 执行 OpenCOOD rotated NMS。
5. 按感知范围过滤。

## 10. 配置模板

推荐 trust 模型目录使用：

```yaml
trust_fusion:
  use_trust_fusion: true
  mode: trust_nms
  consistency_mode: leave_one_out
  min_reference_agents: 1
  min_matched_boxes: 1
  iou_thr: 0.5
  skip_box_thr: 1.0e-4
  update_rate: 0.1
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
    vehicle_id_column: vehicle_id
    reputation_column: reputation
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

注意：

- `drop_below` 不宜过高。短序列冷启动时若设为 `0.9`，非 ego CAV 很容易长期被过滤。
- `velocity_sigma` 不宜过小。过小会把正常车辆速度误判为低 pose motion score。
- 当前不再配置 `wbf_3d`、`physical_power`、`reputation_power`。

## 11. 日志

开启：

```yaml
trust_fusion:
  log_reputation: true
  log_dir: "logs/trust/<exp_name>"
```

输出：

```text
reputation.jsonl
physical.jsonl
frame_summary.csv
```

`reputation.jsonl` 记录每帧每车信誉和 evidence：

```json
{
  "frame": 0,
  "mode": "trust_nms",
  "physical_enabled": true,
  "cavs": {
    "4288": {
      "reputation_before": 0.5,
      "evidence_score": 0.86,
      "motion_score": 0.99,
      "consensus_motion_score": 0.75,
      "reputation_after": 0.53,
      "num_boxes_before": 62,
      "num_boxes_after": 62
    }
  }
}
```

`physical.jsonl` 是 box 级物理证据，包含：

```text
trust_id
box_index
residual
motion_score
pose_motion_score
consensus_residual
consensus_motion_score
physical_score
reason
```

`frame_summary.csv` 记录每帧聚合指标：

```text
frame,scenario_index,timestamp,num_cavs,num_boxes_before,num_boxes_after,num_filtered_cavs,mode
```

## 12. 实验命令

短序列验证：

```bash
cd /home/wcp/c4/OpenCOOD-main
conda run -n torch118 python opencood/tools/inference.py \
  --model_dir opencood/model_weight/pointpillar_late_fusion_trust \
  --fusion_method late \
  --frame_index 1 \
  --num_frames 5 \
  --max_frames 10 \
  --save_vis_dir logs/trust/test4_Physcial \
  --color_mode constant \
  --headless \
  --num_workers 0
```

完整验证集：

```bash
conda run -n torch118 python opencood/tools/inference.py \
  --model_dir opencood/model_weight/pointpillar_late_fusion_trust \
  --fusion_method late \
  --max_frames 0 \
  --num_workers 0
```

`--frame_index` 和 `--num_frames` 主要控制定帧可视化导出；AP 统计由 `--max_frames` 控制。

## 13. 测试

当前关键单测：

```bash
cd /home/wcp/c4/OpenCOOD-main
conda run -n torch118 python -m pytest \
  tests/test_trust_late_fusion.py \
  tests/test_trust_reputation_source.py \
  tests/test_trust_physical_consistency.py \
  tests/test_trust_track_association.py \
  -q
```

覆盖内容：

- leave-one-out voting 不自证。
- 低信誉 CAV 过滤。
- 中性信誉不改变 score。
- JSON / DIVA CSV reputation source。
- 物理一致性 residual / score。
- track association 跨帧残差。

## 14. 验收标准

当前版本完成标准：

- `use_trust_fusion=false` 时不影响原 late fusion。
- `use_trust_fusion=true` 时能够完成 late inference。
- reputation 能按 CAV 维度跨帧更新。
- voting 使用 leave-one-out，不让目标 CAV 参与自证。
- physical consistency 使用真实 pose / track / consensus 证据。
- 低信誉非 ego CAV 会被过滤，保留 CAV 的 score 会按 reputation 加权。
- 最终输出仍为 OpenCOOD 标准 `pred_box3d_tensor, pred_score, gt_box_tensor`。
- 不影响 early / intermediate fusion。
- 不依赖 3D WBF。

## 15. 风险与维护边界

- 信誉状态必须在主进程后处理阶段维护，不要放入 DataLoader worker。
- `drop_below` 与 `positive_rate` 必须配套调参；短序列冷启动时高阈值会导致非 ego CAV 被过度过滤。
- `frame_interval` 应与数据集实际时间间隔一致，否则 pose velocity score 会失真。
- `velocity_sigma` 应按车辆正常速度范围设置，过小会误伤正常 CAV。
- `consensus_residual` 依赖其他 CAV 的检测覆盖；无参考 agent 时应返回 unknown，而不是惩罚。
- DIVA / RSU 外部 ID 必须通过 `id_map` 或 `original_cav_id` 对齐。
- 3D WBF 已从当前运行框架中删除；如未来重新评估，应作为独立实验分支实现，不应混入当前 `trust_nms` 主链路。
