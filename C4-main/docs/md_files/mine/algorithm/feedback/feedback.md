# CRB Late Trust 一致性判断自我证明问题反馈

## 1. 问题摘要

当前 CRB late trust 逻辑中的重叠视场投票一致性判断存在一个核心问题：

```text
某个 CAV 的检测框先参与生成 fused boxes，
随后又用该 CAV 自己的检测框去匹配这些 fused boxes，
并据此判断该 CAV 是否 consistent。
```

这会导致“自我证明”现象。也就是说，一个车辆的检测结果可能因为参与了融合结果本身，而更容易被判定为一致，从而提升自己的信誉分数。

该问题不是 OpenCOOD 接入时新引入的，而是 CRB 当前提供的 late trust 实现逻辑中已经存在。OpenCOOD 第一阶段接入时沿用了该逻辑，因此也继承了这个问题。

## 2. 当前逻辑流程

当前 late trust 推理链路大致是：

```text
每个 CAV 独立检测
  -> 得到每车 boxes / scores / labels
  -> 所有 CAV 的检测结果组成 detections_dict
  -> fuse(detections_dict) 生成 fused boxes
  -> 每个 CAV 再拿自己的 boxes 去匹配 fused boxes
  -> 若匹配比例超过阈值，则该 CAV 被判 consistent
  -> consistent 则 reputation += update_rate
  -> inconsistent 则 reputation -= update_rate
```

问题发生在：

```text
fused boxes 是由所有 CAV 共同生成的，包含待评估 CAV 自己。
```

因此，在评估 CAV A 是否可信时，融合结果中已经包含了 CAV A 的检测框。

## 3. CRB 源码证据

### 3.1 CRB late_fusion_dataset 中的调用方式

CRB 文件：

```text
CRB-V2V-CPABDS-master/CRB-V2V-CPABDS-master/C4-main/opencood/data_utils/datasets/late_fusion_dataset.py
```

相关逻辑：

```python
if self.use_trust_fusion and self.voting_system is not None and detections_dict:
    fused_output = self.voting_system.fuse(detections_dict)
    self.voting_system.update_reputations(fused_output, detections_dict)
```

这里的 `detections_dict` 包含当前帧所有 CAV 的检测结果。`fuse(detections_dict)` 使用全体 CAV 生成 `fused_output`，随后 `update_reputations(fused_output, detections_dict)` 再用同一批 CAV 的原始检测去更新信誉。

### 3.2 CRB overlap_field_voting 中 fuse 使用所有 agent

CRB 文件：

```text
CRB-V2V-CPABDS-master/CRB-V2V-CPABDS-master/C4-main/opencood/data_utils/datasets/overlap_field_voting.py
```

相关逻辑：

```python
def fuse(self, detections_dict):
    agent_ids = list(detections_dict.keys())
    reputation_scores = [self.get_reputation(aid) for aid in agent_ids]

    boxes_list = [detections_dict[aid]['boxes'] for aid in agent_ids]
    scores_list = [detections_dict[aid]['scores'] for aid in agent_ids]
    labels_list = [detections_dict[aid]['labels'] for aid in agent_ids]

    return self.voter.vote_detection_level(
        boxes_list,
        scores_list,
        labels_list,
        reputation_scores
    )
```

这里没有排除正在被评估的 CAV。

### 3.3 CRB update_reputations 再逐车回查一致性

同一文件中：

```python
def update_reputations(self, fused_output, detections_dict):
    agent_ids = list(detections_dict.keys())
    return self.reputation_manager.batch_update_from_voting(
        fused_output, detections_dict, agent_ids
    )
```

`batch_update_from_voting` 中再逐个车辆判断：

```python
for vehicle_id in vehicle_ids:
    detections = original_detections.get(vehicle_id, {})
    vehicle_boxes = detections.get('boxes', [])
    vehicle_labels = detections.get('labels', [])

    for i, vbox in enumerate(vehicle_boxes):
        for j, fbox in enumerate(fused_boxes):
            iou = self._calculate_iou(vbox, fbox)
            if iou > iou_thr:
                total_matchable += 1
                if vlabel == fused_labels[j]:
                    consistent_count += 1
                break

    is_consistent = total_matchable > 0 and (consistent_count / total_matchable) > 0.7
    self.update_from_voting_consistency(vehicle_id, is_consistent)
```

这里的 `fused_boxes` 已经包含该 `vehicle_id` 自己参与投票得到的结果。

## 4. OpenCOOD 当前接入状态

OpenCOOD 第一阶段接入后，为了保持与 CRB 逻辑一致，也采用了相同的流程：

```text
fuse(all CAV detections)
  -> compute_consistency(all CAV detections against fused boxes)
  -> update reputation
```

OpenCOOD 文件：

```text
OpenCOOD-main/opencood/trust/late_trust_fusion.py
OpenCOOD-main/opencood/trust/overlap_field_voting.py
```

对应逻辑：

```python
fused_output = self.voting_system.fuse(detections_dict)
consistency = self.voting_system.compute_consistency(
    fused_output,
    detections_dict,
    iou_thr=self.voting_system.voter.iou_thr
)
```

因此当前 OpenCOOD-main 不是新增了这个问题，而是继承了 CRB 算法逻辑中的该问题。

## 5. 实验现象

在 `pointpillar_late_fusion` 与 `pointpillar_late_fusion_trust` 上分别运行前 50 帧推理：

```text
official late:
AP@0.3 = 0.9570409296
AP@0.5 = 0.9529257472
AP@0.7 = 0.8653294528

trust late:
AP@0.3 = 0.9570409296
AP@0.5 = 0.9529257472
AP@0.7 = 0.8672298021
```

结果显示：

```text
AP@0.3 完全相同
AP@0.5 完全相同
AP@0.7 仅有极小提升
```

进一步查看 trust 日志：

```text
frames: 50
agents: 3

4279 ego:
  consistent 50 / 50
  reputation 1.0 -> 1.0
  boxes 2616 -> 2616
  filtered_frames 0

4288 cav:
  consistent 50 / 50
  reputation 0.5 -> 1.0
  boxes 3078 -> 3078
  filtered_frames 0

4297 cav:
  consistent 50 / 50
  reputation 0.5 -> 1.0
  boxes 2204 -> 2204
  filtered_frames 0
```

现象说明：

```text
所有协作车每一帧都被判定为 consistent；
非 ego CAV 信誉很快从 0.5 增长到 1.0；
没有任何车辆或检测框被过滤；
trust 策略很快退化为普通 late fusion。
```

## 6. 影响分析

该问题会带来以下影响：

1. 信誉过快上涨

   正常情况下，协作车只要连续几帧被判 consistent，就会快速涨到 1.0。

2. 异常车辆可能自证一致

   如果某辆车上报的异常框参与了 fused boxes 的生成，那么该车辆后续用自己的框去匹配 fused boxes 时，可能仍然被判定为一致。

3. 策略区分度不足

   当前 trust 策略在正常数据上几乎不改变结果；在异常数据上，也可能因为自我证明而降低恶意检测敏感性。

4. 低信誉过滤不易触发

   因为车辆较容易被判 consistent，reputation 会持续升高，难以下降到 `drop_below` 阈值以下。

## 7. 推荐修复方案：Leave-One-Out 一致性判断

建议将一致性判断改为 leave-one-out voting。

核心思想：

```text
评估 CAV A 是否 consistent 时，
不能使用 CAV A 自己参与生成的 fused boxes。

应使用除 CAV A 之外的其他 CAV 检测结果生成 consensus，
再用 CAV A 的检测结果去匹配该 consensus。
```

推荐流程：

```text
for each target_cav in detections_dict:
    reference_detections = detections_dict excluding target_cav

    if reference_detections is empty:
        target_cav 不更新或标记为 unknown

    fused_without_target = fuse(reference_detections)

    consistency[target_cav] = compare(
        target_cav_detections,
        fused_without_target
    )

    update reputation by consistency[target_cav]
```

伪代码：

```python
def compute_consistency_leave_one_out(detections_dict):
    consistency = {}

    for target_id, target_det in detections_dict.items():
        ref_dets = {
            cav_id: det
            for cav_id, det in detections_dict.items()
            if cav_id != target_id
        }

        if len(ref_dets) == 0:
            consistency[target_id] = None
            continue

        fused_ref = fuse(ref_dets)
        consistency[target_id] = compare_to_fused(
            target_det,
            fused_ref
        )

    return consistency
```

