# -*- coding: utf-8 -*-
"""Overlap-field voting for detection-level trust updates."""

import numpy as np
import sys  # 添加这个导入


class OverlapFieldVoter:
    """Cluster 2D detections with score-reputation weighted voting."""

    def __init__(self, iou_thr=0.5, skip_box_thr=1e-4):
        self.iou_thr = float(iou_thr)
        self.skip_box_thr = float(skip_box_thr)

    @staticmethod
    def empty_output():
        """Return the standard empty voting output tuple."""
        return np.zeros((0, 4), dtype=np.float32), \
            np.zeros((0,), dtype=np.float32), \
            np.zeros((0,), dtype=np.int32)

    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate IoU between two 2D boxes [x1, y1, x2, y2]."""
        # 直接计算，假设输入已经是数值类型
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        # 面积直接计算，不需要max（因为框本身是有效的）
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter_area
        
        return inter_area / union if union > 0.0 else 0.0

    def vote_detection_level(self, boxes_list, scores_list, labels_list,
                             reputation_scores=None):
        # ✅ 新增：输入验证
        # 检查是否为空
        if not boxes_list:
            return self.empty_output()
        
        # 检查三个列表长度是否一致
        if not (len(boxes_list) == len(scores_list) == len(labels_list)):
            raise ValueError(
                f"boxes_list, scores_list, labels_list must have same length. "
                f"Got {len(boxes_list)}, {len(scores_list)}, {len(labels_list)}"
            )
        
        # 检查信誉分长度是否匹配
        if reputation_scores is not None and len(reputation_scores) != len(boxes_list):
            raise ValueError(
                f"reputation_scores length ({len(reputation_scores)}) "
                f"must match boxes_list length ({len(boxes_list)})"
            )

        flattened = []
        for agent_idx, boxes in enumerate(boxes_list):
            reputation = 1.0 if reputation_scores is None else \
                float(reputation_scores[agent_idx])
            for box_idx, box in enumerate(boxes):
                score = float(scores_list[agent_idx][box_idx])
                label = int(labels_list[agent_idx][box_idx])
                weight = score * max(reputation, 0.0)
                if weight <= self.skip_box_thr + sys.float_info.epsilon:
                    continue
                flattened.append({
                    'box': np.asarray(box, dtype=np.float32),
                    'score': score,
                    'label': label,
                    'weight': weight,
                })

        if not flattened:
            return self.empty_output()

        flattened.sort(key=lambda item: item['weight'], reverse=True)
        clusters = []
        for det in flattened:
            assigned = False
            for cluster in clusters:
                if det['label'] != cluster['label']:
                    continue
                if self.calculate_iou(det['box'], cluster['mean_box']) < \
                        self.iou_thr:
                    continue
                cluster['box_sum'] += det['box'] * det['weight']
                cluster['score_sum'] += det['score'] * det['weight']
                cluster['sum_weight'] += det['weight']
                cluster['mean_box'] = cluster['box_sum'] / \
                    cluster['sum_weight']
                assigned = True
                break
            if not assigned:
                clusters.append({
                    'label': det['label'],
                    'mean_box': det['box'].copy(),
                    'box_sum': det['box'] * det['weight'],
                    'score_sum': det['score'] * det['weight'],
                    'sum_weight': det['weight'],
                })

        fused_boxes = []
        fused_scores = []
        fused_labels = []
        for cluster in clusters:
            if cluster['sum_weight'] <= self.skip_box_thr + sys.float_info.epsilon:
                continue
            fused_boxes.append(cluster['mean_box'])
            fused_scores.append(cluster['score_sum'] / cluster['sum_weight'])
            fused_labels.append(cluster['label'])

        if not fused_boxes:
            return self.empty_output()

        return np.asarray(fused_boxes, dtype=np.float32), \
            np.asarray(fused_scores, dtype=np.float32), \
            np.asarray(fused_labels, dtype=np.int32)


class OverlapFieldVotingSystem:
    """Build voting consensus and check leave-one-out consistency."""

    def __init__(self, iou_thr=0.5, skip_box_thr=1e-4,
                 min_reference_agents=1, min_matched_boxes=1):
        self.voter = OverlapFieldVoter(iou_thr=iou_thr,
                                       skip_box_thr=skip_box_thr)
        self.min_reference_agents = int(min_reference_agents)
        self.min_matched_boxes = int(min_matched_boxes)

    def fuse(self, detections_dict):
        agent_ids = list(detections_dict.keys())
        reputations = [detections_dict[agent_id].get('reputation', 1.0)
                       for agent_id in agent_ids]
        boxes_list = [detections_dict[agent_id]['boxes']
                      for agent_id in agent_ids]
        scores_list = [detections_dict[agent_id]['scores']
                       for agent_id in agent_ids]
        labels_list = [detections_dict[agent_id]['labels']
                       for agent_id in agent_ids]
        return self.voter.vote_detection_level(boxes_list, scores_list,
                                               labels_list, reputations)

    def compute_consistency_leave_one_out(self, detections_dict, iou_thr=0.5):
        """Evaluate each agent against consensus formed without itself.

        Returns
        -------
        dict
            ``agent_id -> bool | None``. ``None`` means there is no reference
            detection set, so the caller should not update that agent's
            reputation for this frame.
        """
        details = self.compute_consistency_details(detections_dict,
                                                   iou_thr=iou_thr)
        return {
            agent_id: item['consistent']
            for agent_id, item in details.items()
        }

    def compute_consistency_details(self, detections_dict, iou_thr=0.5):
        """Return leave-one-out consistency and debug metadata."""
        details = {}
        for target_id, target_det in detections_dict.items():
            reference_detections = {
                agent_id: detections
                for agent_id, detections in detections_dict.items()
                if agent_id != target_id
            }
            reference_count = len(reference_detections)
            if reference_count < self.min_reference_agents:
                details[target_id] = {
                    'consistent': None,
                    'reason': 'insufficient_reference_agents',
                    'reference_agent_count': reference_count,
                    'matched_boxes': 0,
                    'unmatched_boxes': len(target_det.get('boxes', [])),
                    'consistency_ratio': None,
                }
                continue

            fused_reference = self.fuse(reference_detections)
            details[target_id] = self.compare_to_fused_details(
                target_det, fused_reference, iou_thr=iou_thr)
            details[target_id]['reference_agent_count'] = reference_count
        return details

    def compare_to_fused(self, detections, fused_output, iou_thr=0.5):
        """Return whether one agent's detections agree with fused boxes."""
        return self.compare_to_fused_details(
            detections,
            fused_output,
            iou_thr=iou_thr)['consistent']

    def compare_to_fused_details(self, detections, fused_output, iou_thr=0.5):
        """Return consistency plus matched/unmatched debug counts."""
        fused_boxes, _, fused_labels = fused_output
        boxes = detections.get('boxes', [])
        labels = detections.get('labels', [])
        if len(fused_boxes) == 0 or len(boxes) == 0:
            return {
                'consistent': False,
                'reason': 'empty_reference_or_target',
                'matched_boxes': 0,
                'unmatched_boxes': len(boxes),
                'consistency_ratio': 0.0,
            }

        matched = 0
        label_matched = 0  # 改名，更清晰
        for box_idx, box in enumerate(boxes):
            label = labels[box_idx] if box_idx < len(labels) else None
            best_idx = None
            best_iou = 0.0
            for fused_idx, fused_box in enumerate(fused_boxes):
                iou = self.voter.calculate_iou(box, fused_box)
                if iou > iou_thr and iou > best_iou:
                    best_idx = fused_idx
                    best_iou = iou
            if best_idx is None:
                continue
            matched += 1
            if label == fused_labels[best_idx]:
                label_matched += 1

        unmatched = len(boxes) - matched
        # ✅ 修复：用"匹配率" = 匹配数量 / 总检测数
        match_ratio = float(matched) / float(len(boxes)) if len(boxes) > 0 else 0.0
        # 额外提供标签匹配率（调试用）
        label_ratio = float(label_matched) / float(matched) if matched > 0 else 0.0

        if matched < self.min_matched_boxes:
            return {
                'consistent': False,
                'reason': 'insufficient_matched_boxes',
                'matched_boxes': matched,
                'unmatched_boxes': unmatched,
                'consistency_ratio': match_ratio,  # ✅ 用匹配率
                'label_consistency': label_ratio,  # ✅ 新增标签一致性
            }
        return {
            'consistent': match_ratio > 0.7,  # ✅ 用匹配率判断
            'reason': 'matched',
            'matched_boxes': matched,
            'unmatched_boxes': unmatched,
            'consistency_ratio': match_ratio,
            'label_consistency': label_ratio,
        }
