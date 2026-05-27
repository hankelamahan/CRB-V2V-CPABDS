# -*- coding: utf-8 -*-
"""Overlap-field voting for detection-level trust updates."""

import numpy as np


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
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, float(box1[2]) - float(box1[0])) * \
            max(0.0, float(box1[3]) - float(box1[1]))
        area2 = max(0.0, float(box2[2]) - float(box2[0])) * \
            max(0.0, float(box2[3]) - float(box2[1]))
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0.0

    def vote_detection_level(self, boxes_list, scores_list, labels_list,
                             reputation_scores=None):
        flattened = []
        for agent_idx, boxes in enumerate(boxes_list):
            reputation = 1.0 if reputation_scores is None else \
                float(reputation_scores[agent_idx])
            for box_idx, box in enumerate(boxes):
                score = float(scores_list[agent_idx][box_idx])
                label = int(labels_list[agent_idx][box_idx])
                weight = score * max(reputation, 0.0)
                if weight <= self.skip_box_thr:
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
            if cluster['sum_weight'] <= self.skip_box_thr:
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
        consistent = 0
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
                consistent += 1

        unmatched = len(boxes) - matched
        ratio = float(consistent) / float(matched) if matched > 0 else 0.0 #cause the label always is 1,so the consistent is always equal to matched,so the ratio is always 1.0.
        if matched < self.min_matched_boxes:
            return {
                'consistent': False,
                'reason': 'insufficient_matched_boxes',
                'matched_boxes': matched,
                'unmatched_boxes': unmatched,
                'consistency_ratio': ratio,
            }
        return {
            'consistent': ratio > 0.7,
            'reason': 'matched',
            'matched_boxes': matched,
            'unmatched_boxes': unmatched,
            'consistency_ratio': ratio,
        }
