# -*- coding: utf-8 -*-
"""Trust-aware late fusion score weighting and reputation updates."""

import json
import os

import torch

from opencood.trust.overlap_field_voting import OverlapFieldVotingSystem
from opencood.trust.reputation_manager import ReputationManager


class LateTrustFusion:
    """Apply centralized reputation and overlap voting to late detections."""

    def __init__(self, trust_config=None, physical_config=None):
        """Create trust fusion state.

        ``physical_config`` is accepted by the dataset integration point, but
        physical consistency is not part of the current late-trust path.
        """
        trust_config = trust_config or {}
        self.enabled = bool(trust_config.get('use_trust_fusion', False))
        self.drop_below = float(trust_config.get(
            'drop_below', trust_config.get('min_reputation', 0.3)))
        self.score_power = float(trust_config.get('score_power', 1.0))
        self.log_reputation = bool(trust_config.get('log_reputation', False))
        self.log_dir = trust_config.get('log_dir', '')
        self.frame_counter = 0

        self.reputation_manager = ReputationManager(trust_config)
        self.voting_system = OverlapFieldVotingSystem(
            iou_thr=trust_config.get('iou_thr', 0.5),
            skip_box_thr=trust_config.get('skip_box_thr', 1e-4),
        )

    def apply(self, cav_detections):
        """Update reputation and weight detection scores in-place logically."""
        if not self.enabled or not cav_detections:
            return cav_detections, {}

        debug = {
            'frame': self.frame_counter,
            'consistency_mode': 'leave_one_out',
            'cavs': {},
        }
        detections_dict = self._build_voting_inputs(cav_detections)
        debug['num_voting_agents'] = len(detections_dict)
        debug['num_voting_boxes'] = sum(
            len(detections['boxes'])
            for detections in detections_dict.values())
        consistency = self._update_reputations(cav_detections,
                                               detections_dict)

        weighted_detections = []
        for det in cav_detections:
            reputation = float(det.get('reputation',
                                       det.get('reputation_before', 1.0)))
            keep = det.get('is_ego', False) or reputation >= self.drop_below
            boxes_before = int(det['boxes3d'].shape[0])
            boxes_after = boxes_before if keep else 0
            if keep and boxes_before > 0:
                weighted = dict(det)
                weight = reputation ** self.score_power
                weighted['scores'] = det['scores'] * weight
                weighted_detections.append(weighted)

            debug['cavs'][str(det['trust_id'])] = {
                'cav_id': str(det['cav_id']),
                'original_cav_id': str(det.get('original_cav_id',
                                               det['cav_id'])),
                'is_ego': bool(det.get('is_ego', False)),
                'reputation_before': float(det.get('reputation_before',
                                                   reputation)),
                'reputation_after': reputation,
                'voting_consistent': consistency.get(det['trust_id']),
                'num_boxes_before': boxes_before,
                'num_boxes_after': boxes_after,
            }

        self._write_log(debug)
        self.frame_counter += 1
        return weighted_detections, debug

    def _build_voting_inputs(self, cav_detections):
        """Attach current reputation and build CPU voting inputs."""
        detections_dict = {}
        for det in cav_detections:
            reputation = self.reputation_manager.get_reputation(
                det['cav_id'],
                det.get('original_cav_id'),
                det.get('is_ego', False))
            det['reputation_before'] = reputation
            det['reputation'] = reputation

            if det['boxes2d'].shape[0] == 0:
                continue

            detections_dict[det['trust_id']] = {
                'boxes': det['boxes2d'].detach().cpu().numpy().tolist(),
                'scores': det['scores'].detach().cpu().numpy().tolist(),
                'labels': det['labels'].detach().cpu().numpy().astype(
                    int).tolist(),
                'reputation': reputation,
            }
        return detections_dict

    def _update_reputations(self, cav_detections, detections_dict):
        """Update CAV reputation from leave-one-out voting consistency."""
        if not detections_dict:
            return {}

        consistency = self.voting_system.compute_consistency_leave_one_out(
            detections_dict,
            iou_thr=self.voting_system.voter.iou_thr)
        for det in cav_detections:
            trust_id = det['trust_id']
            is_consistent = consistency.get(trust_id)
            if is_consistent is None:
                continue

            det['reputation'] = self.reputation_manager.update_from_voting(
                det['cav_id'],
                is_consistent,
                det.get('original_cav_id'),
                det.get('is_ego', False))
        return consistency

    def _write_log(self, debug):
        if not self.log_reputation or not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, 'reputation.jsonl')
        with open(path, 'a') as f:
            f.write(json.dumps(debug, sort_keys=True) + '\n')


def merge_and_nms(cav_detections, nms_thresh):
    """Merge per-CAV projected 3D boxes and run OpenCOOD rotated NMS."""
    from opencood.utils import box_utils

    final_boxes = []
    final_scores = []
    for det in cav_detections:
        if det['boxes3d'].shape[0] == 0:
            continue
        final_boxes.append(det['boxes3d'])
        final_scores.append(det['scores'])

    if not final_boxes:
        return None, None

    pred_box3d_tensor = torch.vstack(final_boxes)
    scores_tensor = torch.cat(final_scores).to(
        device=pred_box3d_tensor.device,
        dtype=pred_box3d_tensor.dtype)

    keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
    keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
    keep_index = torch.logical_and(keep_index_1, keep_index_2)
    pred_box3d_tensor = pred_box3d_tensor[keep_index]
    scores_tensor = scores_tensor[keep_index]

    if pred_box3d_tensor.shape[0] == 0:
        return None, None

    keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                       scores_tensor,
                                       nms_thresh)
    pred_box3d_tensor = pred_box3d_tensor[keep_index]
    scores_tensor = scores_tensor[keep_index]

    mask = box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor)
    pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
    scores_tensor = scores_tensor[mask]

    if pred_box3d_tensor.shape[0] == 0:
        return None, None

    return pred_box3d_tensor, scores_tensor
