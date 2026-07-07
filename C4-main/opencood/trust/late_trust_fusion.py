# -*- coding: utf-8 -*-
"""Trust-aware late fusion orchestration."""

import torch

from opencood.trust.motion_state_buffer import MotionStateBuffer
from opencood.trust.overlap_field_voting import OverlapFieldVotingSystem
from opencood.trust.physical_consistency_manager import (
    PhysicalConsistencyManager,
)
from opencood.trust.reputation_manager import ReputationManager
from opencood.trust.track_association import TrackAssociation
from opencood.trust.trust_logger import TrustLogger


class LateTrustFusion:
    """Coordinate reputation, physical evidence and final late fusion."""

    def __init__(self, trust_config=None, physical_config=None,
                 reputation_update_config=None, track_config=None):
        trust_config = trust_config or {}
        physical_config = physical_config or {}
        track_config = track_config or {}
        if reputation_update_config:
            trust_config = dict(trust_config)
            trust_config['reputation_update'] = reputation_update_config

        self.enabled = bool(trust_config.get('use_trust_fusion', False))
        self.mode = trust_config.get('mode', 'trust_nms')
        if self.mode != 'trust_nms':
            self.mode = 'trust_nms'
        self.drop_below = float(trust_config.get(
            'drop_below', trust_config.get('min_reputation', 0.3)))
        self.score_power = float(trust_config.get('score_power', 1.0))
        self.frame_counter = 0
        self.use_physical = bool(physical_config.get(
            'use_physical_consistency', False))

        self.reputation_manager = ReputationManager(trust_config)
        self.voting_system = OverlapFieldVotingSystem(
            iou_thr=trust_config.get('iou_thr', 0.5),
            skip_box_thr=trust_config.get('skip_box_thr', 1e-4),
            min_reference_agents=trust_config.get('min_reference_agents', 1),
            min_matched_boxes=trust_config.get('min_matched_boxes', 1),
        )
        motion_config = dict(physical_config)
        track_runtime_config = dict(physical_config)
        track_runtime_config.update(track_config)
        self.motion_buffer = MotionStateBuffer(motion_config)
        self.track_association = TrackAssociation(track_runtime_config)
        self.physical_manager = PhysicalConsistencyManager(physical_config)

        self.logger = TrustLogger(
            enabled=trust_config.get('log_reputation', False),
            log_dir=trust_config.get('log_dir', ''))

    def apply(self, cav_detections, frame_context=None):
        """Apply trust pipeline and return detections plus debug metadata."""
        if not self.enabled or not cav_detections:
            return cav_detections, {}

        frame_context = self._frame_context(cav_detections, frame_context)
        debug = {
            'frame': self.frame_counter,
            'mode': self.mode,
            'consistency_mode': 'leave_one_out',
            'physical_enabled': self.use_physical,
            'cavs': {},
            'physical': [],
        }
        #init the reputation
        detections_dict = self._attach_reputation_and_voting_inputs(
            cav_detections)
        voting_details = self._compute_voting(detections_dict)
        debug['num_voting_agents'] = len(detections_dict)
        debug['num_voting_boxes'] = sum(
            len(detections['boxes'])
            for detections in detections_dict.values())

        physical_records = []
        if self.use_physical:
            track_evidence = self._compute_track_evidence(cav_detections)
            physical_records = self.physical_manager.annotate_detections(
                cav_detections,
                track_evidence_by_id=track_evidence,
                frame_context=frame_context)
            debug['physical'] = physical_records

        self._update_reputations(cav_detections, voting_details,
                                 use_physical=self.use_physical)
        output_detections = self._prepare_output_detections(cav_detections)
        self._fill_debug(debug, cav_detections, voting_details,
                         output_detections, frame_context)

        self._write_logs(debug, physical_records)
        self.frame_counter += 1
        return output_detections, debug

    def _attach_reputation_and_voting_inputs(self, cav_detections):
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

    def _compute_voting(self, detections_dict):
        if not detections_dict:
            return {}
        return self.voting_system.compute_consistency_details(
            detections_dict,
            iou_thr=self.voting_system.voter.iou_thr)

    def _compute_track_evidence(self, cav_detections):
        track_evidence = {}
        for det in cav_detections:
            trust_id = str(det['trust_id'])
            pose_evidence = self.motion_buffer.update_pose(
                det.get('scenario_index', -1),
                trust_id,
                det.get('timestamp_index', -1),
                timestamp=det.get('timestamp', ''),
                lidar_pose=det.get('lidar_pose'))
            det['pose_evidence'] = pose_evidence
            det['pose_motion_score'] = self._pose_motion_score(pose_evidence)
            track_evidence[trust_id] = self.track_association.update(
                det.get('scenario_index', -1),
                trust_id,
                det.get('timestamp_index', -1),
                det['boxes3d'],
                det.get('boxes2d'))
        return track_evidence

    def _pose_motion_score(self, pose_evidence):
        if not pose_evidence:
            return None
        if pose_evidence.get('status') == 'valid':
            return self.physical_manager.score_velocity(
                pose_evidence.get('velocity_xy'))
        if pose_evidence.get('reason') == 'speed_exceeds_limit':
            return 0.0
        return None

    def _update_reputations(self, cav_detections, voting_details,
                        use_physical=False):
        for det in cav_detections:
            trust_id = det['trust_id']
            voting_detail = voting_details.get(trust_id, {})
            is_consistent = voting_detail.get('consistent')
            
            if use_physical:
                voting_score = None if is_consistent is None else \
                    float(bool(is_consistent))
                evidence_score = self.physical_manager.combine_evidence(
                    voting_score=voting_score,
                    motion_score=det.get('motion_score'),
                    consensus_motion_score=det.get('consensus_motion_score'))
                det['evidence_score'] = evidence_score
                
                # ✅ 修复：如果 evidence_score 为 None，保持信誉不变
                if evidence_score is None:
                    det['reputation'] = det.get('reputation_before', det.get('reputation', 1.0))
                else:
                    det['reputation'] = \
                        self.reputation_manager.update_from_evidence(
                            det['cav_id'],
                            evidence_score,
                            det.get('original_cav_id'),
                            det.get('is_ego', False))
                continue

            # ✅ 修复：当 is_consistent is None 时，也更新 reputation
            if is_consistent is None:
                det['evidence_score'] = None
                # ✅ 新增：保持信誉不变（不更新）
                det['reputation'] = det.get('reputation_before', det.get('reputation', 1.0))
                continue
                
            det['evidence_score'] = float(bool(is_consistent))
            det['reputation'] = self.reputation_manager.update_from_voting(
                det['cav_id'],
                is_consistent,
                det.get('original_cav_id'),
                det.get('is_ego', False))

    def _prepare_output_detections(self, cav_detections):
        output = []
        for det in cav_detections:
            # ✅ 修复：明确优先级，并确保有默认值
            reputation = det.get('reputation')
            if reputation is None:
                reputation = det.get('reputation_before')
            if reputation is None:
                # 如果都没有，使用当前信誉管理器中的值
                reputation = self.reputation_manager.get_reputation(
                    det['cav_id'],
                    det.get('original_cav_id'),
                    det.get('is_ego', False))
            reputation = float(reputation)
            
            keep = det.get('is_ego', False) or reputation >= self.drop_below
            if not keep:
                continue
            weighted = dict(det)
            weight = reputation ** self.score_power
            weighted['scores'] = det['scores'] * weight
            output.append(weighted)
        return output

    def _fill_debug(self, debug, cav_detections, voting_details,
                    output_detections, frame_context):
        kept_ids = {str(det['trust_id']) for det in output_detections}
        for det in cav_detections:
            trust_id = str(det['trust_id'])
            voting_detail = voting_details.get(det['trust_id'], {})
            boxes_before = int(det['boxes3d'].shape[0])
            boxes_after = boxes_before if trust_id in kept_ids else 0
            debug['cavs'][trust_id] = {
                'cav_id': str(det['cav_id']),
                'original_cav_id': str(det.get('original_cav_id',
                                               det['cav_id'])),
                'is_ego': bool(det.get('is_ego', False)),
                'reputation_before': float(det.get('reputation_before',
                                                   det.get('reputation',
                                                           1.0))),
                'voting_consistent': voting_detail.get('consistent'),
                'voting_reason': voting_detail.get('reason'),
                'voting_reference_agent_count': voting_detail.get(
                    'reference_agent_count'),
                'voting_matched_boxes': voting_detail.get('matched_boxes'),
                'voting_unmatched_boxes': voting_detail.get(
                    'unmatched_boxes'),
                'voting_consistency_ratio': voting_detail.get(
                    'consistency_ratio'),
                'physical_score': det.get('physical_score'),
                'motion_score': det.get('motion_score'),
                'consensus_motion_score': det.get('consensus_motion_score'),
                'evidence_score': det.get('evidence_score'),
                'reputation_after': float(det.get('reputation',
                                                  det.get(
                                                      'reputation_before',
                                                      1.0))),
                'num_boxes_before': boxes_before,
                'num_boxes_after': boxes_after,
            }
        debug['summary'] = {
            'frame': self.frame_counter,
            'scenario_index': frame_context.get('scenario_index', -1),
            'timestamp': frame_context.get('timestamp', ''),
            'num_cavs': len(cav_detections),
            'num_boxes_before': sum(int(det['boxes3d'].shape[0])
                                    for det in cav_detections),
            'num_boxes_after': sum(int(det['boxes3d'].shape[0])
                                   for det in output_detections),
            'num_filtered_cavs': len(cav_detections) - len(output_detections),
            'mode': self.mode,
        }

    def _write_logs(self, debug, physical_records):
        self.logger.log_reputation(debug)
        self.logger.log_physical(physical_records)
        summary = dict(debug.get('summary', {}))
        self.logger.log_frame_summary(summary)

    def _frame_context(self, cav_detections, frame_context):
        if frame_context is not None:
            return dict(frame_context)
        
        # ✅ 修复：检查 cav_detections 是否为空
        if not cav_detections:
            return {
                'frame': self.frame_counter,
                'frame_index': self.frame_counter,
                'scenario_index': -1,
                'timestamp': '',
                'timestamp_index': -1,
                'num_cavs': 0,
                'ego_id': 'unknown',
            }
        
        first = cav_detections[0]
        ego = next((det for det in cav_detections
                    if det.get('is_ego', False)), first)
        return {
            'frame': self.frame_counter,
            'frame_index': self.frame_counter,
            'scenario_index': first.get('scenario_index', -1),
            'timestamp': first.get('timestamp', ''),
            'timestamp_index': first.get('timestamp_index', -1),
            'num_cavs': len(cav_detections),
            'ego_id': str(ego.get('trust_id', ego.get('cav_id', 'ego'))),
        }


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
