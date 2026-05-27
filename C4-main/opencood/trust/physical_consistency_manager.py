# -*- coding: utf-8 -*-
"""Physical consistency evidence calculators for trust-aware late fusion."""

import numpy as np
import torch


def physical_score(residual, sigma=5.0):
    return np.exp(-float(residual) ** 2 / float(sigma) ** 2)


def trajectory_score(vel_history, window=5, sigma_v=1.0):
    if len(vel_history) < window:
        return 0.5
    vel_array = np.asarray(vel_history, dtype=np.float32)
    vel_var = np.var(vel_array, axis=0)
    return float(np.clip(np.exp(-np.sum(vel_var) / float(sigma_v) ** 2),
                         0.0, 1.0))


def rsu_score(neighbor_votes, sigma_r=0.5):
    if not neighbor_votes:
        return 0.5
    avg_vote = np.mean(neighbor_votes)
    return float(np.clip(np.exp(-(1 - avg_vote) ** 2 / float(sigma_r) ** 2),
                         0.0, 1.0))


class PhysicalConsistencyManager:
    def __init__(self, config=None):
        config = config or {}
        self.enabled = bool(config.get('use_physical_consistency', False))
        self.history_window = int(config.get('history_window', 5))
        self.residual_sigma = float(config.get('residual_sigma', 5.0))
        self.velocity_sigma = float(config.get('velocity_sigma', 3.0))
        self.max_valid_speed = float(config.get('max_valid_speed', 40.0))
        self.weights = config.get('weights', {
            'voting': 0.4,
            'motion': 0.3,
            'consensus_motion': 0.3,
        })
        self.vel_history = {}

    def score_residual(self, residual):
        if residual is None:
            return None
        return float(np.clip(physical_score(residual, self.residual_sigma),
                             0.0, 1.0))

    def score_velocity(self, velocity_xy):
        if velocity_xy is None:
            return None
        speed = float(np.linalg.norm(np.asarray(velocity_xy,
                                                dtype=np.float32)))
        if speed > self.max_valid_speed:
            return 0.0
        return float(np.clip(np.exp(-(speed ** 2) /
                                    max(self.velocity_sigma ** 2, 1e-6)),
                             0.0, 1.0))

    def combine_evidence(self, voting_score=None, motion_score=None,
                         consensus_motion_score=None):
        """Combine non-None evidence using configured normalized weights."""
        evidence = {
            'voting': voting_score,
            'motion': motion_score,
            'consensus_motion': consensus_motion_score,
        }
        weighted_sum = 0.0
        weight_sum = 0.0
        for name, score in evidence.items():
            if score is None:
                continue
            weight = float(self.weights.get(name, 0.0))
            if weight <= 0:
                continue
            weighted_sum += weight * float(score)
            weight_sum += weight
        if weight_sum <= 0:
            return None
        return float(np.clip(weighted_sum / weight_sum, 0.0, 1.0))

    def annotate_detections(self, cav_detections, track_evidence_by_id=None,
                            frame_context=None):
        """Attach per-box physical scores and return log records."""
        track_evidence_by_id = track_evidence_by_id or {}
        consensus_by_id = self.compute_consensus_scores(cav_detections)
        records = []
        for det in cav_detections:
            trust_id = str(det.get('trust_id', det.get('cav_id')))
            track_records = track_evidence_by_id.get(trust_id, [])
            consensus_records = consensus_by_id.get(trust_id, [])
            pose_motion_score = det.get('pose_motion_score')
            num_boxes = int(det['boxes3d'].shape[0])
            per_box_scores = []
            motion_scores = []
            consensus_scores = []
            for box_idx in range(num_boxes):
                track_record = track_records[box_idx] \
                    if box_idx < len(track_records) else {}
                consensus_record = consensus_records[box_idx] \
                    if box_idx < len(consensus_records) else {}
                motion_score = self.score_residual(
                    track_record.get('residual'))
                box_motion_score = self._mean_scores([
                    motion_score,
                    pose_motion_score,
                ])
                consensus_score = consensus_record.get('score')
                combined = self.combine_evidence(
                    motion_score=box_motion_score,
                    consensus_motion_score=consensus_score)
                per_box_scores.append(combined)
                if box_motion_score is not None:
                    motion_scores.append(box_motion_score)
                if consensus_score is not None:
                    consensus_scores.append(consensus_score)
                records.append(self._record(frame_context, trust_id, box_idx,
                                            track_record, motion_score,
                                            pose_motion_score,
                                            consensus_record,
                                            consensus_score, combined))

            det['physical_scores'] = per_box_scores
            cav_scores = [score for score in per_box_scores
                          if score is not None]
            det['physical_score'] = float(np.mean(cav_scores)) \
                if cav_scores else None
            det['motion_score'] = float(np.mean(motion_scores)) \
                if motion_scores else None
            det['consensus_motion_score'] = float(np.mean(consensus_scores)) \
                if consensus_scores else None
        return records

    def update_vel_history(self, vehicle_id, velocity):
        vehicle_id = str(vehicle_id)
        history = self.vel_history.setdefault(vehicle_id, [])
        history.append(np.asarray(velocity, dtype=np.float32))
        if len(history) > self.history_window:
            history.pop(0)

    def compute_all_scores(self, residual, vehicle_id, velocity):
        phy = physical_score(residual, self.residual_sigma)
        self.update_vel_history(vehicle_id, velocity)
        traj = trajectory_score(self.vel_history[str(vehicle_id)],
                                self.history_window,
                                self.velocity_sigma)
        vel = self.score_velocity(velocity)
        fused = self.combine_evidence(motion_score=phy,
                                      consensus_motion_score=vel)
        return {
            'physical': float(phy),
            'trajectory': float(traj),
            'velocity': vel,
            'fused': fused,
        }

    def compute_consensus_scores(self, cav_detections):
        """Compute leave-one-out 3D center residual scores per detection."""
        centers_by_id = {}
        labels_by_id = {}
        for det in cav_detections:
            trust_id = str(det.get('trust_id', det.get('cav_id')))
            centers_by_id[trust_id] = self._centers(det.get('boxes3d'))
            labels_by_id[trust_id] = self._labels(det.get('labels'),
                                                  len(centers_by_id[trust_id]))

        output = {}
        for det in cav_detections:
            trust_id = str(det.get('trust_id', det.get('cav_id')))
            centers = centers_by_id.get(trust_id,
                                        np.zeros((0, 3), dtype=np.float32))
            labels = labels_by_id.get(trust_id, [])
            records = []
            for box_idx, center in enumerate(centers):
                label = labels[box_idx] if box_idx < len(labels) else None
                references = []
                for ref_id, ref_centers in centers_by_id.items():
                    if ref_id == trust_id:
                        continue
                    ref_labels = labels_by_id.get(ref_id, [])
                    for ref_idx, ref_center in enumerate(ref_centers):
                        ref_label = ref_labels[ref_idx] \
                            if ref_idx < len(ref_labels) else None
                        if label is not None and ref_label != label:
                            continue
                        references.append(ref_center)
                if not references:
                    records.append({
                        'residual': None,
                        'score': None,
                        'reason': 'no_reference_agent',
                    })
                    continue
                ref_arr = np.asarray(references, dtype=np.float32)
                distances = np.linalg.norm(ref_arr - center, axis=1)
                residual = float(np.min(distances))
                records.append({
                    'residual': residual,
                    'score': self.score_residual(residual),
                    'reason': 'matched_reference',
                })
            output[trust_id] = records
        return output

    @staticmethod
    def get_vote(fused_score):
        return -1 if float(fused_score) < 0.6 else 1

    @staticmethod
    def _centers(boxes3d):
        if boxes3d is None or boxes3d.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if isinstance(boxes3d, torch.Tensor):
            arr = boxes3d.detach().cpu().numpy()
        else:
            arr = np.asarray(boxes3d)
        return np.mean(arr[:, [0, 3, 5, 6], :], axis=1).astype(np.float32)

    @staticmethod
    def _labels(labels, size):
        if labels is None:
            return [1 for _ in range(size)]
        if isinstance(labels, torch.Tensor):
            return labels.detach().cpu().numpy().astype(int).tolist()
        return np.asarray(labels).astype(int).tolist()

    @staticmethod
    def _mean_scores(scores):
        valid = [float(score) for score in scores if score is not None]
        if not valid:
            return None
        return float(np.mean(valid))

    @staticmethod
    def _record(frame_context, trust_id, box_idx, track_record, motion_score,
                pose_motion_score, consensus_record, consensus_score,
                combined):
        payload = dict(frame_context or {})
        payload.update({
            'trust_id': trust_id,
            'box_index': int(box_idx),
            'track_id': track_record.get('track_id'),
            'residual': track_record.get('residual'),
            'motion_score': motion_score,
            'pose_motion_score': pose_motion_score,
            'consensus_residual': consensus_record.get('residual'),
            'consensus_motion_score': consensus_score,
            'physical_score': combined,
            'used_for_update': combined is not None,
            'reason': track_record.get('reason',
                                       consensus_record.get('reason')),
        })
        return payload
