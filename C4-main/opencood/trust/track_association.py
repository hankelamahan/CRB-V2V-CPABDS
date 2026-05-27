# -*- coding: utf-8 -*-
"""Detection track association for physical consistency evidence."""

import itertools

import numpy as np
import torch

from opencood.utils import box_utils


class TrackAssociation:
    """Associate per-CAV detections across frames with greedy matching."""

    def __init__(self, config=None):
        config = config or {}
        self.max_center_distance = float(config.get('max_center_distance',
                                                    4.0))
        self.min_bev_iou = float(config.get('min_bev_iou', 0.1))
        self.max_age = int(config.get('max_age', 3))
        self.frame_interval = float(config.get('frame_interval', 0.2))
        self._tracks = {}
        self._next_track_id = itertools.count()

    def update(self, scenario_index, trust_id, timestamp_index, boxes3d,
               boxes2d=None):
        """Update tracks and return one evidence dict per input detection."""
        scenario_index = int(scenario_index)
        trust_id = str(trust_id)
        timestamp_index = int(timestamp_index)
        key = (scenario_index, trust_id)
        centers = self._box_centers(boxes3d)
        boxes2d_np = self._boxes2d(boxes3d, boxes2d)
        tracks = self._tracks.setdefault(key, {})
        self._age_tracks(tracks, timestamp_index)

        evidence = [self._unknown_record(i) for i in range(len(centers))]
        if len(centers) == 0:
            self._drop_expired(tracks)
            return evidence

        pairs = []
        for box_idx, center in enumerate(centers):
            for track_id, track in tracks.items():
                age = timestamp_index - track['last_timestamp_index']
                if age <= 0 or age > self.max_age:
                    continue
                dist = float(np.linalg.norm(center - track['last_center']))
                iou = self._iou(boxes2d_np[box_idx], track['last_box2d'])
                if dist > self.max_center_distance or iou < self.min_bev_iou:
                    continue
                pairs.append((dist - iou, box_idx, track_id, dist, iou))

        matched_boxes = set()
        matched_tracks = set()
        for _, box_idx, track_id, dist, iou in sorted(pairs):
            if box_idx in matched_boxes or track_id in matched_tracks:
                continue
            track = tracks[track_id]
            dt = (timestamp_index - track['last_timestamp_index']) * \
                self.frame_interval
            residual = None
            predicted_center = None
            if track.get('last_velocity') is not None and dt > 0:
                predicted_center = track['last_center'] + \
                    track['last_velocity'] * dt
                residual = float(np.linalg.norm(centers[box_idx] -
                                                predicted_center))
            velocity = None
            if dt > 0:
                velocity = (centers[box_idx] - track['last_center']) / dt
            evidence[box_idx] = {
                'box_index': box_idx,
                'track_id': track_id,
                'matched': True,
                'match_distance': dist,
                'match_iou': iou,
                'residual': residual,
                'predicted_center': None if predicted_center is None
                else predicted_center.tolist(),
                'velocity_xy': None if velocity is None
                else velocity[:2].tolist(),
                'reason': 'matched',
            }
            self._update_track(track, timestamp_index, centers[box_idx],
                               boxes2d_np[box_idx], velocity)
            matched_boxes.add(box_idx)
            matched_tracks.add(track_id)

        for box_idx, center in enumerate(centers):
            if box_idx in matched_boxes:
                continue
            track_id = self._new_track_id(trust_id)
            tracks[track_id] = {
                'track_id': track_id,
                'last_center': center,
                'last_box2d': boxes2d_np[box_idx],
                'last_velocity': None,
                'last_timestamp_index': timestamp_index,
                'age': 0,
            }
            evidence[box_idx] = {
                'box_index': box_idx,
                'track_id': track_id,
                'matched': False,
                'match_distance': None,
                'match_iou': None,
                'residual': None,
                'predicted_center': None,
                'velocity_xy': None,
                'reason': 'new_track',
            }

        self._drop_expired(tracks)
        return evidence

    @staticmethod
    def _box_centers(boxes3d):
        if boxes3d is None or boxes3d.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if isinstance(boxes3d, torch.Tensor):
            arr = boxes3d.detach().cpu().numpy()
        else:
            arr = np.asarray(boxes3d)
        return np.mean(arr[:, [0, 3, 5, 6], :], axis=1).astype(np.float32)

    @staticmethod
    def _boxes2d(boxes3d, boxes2d):
        if boxes2d is not None:
            if isinstance(boxes2d, torch.Tensor):
                return boxes2d.detach().cpu().numpy().astype(np.float32)
            return np.asarray(boxes2d, dtype=np.float32)
        if boxes3d is None or boxes3d.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return box_utils.corner_to_standup_box_torch(boxes3d).detach().cpu() \
            .numpy().astype(np.float32)

    @staticmethod
    def _iou(box_a, box_b):
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * \
            max(0.0, float(box_a[3]) - float(box_a[1]))
        area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * \
            max(0.0, float(box_b[3]) - float(box_b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _new_track_id(self, trust_id):
        return '%s-%d' % (trust_id, next(self._next_track_id))

    @staticmethod
    def _unknown_record(box_idx):
        return {
            'box_index': box_idx,
            'track_id': None,
            'matched': False,
            'match_distance': None,
            'match_iou': None,
            'residual': None,
            'predicted_center': None,
            'velocity_xy': None,
            'reason': 'no_track',
        }

    @staticmethod
    def _update_track(track, timestamp_index, center, box2d, velocity):
        track['last_center'] = center
        track['last_box2d'] = box2d
        track['last_velocity'] = velocity
        track['last_timestamp_index'] = timestamp_index
        track['age'] = 0

    @staticmethod
    def _age_tracks(tracks, timestamp_index):
        for track in tracks.values():
            track['age'] = max(0, timestamp_index -
                               track['last_timestamp_index'])

    def _drop_expired(self, tracks):
        expired = [
            track_id
            for track_id, track in tracks.items()
            if track.get('age', 0) > self.max_age
        ]
        for track_id in expired:
            del tracks[track_id]