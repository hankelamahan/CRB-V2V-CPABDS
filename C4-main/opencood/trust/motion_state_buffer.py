# -*- coding: utf-8 -*-
"""Cross-frame CAV pose state for physical consistency checks."""

from collections import deque

import numpy as np


class MotionStateBuffer:
    """Track per-scenario CAV pose velocity in the post-process path."""

    def __init__(self, config=None):
        config = config or {}
        self.frame_interval = float(config.get('frame_interval', 0.2))
        self.history_window = int(config.get('history_window', 5))
        self.max_valid_speed = float(config.get('max_valid_speed', 40.0))
        self._states = {}

    def update_pose(self, scenario_index, trust_id, timestamp_index,
                    timestamp='', lidar_pose=None):
        """Update a CAV pose and return velocity evidence for this frame."""
        key = (int(scenario_index), str(trust_id))
        pose_xy = self._pose_xy(lidar_pose)
        if pose_xy is None or timestamp_index is None:
            return {
                'status': 'unknown',
                'reason': 'missing_pose_or_time',
                'velocity_xy': None,
                'speed': None,
            }

        timestamp_index = int(timestamp_index)
        state = self._states.get(key)
        if state is None:
            self._states[key] = self._new_state(timestamp_index, timestamp,
                                                lidar_pose, pose_xy)
            return {
                'status': 'unknown',
                'reason': 'cold_start',
                'velocity_xy': None,
                'speed': None,
            }

        delta_index = timestamp_index - state['last_timestamp_index']
        if delta_index <= 0:
            return {
                'status': 'invalid',
                'reason': 'non_increasing_timestamp',
                'velocity_xy': None,
                'speed': None,
            }

        dt = delta_index * self.frame_interval
        velocity_xy = (pose_xy - state['last_pose_xy']) / max(dt, 1e-6) # get the velocity in x and y direction
        speed = float(np.linalg.norm(velocity_xy)) # get the speed by calculating the norm of the velocity vector.
        status = 'valid' if speed <= self.max_valid_speed else 'invalid'
        reason = '' if status == 'valid' else 'speed_exceeds_limit'

        state['last_timestamp_index'] = timestamp_index
        state['last_timestamp'] = timestamp
        state['last_lidar_pose'] = lidar_pose
        state['last_pose_xy'] = pose_xy
        state['velocity_xy'] = velocity_xy
        state['pose_history'].append({
            'timestamp_index': timestamp_index,
            'timestamp': timestamp,
            'pose_xy': pose_xy.copy(),
            'velocity_xy': velocity_xy.copy(),
            'speed': speed,
        })
        return {
            'status': status,
            'reason': reason,
            'velocity_xy': velocity_xy.tolist(),
            'speed': speed,
            'dt': dt,
        }

    def get_state(self, scenario_index, trust_id):
        return self._states.get((int(scenario_index), str(trust_id)))

    def reset_scenario(self, scenario_index):
        scenario_index = int(scenario_index)
        self._states = {
            key: state
            for key, state in self._states.items()
            if key[0] != scenario_index
        }

    def _new_state(self, timestamp_index, timestamp, lidar_pose, pose_xy):
        return {
            'last_timestamp_index': int(timestamp_index),
            'last_timestamp': timestamp,
            'last_lidar_pose': lidar_pose,
            'last_pose_xy': pose_xy,
            'velocity_xy': None,
            'pose_history': deque(maxlen=self.history_window),
        }

    @staticmethod
    def _pose_xy(lidar_pose):
        if lidar_pose is None:
            return None
        arr = np.asarray(lidar_pose, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 2:
            return None
        return arr[:2]
