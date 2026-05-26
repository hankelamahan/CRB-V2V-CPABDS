# -*- coding: utf-8 -*-
"""Physical consistency reputation helpers.

The first late-fusion integration keeps this module available but disabled by
default. Real use should pass measured residuals and velocities instead of
constant placeholders.
"""

import numpy as np


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
        self.reputation = {}
        self.vel_history = {}
        self.neighbor_votes = {}
        self.beta = float(config.get('update_rate', 0.1))
        self.min_reputation = float(config.get('min_reputation', 0.0))
        self.default_reputation = float(config.get('default_reputation', 0.5))
        self.history_window = int(config.get('history_window', 5))
        self.residual_sigma = float(config.get('residual_sigma', 5.0))
        self.velocity_sigma = float(config.get('velocity_sigma', 1.0))
        self.weights = config.get('weights', {
            'physical': 0.4,
            'trajectory': 0.3,
            'rsu': 0.3,
        })

    def get_reputation(self, vehicle_id):
        return self.reputation.get(str(vehicle_id), self.default_reputation)

    def update_vel_history(self, vehicle_id, velocity):
        vehicle_id = str(vehicle_id)
        history = self.vel_history.setdefault(vehicle_id, [])
        history.append(np.asarray(velocity, dtype=np.float32))
        if len(history) > self.history_window:
            history.pop(0)

    def get_neighbor_votes(self, vehicle_id):
        return self.neighbor_votes.setdefault(str(vehicle_id), [])

    def update_neighbor_votes(self, vehicle_id, vote):
        votes = self.get_neighbor_votes(vehicle_id)
        votes.append(vote)
        if len(votes) > 10:
            votes.pop(0)

    def compute_all_scores(self, residual, vehicle_id, velocity):
        phy = physical_score(residual, self.residual_sigma)
        self.update_vel_history(vehicle_id, velocity)
        traj = trajectory_score(self.vel_history[str(vehicle_id)],
                                self.history_window,
                                self.velocity_sigma)
        rsu = rsu_score(self.get_neighbor_votes(vehicle_id))
        fused = self.weights['physical'] * phy + \
            self.weights['trajectory'] * traj + self.weights['rsu'] * rsu
        return {
            'physical': float(phy),
            'trajectory': float(traj),
            'rsu': float(rsu),
            'fused': float(np.clip(fused, 0.0, 1.0)),
        }

    def update_reputation(self, vehicle_id, fused_score):
        vehicle_id = str(vehicle_id)
        current = self.reputation.get(vehicle_id, self.default_reputation)
        updated = current + self.beta * (float(fused_score) - current)
        self.reputation[vehicle_id] = float(np.clip(updated,
                                                    self.min_reputation, 1.0))
        return self.reputation[vehicle_id]

    @staticmethod
    def get_vote(fused_score):
        return -1 if float(fused_score) < 0.6 else 1

