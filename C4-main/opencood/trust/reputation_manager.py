# -*- coding: utf-8 -*-
"""Centralized reputation state for trust-aware late fusion."""

import json

from opencood.trust.id_mapper import VehicleIdMapper, resolve_path


class ReputationManager:
    """Manage initial reputations and online consistency updates."""

    def __init__(self, config=None):
        config = config or {}
        self.default_reputation = float(config.get('default_reputation', 0.5))
        self.min_reputation = float(config.get('min_reputation', 0.0))
        self.max_reputation = float(config.get('max_reputation', 1.0))
        self.update_rate = float(config.get('update_rate', 0.1))
        self.ego_reputation = float(config.get('ego_reputation', 1.0))
        self.id_mapper = VehicleIdMapper(config.get('id_map', ''))
        self.reputations = {}
        self.load_reputations(config.get('reputation_map', ''))

    def load_reputations(self, reputation_map_path):
        resolved = resolve_path(reputation_map_path)
        if not resolved:
            return
        with open(resolved, 'r') as f:
            loaded = json.load(f)
        for vehicle_id, score in loaded.items():
            self.set_reputation(vehicle_id, score, already_mapped=True)

    def external_id(self, cav_id, original_cav_id=None, is_ego=False):
        return self.id_mapper.map(cav_id, original_cav_id, is_ego)

    def _clip(self, score):
        return max(self.min_reputation, min(self.max_reputation, float(score)))

    def get_reputation(self, cav_id, original_cav_id=None, is_ego=False):
        if is_ego:
            return self.ego_reputation
        external_id = self.external_id(cav_id, original_cav_id, is_ego)
        return self.reputations.get(external_id, self.default_reputation)

    def set_reputation(self, vehicle_id, score, already_mapped=False):
        external_id = str(vehicle_id) if already_mapped else self.external_id(
            vehicle_id)
        self.reputations[external_id] = self._clip(score)

    def update_from_voting(self, cav_id, is_consistent, original_cav_id=None,
                           is_ego=False):
        if is_ego:
            return self.ego_reputation
        current = self.get_reputation(cav_id, original_cav_id, is_ego)
        delta = self.update_rate if is_consistent else -self.update_rate
        updated = self._clip(current + delta)
        external_id = self.external_id(cav_id, original_cav_id, is_ego)
        self.reputations[external_id] = updated
        return updated

    def update_from_physical(self, cav_id, physical_score,
                             original_cav_id=None, is_ego=False):
        if is_ego:
            return self.ego_reputation
        current = self.get_reputation(cav_id, original_cav_id, is_ego)
        updated = self._clip(current + self.update_rate *
                             (float(physical_score) - current))
        external_id = self.external_id(cav_id, original_cav_id, is_ego)
        self.reputations[external_id] = updated
        return updated

    def get_all(self):
        return dict(self.reputations)

