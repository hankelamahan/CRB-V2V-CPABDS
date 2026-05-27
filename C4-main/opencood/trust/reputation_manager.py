# -*- coding: utf-8 -*-
"""Centralized reputation state for trust-aware late fusion."""

import json

from opencood.trust.id_mapper import VehicleIdMapper, resolve_path
from opencood.trust.reputation_cache import VehicleReputationCache
from opencood.trust.reputation_source import build_reputation_source


class ReputationManager:
    """Manage initial reputations and online consistency updates."""

    def __init__(self, config=None):
        config = config or {}
        update_config = config.get('reputation_update', {})
        self.default_reputation = float(config.get('default_reputation', 0.5))
        self.min_reputation = float(config.get('min_reputation', 0.0))
        self.max_reputation = float(config.get('max_reputation', 1.0))
        self.update_rate = float(config.get('update_rate', 0.1))
        self.positive_rate = float(update_config.get(
            'positive_rate', config.get('positive_rate', self.update_rate)))
        self.negative_rate = float(update_config.get(
            'negative_rate', config.get('negative_rate', self.update_rate)))
        self.unknown_rate = float(update_config.get(
            'unknown_rate', config.get('unknown_rate', 0.0)))
        self.good_thr = float(update_config.get(
            'good_thr', config.get('good_thr', 0.7)))
        self.bad_thr = float(update_config.get(
            'bad_thr', config.get('bad_thr', 0.4)))
        self.max_per_frame_delta = float(update_config.get(
            'max_per_frame_delta',
            config.get('max_per_frame_delta', self.update_rate)))
        self.ego_reputation = float(config.get('ego_reputation', 1.0))
        self.id_mapper = VehicleIdMapper(config.get('id_map', ''))
        self.source = build_reputation_source(config)
        self.cache = VehicleReputationCache(
            capacity=int(config.get('cache_capacity', 100)),
            ttl=float(config.get('cache_ttl', 300)),
            server_sync_callback=self.source.query)
        self.reputations = {}
        self.load_reputations(config.get('reputation_map', ''))
        self.load_source_reputations()
        self.pending_updates = {}

    def load_reputations(self, reputation_map_path):
        resolved = resolve_path(reputation_map_path)
        if not resolved:
            return
        with open(resolved, 'r') as f:
            loaded = json.load(f)
        for vehicle_id, score in loaded.items():
            self.set_reputation(vehicle_id, score, already_mapped=True)

    def load_source_reputations(self):
        for vehicle_id, score in self.source.load_initial().items():
            self.set_reputation(vehicle_id, score, already_mapped=True)

    def external_id(self, cav_id, original_cav_id=None, is_ego=False):
        return self.id_mapper.map(cav_id, original_cav_id, is_ego)

    def _clip(self, score):
        return max(self.min_reputation, min(self.max_reputation, float(score)))

    def get_reputation(self, cav_id, original_cav_id=None, is_ego=False):
        if is_ego:
            return self.ego_reputation
        external_id = self.external_id(cav_id, original_cav_id, is_ego)
        if external_id in self.reputations:
            return self.reputations[external_id]
        cached = self.cache.get(external_id, default=None)
        if cached is None:
            return self.default_reputation
        clipped = self._clip(cached)
        self.reputations[external_id] = clipped
        return clipped

    def set_reputation(self, vehicle_id, score, already_mapped=False):
        external_id = str(vehicle_id) if already_mapped else self.external_id(
            vehicle_id)
        clipped = self._clip(score)
        self.reputations[external_id] = clipped
        self.cache.update(external_id, clipped)

    def update_from_voting(self, cav_id, is_consistent, original_cav_id=None,
                           is_ego=False):
        if is_ego:
            return self.ego_reputation
        current = self.get_reputation(cav_id, original_cav_id, is_ego)
        delta = self.update_rate if is_consistent else -self.update_rate
        updated = self._clip(current + delta)
        external_id = self.external_id(cav_id, original_cav_id, is_ego)
        self.reputations[external_id] = updated
        self.cache.update(external_id, updated)
        self.pending_updates[external_id] = updated
        return updated

    def update_from_evidence(self, cav_id, evidence_score,
                             original_cav_id=None, is_ego=False):
        """Update reputation with asymmetric evidence-score dynamics."""
        if is_ego:
            return self.ego_reputation
        current = self.get_reputation(cav_id, original_cav_id, is_ego)
        if evidence_score is None:
            if self.unknown_rate <= 0:
                return current
            target = self.default_reputation
            raw_delta = self.unknown_rate * (target - current)
        else:
            score = float(evidence_score)
            if score >= self.good_thr:
                raw_delta = self.positive_rate * (score - current)
            elif score <= self.bad_thr:
                raw_delta = -self.negative_rate * (current - score)
            else:
                raw_delta = 0.0

        delta = max(-self.max_per_frame_delta,
                    min(self.max_per_frame_delta, raw_delta))
        updated = self._clip(current + delta)
        external_id = self.external_id(cav_id, original_cav_id, is_ego)
        self.reputations[external_id] = updated
        self.cache.update(external_id, updated)
        self.pending_updates[external_id] = updated
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
        self.cache.update(external_id, updated)
        self.pending_updates[external_id] = updated
        return updated

    def flush_updates(self):
        if not self.pending_updates:
            return
        self.source.push_updates(dict(self.pending_updates))
        self.pending_updates.clear()

    def get_all(self):
        return dict(self.reputations)
