# -*- coding: utf-8 -*-
"""Offline and optional online reputation sources."""

import csv
import json

from opencood.trust.id_mapper import resolve_path


class ReputationSource:
    """Base interface for external reputation providers."""

    def load_initial(self):
        return {}

    def query(self, external_id):
        return None

    def push_updates(self, updates):
        return None


class NoopReputationSource(ReputationSource):
    """Source used when no external reputation provider is configured."""


class JsonReputationSource(ReputationSource):
    """Load reputation values from a JSON map or list of records."""

    def __init__(self, path):
        self.path = resolve_path(path)
        self.reputations = {}
        if self.path:
            self.reputations = self._load_file()

    def _load_file(self):
        with open(self.path, 'r') as f:
            loaded = json.load(f)

        if isinstance(loaded, dict):
            records = loaded.get('reputations', loaded)
            if isinstance(records, dict):
                return {
                    str(vehicle_id): float(score)
                    for vehicle_id, score in records.items()
                }
            loaded = records

        reputations = {}
        if isinstance(loaded, list):
            for item in loaded:
                if not isinstance(item, dict):
                    continue
                vehicle_id = item.get('vehicle_id', item.get('id'))
                score = item.get('reputation', item.get('score'))
                if vehicle_id is None or score is None:
                    continue
                reputations[str(vehicle_id)] = float(score)
        return reputations

    def load_initial(self):
        return dict(self.reputations)

    def query(self, external_id):
        return self.reputations.get(str(external_id))

    def push_updates(self, updates):
        self.reputations.update({
            str(vehicle_id): float(score)
            for vehicle_id, score in updates.items()
        })


class CsvDivaReputationSource(ReputationSource):
    """Load DIVA-style reputation values from a CSV file."""

    def __init__(self, path, vehicle_id_column='vehicle_id',
                 reputation_column='reputation'):
        self.path = resolve_path(path)
        self.vehicle_id_column = vehicle_id_column
        self.reputation_column = reputation_column
        self.reputations = {}
        if self.path:
            self.reputations = self._load_file()

    def _load_file(self):
        reputations = {}
        with open(self.path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vehicle_id = row.get(self.vehicle_id_column)
                score = row.get(self.reputation_column)
                if vehicle_id in (None, '') or score in (None, ''):
                    continue
                reputations[str(vehicle_id)] = float(score)
        return reputations

    def load_initial(self):
        return dict(self.reputations)

    def query(self, external_id):
        return self.reputations.get(str(external_id))

    def push_updates(self, updates):
        self.reputations.update({
            str(vehicle_id): float(score)
            for vehicle_id, score in updates.items()
        })


class RsuHttpReputationSource(ReputationSource):
    """Placeholder for a future RSU HTTP integration.

    The second-stage implementation keeps the interface available without
    introducing a network dependency in offline inference or unit tests.
    """

    def __init__(self, config=None):
        self.config = config or {}


def build_reputation_source(config=None):
    """Create a reputation source from a trust_fusion config fragment."""
    config = config or {}
    source_config = config.get('reputation_source', {}) or {}
    source_type = str(source_config.get('type', 'none')).lower()

    if source_type in ('none', 'noop', ''):
        return NoopReputationSource()
    if source_type == 'json':
        return JsonReputationSource(source_config.get('path', ''))
    if source_type in ('diva_csv', 'csv'):
        return CsvDivaReputationSource(
            source_config.get('path', ''),
            vehicle_id_column=source_config.get('vehicle_id_column',
                                                'vehicle_id'),
            reputation_column=source_config.get('reputation_column',
                                                'reputation'))
    if source_type == 'rsu_http':
        return RsuHttpReputationSource(source_config)
    raise ValueError('Unsupported reputation_source type: %s' % source_type)
