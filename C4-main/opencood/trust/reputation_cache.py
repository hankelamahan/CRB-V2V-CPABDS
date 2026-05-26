# -*- coding: utf-8 -*-
"""Local LRU reputation cache."""

import time
from collections import OrderedDict


class VehicleReputationCache:
    """Cache vehicle reputations with a TTL and optional server callback."""

    def __init__(self, capacity=100, ttl=300, server_sync_callback=None):
        self.capacity = capacity
        self.ttl = ttl
        self._cache = OrderedDict()
        self._server_sync_callback = server_sync_callback

    def _is_valid(self, timestamp):
        return (time.time() - timestamp) < self.ttl

    def get(self, vehicle_id, default=0.5):
        vehicle_id = str(vehicle_id)
        if vehicle_id in self._cache:
            reputation, timestamp = self._cache[vehicle_id]
            if self._is_valid(timestamp):
                self._cache.move_to_end(vehicle_id)
                return reputation
            del self._cache[vehicle_id]

        if self._server_sync_callback is None:
            return default

        reputation = self._server_sync_callback(vehicle_id)
        self.update(vehicle_id, reputation)
        return reputation

    def update(self, vehicle_id, reputation):
        vehicle_id = str(vehicle_id)
        self._cache[vehicle_id] = (float(reputation), time.time())
        self._cache.move_to_end(vehicle_id)
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def batch_update(self, updates):
        for vehicle_id, reputation in updates.items():
            self.update(vehicle_id, reputation)

    def get_all(self):
        valid = {}
        for vehicle_id, (reputation, timestamp) in list(self._cache.items()):
            if self._is_valid(timestamp):
                valid[vehicle_id] = reputation
            else:
                del self._cache[vehicle_id]
        return valid

