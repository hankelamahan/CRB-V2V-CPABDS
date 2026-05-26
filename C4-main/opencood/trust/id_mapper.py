# -*- coding: utf-8 -*-
"""Vehicle id mapping helpers for trust modules."""

import json
import os


def resolve_path(path):
    """Resolve a config path against cwd and the repository root."""
    if not path:
        return None

    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded if os.path.exists(expanded) else None

    candidates = [
        os.path.abspath(expanded),
        os.path.abspath(os.path.join(os.getcwd(), expanded)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',
                                     expanded)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


class VehicleIdMapper:
    """Map OpenCOOD CAV ids to external reputation ids."""

    def __init__(self, id_map_path='', mapping=None):
        self.mapping = {}
        if mapping:
            self.mapping.update({str(k): str(v) for k, v in mapping.items()})
        resolved = resolve_path(id_map_path)
        if resolved:
            with open(resolved, 'r') as f:
                loaded = json.load(f)
            self.mapping.update({str(k): str(v) for k, v in loaded.items()})

    def map(self, cav_id, original_cav_id=None, is_ego=False):
        """Return the external id used by the reputation source."""
        if original_cav_id is not None:
            raw_id = str(original_cav_id)
        else:
            raw_id = str(cav_id)
        return self.mapping.get(raw_id, raw_id)

