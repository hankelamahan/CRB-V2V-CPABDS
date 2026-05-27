# -*- coding: utf-8 -*-

import csv
import json
import os
import tempfile
import unittest

from opencood.trust.id_mapper import VehicleIdMapper
from opencood.trust.reputation_manager import ReputationManager
from opencood.trust.reputation_source import (
    CsvDivaReputationSource,
    JsonReputationSource,
)


class TestReputationSource(unittest.TestCase):
    def test_json_reputation_source_loads_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'reputation.json')
            with open(path, 'w') as f:
                json.dump({'did:iota:vehicle:4288': 0.2}, f)

            source = JsonReputationSource(path)

            self.assertEqual(
                source.load_initial(),
                {'did:iota:vehicle:4288': 0.2})

    def test_diva_csv_converts_to_reputation_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'diva.csv')
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f,
                                        fieldnames=['vehicle_id',
                                                    'reputation'])
                writer.writeheader()
                writer.writerow({
                    'vehicle_id': 'did:iota:vehicle:4297',
                    'reputation': '0.8',
                })

            source = CsvDivaReputationSource(path)

            self.assertEqual(
                source.load_initial(),
                {'did:iota:vehicle:4297': 0.8})

    def test_id_map_aligns_opencood_id_to_external_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            id_map_path = os.path.join(tmpdir, 'id_map.json')
            reputation_path = os.path.join(tmpdir, 'reputation.json')
            with open(id_map_path, 'w') as f:
                json.dump({'4288': 'did:iota:vehicle:4288'}, f)
            with open(reputation_path, 'w') as f:
                json.dump({'did:iota:vehicle:4288': 0.3}, f)

            manager = ReputationManager({
                'default_reputation': 0.5,
                'id_map': id_map_path,
                'reputation_source': {
                    'type': 'json',
                    'path': reputation_path,
                },
            })

            self.assertAlmostEqual(
                manager.get_reputation('4288', original_cav_id='4288'), 0.3)

    def test_vehicle_id_mapper_prefers_original_cav_id(self):
        mapper = VehicleIdMapper(mapping={
            'raw_1': 'did:iota:vehicle:raw_1',
        })

        self.assertEqual(
            mapper.map('ego_alias', original_cav_id='raw_1'),
            'did:iota:vehicle:raw_1')


if __name__ == '__main__':
    unittest.main()
