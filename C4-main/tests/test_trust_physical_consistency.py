# -*- coding: utf-8 -*-

import unittest

import torch

from opencood.trust.motion_state_buffer import MotionStateBuffer
from opencood.trust.physical_consistency_manager import (
    PhysicalConsistencyManager,
)
from opencood.trust.reputation_manager import ReputationManager
from opencood.trust.track_association import TrackAssociation
from opencood.utils import box_utils


def make_box(x, y=0.0):
    center = torch.tensor([[x, y, 0.0, 1.5, 1.6, 4.0, 0.0]],
                          dtype=torch.float32)
    return box_utils.boxes_to_corners_3d(center, order='hwl')


def make_det(cav_id, box):
    boxes2d = box_utils.corner_to_standup_box_torch(box)
    return {
        'cav_id': cav_id,
        'trust_id': cav_id,
        'boxes3d': box,
        'boxes2d': boxes2d,
        'scores': torch.tensor([0.9], dtype=torch.float32),
        'labels': torch.ones(1, dtype=torch.long),
    }


class TestPhysicalConsistency(unittest.TestCase):
    def test_single_frame_physical_score_is_unknown(self):
        manager = PhysicalConsistencyManager({
            'use_physical_consistency': True,
        })
        det = make_det('cav1', make_box(0.0))
        records = manager.annotate_detections([det], {
            'cav1': [{
                'box_index': 0,
                'track_id': 'cav1-0',
                'residual': None,
                'reason': 'new_track',
            }],
        })

        self.assertIsNone(det['physical_score'])
        self.assertIsNone(det['physical_scores'][0])
        self.assertFalse(records[0]['used_for_update'])

    def test_time_backwards_skips_pose_update(self):
        buffer = MotionStateBuffer({'frame_interval': 1.0})
        buffer.update_pose(0, 'cav1', 2, lidar_pose=[2.0, 0.0, 0.0])
        evidence = buffer.update_pose(0, 'cav1', 1,
                                      lidar_pose=[1.0, 0.0, 0.0])

        self.assertEqual(evidence['status'], 'invalid')
        self.assertEqual(evidence['reason'], 'non_increasing_timestamp')

    def test_real_track_residual_drives_physical_score(self):
        assoc = TrackAssociation({
            'frame_interval': 1.0,
            'max_center_distance': 4.0,
            'min_bev_iou': 0.0,
        })
        manager = PhysicalConsistencyManager({
            'use_physical_consistency': True,
            'residual_sigma': 1.0,
        })
        assoc.update(0, 'cav1', 0, make_box(0.0))
        assoc.update(0, 'cav1', 1, make_box(1.0))
        track_records = assoc.update(0, 'cav1', 2, make_box(2.0))
        det = make_det('cav1', make_box(2.0))
        manager.annotate_detections([det], {'cav1': track_records})

        self.assertIsNotNone(det['physical_score'])
        self.assertGreater(det['physical_score'], 0.9)

    def test_unknown_evidence_does_not_update_reputation(self):
        manager = ReputationManager({
            'default_reputation': 0.5,
            'unknown_rate': 0.0,
        })
        updated = manager.update_from_evidence('cav1', None)

        self.assertAlmostEqual(updated, 0.5)

    def test_consensus_uses_other_cavs_as_reference(self):
        manager = PhysicalConsistencyManager({
            'use_physical_consistency': True,
            'residual_sigma': 2.0,
        })
        det_a = make_det('a', make_box(0.0))
        det_b = make_det('b', make_box(0.2))
        manager.annotate_detections([det_a, det_b])

        self.assertIsNotNone(det_a['consensus_motion_score'])
        self.assertGreater(det_a['consensus_motion_score'], 0.9)

    def test_pose_velocity_evidence_can_lower_physical_score(self):
        manager = PhysicalConsistencyManager({
            'use_physical_consistency': True,
        })
        det = make_det('cav1', make_box(0.0))
        det['pose_motion_score'] = 0.0
        manager.annotate_detections([det], {
            'cav1': [{
                'box_index': 0,
                'track_id': 'cav1-0',
                'residual': None,
                'reason': 'new_track',
            }],
        })

        self.assertEqual(det['motion_score'], 0.0)
        self.assertEqual(det['physical_score'], 0.0)


if __name__ == '__main__':
    unittest.main()
