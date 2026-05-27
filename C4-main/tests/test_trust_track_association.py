# -*- coding: utf-8 -*-

import unittest

import torch

from opencood.trust.track_association import TrackAssociation
from opencood.utils import box_utils


def make_box(x, y=0.0):
    center = torch.tensor([[x, y, 0.0, 1.5, 1.6, 4.0, 0.0]],
                          dtype=torch.float32)
    return box_utils.boxes_to_corners_3d(center, order='hwl')


class TestTrackAssociation(unittest.TestCase):
    def test_history_insufficient_until_track_has_velocity(self):
        assoc = TrackAssociation({
            'frame_interval': 1.0,
            'max_center_distance': 4.0,
            'min_bev_iou': 0.0,
        })
        first = make_box(0.0)
        second = make_box(1.0)

        first_evidence = assoc.update(0, 'cav1', 0, first)
        second_evidence = assoc.update(0, 'cav1', 1, second)

        self.assertIsNone(first_evidence[0]['residual'])
        self.assertIsNone(second_evidence[0]['residual'])
        self.assertTrue(second_evidence[0]['matched'])

    def test_constant_velocity_residual_is_real_value(self):
        assoc = TrackAssociation({
            'frame_interval': 1.0,
            'max_center_distance': 4.0,
            'min_bev_iou': 0.0,
        })
        assoc.update(0, 'cav1', 0, make_box(0.0))
        assoc.update(0, 'cav1', 1, make_box(1.0))
        evidence = assoc.update(0, 'cav1', 2, make_box(2.2))

        self.assertTrue(evidence[0]['matched'])
        self.assertAlmostEqual(evidence[0]['residual'], 0.2, places=5)

    def test_different_scenarios_do_not_share_tracks(self):
        assoc = TrackAssociation({
            'frame_interval': 1.0,
            'max_center_distance': 4.0,
            'min_bev_iou': 0.0,
        })
        assoc.update(0, 'cav1', 0, make_box(0.0))
        evidence = assoc.update(1, 'cav1', 1, make_box(1.0))

        self.assertFalse(evidence[0]['matched'])
        self.assertIsNone(evidence[0]['residual'])


if __name__ == '__main__':
    unittest.main()
