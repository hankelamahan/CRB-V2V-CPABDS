# -*- coding: utf-8 -*-

import unittest

import torch

from opencood.trust.late_trust_fusion import LateTrustFusion
from opencood.trust.overlap_field_voting import OverlapFieldVoter


class TestOverlapFieldVoter(unittest.TestCase):
    def test_weighted_overlap_clusters_boxes(self):
        voter = OverlapFieldVoter(iou_thr=0.5)
        boxes, scores, labels = voter.vote_detection_level(
            boxes_list=[
                [[0.0, 0.0, 2.0, 2.0]],
                [[0.1, 0.1, 2.1, 2.1]],
            ],
            scores_list=[[0.9], [0.8]],
            labels_list=[[1], [1]],
            reputation_scores=[1.0, 0.5],
        )

        self.assertEqual(boxes.shape[0], 1)
        self.assertEqual(labels.tolist(), [1])
        self.assertGreater(scores[0], 0.0)


class TestLateTrustFusion(unittest.TestCase):
    @staticmethod
    def _det(cav_id, boxes, scores, is_ego=False):
        boxes2d = torch.tensor(boxes, dtype=torch.float32)
        num_boxes = boxes2d.shape[0]
        boxes3d = torch.zeros((num_boxes, 8, 3), dtype=torch.float32)
        labels = torch.ones(num_boxes, dtype=torch.long)
        return {
            'cav_id': cav_id,
            'original_cav_id': cav_id,
            'trust_id': cav_id,
            'is_ego': is_ego,
            'boxes2d': boxes2d,
            'boxes3d': boxes3d,
            'scores': torch.tensor(scores, dtype=torch.float32),
            'labels': labels,
        }

    def test_low_reputation_non_ego_is_filtered(self):
        fusion = LateTrustFusion({
            'use_trust_fusion': True,
            'default_reputation': 0.1,
            'ego_reputation': 1.0,
            'drop_below': 0.3,
            'update_rate': 0.0,
        })
        detections, _ = fusion.apply([
            self._det('ego', [[0.0, 0.0, 2.0, 2.0]], [0.9], is_ego=True),
            self._det('bad_cav', [[5.0, 5.0, 7.0, 7.0]], [0.9]),
        ])

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]['cav_id'], 'ego')

    def test_neutral_reputation_preserves_scores(self):
        fusion = LateTrustFusion({
            'use_trust_fusion': True,
            'default_reputation': 1.0,
            'ego_reputation': 1.0,
            'drop_below': 0.0,
            'update_rate': 0.0,
        })
        detections, _ = fusion.apply([
            self._det('cav_1', [[0.0, 0.0, 2.0, 2.0]], [0.75]),
        ])

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(float(detections[0]['scores'][0]), 0.75)

    def test_single_agent_does_not_self_confirm(self):
        fusion = LateTrustFusion({
            'use_trust_fusion': True,
            'default_reputation': 0.5,
            'drop_below': 0.0,
            'update_rate': 0.2,
        })
        _, debug = fusion.apply([
            self._det('cav_1', [[0.0, 0.0, 2.0, 2.0]], [0.9]),
        ])

        self.assertIsNone(debug['cavs']['cav_1']['voting_consistent'])
        self.assertAlmostEqual(
            debug['cavs']['cav_1']['reputation_after'], 0.5)

    def test_leave_one_out_penalizes_unmatched_agent(self):
        fusion = LateTrustFusion({
            'use_trust_fusion': True,
            'default_reputation': 0.5,
            'ego_reputation': 1.0,
            'drop_below': 0.0,
            'update_rate': 0.1,
        })
        _, debug = fusion.apply([
            self._det('ego', [[0.0, 0.0, 2.0, 2.0]], [0.9], is_ego=True),
            self._det('bad_cav', [[5.0, 5.0, 7.0, 7.0]], [0.9]),
        ])

        self.assertFalse(debug['cavs']['bad_cav']['voting_consistent'])
        self.assertAlmostEqual(
            debug['cavs']['bad_cav']['reputation_after'], 0.4)


if __name__ == '__main__':
    unittest.main()
