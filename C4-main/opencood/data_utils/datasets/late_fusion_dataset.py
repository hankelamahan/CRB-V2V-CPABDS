# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for late fusion
"""
import random
import math
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.data_utils.datasets
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.trust.late_trust_fusion import LateTrustFusion, merge_and_nms
from opencood.utils import box_utils
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


class LateFusionDataset(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """
    def __init__(self, params, visualize, train=True):
        super(LateFusionDataset, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = build_postprocessor(params['postprocess'], train)
        trust_params = params.get('trust_fusion', {})
        self.trust_fusion = None
        self.last_trust_debug = None
        self.last_gt_cav_labels = None
        if trust_params.get('use_trust_fusion', False):
            self.trust_fusion = LateTrustFusion(
                trust_params,
                params.get('physical_consistency', {}),
                reputation_update_config=params.get('reputation_update', {}),
                track_config=params.get('track_association', {}),
            )

    def __getitem__(self, idx):
        sample_metadata = None
        if not self.train:
            sample_metadata = self._resolve_sample_metadata(idx)

        base_data_dict = self.retrieve_base_data(idx)
        if sample_metadata is not None:
            for cav_content in base_data_dict.values():
                cav_content.update(sample_metadata)

        if self.train:
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            reformat_data_dict = self.get_item_test(base_data_dict)

        return reformat_data_dict

    def _resolve_sample_metadata(self, idx):
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break

        scenario_database = self.scenario_database[scenario_index]
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        return {
            'scenario_index': scenario_index,
            'timestamp_index': timestamp_index,
            'timestamp_key': timestamp_key,
        }

    def get_item_single_car(self, selected_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # remove points that hit ego vehicle
        lidar_np = mask_ego_points(lidar_np)

        # generate the bounding box(n, 7) under the cav's space
        # set the vehicles in the real world(GT BOXES) to the CAV's LIDAR coordinate system.
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       selected_cav_base[
                                                           'params'][
                                                           'lidar_pose'])
        # data augmentation
        lidar_np, object_bbx_center, object_bbx_mask = \
            self.augment(lidar_np, object_bbx_center, object_bbx_mask)

        if self.visualize:
            selected_cav_processed.update({'origin_lidar': lidar_np})

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(lidar_np)
        selected_cav_processed.update({'processed_lidar': lidar_dict})

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()
        selected_cav_processed.update({'anchor_box': anchor_box})

        selected_cav_processed.update({'object_bbx_center': object_bbx_center,
                                       'object_bbx_mask': object_bbx_mask,
                                       'object_ids': object_ids})

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=object_bbx_mask)
        selected_cav_processed.update({'label_dict': label_dict})

        return selected_cav_processed

    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()

        # during training, we return a random cav's data
        if not self.visualize:
            selected_cav_id, selected_cav_base = \
                random.choice(list(base_data_dict.items()))
        else:
            selected_cav_id, selected_cav_base = \
                list(base_data_dict.items())[0]

        selected_cav_processed = self.get_item_single_car(selected_cav_base)
        processed_data_dict.update({'ego': selected_cav_processed})

        return processed_data_dict

    def get_item_test(self, base_data_dict):
        processed_data_dict = OrderedDict()
        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0])**2 + (
                                      selected_cav_base['params'][
                                          'lidar_pose'][1] - ego_lidar_pose[
                                          1])**2)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            # find the transformation matrix from current cav to ego.
            cav_lidar_pose = selected_cav_base['params']['lidar_pose']
            transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)

            selected_cav_processed = \
                self.get_item_single_car(selected_cav_base)
            selected_cav_processed.update({'transformation_matrix':
                                               transformation_matrix})
            selected_cav_processed.update({
                'cav_id': cav_id,
                'original_cav_id': cav_id,
                'is_ego': cav_id == ego_id,
                'timestamp': selected_cav_base.get('timestamp_key', ''),
                'timestamp_index': selected_cav_base.get('timestamp_index', -1),
                'scenario_index': selected_cav_base.get('scenario_index', -1),
                'lidar_pose': cav_lidar_pose,
                'ego_lidar_pose': ego_lidar_pose,
            })
            update_cav = "ego" if cav_id == ego_id else cav_id
            processed_data_dict.update({update_cav: selected_cav_processed})

        return processed_data_dict

    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}

        # for late fusion, we also need to stack the lidar for better
        # visualization
        if self.visualize:
            projected_lidar_list = []
            origin_lidar = []

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content[
                            'anchor_box']))})
            if self.visualize:
                transformation_matrix = cav_content['transformation_matrix']
                origin_lidar = [cav_content['origin_lidar']]

                projected_lidar = cav_content['origin_lidar']
                projected_lidar[:, :3] = \
                    box_utils.project_points_by_matrix_torch(
                        projected_lidar[:, :3],
                        transformation_matrix)
                projected_lidar_list.append(projected_lidar)

            # processed lidar dictionary
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['processed_lidar']])
            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(
                    np.array(cav_content['transformation_matrix'])).float()

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'processed_lidar': processed_lidar_torch_dict,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids,
                                        'transformation_matrix': transformation_matrix_torch})
            for meta_key in ['cav_id', 'original_cav_id', 'is_ego',
                             'timestamp', 'timestamp_index',
                             'scenario_index', 'lidar_pose',
                             'ego_lidar_pose']:
                if meta_key in cav_content:
                    output_dict[cav_id][meta_key] = cav_content[meta_key]

            if self.visualize:
                origin_lidar = \
                    np.array(
                        downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})

        if self.visualize:
            projected_lidar_stack = torch.from_numpy(
                np.vstack(projected_lidar_list))
            output_dict['ego'].update({'origin_lidar': projected_lidar_stack})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        if self.trust_fusion is not None:
            return self.post_process_trust(data_dict, output_dict)

        self.last_trust_debug = None
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)
        self.last_gt_cav_labels = self._label_gt_boxes_with_cav_ids(
            data_dict,
            gt_box_tensor)

        return pred_box_tensor, pred_score, gt_box_tensor

    def post_process_trust(self, data_dict, output_dict):
        if not hasattr(self.post_processor, 'delta_to_boxes3d'):
            raise NotImplementedError(
                'Trust-aware late fusion currently supports voxel-style '
                'postprocessors with delta_to_boxes3d().')

        cav_detections = []
        for cav_id, cav_content in data_dict.items():
            assert cav_id in output_dict
            decoded = self._decode_single_cav_detection(
                cav_id,
                cav_content, #for gt and transformation
                output_dict[cav_id]) #for predicting
            if decoded is not None:
                cav_detections.append(decoded)

        frame_context = self._trust_frame_context(cav_detections)
        trusted_detections, trust_debug = self.trust_fusion.apply(
            cav_detections,
            frame_context=frame_context)
        self.last_trust_debug = trust_debug
        pred_box_tensor, pred_score = merge_and_nms(
            trusted_detections,
            self.post_processor.params['nms_thresh'])
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)
        self.last_gt_cav_labels = self._label_gt_boxes_with_cav_ids(
            data_dict,
            gt_box_tensor)

        return pred_box_tensor, pred_score, gt_box_tensor

    def _label_gt_boxes_with_cav_ids(self, data_dict, gt_box_tensor):
        """
        Build per-GT-box labels for CAV self vehicles only.

        The GT tensor is a de-duplicated scene-level box list, while the
        original CAV ids live in each CAV's object_ids. Match by projected box
        center so visualization labels stay aligned with the final GT tensor.
        """
        if gt_box_tensor is None:
            return []

        num_gt = int(gt_box_tensor.shape[0])
        labels = [None] * num_gt
        if num_gt == 0:
            return labels

        cav_ids = set()
        for cav_key, cav_content in data_dict.items():
            original_cav_id = cav_content.get('original_cav_id', cav_key)
            cav_ids.add(str(original_cav_id))
        if not cav_ids:
            return labels

        gt_centers = gt_box_tensor[:, :, :2].mean(dim=1)
        best_match_by_cav = {}
        order = self.post_processor.params['order']

        for cav_content in data_dict.values():
            object_ids = cav_content.get('object_ids', [])
            object_bbx_center = cav_content.get('object_bbx_center')
            object_bbx_mask = cav_content.get('object_bbx_mask')
            transformation_matrix = cav_content.get('transformation_matrix')
            if not object_ids or object_bbx_center is None or \
                    object_bbx_mask is None or transformation_matrix is None:
                continue

            if len(object_bbx_center.shape) > 2:
                object_bbx_center = object_bbx_center[0]
            if len(object_bbx_mask.shape) > 1:
                object_bbx_mask = object_bbx_mask[0]

            valid_boxes = object_bbx_center[object_bbx_mask == 1]
            valid_count = min(len(object_ids), int(valid_boxes.shape[0]))
            for box_idx in range(valid_count):
                cav_id = str(object_ids[box_idx])
                if cav_id not in cav_ids:
                    continue

                object_box = valid_boxes[box_idx:box_idx + 1]
                object_corner = box_utils.boxes_to_corners_3d(
                    object_box,
                    order)
                projected_corner = box_utils.project_box3d(
                    object_corner.float(),
                    transformation_matrix)
                candidate_center = projected_corner[:, :, :2].mean(dim=1)[0]
                distances = torch.norm(
                    gt_centers - candidate_center.reshape(1, 2),
                    dim=1)
                distance, gt_idx = torch.min(distances, dim=0)
                distance = float(distance.detach().cpu().item())
                if distance > 3.0:
                    continue

                previous = best_match_by_cav.get(cav_id)
                if previous is None or distance < previous[1]:
                    best_match_by_cav[cav_id] = (
                        int(gt_idx.detach().cpu().item()),
                        distance)

        used_gt_indices = set()
        for cav_id, (gt_idx, distance) in sorted(
                best_match_by_cav.items(),
                key=lambda item: item[1][1]):
            if gt_idx in used_gt_indices:
                continue
            labels[gt_idx] = cav_id
            used_gt_indices.add(gt_idx)

        return labels

    def _trust_frame_context(self, cav_detections):
        if not cav_detections:
            return {
                'frame': self.trust_fusion.frame_counter,
                'frame_index': self.trust_fusion.frame_counter,
                'scenario_index': -1,
                'timestamp': '',
                'timestamp_index': -1,
                'num_cavs': 0,
                'ego_id': 'ego',
            }
        first = cav_detections[0]
        ego_det = next((det for det in cav_detections
                        if det.get('is_ego', False)), first)
        return {
            'frame': self.trust_fusion.frame_counter,
            'frame_index': self.trust_fusion.frame_counter,
            'scenario_index': first.get('scenario_index', -1),
            'timestamp': first.get('timestamp', ''),
            'timestamp_index': first.get('timestamp_index', -1),
            'num_cavs': len(cav_detections),
            'ego_id': str(ego_det.get('trust_id',
                                      ego_det.get('cav_id', 'ego'))),
        }

    def _decode_single_cav_detection(self, cav_id, cav_content, cav_output):
        transformation_matrix = cav_content['transformation_matrix']
        anchor_box = cav_content['anchor_box']
        post_params = self.post_processor.params

        prob = cav_output['psm']
        prob = torch.sigmoid(prob.permute(0, 2, 3, 1)).reshape(1, -1)
        reg = cav_output['rm']
        batch_box3d = self.post_processor.delta_to_boxes3d(reg, anchor_box)
        mask = torch.gt(prob, post_params['target_args']['score_threshold'])
        mask = mask.view(1, -1)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

        assert batch_box3d.shape[0] == 1
        boxes3d = torch.masked_select(batch_box3d[0],
                                      mask_reg[0]).view(-1, 7)
        scores = torch.masked_select(prob[0], mask[0])
        if boxes3d.shape[0] == 0:
            return None

        boxes3d_corner = box_utils.boxes_to_corners_3d(
            boxes3d,
            order=post_params['order'])
        projected_boxes3d = box_utils.project_box3d(boxes3d_corner,
                                                    transformation_matrix)
        projected_boxes2d = box_utils.corner_to_standup_box_torch(
            projected_boxes3d)
        labels = torch.ones(scores.shape[0],
                            dtype=torch.long,
                            device=scores.device)

        original_cav_id = cav_content.get('original_cav_id', cav_id)
        is_ego = cav_content.get('is_ego', cav_id == 'ego')
        trust_id = self.trust_fusion.reputation_manager.external_id(
            cav_id,
            original_cav_id,
            is_ego)

        return {
            'cav_id': cav_id,
            'original_cav_id': original_cav_id,
            'trust_id': trust_id,
            'is_ego': is_ego,
            'scenario_index': cav_content.get('scenario_index', -1),
            'timestamp': cav_content.get('timestamp', ''),
            'timestamp_index': cav_content.get('timestamp_index', -1),
            'boxes3d': projected_boxes3d,
            'boxes2d': projected_boxes2d,
            'scores': scores,
            'labels': labels,
            'lidar_pose': cav_content.get('lidar_pose'),
            'ego_lidar_pose': cav_content.get('ego_lidar_pose'),
        }
