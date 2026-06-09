# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for intermediate fusion
"""
import random
import math
import warnings
from collections import OrderedDict

import numpy as np
import torch

import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2

from opencood.data_utils.datasets.overlap_field_voting import OverlapFieldVotingSystem
from reputation_client import ReputationClient



class ReputationClientAdapter:
    """
    Bridges ReputationClient (HTTP-backed) with the interface expected by
    IntermediateFusionDataset: get per-vehicle reputation scores and report
    fused detection results back to the server.

    Ego vehicle always returns 1.0 without a network round-trip.
    numpy arrays are serialised to plain lists before being handed to the
    HTTP client.
    """

    def __init__(self, client: ReputationClient, ego_id: str = "ego"):
        self._client = client
        self._ego_id = str(ego_id)

    def get_reputation(self, vehicle_id: str) -> float:
        if str(vehicle_id) == self._ego_id:
            return 1.0
        return self._client.get_reputation(str(vehicle_id))

    def get_batch_reputations(self, vehicle_ids) -> dict:
        non_ego = [str(v) for v in vehicle_ids if str(v) != self._ego_id]
        result = self._client.get_batch_reputations(non_ego)
        for vid in vehicle_ids:
            if str(vid) == self._ego_id:
                result[str(vid)] = 1.0
        return result

    def report_fused_boxes(
        self,
        fused_boxes,
        fused_scores,
        fused_labels,
        cav_detections: dict,
    ) -> dict:
        """Upload fused boxes and per-CAV detections; return updated reputations."""
        def _to_list(x):
            return x.tolist() if hasattr(x, "tolist") else list(x)

        serialised_detections = {}
        for cav_id, det in cav_detections.items():
            serialised_detections[str(cav_id)] = {
                "boxes":  _to_list(det["boxes"]),
                "scores": _to_list(det["scores"]),
                "labels": _to_list(det["labels"]),
            }

        return self._client.report_fused_boxes(
            reporter_id=self._ego_id,
            fused_boxes=_to_list(fused_boxes),
            fused_scores=_to_list(fused_scores),
            fused_labels=_to_list(fused_labels),
            cav_detections=serialised_detections,
        )



class IntermediateFusionDataset(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """
    def __init__(self, params, visualize, train=True):
        super(IntermediateFusionDataset, self). \
            __init__(params, visualize, train)

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.proj_first = True
        if 'proj_first' in params['fusion']['args'] and \
            not params['fusion']['args']['proj_first']:
            self.proj_first = False

        # whether there is a time delay between the time that cav project
        # lidar to ego and the ego receive the delivered feature
        self.cur_ego_pose_flag = True if 'cur_ego_pose_flag' not in \
            params['fusion']['args'] else \
            params['fusion']['args']['cur_ego_pose_flag']

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

        # ========== [新增] 初始化重叠视场投票系统 ==========
        # 从配置文件中读取参数，如果没有则使用默认值
        trust_params = params.get('trust_fusion', {})
        self.use_trust_fusion = trust_params.get('use_trust_fusion', True)
        self.reputation_adapter = None

        if self.use_trust_fusion:
            server_url = trust_params.get('reputation_server_url', 'http://localhost:8888')
            reputation_client = ReputationClient(
                server_url=server_url,
                cache_capacity=trust_params.get('cache_capacity', 100),
                cache_ttl=trust_params.get('cache_ttl', 60),
            )
            self.reputation_adapter = ReputationClientAdapter(
                client=reputation_client,
                ego_id=str(params.get('ego_id', 'ego'))
            )

        self.current_frame_detections = {}
        # ========== [新增结束] ==========

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx,
                                                 cur_ego_pose_flag=self.cur_ego_pose_flag)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break
        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = \
            self.get_pairwise_transformation(base_data_dict,
                                             self.max_cav)

        processed_features = []
        object_stack = []
        object_id_stack = []

        # prior knowledge for time delay correction and indicating data type
        # (V2V vs V2i)
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []

        # ========== [新增] 存储各车的CAV ID和检测结果，用于投票融合 ==========
        cav_id_list = []  # 存储参与融合的车辆ID
        cav_boxes_list = []  # 存储各车的检测框
        cav_scores_list = []  # 存储各车的置信度
        cav_labels_list = []  # 存储各车的标签
        # ========== [新增结束] ==========

        if self.visualize:
            projected_lidar_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            object_stack.append(selected_cav_processed['object_bbx_center'])
            object_id_stack += selected_cav_processed['object_ids']
            processed_features.append(
                selected_cav_processed['processed_features'])

            velocity.append(selected_cav_processed['velocity'])
            time_delay.append(float(selected_cav_base['time_delay']))
            # this is only useful when proj_first = True, and communication
            # delay is considered. Right now only V2X-ViT utilizes the
            # spatial_correction. There is a time delay when the cavs project
            # their lidar to ego and when the ego receives the feature, and
            # this variable is used to correct such pose difference (ego_t-1 to
            # ego_t)
            spatial_correction_matrix.append(
                selected_cav_base['params']['spatial_correction_matrix'])
            infra.append(1 if int(cav_id) < 0 else 0)

            # ========== [新增] 收集各车的检测结果，用于后续投票融合 ==========
            if self.use_trust_fusion and self.reputation_adapter is not None:
                # 获取该车的检测框（已在get_item_single_car中转换为ego坐标系）
                cav_boxes = selected_cav_processed.get('object_bbx_center', [])
                # 注意：这里需要将7维检测框转换为WBF需要的4维[x1,y1,x2,y2]格式
                # 假设检测框格式为 [x, y, z, w, l, h, heading]，取前两维和宽高计算2D框
                if len(cav_boxes) > 0:
                    # 将3D检测框转换为2D边界框（归一化坐标）
                    # 这是一个简化转换，实际需要根据相机内参或投影矩阵计算
                    boxes_2d = self._convert_3d_to_2d_boxes(cav_boxes)
                    cav_boxes_list.append(boxes_2d)
                    # 使用检测框的置信度（这里用默认值0.8，实际应从检测结果中获取）
                    cav_scores_list.append([0.8] * len(cav_boxes))
                    # 使用默认标签（1表示车辆）
                    cav_labels_list.append([1] * len(cav_boxes))
                    cav_id_list.append(cav_id)
                    
                    # 存储原始检测结果，用于后续信誉更新
                    self.current_frame_detections[cav_id] = {
                        'boxes': boxes_2d,
                        'scores': [0.8] * len(cav_boxes),
                        'labels': [1] * len(cav_boxes)
                    }
            # ========== [新增结束] ==========

            if self.visualize:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])

        # ==========  执行重叠视场投票融合 ==========
        fused_boxes = None
        fused_scores = None
        fused_labels = None
        
        if self.use_trust_fusion and self.voting_system is not None and len(cav_id_list) > 0:
            # 获取各车的信誉值
            trust_scores = [self.voting_system.get_reputation(vid) for vid in cav_id_list]
            # 存储当前帧信誉值，用于后续更新
            for vid, score in zip(cav_id_list, trust_scores):
                self.current_frame_reputations[vid] = score
            
            # 构建检测结果字典
            detections_dict = {}
            for i, vid in enumerate(cav_id_list):
                detections_dict[vid] = {
                    'boxes': cav_boxes_list[i],
                    'scores': cav_scores_list[i],
                    'labels': cav_labels_list[i]
                }
            
            # 执行投票融合
            fused_boxes, fused_scores, fused_labels = self.voting_system.fuse(detections_dict)
        # ========== ==========

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(processed_features)
        merged_feature_dict = self.merge_features_to_dict(processed_features)

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        # pad dv, dt, infra to max_cav
        velocity = velocity + (self.max_cav - len(velocity)) * [0.]
        time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.max_cav - len(infra)) * [0.]
        spatial_correction_matrix = np.stack(spatial_correction_matrix)
        padding_eye = np.tile(np.eye(4)[None],(self.max_cav - len(
                                               spatial_correction_matrix),1,1))
        spatial_correction_matrix = np.concatenate([spatial_correction_matrix,
                                                   padding_eye], axis=0)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'velocity': velocity,
             'time_delay': time_delay,
             'infra': infra,
             'spatial_correction_matrix': spatial_correction_matrix,
             'pairwise_t_matrix': pairwise_t_matrix})

        # ========== [新增] 将融合结果添加到输出字典中 ==========
        if self.use_trust_fusion and fused_boxes is not None:
            processed_data_dict['ego'].update({
                'fused_boxes': fused_boxes,
                'fused_scores': fused_scores,
                'fused_labels': fused_labels,
                'cav_id_list': cav_id_list
            })
        # ========== [新增结束] ==========

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})
        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            selected_cav_base['params']['transformation_matrix']

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       ego_pose)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                         transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # velocity
        velocity = selected_cav_base['params']['ego_speed']
        # normalize veloccity by average speed 30 km/h
        velocity = velocity / 30

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             'velocity': velocity})

        return selected_cav_processed

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    # ========== [新增] 辅助方法：3D检测框转2D边界框 ==========
    def _convert_3d_to_2d_boxes(self, boxes_3d, image_size=800):
        """
        将3D检测框转换为2D边界框（用于WBF融合）
        
        参数:
            boxes_3d: numpy array, 形状(N, 7)，格式[x, y, z, w, l, h, heading]
            image_size: 图像尺寸，用于归一化坐标
        
        返回:
            boxes_2d: list of [x1, y1, x2, y2] 归一化坐标
        """
        boxes_2d = []
        for box in boxes_3d:
            # 简化转换：使用x和y坐标加上宽度和长度
            x, y, w, l = box[0], box[1], box[3], box[4]
            # 计算2D边界框
            x1 = (x - w/2) / image_size
            y1 = (y - l/2) / image_size
            x2 = (x + w/2) / image_size
            y2 = (y + l/2) / image_size
            # 限制在[0,1]范围内
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))
            boxes_2d.append([x1, y1, x2, y2])
        return boxes_2d
    # ========== [新增结束] ==========

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []

        # used for PriorEncoding for models
        velocity = []
        time_delay = []
        infra = []

        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        # used for correcting the spatial transformation between delayed timestamp
        # and current timestamp
        spatial_correction_matrix_list = []

        # ========== [新增] 收集融合结果用于batch处理 ==========
        fused_boxes_list = []
        fused_scores_list = []
        fused_labels_list = []
        cav_id_lists = []
        # ========== [新增结束] ==========

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])

            processed_lidar_list.append(ego_dict['processed_lidar'])
            record_len.append(ego_dict['cav_num'])
            label_dict_list.append(ego_dict['label_dict'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            spatial_correction_matrix_list.append(
                ego_dict['spatial_correction_matrix'])

            # ========== [新增] 收集融合结果 ==========
            if 'fused_boxes' in ego_dict:
                fused_boxes_list.append(ego_dict['fused_boxes'])
                fused_scores_list.append(ego_dict['fused_scores'])
                fused_labels_list.append(ego_dict['fused_labels'])
                cav_id_lists.append(ego_dict.get('cav_id_list', []))
            # ========== [新增结束] ==========

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])
        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav)
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix_list = \
            torch.from_numpy(np.array(spatial_correction_matrix_list))
        # (B, max_cav, 3)
        prior_encoding = \
            torch.stack([velocity, time_delay, infra], dim=-1).float()
        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'object_ids': object_ids[0],
                                   'prior_encoding': prior_encoding,
                                   'spatial_correction_matrix': spatial_correction_matrix_list,
                                   'pairwise_t_matrix': pairwise_t_matrix})

        # ========== [新增] 将融合结果添加到batch输出 ==========
        if len(fused_boxes_list) > 0:
            output_dict['ego'].update({
                'fused_boxes_batch': fused_boxes_list,
                'fused_scores_batch': fused_scores_list,
                'fused_labels_batch': fused_labels_list,
                'cav_id_lists': cav_id_lists
            })
        # ========== [新增结束] ==========

        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(
                    batch[0]['ego'][
                        'anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch})

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
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            pairwise_t_matrix[:, :] = np.identity(4)
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                t_list.append(cav_content['params']['transformation_matrix'])

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i == j:
                        t_matrix = np.eye(4)
                        pairwise_t_matrix[i, j] = t_matrix
                        continue
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix

    def get_vehicle_reputation(self, vehicle_id):
        """Get reputation score for a vehicle via the remote server."""
        if self.reputation_adapter is not None:
            return self.reputation_adapter.get_reputation(vehicle_id)
        return 0.5

    def get_batch_reputations(self, vehicle_ids):
        """Batch-fetch reputation scores for multiple vehicles."""
        if self.reputation_adapter is not None:
            return self.reputation_adapter.get_batch_reputations(vehicle_ids)
        return {str(vid): 0.5 for vid in vehicle_ids}
