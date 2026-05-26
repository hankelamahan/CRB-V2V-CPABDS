# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_vis_dir', type=str, default='',
                        help='directory for targeted visualization export.')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--frame_index', type=int, default=None,
                        help='starting frame index for visualization export. '
                             'Negative values count from the end.')
    parser.add_argument('--frame_indices', type=str, default='',
                        help='comma-separated explicit frame indices for '
                             'targeted export, e.g. "0,167,608". Negative '
                             'values count from the end.')
    parser.add_argument('--num_frames', type=int, default=1,
                        help='number of consecutive frames to export. '
                             'Use 0 or a negative value to export until the '
                             'dataset end.')
    parser.add_argument('--color_mode', type=str, default='constant',
                        choices=['constant', 'intensity', 'z-value'],
                        help='lidar color rendering mode for visualization.')
    parser.add_argument('--width', type=int, default=1920,
                        help='visualization width in pixels.')
    parser.add_argument('--height', type=int, default=1080,
                        help='visualization height in pixels.')
    parser.add_argument('--point_size', type=float, default=1.0,
                        help='Open3D point size.')
    parser.add_argument('--background', type=str, default='dark',
                        choices=sorted(vis_utils.BACKGROUND_PRESETS.keys()),
                        help='background preset for visualization output.')
    parser.add_argument('--headless', action='store_true',
                        help='use matplotlib fallback when saving images '
                             'without a render window.')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='limit the number of evaluated frames. '
                             'Use 0 or a negative value to process the full '
                             'validation set.')
    parser.add_argument('--pred_only', action='store_true',
                        help='only render prediction boxes.')
    parser.add_argument('--gt_only', action='store_true',
                        help='only render ground-truth boxes.')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                    'the results in single ' \
                                                    'image mode or video mode'
    assert not (opt.pred_only and opt.gt_only), \
        '--pred_only and --gt_only can not be used together.'

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    targeted_save_dir = opt.save_vis_dir
    selected_frame_indices = set()
    if not targeted_save_dir and opt.frame_index is not None:
        targeted_save_dir = os.path.join(opt.model_dir, 'vis_selected')
    if targeted_save_dir:
        os.makedirs(targeted_save_dir, exist_ok=True)
        if opt.frame_indices:
            selected_frame_indices = set()
            for raw_index in opt.frame_indices.split(','):
                raw_index = raw_index.strip()
                if not raw_index:
                    continue
                resolved_indices = vis_utils.resolve_frame_indices(
                    len(opencood_dataset),
                    int(raw_index),
                    1
                )
                selected_frame_indices.update(resolved_indices)
        else:
            selected_frame_indices = set(
                vis_utils.resolve_frame_indices(len(opencood_dataset),
                                                opt.frame_index,
                                                opt.num_frames)
            )

    effective_max_frames = opt.max_frames
    if selected_frame_indices:
        max_selected_index = max(selected_frame_indices) + 1
        if effective_max_frames <= 0 or effective_max_frames < max_selected_index:
            effective_max_frames = max_selected_index

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation.
    # also store the confidence score for each prediction
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=opt.width, height=opt.height)
        vis_utils.configure_visualizer(vis,
                                       point_size=opt.point_size,
                                       background_color=opt.background,
                                       show_coordinate_frame=True)

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    for i, batch_data in tqdm(enumerate(data_loader)):
        if effective_max_frames > 0 and i >= effective_max_frames:
            break
        # print(i)
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                   gt_box_tensor,
                                                   batch_data['ego'][
                                                       'origin_lidar'][0],
                                                   i,
                                                   npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if targeted_save_dir and i in selected_frame_indices:
                targeted_save_path = os.path.join(
                    targeted_save_dir,
                    '%s_%s_frame_%05d.png' % (
                        opt.fusion_method, opt.color_mode, i))
                vis_utils.visualize_single_sample_output_gt(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'],
                    show_vis=False,
                    save_path=targeted_save_path,
                    mode=opt.color_mode,
                    width=opt.width,
                    height=opt.height,
                    point_size=opt.point_size,
                    background_color=opt.background,
                    show_pred=not opt.gt_only,
                    show_gt=not opt.pred_only,
                    pc_range=hypes['preprocess']['cav_lidar_range'],
                    headless=opt.headless)
                print('saved frame %d to %s' % (i, targeted_save_path))

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        vis_pcd,
                        mode=opt.color_mode
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  opt.global_sort_detections)
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
