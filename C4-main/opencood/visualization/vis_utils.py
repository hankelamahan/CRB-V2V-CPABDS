# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import time

import cv2
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import cm

from opencood.utils import box_utils
from opencood.utils import common_utils

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
BACKGROUND_PRESETS = {
    'dark': (0.05, 0.05, 0.05),
    'black': (0.0, 0.0, 0.0),
    'white': (1.0, 1.0, 1.0),
}


def ensure_parent_dir(path):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def get_background_color(background):
    if isinstance(background, str):
        if background not in BACKGROUND_PRESETS:
            raise ValueError('Unsupported background preset %s' % background)
        return np.asarray(BACKGROUND_PRESETS[background], dtype=float)

    background_array = np.asarray(background, dtype=float)
    if background_array.shape != (3,):
        raise ValueError('background color must be an RGB triplet.')
    return background_array


def get_foreground_color(background):
    background_rgb = get_background_color(background)
    return 'black' if np.mean(background_rgb) >= 0.5 else 'white'


def resolve_frame_indices(total_frames, frame_index=None, num_frames=1):
    """
    Resolve a start frame and frame count into concrete dataset indices.

    Parameters
    ----------
    total_frames : int
        Dataset length.
    frame_index : int or None
        Starting frame. Negative values index from the end.
    num_frames : int
        Number of consecutive frames. Non-positive values mean until the end.

    Returns
    -------
    list[int]
        Concrete frame indices.
    """
    if total_frames <= 0:
        return []

    if frame_index is None:
        start_index = 0
    else:
        start_index = frame_index if frame_index >= 0 else \
            total_frames + frame_index

    if start_index < 0 or start_index >= total_frames:
        raise IndexError('frame_index %s is out of range for %s frames' %
                         (frame_index, total_frames))

    if num_frames is None or num_frames <= 0:
        end_index = total_frames
    else:
        end_index = min(total_frames, start_index + num_frames)

    return list(range(start_index, end_index))


def configure_visualizer(vis,
                         point_size=1.0,
                         background_color='dark',
                         show_coordinate_frame=False):
    opt = vis.get_render_option()
    opt.background_color = get_background_color(background_color)
    opt.point_size = point_size
    opt.show_coordinate_frame = show_coordinate_frame


def show_o3d_visualization(elements,
                           width=1920,
                           height=1080,
                           point_size=1.0,
                           background_color='black',
                           show_coordinate_frame=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    configure_visualizer(vis,
                         point_size=point_size,
                         background_color=background_color,
                         show_coordinate_frame=show_coordinate_frame)

    for element in elements:
        vis.add_geometry(element)
        vis.update_geometry(element)

    vis.run()
    vis.destroy_window()


def bbx2linset(bbx_corner, order='hwl', color=(0, 1, 0)):
    """
    Convert the torch tensor bounding box to o3d lineset for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    line_set : list
        The list containing linsets.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)
    else:
        bbx_corner = np.asarray(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner,
                                                   order)
    else:
        bbx_corner = bbx_corner.copy()

    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [list(color) for _ in range(len(lines))]
    bbx_linset = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbx)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bbx_linset.append(line_set)

    return bbx_linset


def bbx2oabb(bbx_corner, order='hwl', color=(0, 0, 1)):
    """
    Convert the torch tensor bounding box to o3d oabb for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    oabbs : list
        The list containing all oriented bounding boxes.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)
    else:
        bbx_corner = np.asarray(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner,
                                                   order)
    else:
        bbx_corner = bbx_corner.copy()
    oabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        oabb = tmp_pcd.get_oriented_bounding_box()
        oabb.color = color
        oabbs.append(oabb)

    return oabbs


def bbx2aabb(bbx_center, order):
    """
    Convert the torch tensor bounding box to o3d aabb for visualization.

    Parameters
    ----------
    bbx_center : torch.Tensor
        shape: (n, 7).

    order: str
        hwl or lwh.

    Returns
    -------
    aabbs : list
        The list containing all o3d.aabb
    """
    if not isinstance(bbx_center, np.ndarray):
        bbx_center = common_utils.torch_tensor_to_numpy(bbx_center)
    else:
        bbx_center = np.asarray(bbx_center)
    bbx_corner = box_utils.boxes_to_corners_3d(bbx_center, order)

    aabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        aabb = tmp_pcd.get_axis_aligned_bounding_box()
        aabb.color = (0, 0, 1)
        aabbs.append(aabb)

    return aabbs


def linset_assign_list(vis,
                       lineset_list1,
                       lineset_list2,
                       update_mode='update'):
    """
    Associate two lists of lineset.

    Parameters
    ----------
    vis : open3d.Visualizer
    lineset_list1 : list
    lineset_list2 : list
    update_mode : str
        Add or update the geometry.
    """
    for j in range(len(lineset_list1)):
        index = j if j < len(lineset_list2) else -1
        lineset_list1[j] = \
            lineset_assign(lineset_list1[j],
                                     lineset_list2[index])
        if update_mode == 'add':
            vis.add_geometry(lineset_list1[j])
        else:
            vis.update_geometry(lineset_list1[j])


def lineset_assign(lineset1, lineset2):
    """
    Assign the attributes of lineset2 to lineset1.

    Parameters
    ----------
    lineset1 : open3d.LineSet
    lineset2 : open3d.LineSet

    Returns
    -------
    The lineset1 object with 2's attributes.
    """

    lineset1.points = lineset2.points
    lineset1.lines = lineset2.lines
    lineset1.colors = lineset2.colors

    return lineset1


def color_encoding(intensity, mode='intensity'):
    """
    Encode the single-channel intensity to 3 channels rgb color.

    Parameters
    ----------
    intensity : np.ndarray
        Lidar intensity, shape (n,)

    mode : str
        The color rendering mode. intensity, z-value and constant are
        supported.

    Returns
    -------
    color : np.ndarray
        Encoded Lidar color, shape (n, 3)
    """
    assert mode in ['intensity', 'z-value', 'constant']

    if mode == 'intensity':
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    elif mode == 'z-value':
        min_value = -1.5
        max_value = 0.5
        norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        colors = m.to_rgba(intensity)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5
        int_color = colors[:, :3]

    elif mode == 'constant':
        # regard all point cloud the same color
        int_color = np.ones((intensity.shape[0], 3))
        int_color[:, 0] *= 247 / 255
        int_color[:, 1] *= 244 / 255
        int_color[:, 2] *= 237 / 255

    return int_color


def visualize_single_sample_output_gt(pred_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis=True,
                                      save_path='',
                                      mode='constant',
                                      width=1920,
                                      height=1080,
                                      point_size=1.0,
                                      background_color='black',
                                      show_pred=True,
                                      show_gt=True,
                                      pc_range=None,
                                      headless=False):
    """
    Visualize the prediction, groundtruth with point cloud together.

    Parameters
    ----------
    pred_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.

    mode : str
        Color rendering mode.
    """

    if len(pcd.shape) == 3:
        pcd = pcd[0]
    origin_lidar = pcd
    if not isinstance(pcd, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(pcd)
    origin_lidar = origin_lidar.copy()

    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)
    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    oabbs_pred = bbx2oabb(pred_tensor, color=(1, 0, 0)) \
        if pred_tensor is not None and show_pred else []
    oabbs_gt = bbx2oabb(gt_tensor, color=(0, 1, 0)) \
        if gt_tensor is not None and show_gt else []

    visualize_elements = [o3d_pcd] + oabbs_pred + oabbs_gt
    if show_vis:
        show_o3d_visualization(visualize_elements,
                               width=width,
                               height=height,
                               point_size=point_size,
                               background_color=background_color)
    if save_path:
        try:
            if headless:
                raise AttributeError('background_color headless fallback')

            save_o3d_visualization(visualize_elements,
                                   save_path,
                                   width=width,
                                   height=height,
                                   point_size=point_size,
                                   background_color=background_color)
        except AttributeError as exc:
            if not headless and "background_color" not in str(exc):
                raise
            save_inference_sample_plt(pred_tensor,
                                      gt_tensor,
                                      pcd,
                                      save_path,
                                      pc_range=pc_range,
                                      mode=mode,
                                      width=width,
                                      height=height,
                                      point_size=point_size,
                                      background_color=background_color,
                                      show_pred=show_pred,
                                      show_gt=show_gt)


def visualize_single_sample_output_bev(pred_box, gt_box, pcd, dataset,
                                       show_vis=True,
                                       save_path=''):
    """
    Visualize the prediction, groundtruth with point cloud together in
    a bev format.

    Parameters
    ----------
    pred_box : torch.Tensor
        (N, 4, 2) prediction.

    gt_box : torch.Tensor
        (N, 4, 2) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.
    """

    if not isinstance(pcd, np.ndarray):
        pcd = common_utils.torch_tensor_to_numpy(pcd)
    if pred_box is not None and not isinstance(pred_box, np.ndarray):
        pred_box = common_utils.torch_tensor_to_numpy(pred_box)
    if gt_box is not None and not isinstance(gt_box, np.ndarray):
        gt_box = common_utils.torch_tensor_to_numpy(gt_box)

    ratio = dataset.params["preprocess"]["args"]["res"]
    L1, W1, H1, L2, W2, H2 = dataset.params["preprocess"]["cav_lidar_range"]
    bev_origin = np.array([L1, W1]).reshape(1, -1)
    # (img_row, img_col)
    bev_map = dataset.project_points_to_bev_map(pcd, ratio)
    # (img_row, img_col, 3)
    bev_map = \
        np.repeat(bev_map[:, :, np.newaxis], 3, axis=-1).astype(np.float32)
    bev_map = bev_map * 255

    if pred_box is not None:
        num_bbx = pred_box.shape[0]
        for i in range(num_bbx):
            bbx = pred_box[i]

            bbx = ((bbx - bev_origin) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            cv2.polylines(bev_map, [bbx], True, (0, 0, 255), 1)

    if gt_box is not None and len(gt_box):
        for i in range(gt_box.shape[0]):
            bbx = gt_box[i][:4, :2]
            bbx = (((bbx - bev_origin)) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            cv2.polylines(bev_map, [bbx], True, (255, 0, 0), 1)

    if show_vis:
        plt.axis("off")
        plt.imshow(bev_map)
        plt.show()
    if save_path:
        plt.axis("off")
        plt.imshow(bev_map)
        plt.savefig(save_path)


def visualize_single_sample_dataloader(batch_data,
                                       o3d_pcd,
                                       order,
                                       key='origin_lidar',
                                       visualize=False,
                                       save_path='',
                                       oabb=False,
                                       mode='constant',
                                       width=1920,
                                       height=1080,
                                       point_size=1.0,
                                       background_color='dark',
                                       include_boxes=True):
    """
    Visualize a single frame of a single CAV for validation of data pipeline.

    Parameters
    ----------
    o3d_pcd : o3d.PointCloud
        Open3d PointCloud.

    order : str
        The bounding box order.

    key : str
        origin_lidar for late fusion and stacked_lidar for early fusion.

    visualize : bool
        Whether to visualize the sample.

    batch_data : dict
        The dictionary that contains current timestamp's data.

    save_path : str
        If set, save the visualization image to the path.

    oabb : bool
        If oriented bounding box is used.
    """

    if o3d_pcd is None:
        o3d_pcd = o3d.geometry.PointCloud()

    origin_lidar = batch_data[key]
    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    # we only visualize the first cav for single sample
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    origin_lidar = origin_lidar.copy()
    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)

    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    object_bbx_center = batch_data['object_bbx_center']
    object_bbx_mask = batch_data['object_bbx_mask']
    object_bbx_center = object_bbx_center[object_bbx_mask == 1]

    aabbs = []
    if include_boxes:
        aabbs = bbx2linset(object_bbx_center, order) if not oabb else \
            bbx2oabb(object_bbx_center, order)
    visualize_elements = [o3d_pcd] + aabbs
    if visualize:
        show_o3d_visualization(visualize_elements,
                               width=width,
                               height=height,
                               point_size=point_size,
                               background_color=background_color,
                               show_coordinate_frame=True)

    if save_path:
        save_o3d_visualization(visualize_elements,
                               save_path,
                               width=width,
                               height=height,
                               point_size=point_size,
                               background_color=background_color,
                               show_coordinate_frame=True)

    return o3d_pcd, aabbs


def visualize_inference_sample_dataloader(pred_box_tensor,
                                          gt_box_tensor,
                                          origin_lidar,
                                          o3d_pcd,
                                          mode='constant'):
    """
    Visualize a frame during inference for video stream.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_box_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    origin_lidar : torch.Tensor
        PointCloud, (N, 4).

    o3d_pcd : open3d.PointCloud
        Used to visualize the pcd.

    mode : str
        lidar point rendering mode.
    """

    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    # we only visualize the first cav for single sample
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    # this is for 2-stage origin lidar, it has different format
    if origin_lidar.shape[1] > 4:
        origin_lidar = origin_lidar[:, 1:]
    origin_lidar = origin_lidar.copy()

    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)

    if not isinstance(pred_box_tensor, np.ndarray):
        pred_box_tensor = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    if not isinstance(gt_box_tensor, np.ndarray):
        gt_box_tensor = common_utils.torch_tensor_to_numpy(gt_box_tensor)

    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    gt_o3d_box = bbx2linset(gt_box_tensor, order='hwl', color=(0, 1, 0))
    pred_o3d_box = bbx2linset(pred_box_tensor, color=(1, 0, 0))

    return o3d_pcd, pred_o3d_box, gt_o3d_box


def visualize_sequence_dataloader(dataloader,
                                  order,
                                  color_mode='constant',
                                  max_frames=None,
                                  width=1920,
                                  height=1080,
                                  point_size=1.0,
                                  background_color='dark',
                                  include_boxes=True):
    """
    Visualize the batch data in animation.

    Parameters
    ----------
    dataloader : torch.Dataloader
        Pytorch dataloader

    order : str
        Bounding box order(N, 7).

    color_mode : str
        Color rendering mode.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    configure_visualizer(vis,
                         point_size=point_size,
                         background_color=background_color,
                         show_coordinate_frame=True)

    # used to visualize lidar points
    vis_pcd = o3d.geometry.PointCloud()
    # used to visualize object bounding box, maximum 50
    vis_aabbs = []
    for _ in range(50):
        vis_aabbs.append(o3d.geometry.LineSet())

    processed_frames = 0
    while True:
        for i_batch, sample_batched in enumerate(dataloader):
            print(i_batch)
            pcd, aabbs = \
                visualize_single_sample_dataloader(sample_batched['ego'],
                                                   vis_pcd,
                                                   order,
                                                   mode=color_mode,
                                                   include_boxes=include_boxes)
            box_geometries = aabbs if len(aabbs) > 0 else \
                [o3d.geometry.LineSet()]
            if i_batch == 0:
                vis.add_geometry(pcd)
                for i in range(len(vis_aabbs)):
                    index = i if i < len(box_geometries) else -1
                    vis_aabbs[i] = lineset_assign(vis_aabbs[i],
                                                  box_geometries[index])
                    vis.add_geometry(vis_aabbs[i])

                for i in range(len(vis_aabbs)):
                    index = i if i < len(box_geometries) else -1
                    vis_aabbs[i] = lineset_assign(vis_aabbs[i],
                                                  box_geometries[index])
                    vis.update_geometry(vis_aabbs[i])

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)
            processed_frames += 1
            if max_frames is not None and processed_frames >= max_frames:
                vis.destroy_window()
                return

    vis.destroy_window()


def save_sequence_sample_plt(batch_data,
                             order,
                             pc_range,
                             save_path,
                             width=1920,
                             height=1080,
                             dpi=100,
                             mode='constant',
                             point_size=1.0,
                             background_color='dark',
                             include_boxes=True):
    """
    Save a single visualization frame with matplotlib. This is used as a
    fallback when Open3D cannot create a render window in headless
    environments.

    Parameters
    ----------
    batch_data : dict
        One sample from the dataloader, usually sample_batched['ego'].
    order : str
        Bounding box order.
    pc_range : list
        Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
    save_path : str
        Output image path.
    """
    origin_lidar = batch_data['origin_lidar']
    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    if origin_lidar.shape[1] > 4:
        origin_lidar = origin_lidar[:, 1:]
    point_colors = color_encoding(origin_lidar[:, -1]
                                  if mode == 'intensity'
                                  else origin_lidar[:, 2],
                                  mode=mode)

    object_bbx_center = batch_data['object_bbx_center']
    object_bbx_mask = batch_data['object_bbx_mask']
    if not isinstance(object_bbx_center, np.ndarray):
        object_bbx_center = common_utils.torch_tensor_to_numpy(
            object_bbx_center)
    if not isinstance(object_bbx_mask, np.ndarray):
        object_bbx_mask = common_utils.torch_tensor_to_numpy(object_bbx_mask)
    if len(object_bbx_center.shape) > 2:
        object_bbx_center = object_bbx_center[0]
    if len(object_bbx_mask.shape) > 1:
        object_bbx_mask = object_bbx_mask[0]

    valid_boxes = object_bbx_center[object_bbx_mask == 1] \
        if include_boxes else None
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    draw_points_boxes_plt(pc_range,
                          points=origin_lidar,
                          point_colors=point_colors,
                          boxes_gt=valid_boxes,
                          save_path=save_path,
                          ax=ax,
                          point_size=point_size,
                          background_color=background_color)


def draw_corner_boxes_plt(boxes_corner,
                          ax,
                          color=None,
                          linewidth_scale=1.0):
    if boxes_corner is None:
        return ax

    boxes_np = boxes_corner
    if not isinstance(boxes_np, np.ndarray):
        boxes_np = common_utils.torch_tensor_to_numpy(boxes_np)
    if len(boxes_np.shape) == 2:
        boxes_np = boxes_np[np.newaxis, ...]

    for box in boxes_np:
        bev = box[:4, :2]
        ax.plot(bev[[0, 1, 2, 3, 0], 0],
                bev[[0, 1, 2, 3, 0], 1],
                color=color,
                linewidth=0.8 * linewidth_scale)
        ax.plot(bev[[2, 3], 0],
                bev[[2, 3], 1],
                color=color,
                linewidth=2.0 * linewidth_scale)
    return ax


def save_inference_sample_plt(pred_tensor,
                              gt_tensor,
                              pcd,
                              save_path,
                              pc_range=None,
                              mode='constant',
                              width=1920,
                              height=1080,
                              dpi=100,
                              point_size=1.0,
                              background_color='dark',
                              show_pred=True,
                              show_gt=True):
    if len(pcd.shape) == 3:
        pcd = pcd[0]

    origin_lidar = pcd
    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    if origin_lidar.shape[1] > 4:
        origin_lidar = origin_lidar[:, 1:]
    origin_lidar = origin_lidar.copy()

    point_colors = color_encoding(origin_lidar[:, -1]
                                  if mode == 'intensity'
                                  else origin_lidar[:, 2],
                                  mode=mode)

    if pc_range is None:
        pc_range = [-140.8, -40, -3, 140.8, 40, 1]

    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    draw_points_boxes_plt(pc_range,
                          points=origin_lidar,
                          point_colors=point_colors,
                          ax=ax,
                          point_size=point_size,
                          background_color=background_color)

    if show_gt and gt_tensor is not None:
        draw_corner_boxes_plt(gt_tensor, ax, color='green')
    if show_pred and pred_tensor is not None:
        draw_corner_boxes_plt(pred_tensor, ax, color='red')

    ensure_parent_dir(save_path)
    ax.figure.savefig(save_path, facecolor=ax.figure.get_facecolor())
    plt.close(ax.figure)


def save_o3d_visualization(element,
                           save_path,
                           width=1920,
                           height=1080,
                           point_size=1.0,
                           background_color='dark',
                           show_coordinate_frame=False):
    """
    Save the open3d drawing to folder.

    Parameters
    ----------
    element : list
        List of o3d.geometry objects.

    save_path : str
        The save path.
    """
    ensure_parent_dir(save_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    configure_visualizer(vis,
                         point_size=point_size,
                         background_color=background_color,
                         show_coordinate_frame=show_coordinate_frame)
    for i in range(len(element)):
        vis.add_geometry(element[i])
        vis.update_geometry(element[i])

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(save_path)
    vis.destroy_window()


def visualize_bev(batch_data):
    bev_input = batch_data["processed_lidar"]["bev_input"]
    label_map = batch_data["label_dict"]["label_map"]
    if not isinstance(bev_input, np.ndarray):
        bev_input = common_utils.torch_tensor_to_numpy(bev_input)

    if not isinstance(label_map, np.ndarray):
        label_map = label_map[0].numpy() if not label_map[0].is_cuda else \
            label_map[0].cpu().detach().numpy()

    if len(bev_input.shape) > 3:
        bev_input = bev_input[0, ...]

    plt.matshow(np.sum(bev_input, axis=0))
    plt.axis("off")
    plt.matshow(label_map[0, :, :])
    plt.axis("off")
    plt.show()


def draw_box_plt(boxes_dec, ax, color=None, linewidth_scale=1.0):
    """
    draw boxes in a given plt ax
    :param boxes_dec: (N, 5) or (N, 7) in metric
    :param ax:
    :return: ax with drawn boxes
    """
    if not len(boxes_dec)>0:
        return ax
    boxes_np= boxes_dec
    if not isinstance(boxes_np, np.ndarray):
        boxes_np = boxes_np.cpu().detach().numpy()
    if boxes_np.shape[-1]>5:
        boxes_np = boxes_np[:, [0, 1, 3, 4, 6]]
    x = boxes_np[:, 0]
    y = boxes_np[:, 1]
    dx = boxes_np[:, 2]
    dy = boxes_np[:, 3]

    x1 = x - dx / 2
    y1 = y - dy / 2
    x2 = x + dx / 2
    y2 = y + dy / 2
    theta = boxes_np[:, 4:5]
    # bl, fl, fr, br
    corners = np.array([[x1, y1],[x1,y2], [x2,y2], [x2, y1]]).transpose(2, 0, 1)
    new_x = (corners[:, :, 0] - x[:, None]) * np.cos(theta) + (corners[:, :, 1]
              - y[:, None]) * (-np.sin(theta)) + x[:, None]
    new_y = (corners[:, :, 0] - x[:, None]) * np.sin(theta) + (corners[:, :, 1]
              - y[:, None]) * (np.cos(theta)) + y[:, None]
    corners = np.stack([new_x, new_y], axis=2)
    for corner in corners:
        ax.plot(corner[[0,1,2,3,0], 0], corner[[0,1,2,3,0], 1], color=color, linewidth=0.5*linewidth_scale)
        # draw front line (
        ax.plot(corner[[2, 3], 0], corner[[2, 3], 1], color=color, linewidth=2*linewidth_scale)
    return ax


def draw_points_boxes_plt(pc_range, points=None, boxes_pred=None, boxes_gt=None, save_path=None,
                          points_c='y.', bbox_gt_c='green', bbox_pred_c='red',
                          point_colors=None, point_size=0.1,
                          background_color='white', return_ax=False, ax=None):
    created_fig = False
    background_rgb = get_background_color(background_color)
    foreground = get_foreground_color(background_color)
    if ax is None:
        ax = plt.figure(figsize=(15, 6)).add_subplot(1, 1, 1)
        created_fig = True
        ax.set_aspect('equal', 'box')
        ax.set(xlim=(pc_range[0], pc_range[3]),
               ylim=(pc_range[1], pc_range[4]))
    else:
        ax.set_aspect('equal', 'box')
        ax.set(xlim=(pc_range[0], pc_range[3]),
               ylim=(pc_range[1], pc_range[4]))
    ax.set_facecolor(background_rgb)
    ax.figure.set_facecolor(background_rgb)
    ax.tick_params(colors=foreground)
    ax.xaxis.label.set_color(foreground)
    ax.yaxis.label.set_color(foreground)
    for spine in ax.spines.values():
        spine.set_color(foreground)
    if points is not None:
        if point_colors is not None:
            ax.scatter(points[:, 0],
                       points[:, 1],
                       s=max(point_size, 0.1),
                       c=point_colors,
                       marker='.',
                       linewidths=0)
        else:
            ax.plot(points[:, 0], points[:, 1], points_c,
                    markersize=max(point_size, 0.1))
    if (boxes_gt is not None) and len(boxes_gt)>0:
        ax = draw_box_plt(boxes_gt, ax, color=bbox_gt_c)
    if (boxes_pred is not None) and len(boxes_pred)>0:
        ax = draw_box_plt(boxes_pred, ax, color=bbox_pred_c)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if save_path:
        ensure_parent_dir(save_path)
        ax.figure.savefig(save_path, facecolor=ax.figure.get_facecolor())
    if return_ax:
        return ax
    if created_fig:
        plt.close(ax.figure)
