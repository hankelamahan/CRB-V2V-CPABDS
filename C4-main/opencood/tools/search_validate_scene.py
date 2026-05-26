# -*- coding: utf-8 -*-
# Author: OpenAI Codex

"""
Index validate scenes and frames with light-weight heuristics so users can
quickly find candidate samples before manual visualization.
"""

import argparse
import csv
import math
import os

import yaml

from opencood.hypes_yaml.yaml_utils import load_yaml


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
)
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'logs',
                                   'validate_scene_index.csv')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Search validate scenes with metadata and simple '
                    'obstacle-like heuristics.'
    )
    parser.add_argument('--validate_dir',
                        type=str,
                        default='',
                        help='validate dataset directory. If empty, the '
                             'script tries local defaults.')
    parser.add_argument('--hypes_yaml',
                        type=str,
                        default='',
                        help='optional YAML config. When provided and '
                             '--validate_dir is empty, validate_dir will be '
                             'loaded from this YAML.')
    parser.add_argument('--output_path',
                        type=str,
                        default='',
                        help='CSV output path. Defaults to '
                             'logs/validate_scene_index.csv.')
    parser.add_argument('--target_cav_count',
                        type=int,
                        default=None,
                        help='only keep scenes with this many positive-ID '
                             'CAVs.')
    parser.add_argument('--require_rsu',
                        action='store_true',
                        help='only keep scenes that contain at least one RSU.')
    parser.add_argument('--exclude_rsu',
                        action='store_true',
                        help='only keep scenes without RSU.')
    parser.add_argument('--min_object_count',
                        type=int,
                        default=0,
                        help='minimum visible object count in a frame.')
    parser.add_argument('--max_ego_speed',
                        type=float,
                        default=None,
                        help='maximum allowed ego speed in a frame.')
    parser.add_argument('--scenario_id',
                        type=str,
                        default='',
                        help='only inspect one scenario directory name.')
    parser.add_argument('--timestamp',
                        type=str,
                        default='',
                        help='only inspect one timestamp, such as 000291.')
    parser.add_argument('--candidate_only',
                        action='store_true',
                        help='only keep frames flagged by candidate '
                             'heuristics.')
    parser.add_argument('--candidate_type',
                        type=str,
                        default='either',
                        choices=['either', 'blocking', 'dense'],
                        help='candidate type used with --candidate_only.')
    parser.add_argument('--max_results',
                        type=int,
                        default=20,
                        help='preview row count printed to stdout.')
    return parser.parse_args()


def resolve_path(path):
    if not path:
        return ''
    if os.path.isabs(path):
        return path

    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path

    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def resolve_validate_dir(opt):
    candidate_paths = []

    if opt.validate_dir:
        candidate_paths.append(resolve_path(opt.validate_dir))

    if opt.hypes_yaml and not opt.validate_dir:
        hypes_yaml = resolve_path(opt.hypes_yaml)
        params = load_yaml(hypes_yaml)
        if 'validate_dir' not in params:
            raise KeyError('validate_dir is not defined in %s' % hypes_yaml)
        candidate_paths.append(resolve_path(params['validate_dir']))

    if not candidate_paths:
        for relative_path in ['v2xset/validate', 'opv2v_data_dumping/validate']:
            candidate_paths.append(resolve_path(relative_path))

    for path in candidate_paths:
        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        'Cannot resolve validate_dir. Provide --validate_dir or --hypes_yaml.'
    )


def resolve_output_path(output_path):
    if output_path:
        path = resolve_path(output_path)
    else:
        path = DEFAULT_OUTPUT_PATH

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return path


def list_scenario_dirs(validate_dir, scenario_id=''):
    scenario_dirs = sorted([
        os.path.join(validate_dir, name)
        for name in os.listdir(validate_dir)
        if os.path.isdir(os.path.join(validate_dir, name))
    ])

    if scenario_id:
        scenario_dirs = [
            path for path in scenario_dirs
            if os.path.basename(path) == scenario_id
        ]

    return scenario_dirs


def split_agent_dirs(scenario_dir):
    raw_agent_ids = sorted([
        name for name in os.listdir(scenario_dir)
        if os.path.isdir(os.path.join(scenario_dir, name))
    ])

    positive_ids = []
    negative_ids = []

    for agent_id in raw_agent_ids:
        if int(agent_id) >= 0:
            positive_ids.append(agent_id)
        else:
            negative_ids.append(agent_id)

    return positive_ids, negative_ids


def list_timestamps(cav_dir):
    yaml_files = sorted([
        name for name in os.listdir(cav_dir)
        if name.endswith('.yaml') and 'additional' not in name
    ])
    return [name.replace('.yaml', '') for name in yaml_files]


def load_frame_yaml(yaml_path):
    with open(yaml_path, 'r') as handle:
        return yaml.safe_load(handle)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def world_to_ego(dx, dy, ego_yaw_degree):
    yaw_rad = math.radians(safe_float(ego_yaw_degree))
    local_x = math.cos(yaw_rad) * dx + math.sin(yaw_rad) * dy
    local_y = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
    return local_x, local_y


def analyze_objects(vehicles, lidar_pose):
    ego_x = safe_float(lidar_pose[0]) if len(lidar_pose) > 0 else 0.0
    ego_y = safe_float(lidar_pose[1]) if len(lidar_pose) > 1 else 0.0
    ego_yaw = safe_float(lidar_pose[4]) if len(lidar_pose) > 4 else 0.0

    nearby_distance = 25.0
    front_distance = 20.0
    front_lateral = 4.0
    static_speed = 0.5

    metrics = {
        'object_count': 0,
        'nearby_object_count': 0,
        'nearby_static_count': 0,
        'front_object_count': 0,
        'front_static_count': 0,
        'nearest_object_distance': -1.0,
        'nearest_front_static_distance': -1.0,
        'candidate_static_object_dense': False,
        'candidate_blocking_object': False,
    }

    nearest_distance = None
    nearest_front_static_distance = None

    for _, object_info in vehicles.items():
        metrics['object_count'] += 1

        location = object_info.get('location', [])
        if len(location) < 2:
            continue

        object_x = safe_float(location[0])
        object_y = safe_float(location[1])
        object_speed = safe_float(object_info.get('speed', 0.0))
        dx = object_x - ego_x
        dy = object_y - ego_y
        distance = math.hypot(dx, dy)

        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance

        is_static = abs(object_speed) <= static_speed

        if distance <= nearby_distance:
            metrics['nearby_object_count'] += 1
            if is_static:
                metrics['nearby_static_count'] += 1

        local_x, local_y = world_to_ego(dx, dy, ego_yaw)
        if 0.0 < local_x <= front_distance and abs(local_y) <= front_lateral:
            metrics['front_object_count'] += 1
            if is_static:
                metrics['front_static_count'] += 1
                if (nearest_front_static_distance is None or
                        distance < nearest_front_static_distance):
                    nearest_front_static_distance = distance

    metrics['nearest_object_distance'] = \
        round(nearest_distance, 3) if nearest_distance is not None else -1.0
    metrics['nearest_front_static_distance'] = \
        round(nearest_front_static_distance, 3) \
        if nearest_front_static_distance is not None else -1.0
    metrics['candidate_static_object_dense'] = \
        metrics['nearby_static_count'] >= 3
    metrics['candidate_blocking_object'] = \
        metrics['front_static_count'] >= 1

    return metrics


def build_candidate_reasons(row):
    reasons = []
    if row['candidate_blocking_object']:
        reasons.append('front_static_object')
    if row['candidate_static_object_dense']:
        reasons.append('dense_static_objects')
    if row['ego_speed'] <= 0.5:
        reasons.append('low_ego_speed')
    return '|'.join(reasons)


def collect_rows(validate_dir):
    rows = []
    scenario_dirs = list_scenario_dirs(validate_dir)
    dataset_index_offset = 0

    for scenario_dir in scenario_dirs:
        scenario_id = os.path.basename(scenario_dir)
        cav_ids, rsu_ids = split_agent_dirs(scenario_dir)
        if not cav_ids:
            continue

        ego_cav_id = cav_ids[0]
        timestamps = list_timestamps(os.path.join(scenario_dir, ego_cav_id))
        frame_count = len(timestamps)

        for frame_index_in_scenario, timestamp in enumerate(timestamps):
            dataset_index = dataset_index_offset + frame_index_in_scenario
            yaml_path = os.path.join(scenario_dir, ego_cav_id,
                                     timestamp + '.yaml')
            frame_meta = load_frame_yaml(yaml_path)
            vehicles = frame_meta.get('vehicles', {}) or {}
            lidar_pose = frame_meta.get('lidar_pose', []) or []
            object_metrics = analyze_objects(vehicles, lidar_pose)
            ego_speed = safe_float(frame_meta.get('ego_speed', 0.0))

            row = {
                'dataset_index': dataset_index,
                'frame_index_in_scenario': frame_index_in_scenario,
                'scenario_id': scenario_id,
                'timestamp': timestamp,
                'ego_cav_id': ego_cav_id,
                'cav_count': len(cav_ids),
                'rsu_count': len(rsu_ids),
                'has_rsu': len(rsu_ids) > 0,
                'frame_count': frame_count,
                'cav_ids': '|'.join(cav_ids),
                'rsu_ids': '|'.join(rsu_ids),
                'ego_speed': round(ego_speed, 6),
                'yaml_path': yaml_path,
            }
            row.update(object_metrics)
            row['candidate_reasons'] = build_candidate_reasons(row)
            rows.append(row)

        dataset_index_offset += frame_count

    return rows


def row_is_candidate(row, candidate_type):
    if candidate_type == 'blocking':
        return row['candidate_blocking_object']
    if candidate_type == 'dense':
        return row['candidate_static_object_dense']
    return (row['candidate_blocking_object'] or
            row['candidate_static_object_dense'])


def filter_rows(rows, opt):
    filtered_rows = []

    for row in rows:
        if opt.scenario_id and row['scenario_id'] != opt.scenario_id:
            continue
        if opt.timestamp and row['timestamp'] != opt.timestamp:
            continue
        if (opt.target_cav_count is not None and
                row['cav_count'] != opt.target_cav_count):
            continue
        if opt.require_rsu and row['rsu_count'] < 1:
            continue
        if opt.exclude_rsu and row['rsu_count'] > 0:
            continue
        if row['object_count'] < opt.min_object_count:
            continue
        if (opt.max_ego_speed is not None and
                row['ego_speed'] > opt.max_ego_speed):
            continue
        if opt.candidate_only and not row_is_candidate(row, opt.candidate_type):
            continue
        filtered_rows.append(row)

    return filtered_rows


def write_csv(rows, output_path):
    if not rows:
        fieldnames = [
            'dataset_index', 'frame_index_in_scenario', 'scenario_id',
            'timestamp', 'ego_cav_id', 'cav_count', 'rsu_count', 'has_rsu',
            'frame_count', 'cav_ids', 'rsu_ids', 'ego_speed',
            'object_count', 'nearby_object_count', 'nearby_static_count',
            'front_object_count', 'front_static_count',
            'nearest_object_distance', 'nearest_front_static_distance',
            'candidate_static_object_dense', 'candidate_blocking_object',
            'candidate_reasons', 'yaml_path'
        ]
    else:
        fieldnames = list(rows[0].keys())

    with open(output_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_preview(rows, max_results):
    preview_columns = [
        'dataset_index', 'scenario_id', 'timestamp', 'cav_count',
        'rsu_count', 'object_count', 'ego_speed',
        'candidate_blocking_object', 'candidate_static_object_dense'
    ]
    preview_rows = rows[:max_results]

    print(','.join(preview_columns))
    for row in preview_rows:
        values = [str(row[column]) for column in preview_columns]
        print(','.join(values))


def summarize_by_scenario(rows):
    summary = {}
    for row in rows:
        scenario_id = row['scenario_id']
        if scenario_id not in summary:
            summary[scenario_id] = {
                'frames': 0,
                'candidate_frames': 0,
                'blocking_frames': 0,
                'dense_frames': 0,
                'cav_count': row['cav_count'],
                'rsu_count': row['rsu_count'],
            }

        summary[scenario_id]['frames'] += 1
        if row['candidate_blocking_object']:
            summary[scenario_id]['blocking_frames'] += 1
            summary[scenario_id]['candidate_frames'] += 1
        elif row['candidate_static_object_dense']:
            summary[scenario_id]['dense_frames'] += 1
            summary[scenario_id]['candidate_frames'] += 1

    return summary


def print_summary(rows):
    scenario_summary = summarize_by_scenario(rows)
    print('scenarios: %d' % len(scenario_summary))
    print('frames: %d' % len(rows))
    if not rows:
        return

    blocking_count = sum(1 for row in rows if row['candidate_blocking_object'])
    dense_count = sum(1 for row in rows if row['candidate_static_object_dense'])
    print('blocking candidates: %d' % blocking_count)
    print('dense static candidates: %d' % dense_count)


def main():
    opt = parse_args()
    validate_dir = resolve_validate_dir(opt)
    output_path = resolve_output_path(opt.output_path)

    rows = collect_rows(validate_dir)
    filtered_rows = filter_rows(rows, opt)

    print('validate_dir: %s' % validate_dir)
    print('output_path: %s' % output_path)
    print_summary(rows)
    print('matched frames: %d' % len(filtered_rows))
    print_preview(filtered_rows, opt.max_results)

    write_csv(filtered_rows, output_path)
    print('saved %d rows to %s' % (len(filtered_rows), output_path))


if __name__ == '__main__':
    main()
