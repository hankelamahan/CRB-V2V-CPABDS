#!/usr/bin/env python3

import argparse
import os

import numpy as np
import torch
import yaml

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare a bootstrap PointPillar intermediate model_dir.'
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        default='opencood/logs/pointpillar_intermediate_bootstrap_v2xset',
        help='target model directory.'
    )
    parser.add_argument(
        '--base_yaml',
        type=str,
        default='opencood/hypes_yaml/point_pillar_intermediate_fusion.yaml',
        help='base yaml used to generate config.yaml.'
    )
    parser.add_argument(
        '--validate_dir',
        type=str,
        default='v2xset/validate',
        help='dataset directory used for inference.'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default='',
        help='optional root_dir override. Defaults to validate_dir.'
    )
    parser.add_argument(
        '--checkpoint_name',
        type=str,
        default='latest.pth',
        help='checkpoint filename written into target_dir.'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='overwrite existing config/checkpoint.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_dir = os.path.abspath(args.target_dir)
    os.makedirs(target_dir, exist_ok=True)

    config_path = os.path.join(target_dir, 'config.yaml')
    checkpoint_path = os.path.join(target_dir, args.checkpoint_name)

    if (os.path.exists(config_path) or os.path.exists(checkpoint_path)) \
            and not args.force:
        print('bootstrap model_dir already exists at %s' % target_dir)
        print('use --force to overwrite it')
        return

    hypes = load_yaml(args.base_yaml)
    hypes['root_dir'] = args.root_dir or args.validate_dir
    hypes['validate_dir'] = args.validate_dir

    model = train_utils.create_model(hypes)
    torch.save(model.state_dict(), checkpoint_path)

    with open(config_path, 'w', encoding='utf-8') as output_file:
        yaml.safe_dump(to_builtin(hypes),
                       output_file,
                       sort_keys=False,
                       default_flow_style=False)

    print('wrote config to %s' % config_path)
    print('wrote bootstrap checkpoint to %s' % checkpoint_path)


def to_builtin(value):
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if __name__ == '__main__':
    main()
