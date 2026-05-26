#!/usr/bin/env python3

import argparse
import os
import shutil
import tempfile
import zipfile

import numpy as np
import yaml

from opencood.hypes_yaml.yaml_utils import load_yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare a runtime model_dir from a downloaded zip model.'
    )
    parser.add_argument(
        '--zip_path',
        type=str,
        default='opencood/model_weight/pointpillar_attentive_fusion.zip',
        help='downloaded zip model path.'
    )
    parser.add_argument(
        '--variant',
        type=str,
        default='pointpillar_attentive_fusion',
        help='model directory name inside the zip.'
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        default='opencood/model_weight/pointpillar_attentive_fusion_runtime_v2xset',
        help='prepared runtime model_dir.'
    )
    parser.add_argument(
        '--validate_dir',
        type=str,
        default='v2xset/validate',
        help='validation dataset path for local inference.'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default='',
        help='optional root_dir override. Defaults to validate_dir.'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='overwrite target_dir if it already exists.'
    )
    return parser.parse_args()


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


def find_variant_dir(extract_root, variant):
    candidates = []
    for dirpath, _, filenames in os.walk(extract_root):
        if 'config.yaml' not in filenames:
            continue
        checkpoint = resolve_checkpoint(dirpath)
        if checkpoint is None:
            continue
        candidates.append(dirpath)

    if not candidates:
        raise FileNotFoundError('No model_dir with config.yaml and checkpoint found in zip.')

    for candidate in candidates:
        if os.path.basename(candidate) == variant:
            return candidate

    candidate_names = ', '.join(sorted(os.path.basename(path) for path in candidates))
    raise FileNotFoundError('Variant %s not found. Available: %s' %
                            (variant, candidate_names))


def resolve_checkpoint(model_dir):
    latest_path = os.path.join(model_dir, 'latest.pth')
    if os.path.exists(latest_path):
        return latest_path

    epoch_files = []
    for filename in os.listdir(model_dir):
        if filename.startswith('net_epoch') and filename.endswith('.pth'):
            epoch_files.append(os.path.join(model_dir, filename))
    if not epoch_files:
        return None

    epoch_files.sort()
    return epoch_files[-1]


def main():
    args = parse_args()
    zip_path = os.path.abspath(args.zip_path)
    target_dir = os.path.abspath(args.target_dir)

    if not os.path.exists(zip_path):
        raise FileNotFoundError('%s not found' % zip_path)

    if os.path.exists(target_dir):
        if not args.force:
            print('runtime model_dir already exists at %s' % target_dir)
            print('use --force to overwrite it')
            return
        shutil.rmtree(target_dir)

    os.makedirs(target_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='pointpillar_model_') as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(temp_dir)

        source_dir = find_variant_dir(temp_dir, args.variant)
        checkpoint_path = resolve_checkpoint(source_dir)
        config_path = os.path.join(source_dir, 'config.yaml')

        hypes = load_yaml(config_path)
        hypes['validate_dir'] = args.validate_dir
        hypes['root_dir'] = args.root_dir or args.validate_dir

        shutil.copy2(checkpoint_path, os.path.join(target_dir, os.path.basename(checkpoint_path)))
        with open(os.path.join(target_dir, 'config.yaml'), 'w', encoding='utf-8') as output_file:
            yaml.safe_dump(to_builtin(hypes),
                           output_file,
                           sort_keys=False,
                           default_flow_style=False)

    print('prepared runtime model_dir at %s' % target_dir)
    print('source zip: %s' % zip_path)
    print('variant: %s' % args.variant)
    print('validate_dir: %s' % args.validate_dir)


if __name__ == '__main__':
    main()
