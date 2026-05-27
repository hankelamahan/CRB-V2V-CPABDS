# -*- coding: utf-8 -*-
"""Structured logging helpers for trust-aware late fusion."""

import csv
import json
import os

import numpy as np
import torch


def _json_ready(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


class TrustLogger:
    """Write reputation, physical evidence and summary logs."""

    def __init__(self, enabled=False, log_dir=''):
        self.enabled = bool(enabled and log_dir)
        self.log_dir = log_dir
        self._summary_header_written = False

    def _path(self, filename):
        os.makedirs(self.log_dir, exist_ok=True)
        return os.path.join(self.log_dir, filename)

    def write_jsonl(self, filename, payload):
        if not self.enabled:
            return
        with open(self._path(filename), 'a') as f:
            f.write(json.dumps(_json_ready(payload), sort_keys=True) + '\n')

    def log_reputation(self, payload):
        self.write_jsonl('reputation.jsonl', payload)

    def log_physical(self, records):
        for record in records:
            self.write_jsonl('physical.jsonl', record)

    def log_frame_summary(self, summary):
        if not self.enabled:
            return
        path = self._path('frame_summary.csv')
        fieldnames = [
            'frame',
            'scenario_index',
            'timestamp',
            'num_cavs',
            'num_boxes_before',
            'num_boxes_after',
            'num_filtered_cavs',
            'mode',
        ]
        file_exists = os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists and not self._summary_header_written:
                writer.writeheader()
                self._summary_header_written = True
            writer.writerow({
                key: _json_ready(summary.get(key, ''))
                for key in fieldnames
            })
