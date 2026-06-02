# -*- coding: utf-8 -*-
"""
增强信誉引擎 — 超越中心化信誉与固定步长更新

核心能力：
1. 自适应信誉更新（非对称步长 + 可配置阈值 + 方差/更新次数元数据）
2. 多证据信任向量融合（直接/间接/全局信任，参考 DCACA）
3. 反风险置信度分配（可疑/异常双阈值分级惩罚）
4. LSTM 预测性信誉（首次作恶提前预警）
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.mixture import GaussianMixture
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class TrustVector:
    """三维信任向量（DCACA 风格）"""
    direct: float = 0.5      # 实时检测结果
    indirect: float = 0.5    # 其他车辆报告
    global_trust: float = 0.5  # 长期信誉

    def fused_score(self, weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> float:
        w_d, w_i, w_g = weights
        return float(np.clip(w_d * self.direct + w_i * self.indirect + w_g * self.global_trust, 0.0, 1.0))

    def to_dict(self) -> Dict[str, float]:
        return {"direct": self.direct, "indirect": self.indirect, "global": self.global_trust}


@dataclass
class VehicleReputationMeta:
    """信誉元数据：支持自适应步长"""
    score: float = 0.5
    variance: float = 0.0
    update_count: int = 0
    consistency_history: deque = field(default_factory=lambda: deque(maxlen=20))
    trust_vector: TrustVector = field(default_factory=TrustVector)
    impression_score: float = 0.5   # MISO-V 风格信用预评级
    warning_level: int = 0        # 0=正常, 1=可疑, 2=异常


@dataclass
class ReputationConfig:
    """可配置信誉参数"""
    default_reputation: float = 0.5
    positive_step: float = 0.05       # 一致时 +0.05
    negative_step: float = 0.1        # 不一致时 -0.1
    min_reputation: float = 0.0       # 允许声誉值降到0
    suspicious_threshold: float = 0.45  # 反风险：可疑阈值
    anomaly_threshold: float = 0.3      # 反风险：异常阈值
    fusion_filter_threshold: float = 0.3
    adaptive_step: bool = True
    trust_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
    lstm_window: int = 10
    lstm_deviation_threshold: float = 0.15


# ---------------------------------------------------------------------------
# LSTM 预测性信誉
# ---------------------------------------------------------------------------

class _ReputationLSTM(nn.Module if HAS_TORCH else object):
    """轻量 LSTM：用行为序列预测下一窗口信誉"""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16):
        if not HAS_TORCH:
            return
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


class PredictiveReputationModel:
    """
    预测性信誉：偏差超阈值则提前预警。
    有 PyTorch 时用 LSTM；否则用线性趋势外推。
    """

    def __init__(self, window: int = 10, deviation_threshold: float = 0.15):
        self.window = window
        self.deviation_threshold = deviation_threshold
        self._sequences: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._lstm = _ReputationLSTM() if HAS_TORCH else None
        if HAS_TORCH and self._lstm is not None:
            self._lstm.eval()

    def record_observation(self, vehicle_id: str, features: np.ndarray):
        """features: [consistency_ratio, direct_trust, indirect_trust, anomaly_score]"""
        self._sequences[vehicle_id].append(features.astype(np.float32))

    def predict_next(self, vehicle_id: str) -> Optional[float]:
        seq = self._sequences.get(vehicle_id)
        if not seq or len(seq) < 3:
            return None

        arr = np.array(list(seq))
        if HAS_TORCH and self._lstm is not None:
            with torch.no_grad():
                x = torch.tensor(arr[np.newaxis, ...], dtype=torch.float32)
                pred = self._lstm(x).item()
            return float(pred)

        # 无 PyTorch：线性外推最近信誉分量
        scores = arr[:, 0]
        t = np.arange(len(scores))
        slope = np.polyfit(t, scores, 1)[0] if len(scores) >= 2 else 0.0
        return float(np.clip(scores[-1] + slope, 0.0, 1.0))

    def check_deviation(self, vehicle_id: str, actual: float) -> Tuple[bool, float]:
        """返回 (是否预警, 偏差量)"""
        pred = self.predict_next(vehicle_id)
        if pred is None:
            return False, 0.0
        deviation = abs(actual - pred)
        return deviation > self.deviation_threshold, deviation


# ---------------------------------------------------------------------------
# 反风险置信度分配（DCACA 双阈值）
# ---------------------------------------------------------------------------

class AntiRiskConfidenceAllocator:
    """
    根据一致性比率变化，在信誉跌破硬阈值前分级降权。
    级别：0=正常, 1=可疑(降权50%), 2=异常(降权至min)
    """

    def __init__(self, suspicious_threshold: float = 0.45, anomaly_threshold: float = 0.3):
        self.suspicious_threshold = suspicious_threshold
        self.anomaly_threshold = anomaly_threshold

    def allocate(self, consistency_ratio: float, current_rep: float) -> Tuple[float, int]:
        """
        返回 (调整后置信度权重, 警告级别)
        """
        if consistency_ratio < self.anomaly_threshold:
            return max(0.0, current_rep * 0.3), 2
        if consistency_ratio < self.suspicious_threshold:
            return current_rep * 0.5, 1
        return current_rep, 0


# ---------------------------------------------------------------------------
# 自适应信誉管理器
# ---------------------------------------------------------------------------

class AdaptiveReputationManager:
    """
    扩展 ReputationManager：
    - 非对称步长 (+0.05 / -0.1)
    - 动态步长（方差大时减小步长，更新次数多时加速收敛）
    - 信任向量融合
    - 反风险 + LSTM 预测联动
    """

    def __init__(self, config: Optional[ReputationConfig] = None):
        self.config = config or ReputationConfig()
        self._vehicles: Dict[str, VehicleReputationMeta] = {}
        self._allocator = AntiRiskConfidenceAllocator(
            self.config.suspicious_threshold,
            self.config.anomaly_threshold,
        )
        self._predictor = PredictiveReputationModel(
            self.config.lstm_window,
            self.config.lstm_deviation_threshold,
        )

    def _get_meta(self, vehicle_id: str) -> VehicleReputationMeta:
        if vehicle_id not in self._vehicles:
            m = VehicleReputationMeta(score=self.config.default_reputation)
            m.trust_vector.global_trust = self.config.default_reputation
            self._vehicles[vehicle_id] = m
        return self._vehicles[vehicle_id]

    def get_trust_score(self, vehicle_id: str) -> float:
        return self._get_meta(vehicle_id).score

    def get_trust_vector(self, vehicle_id: str) -> TrustVector:
        return self._get_meta(vehicle_id).trust_vector

    def get_fusion_weight(self, vehicle_id: str) -> float:
        """融合用权重：综合信任向量 + 反风险调整"""
        meta = self._get_meta(vehicle_id)
        fused = meta.trust_vector.fused_score(self.config.trust_weights)
        if meta.warning_level == 2:
            return 0.0
        if meta.warning_level == 1:
            return fused * 0.5
        return fused

    def set_trust_score(self, vehicle_id: str, score: float):
        meta = self._get_meta(vehicle_id)
        meta.score = max(self.config.min_reputation, min(1.0, score))
        meta.trust_vector.global_trust = meta.score

    def _compute_dynamic_step(self, meta: VehicleReputationMeta, is_positive: bool) -> float:
        base = self.config.positive_step if is_positive else self.config.negative_step
        if not self.config.adaptive_step:
            return base
        # 方差大 → 环境不稳定 → 减小步长
        var_factor = 1.0 / (1.0 + meta.variance * 5.0)
        # 更新次数多 → 加速收敛
        count_factor = min(1.5, 1.0 + meta.update_count * 0.01)
        if is_positive:
            return base * var_factor * count_factor
        return base * (2.0 - var_factor) * count_factor  # 惩罚侧更激进

    def update_trust_vector(
        self,
        vehicle_id: str,
        direct: float,
        indirect: float,
        global_trust: Optional[float] = None,
    ):
        meta = self._get_meta(vehicle_id)
        meta.trust_vector.direct = float(np.clip(direct, 0.0, 1.0))
        meta.trust_vector.indirect = float(np.clip(indirect, 0.0, 1.0))
        if global_trust is not None:
            meta.trust_vector.global_trust = float(np.clip(global_trust, 0.0, 1.0))

    def update_from_evidence(
        self,
        vehicle_id: str,
        is_consistent: bool,
        consistency_ratio: float = 1.0,
        direct_trust: Optional[float] = None,
        indirect_reports: Optional[List[float]] = None,
    ) -> Dict:
        """
        多证据融合更新入口。
        返回本次更新摘要（供 RSU/日志使用）。
        """
        meta = self._get_meta(vehicle_id)
        current = meta.score

        if direct_trust is not None:
            meta.trust_vector.direct = float(np.clip(direct_trust, 0.0, 1.0))
        if indirect_reports:
            meta.trust_vector.indirect = float(np.clip(np.mean(indirect_reports), 0.0, 1.0))

        # 反风险置信度分配
        adjusted_weight, warning_level = self._allocator.allocate(consistency_ratio, current)
        meta.warning_level = warning_level

        # 记录 LSTM 序列
        anomaly_score = 1.0 - consistency_ratio
        self._predictor.record_observation(
            vehicle_id,
            np.array([consistency_ratio, meta.trust_vector.direct,
                      meta.trust_vector.indirect, anomaly_score]),
        )

        # 预测偏差预警（首次作恶：直接+间接信任骤降）
        fused_before = meta.trust_vector.fused_score(self.config.trust_weights)
        early_warn, deviation = self._predictor.check_deviation(vehicle_id, fused_before)

        step = self._compute_dynamic_step(meta, is_consistent)
        if is_consistent:
            new_score = min(1.0, current + step)
        else:
            new_score = max(self.config.min_reputation, current - step)
            # 不一致时额外参考信任向量（持证作恶：直接信任立即拉低）
            tv_penalty = meta.trust_vector.fused_score(self.config.trust_weights)
            new_score = min(new_score, tv_penalty)

        if early_warn:
            new_score = max(self.config.min_reputation, new_score - step * 0.5)

        meta.score = new_score
        meta.trust_vector.global_trust = new_score
        meta.update_count += 1
        meta.consistency_history.append(1.0 if is_consistent else 0.0)
        if len(meta.consistency_history) > 1:
            meta.variance = float(np.var(list(meta.consistency_history)))

        return {
            "vehicle_id": vehicle_id,
            "old_score": current,
            "new_score": new_score,
            "step_applied": step,
            "warning_level": warning_level,
            "early_warning": early_warn,
            "deviation": deviation,
            "fusion_weight": self.get_fusion_weight(vehicle_id),
            "trust_vector": meta.trust_vector.to_dict(),
        }

    def update_from_voting_consistency(self, vehicle_id: str, is_consistent: bool):
        ratio = 1.0 if is_consistent else 0.0
        return self.update_from_evidence(vehicle_id, is_consistent, consistency_ratio=ratio)

    def batch_update_from_voting(
        self,
        fused_result,
        original_detections: Dict,
        vehicle_ids: List[str],
        iou_thr: float = 0.5,
    ) -> Dict[str, bool]:
        fused_boxes, fused_scores, fused_labels = fused_result
        consistency_dict = {}

        for vehicle_id in vehicle_ids:
            if vehicle_id not in original_detections:
                consistency_dict[vehicle_id] = False
                self.update_from_voting_consistency(vehicle_id, False)
                continue

            detections = original_detections[vehicle_id]
            vehicle_boxes = detections.get("boxes", [])
            vehicle_labels = detections.get("labels", [])

            if len(vehicle_boxes) == 0:
                consistency_dict[vehicle_id] = False
                self.update_from_voting_consistency(vehicle_id, False)
                continue

            consistent_count = 0
            total_matchable = 0
            for i, vbox in enumerate(vehicle_boxes):
                vlabel = vehicle_labels[i] if i < len(vehicle_labels) else None
                for j, fbox in enumerate(fused_boxes):
                    iou = _calculate_iou(vbox, fbox)
                    if iou > iou_thr:
                        total_matchable += 1
                        if vlabel == fused_labels[j]:
                            consistent_count += 1
                        break

            if total_matchable > 0:
                ratio = consistent_count / total_matchable
                is_consistent = ratio > 0.7
            else:
                ratio = 0.0
                is_consistent = False

            consistency_dict[vehicle_id] = is_consistent
            self.update_from_evidence(vehicle_id, is_consistent, consistency_ratio=ratio)

        return consistency_dict

    def get_all_reputations(self) -> Dict[str, float]:
        return {vid: m.score for vid, m in self._vehicles.items()}

    def get_metadata(self, vehicle_id: str) -> Dict:
        meta = self._get_meta(vehicle_id)
        return {
            "score": meta.score,
            "variance": meta.variance,
            "update_count": meta.update_count,
            "impression_score": meta.impression_score,
            "warning_level": meta.warning_level,
            "trust_vector": meta.trust_vector.to_dict(),
        }

    def set_impression_score(self, vehicle_id: str, score: float):
        self._get_meta(vehicle_id).impression_score = float(np.clip(score, 0.0, 1.0))


def _calculate_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0
