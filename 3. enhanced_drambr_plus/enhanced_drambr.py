# -*- coding: utf-8 -*-
"""
Enhanced DRAMBR+ — 超越原版 DRAMBR 的两阶段防御架构

架构：
  车辆端 LMDM（轻量本地检测）→ 即时降权 / 离线缓存
  RSU 端 三重确认（DBSCAN + IsolationForest + GMM）+ 印象分加权
  预测性信誉（LSTM）联动

解决原版痛点：
  1. 无 RSU 覆盖：车辆端 LMDM 立即响应 + 离线队列
  2. RSU 瓶颈：批量聚类 + 印象分预筛，减少无效上报
  3. 首次作恶：印象分 + 直接信任骤降 + LSTM 偏差预警
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from reputation_engine import (
    AdaptiveReputationManager,
    ReputationConfig,
    HAS_SKLEARN,
)

if HAS_SKLEARN:
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest
    from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------
# 车辆端 LMDM
# ---------------------------------------------------------------------------

@dataclass
class LMDMReport:
    """车辆端本地误行为检测报告"""
    vehicle_id: str
    timestamp: float
    position_error: float
    velocity_error: float
    consistency_local: float
    is_suspicious: bool
    features: np.ndarray = field(repr=False)


class LocalMisbehaviorDetectionModule:
    """
    LMDM：车辆端轻量检测，无需 RSU 即可在 1-2 个周期内降权。
    特征：[position_error, velocity_error, timestamp_error, msg_freq_error]
    """

    def __init__(self, local_threshold: float = 0.15):
        self.local_threshold = local_threshold
        self._pending_reports: List[LMDMReport] = []

    def detect(self, vehicle_id: str, observation: Dict) -> LMDMReport:
        pos_err = min(observation.get("position_error", 0.0) / 0.5, 1.0)
        vel_err = min(observation.get("velocity_error", 0.0) / 0.4, 1.0)
        ts_err = min(observation.get("timestamp_error", 0.0) / 0.3, 1.0)
        freq = observation.get("message_frequency", 10.0)
        freq_err = min(abs(freq - 10.0) / 3.0, 1.0)

        features = np.array([pos_err, vel_err, ts_err, freq_err], dtype=np.float32)
        anomaly_score = float(np.mean(features))
        consistency_local = 1.0 - anomaly_score
        is_suspicious = anomaly_score > self.local_threshold

        report = LMDMReport(
            vehicle_id=vehicle_id,
            timestamp=time.time(),
            position_error=pos_err,
            velocity_error=vel_err,
            consistency_local=consistency_local,
            is_suspicious=is_suspicious,
            features=features,
        )
        self._pending_reports.append(report)
        return report

    def pop_pending(self) -> List[LMDMReport]:
        reports = self._pending_reports[:]
        self._pending_reports.clear()
        return reports


# ---------------------------------------------------------------------------
# 离线信誉缓存（无 RSU 区域）
# ---------------------------------------------------------------------------

class OfflineReputationBuffer:
    """无 RSU 时缓存 LMDM 报告，回到覆盖区后批量同步"""

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)

    def enqueue(self, report: LMDMReport):
        self._buffer.append(report)

    def flush(self) -> List[LMDMReport]:
        items = list(self._buffer)
        self._buffer.clear()
        return items

    def size(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# RSU 端三重确认聚类
# ---------------------------------------------------------------------------

class RSUClusterAnalyzer:
    """
    RSU 批量分析：DBSCAN 聚类 → IsolationForest 异常 → GMM 分类
    印象分作为 DBSCAN 样本权重（MISO-V 双重确认）
    """

    def __init__(
        self,
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 2,
        contamination: float = 0.2,
        n_gmm_components: int = 2,
    ):
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.contamination = contamination
        self.n_gmm_components = n_gmm_components

    def analyze_batch(
        self,
        feature_matrix: np.ndarray,
        vehicle_ids: List[str],
        impression_scores: Dict[str, float],
    ) -> Dict[str, Dict]:
        """
        对一批车辆特征做三重确认。
        返回 {vehicle_id: {cluster_label, is_anomaly, gmm_prob, final_risk}}
        """
        n = len(vehicle_ids)
        if n == 0:
            return {}

        X = np.asarray(feature_matrix, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        results = {vid: {} for vid in vehicle_ids}

        # --- DBSCAN（印象分加权：低印象分样本距离放大）---
        sample_weight = np.array([
            2.0 - impression_scores.get(vid, 0.5) for vid in vehicle_ids
        ])
        X_weighted = X * sample_weight[:, np.newaxis]

        if HAS_SKLEARN and n >= self.dbscan_min_samples:
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            labels = db.fit_predict(X_weighted)
        else:
            labels = np.zeros(n, dtype=int)

        for i, vid in enumerate(vehicle_ids):
            results[vid]["cluster_label"] = int(labels[i])
            results[vid]["is_noise"] = labels[i] == -1

        # --- Isolation Forest ---
        if HAS_SKLEARN and n >= 5:
            iso = IsolationForest(contamination=self.contamination, random_state=42)
            iso_pred = iso.fit_predict(X)
            for i, vid in enumerate(vehicle_ids):
                results[vid]["iso_anomaly"] = iso_pred[i] == -1
        else:
            for vid in vehicle_ids:
                results[vid]["iso_anomaly"] = False

        # --- GMM 概率 ---
        if HAS_SKLEARN and n >= self.n_gmm_components * 2:
            gmm = GaussianMixture(
                n_components=self.n_gmm_components, random_state=42, max_iter=50
            )
            gmm.fit(X)
            probs = gmm.predict_proba(X)
            # 取最异常分量（均值 L2 范数最大）
            comp_means = [np.linalg.norm(gmm.means_[k]) for k in range(self.n_gmm_components)]
            risky_comp = int(np.argmax(comp_means))
            for i, vid in enumerate(vehicle_ids):
                results[vid]["gmm_risk_prob"] = float(probs[i, risky_comp])
        else:
            for vid in vehicle_ids:
                results[vid]["gmm_risk_prob"] = 0.5

        # --- 综合风险分 ---
        for i, vid in enumerate(vehicle_ids):
            r = results[vid]
            risk = 0.0
            if r.get("is_noise"):
                risk += 0.35
            if r.get("iso_anomaly"):
                risk += 0.35
            risk += 0.3 * r.get("gmm_risk_prob", 0.5)
            imp = impression_scores.get(vid, 0.5)
            risk += 0.2 * (1.0 - imp)
            r["final_risk"] = float(np.clip(risk, 0.0, 1.0))
            r["is_malicious"] = r["final_risk"] > 0.45

        return results


# ---------------------------------------------------------------------------
# Enhanced DRAMBR+ 主系统
# ---------------------------------------------------------------------------

class EnhancedDRAMBR:
    """
    整合 LMDM + RSU 聚类 + 自适应信誉 + 离线缓冲。
    目标：首次作恶 1-2 个报告周期内识别并降权。
    兼容 BaselineAlgorithm 接口（name / reputations / initialize_reputations）。
    """

    name = "EnhancedDRAMBR"

    def __init__(self, config: Optional[ReputationConfig] = None):
        self.config = config or ReputationConfig()
        self.reputation_manager = AdaptiveReputationManager(self.config)
        self.lmdm = LocalMisbehaviorDetectionModule()
        self.rsu_analyzer = RSUClusterAnalyzer()
        self.offline_buffer = OfflineReputationBuffer()
        self._rsu_coverage: Dict[str, bool] = {}  # vehicle_id -> in_coverage
        self._interaction_count = defaultdict(int)
        self.reputations: Dict[str, float] = {}

    def set_rsu_coverage(self, vehicle_id: str, in_coverage: bool):
        self._rsu_coverage[vehicle_id] = in_coverage

    def initialize_reputations(self, vehicle_ids: List[str], initial_value: float = 0.5):
        for vid in vehicle_ids:
            self.reputation_manager.set_trust_score(vid, initial_value)
            self.reputation_manager.set_impression_score(vid, initial_value)
            self.reputations[vid] = initial_value

    def get_reputation(self, vehicle_id: str) -> float:
        return self.reputation_manager.get_trust_score(vehicle_id)

    def get_all_reputations(self) -> Dict[str, float]:
        return self.reputation_manager.get_all_reputations()

    def _sync_reputations_dict(self):
        self.reputations = self.reputation_manager.get_all_reputations()

    def process_vehicle_observation(
        self,
        vehicle_id: str,
        observation: Dict,
        neighbor_reports: Optional[List[float]] = None,
    ) -> Dict:
        """
        单车辆观测处理流程（车辆端 + 可选 RSU）。
        """
        self._interaction_count[vehicle_id] += 1
        in_coverage = self._rsu_coverage.get(vehicle_id, True)

        # Step 1: LMDM 本地检测（始终执行，低计算开销）
        report = self.lmdm.detect(vehicle_id, observation)
        direct_trust = report.consistency_local

        # Step 2: 车辆端即时响应（无 RSU 也执行）
        if report.is_suspicious:
            self.reputation_manager.update_from_evidence(
                vehicle_id,
                is_consistent=False,
                consistency_ratio=report.consistency_local,
                direct_trust=direct_trust,
                indirect_reports=neighbor_reports,
            )

        if not in_coverage:
            self.offline_buffer.enqueue(report)
            self._sync_reputations_dict()
            return {
                "vehicle_id": vehicle_id,
                "mode": "offline_lmdm",
                "local_suspicious": report.is_suspicious,
                "reputation": self.get_reputation(vehicle_id),
            }

        # Step 3: RSU 在线 — 正常证据更新
        is_consistent = not report.is_suspicious
        update_info = self.reputation_manager.update_from_evidence(
            vehicle_id,
            is_consistent=is_consistent,
            consistency_ratio=report.consistency_local,
            direct_trust=direct_trust,
            indirect_reports=neighbor_reports,
        )
        self._sync_reputations_dict()
        return {"mode": "online", **update_info}

    def process_rsu_batch(
        self,
        reports: List[LMDMReport],
        impression_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict]:
        """
        RSU 批量三重确认（覆盖区恢复或周期性批处理）。
        """
        if not reports:
            return {}

        vehicle_ids = [r.vehicle_id for r in reports]
        X = np.vstack([r.features for r in reports])

        if impression_scores is None:
            impression_scores = {
                vid: self.reputation_manager._get_meta(vid).impression_score
                for vid in vehicle_ids
            }

        cluster_results = self.rsu_analyzer.analyze_batch(X, vehicle_ids, impression_scores)

        for vid, result in cluster_results.items():
            if result.get("is_malicious"):
                self.reputation_manager.update_from_evidence(
                    vid,
                    is_consistent=False,
                    consistency_ratio=0.2,
                    direct_trust=0.2,
                )
                # 更新印象分（双重确认）
                self.reputation_manager.set_impression_score(vid, 0.15)
            else:
                # 正常车辆印象分缓慢上升
                meta = self.reputation_manager._get_meta(vid)
                new_imp = min(1.0, meta.impression_score + 0.02)
                self.reputation_manager.set_impression_score(vid, new_imp)

        self._sync_reputations_dict()
        return cluster_results

    def sync_offline_buffer(self) -> Dict[str, Dict]:
        """车辆回到 RSU 覆盖区后，冲刷离线缓冲并批处理"""
        reports = self.offline_buffer.flush()
        if reports:
            return self.process_rsu_batch(reports)
        return {}

    def update_reputation(self, vehicle_id: str, observation: Dict):
        """兼容 BaselineAlgorithm 接口"""
        self.process_vehicle_observation(vehicle_id, observation)

    def get_statistics(self) -> Dict:
        reps = list(self.reputations.values()) if self.reputations else [0.5]
        warned = sum(
            1 for vid in self.reputations
            if self.reputation_manager._get_meta(vid).warning_level > 0
        )
        return {
            "total_interactions": sum(self._interaction_count.values()),
            "avg_reputation": float(np.mean(reps)),
            "std_reputation": float(np.std(reps)),
            "warned_vehicles": warned,
            "offline_buffer_size": self.offline_buffer.size(),
            "has_sklearn": HAS_SKLEARN,
        }

    def get_fusion_weights(self, vehicle_ids: List[str]) -> List[float]:
        """供 WBF 信誉加权融合使用"""
        threshold = self.config.fusion_filter_threshold
        weights = []
        for vid in vehicle_ids:
            w = self.reputation_manager.get_fusion_weight(vid)
            weights.append(w if w >= threshold else 0.0)
        return weights
