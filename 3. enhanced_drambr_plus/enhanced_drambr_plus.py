# -*- coding: utf-8 -*-
"""
Enhanced DRAMBR+ — 超越原版 DRAMBR 的全面防御系统

核心升级：
1. 多维信任向量系统（直接/间接/全局信任融合）
2. LSTM 预测性信誉（提前识别首次作恶）
3. 动态风险置信度分配（反风险机制）
4. 紧急刹车欺诈专项检测
5. 车辆端轻量级预筛选
6. RSU 端深度三重确认
"""

from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from advanced_trust_system import MultiDimensionalTrustManager
from predictive_reputation import PredictiveReputationEngine, ReputationTrendAnalyzer
from emergency_brake_fraud_detector import EmergencyBrakeFraudDetector, BrakeEvent
from enhanced_drambr import (
    LocalMisbehaviorDetectionModule,
    OfflineReputationBuffer,
    RSUClusterAnalyzer,
    LMDMReport
)


@dataclass
class EnhancedSecurityEvent:
    """增强安全事件"""
    vehicle_id: str
    timestamp: float
    event_type: str
    severity: str
    trust_vector: Dict
    prediction_deviation: float
    fraud_indicators: Dict
    recommended_action: str
    details: Dict = field(default_factory=dict)


class VehicleSidePreFilter:
    """车辆端轻量级预筛选器"""
    
    def __init__(self, quick_threshold: float = 0.3):
        self.quick_threshold = quick_threshold
        self._local_cache: Dict[str, float] = {}
    
    def quick_check(self, vehicle_id: str, observation: Dict) -> Tuple[bool, float]:
        """快速本地检查（毫秒级）"""
        pos_err = observation.get("position_error", 0.0)
        vel_err = observation.get("velocity_error", 0.0)
        quick_risk = (pos_err + vel_err) / 2.0
        cached_risk = self._local_cache.get(vehicle_id, 0.0)
        risk_increase = quick_risk - cached_risk
        self._local_cache[vehicle_id] = quick_risk
        should_report = quick_risk > self.quick_threshold or risk_increase > 0.2
        return should_report, quick_risk


class EnhancedDRAMBRPlus:
    """Enhanced DRAMBR+ 主系统"""
    
    name = "EnhancedDRAMBR+"
    
    def __init__(
        self,
        enable_prediction: bool = True,
        enable_brake_fraud_detection: bool = True,
        enable_vehicle_prefilter: bool = True
    ):
        self.enable_prediction = enable_prediction
        self.enable_brake_fraud_detection = enable_brake_fraud_detection
        self.enable_vehicle_prefilter = enable_vehicle_prefilter
        
        self.trust_manager = MultiDimensionalTrustManager(
            w_direct=0.5, w_indirect=0.35, w_global=0.15, enable_dynamic_weights=True
        )
        self.lmdm = LocalMisbehaviorDetectionModule(local_threshold=0.15)
        self.rsu_analyzer = RSUClusterAnalyzer()
        self.offline_buffer = OfflineReputationBuffer(max_size=500)
        
        if self.enable_prediction:
            self.predictor = PredictiveReputationEngine(
                sequence_length=20, deviation_threshold=0.15, confidence_threshold=0.7
            )
            self.trend_analyzer = ReputationTrendAnalyzer(window_size=10)
        
        if self.enable_brake_fraud_detection:
            self.brake_detector = EmergencyBrakeFraudDetector(
                max_deceleration=9.8, min_emergency_deceleration=4.0
            )
        
        if self.enable_vehicle_prefilter:
            self.prefilter = VehicleSidePreFilter(quick_threshold=0.3)
        
        self._rsu_coverage: Dict[str, bool] = {}
        self._security_events: List[EnhancedSecurityEvent] = []
        self._interaction_count = defaultdict(int)
        self.reputations: Dict[str, float] = {}
    
    def initialize_reputations(self, vehicle_ids: List[str], initial_value: float = 0.5):
        for vid in vehicle_ids:
            profile = self.trust_manager._get_profile(vid)
            profile.trust_vector.direct_trust = initial_value
            profile.trust_vector.indirect_trust = initial_value
            profile.trust_vector.global_trust = initial_value
            self.reputations[vid] = initial_value
    
    def set_rsu_coverage(self, vehicle_id: str, in_coverage: bool):
        self._rsu_coverage[vehicle_id] = in_coverage
    
    def process_vehicle_observation(
        self,
        vehicle_id: str,
        observation: Dict,
        neighbor_reports: Optional[List[float]] = None,
        neighbor_observations: Optional[List[Dict]] = None
    ) -> Dict:
        """处理车辆观测（主入口）"""
        self._interaction_count[vehicle_id] += 1
        in_coverage = self._rsu_coverage.get(vehicle_id, True)
        
        if self.enable_vehicle_prefilter:
            should_report, quick_risk = self.prefilter.quick_check(vehicle_id, observation)
            if not should_report and in_coverage:
                return {
                    'vehicle_id': vehicle_id, 'mode': 'prefilter_passed',
                    'quick_risk': quick_risk, 'reputation': self.get_reputation(vehicle_id)
                }
        
        lmdm_report = self.lmdm.detect(vehicle_id, observation)
        
        trust_update = self.trust_manager.update_trust(
            vehicle_id,
            consistency_ratio=lmdm_report.consistency_local,
            neighbor_reports=neighbor_reports,
            is_consistent=not lmdm_report.is_suspicious
        )
        
        current_trust = trust_update['final_score']
        
        prediction_result = None
        early_warning_score = 0.0
        
        if self.enable_prediction:
            prediction_result = self.predictor.check_deviation(vehicle_id, current_trust)
            early_warning_score = self.predictor.get_early_warning_score(vehicle_id)
            
            if prediction_result and prediction_result.is_anomalous:
                profile = self.trust_manager._get_profile(vehicle_id)
                profile.trust_vector.direct_trust *= 0.7
                profile.adjust_thresholds()
        
        brake_fraud_result = None
        if self.enable_brake_fraud_detection and 'brake_event' in observation:
            brake_fraud_result = self._check_brake_fraud(
                vehicle_id, observation['brake_event'], neighbor_observations
            )
        
        if not in_coverage:
            self.offline_buffer.enqueue(lmdm_report)
            mode = 'offline_cached'
        else:
            mode = 'online_processed'
        
        if self._should_generate_security_event(
            trust_update, early_warning_score, prediction_result, brake_fraud_result
        ):
            self._generate_security_event(
                vehicle_id, trust_update, prediction_result, brake_fraud_result, early_warning_score
            )
        
        self.reputations[vehicle_id] = current_trust
        
        return {
            'vehicle_id': vehicle_id, 'mode': mode, 'trust_update': trust_update,
            'prediction_result': prediction_result, 'brake_fraud_result': brake_fraud_result,
            'early_warning_score': early_warning_score, 'reputation': current_trust,
            'in_coverage': in_coverage
        }
    
    def _check_brake_fraud(
        self, vehicle_id: str, brake_event_data: Dict, neighbor_observations: Optional[List[Dict]]
    ) -> Optional[Dict]:
        brake_event = BrakeEvent(
            vehicle_id=vehicle_id,
            timestamp=brake_event_data.get('timestamp', time.time()),
            position=np.array(brake_event_data.get('position', [0, 0, 0])),
            velocity=brake_event_data.get('velocity', 0.0),
            acceleration=brake_event_data.get('acceleration', 0.0),
            brake_intensity=brake_event_data.get('brake_intensity', 0.0),
            is_emergency=brake_event_data.get('is_emergency', False),
            reported_reason=brake_event_data.get('reason', '')
        )
        
        fraud_result = self.brake_detector.detect_fraud(brake_event, neighbor_observations)
        
        if fraud_result['is_fraud']:
            profile = self.trust_manager._get_profile(vehicle_id)
            profile.trust_vector.direct_trust *= 0.5
            profile.malicious_count += 1
        
        return fraud_result
    
    def _should_generate_security_event(
        self,
        trust_update: Dict,
        early_warning_score: float,
        prediction_result,
        brake_fraud_result: Optional[Dict],
    ) -> bool:
        """判断是否需要生成安全事件"""
        risk_level = trust_update['risk_level']
        if risk_level in ('HIGH', 'CRITICAL'):
            return True
        if early_warning_score >= 0.2:
            return True
        if prediction_result and prediction_result.is_anomalous:
            return True
        if brake_fraud_result and brake_fraud_result.get('is_fraud'):
            return True
        if risk_level == 'MEDIUM' and early_warning_score >= 0.1:
            return True
        return False
    
    def _resolve_event_severity(
        self,
        trust_update: Dict,
        early_warning_score: float,
        prediction_result,
        brake_fraud_result: Optional[Dict],
    ) -> str:
        """综合各信号确定事件严重等级"""
        severity = trust_update['risk_level']
        
        if brake_fraud_result and brake_fraud_result.get('is_fraud'):
            fraud_score = brake_fraud_result.get('fraud_score', 0.0)
            if fraud_score >= 0.55:
                severity = 'HIGH'
            elif severity == 'LOW':
                severity = 'MEDIUM'
        
        if prediction_result and prediction_result.is_anomalous and severity == 'LOW':
            severity = 'MEDIUM'
        
        if early_warning_score >= 0.8:
            if severity in ('LOW', 'MEDIUM'):
                severity = 'HIGH'
            elif severity == 'HIGH':
                severity = 'CRITICAL'
        elif early_warning_score >= 0.2 and severity == 'LOW':
            severity = 'MEDIUM'
        
        return severity
    
    def _generate_security_event(
        self, vehicle_id: str, trust_update: Dict, prediction_result,
        brake_fraud_result: Optional[Dict], early_warning_score: float
    ) -> None:
        severity = self._resolve_event_severity(
            trust_update, early_warning_score, prediction_result, brake_fraud_result
        )
        
        fraud_score = (
            brake_fraud_result.get('fraud_score', 0.0) if brake_fraud_result else 0.0
        )
        if severity == 'CRITICAL' or early_warning_score >= 0.8 or fraud_score >= 0.6:
            recommended_action = 'ISOLATE'
        elif severity in ('HIGH',) or early_warning_score >= 0.2 or (
            brake_fraud_result and brake_fraud_result.get('is_fraud')
        ):
            recommended_action = 'REDUCE_WEIGHT'
        else:
            recommended_action = 'MONITOR'
        
        event_type = 'TRUST_ANOMALY'
        if prediction_result and prediction_result.is_anomalous:
            event_type = 'PREDICTION_ANOMALY'
        if brake_fraud_result and brake_fraud_result['is_fraud']:
            event_type = 'BRAKE_FRAUD'
        
        event = EnhancedSecurityEvent(
            vehicle_id=vehicle_id, timestamp=time.time(), event_type=event_type,
            severity=severity, trust_vector=trust_update['trust_vector'],
            prediction_deviation=prediction_result.deviation if prediction_result else 0.0,
            fraud_indicators={'brake_fraud': brake_fraud_result or {}, 'early_warning': early_warning_score},
            recommended_action=recommended_action, details=trust_update
        )
        
        self._security_events.append(event)
    
    def process_rsu_batch(self, reports: Optional[List[LMDMReport]] = None, flush_offline: bool = False) -> Dict[str, Dict]:
        if flush_offline:
            reports = self.offline_buffer.flush()
        
        if not reports:
            return {}
        
        vehicle_ids = [r.vehicle_id for r in reports]
        X = np.vstack([r.features for r in reports])
        
        impression_scores = {
            vid: self.trust_manager._get_profile(vid).trust_vector.global_trust
            for vid in vehicle_ids
        }
        
        cluster_results = self.rsu_analyzer.analyze_batch(X, vehicle_ids, impression_scores)
        
        for vid, result in cluster_results.items():
            profile = self.trust_manager._get_profile(vid)
            if result.get('is_malicious'):
                profile.trust_vector.direct_trust *= 0.6
                profile.trust_vector.global_trust *= 0.8
                profile.malicious_count += 1
            else:
                profile.trust_vector.global_trust = min(1.0, profile.trust_vector.global_trust + 0.02)
        
        self._sync_reputations()
        return cluster_results
    
    def get_reputation(self, vehicle_id: str) -> float:
        return self.trust_manager.get_trust_score(vehicle_id)
    
    def get_all_reputations(self) -> Dict[str, float]:
        return {vid: self.trust_manager.get_trust_score(vid) for vid in self.trust_manager._profiles.keys()}
    
    def _sync_reputations(self):
        self.reputations = self.get_all_reputations()
    
    def get_fusion_weights(self, vehicle_ids: List[str], threshold: float = 0.3) -> List[float]:
        return [self.get_reputation(vid) if self.get_reputation(vid) >= threshold else 0.0 for vid in vehicle_ids]
    
    def get_security_events(self, severity_filter: Optional[List[str]] = None, limit: int = 50) -> List[EnhancedSecurityEvent]:
        events = self._security_events[-limit:]
        if severity_filter:
            events = [e for e in events if e.severity in severity_filter]
        return events
    
    def get_statistics(self) -> Dict:
        reps = list(self.reputations.values()) if self.reputations else [0.5]
        
        stats = {
            'system': 'EnhancedDRAMBR+',
            'total_interactions': sum(self._interaction_count.values()),
            'tracked_vehicles': len(self.trust_manager._profiles),
            'avg_reputation': float(np.mean(reps)),
            'std_reputation': float(np.std(reps)),
            'security_events': len(self._security_events),
            'offline_buffer_size': self.offline_buffer.size(),
            'modules': {
                'prediction': self.enable_prediction,
                'brake_fraud': self.enable_brake_fraud_detection,
                'prefilter': self.enable_vehicle_prefilter
            }
        }
        
        if self.enable_prediction:
            stats['prediction'] = self.predictor.get_statistics()
        if self.enable_brake_fraud_detection:
            stats['brake_fraud'] = self.brake_detector.get_statistics()
        
        return stats
    
    def update_reputation(self, vehicle_id: str, observation: Dict):
        self.process_vehicle_observation(vehicle_id, observation)
