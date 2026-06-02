# -*- coding: utf-8 -*-
"""
多维信任向量系统 (Multi-Dimensional Trust Vector System)

融合三个维度的信任评估：
1. 直接信任 (Direct Trust): 实时检测结果
2. 间接信任 (Indirect Trust): 其他车辆报告
3. 全局信任 (Global Trust): 长期信誉历史

参考 DCACA 和 DIVA 的信任融合机制
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TrustVector:
    """三维信任向量"""
    direct_trust: float = 0.5
    indirect_trust: float = 0.5
    global_trust: float = 0.5
    timestamp: float = field(default_factory=time.time)
    
    def get_weighted_score(
        self, 
        w_direct: float = 0.4, 
        w_indirect: float = 0.3, 
        w_global: float = 0.3
    ) -> float:
        """加权融合三维信任"""
        return (
            w_direct * self.direct_trust + 
            w_indirect * self.indirect_trust + 
            w_global * self.global_trust
        )
    
    def to_dict(self) -> Dict:
        return {
            'direct': self.direct_trust,
            'indirect': self.indirect_trust,
            'global': self.global_trust,
            'weighted': self.get_weighted_score()
        }


@dataclass
class VehicleTrustProfile:
    """车辆信任档案"""
    vehicle_id: str
    trust_vector: TrustVector = field(default_factory=TrustVector)
    interaction_count: int = 0
    malicious_count: int = 0
    consistent_count: int = 0
    last_update: float = field(default_factory=time.time)
    
    suspicious_threshold: float = 0.4
    anomaly_threshold: float = 0.25
    
    trust_history: List[float] = field(default_factory=list)
    max_history_len: int = 50
    
    def update_direct_trust(self, consistency_ratio: float, decay: float = 0.65):
        """更新直接信任（基于实时检测）- 优化版V4：更激进的响应"""
        old = self.trust_vector.direct_trust
        if consistency_ratio < 0.1:
            decay = 0.95
        elif consistency_ratio < 0.2:
            decay = 0.85
        elif consistency_ratio < 0.35:
            decay = 0.75
        self.trust_vector.direct_trust = (1 - decay) * old + decay * consistency_ratio
        self.trust_vector.direct_trust = np.clip(self.trust_vector.direct_trust, 0.0, 1.0)
    
    def update_indirect_trust(self, neighbor_reports: List[float], decay: float = 0.65):
        """更新间接信任（基于邻居报告）- 优化版V4：更激进的响应"""
        if not neighbor_reports:
            return
        
        avg_report = np.mean(neighbor_reports)
        if avg_report < 0.1:
            decay = 0.95
        elif avg_report < 0.2:
            decay = 0.85
        elif avg_report < 0.35:
            decay = 0.75
        old = self.trust_vector.indirect_trust
        self.trust_vector.indirect_trust = (1 - decay) * old + decay * avg_report
        self.trust_vector.indirect_trust = np.clip(self.trust_vector.indirect_trust, 0.0, 1.0)
    
    def update_global_trust(self, decay: float = 0.5):
        """更新全局信任（基于长期历史）- 优化版V4：更激进的响应"""
        if self.interaction_count == 0:
            return
        
        consistency_rate = self.consistent_count / self.interaction_count
        old = self.trust_vector.global_trust
        self.trust_vector.global_trust = (1 - decay) * old + decay * consistency_rate
        self.trust_vector.global_trust = np.clip(self.trust_vector.global_trust, 0.0, 1.0)
    
    def record_interaction(self, is_consistent: bool):
        """记录交互结果"""
        self.interaction_count += 1
        if is_consistent:
            self.consistent_count += 1
        else:
            self.malicious_count += 1
        
        current_score = self.trust_vector.get_weighted_score()
        self.trust_history.append(current_score)
        if len(self.trust_history) > self.max_history_len:
            self.trust_history.pop(0)
        
        self.last_update = time.time()
    
    def adjust_thresholds(self):
        """动态调整阈值（反风险置信度分配）"""
        current_trust = self.trust_vector.get_weighted_score()
        
        if current_trust < 0.3:
            self.suspicious_threshold = 0.5
            self.anomaly_threshold = 0.35
        elif current_trust < 0.5:
            self.suspicious_threshold = 0.45
            self.anomaly_threshold = 0.3
        else:
            self.suspicious_threshold = 0.4
            self.anomaly_threshold = 0.25
    
    def get_risk_level(self) -> Tuple[str, float]:
        """获取风险等级"""
        score = self.trust_vector.get_weighted_score()
        
        if score < self.anomaly_threshold:
            return "CRITICAL", 1.0 - score
        elif score < self.suspicious_threshold:
            return "HIGH", 1.0 - score
        elif score < 0.6:
            return "MEDIUM", 1.0 - score
        else:
            return "LOW", 1.0 - score


class MultiDimensionalTrustManager:
    """多维信任管理器"""
    
    def __init__(
        self,
        w_direct: float = 0.55,
        w_indirect: float = 0.35,
        w_global: float = 0.1,
        enable_dynamic_weights: bool = True
    ):
        self.w_direct = w_direct
        self.w_indirect = w_indirect
        self.w_global = w_global
        self.enable_dynamic_weights = enable_dynamic_weights
        
        self._profiles: Dict[str, VehicleTrustProfile] = {}
    
    def _get_profile(self, vehicle_id: str) -> VehicleTrustProfile:
        """获取或创建车辆信任档案"""
        if vehicle_id not in self._profiles:
            self._profiles[vehicle_id] = VehicleTrustProfile(vehicle_id=vehicle_id)
        return self._profiles[vehicle_id]
    
    def update_trust(
        self,
        vehicle_id: str,
        consistency_ratio: float,
        neighbor_reports: Optional[List[float]] = None,
        is_consistent: bool = True
    ) -> Dict:
        """综合更新三维信任"""
        profile = self._get_profile(vehicle_id)
        
        profile.update_direct_trust(consistency_ratio)
        if neighbor_reports:
            profile.update_indirect_trust(neighbor_reports)
        profile.update_global_trust()
        
        profile.record_interaction(is_consistent)
        
        profile.adjust_thresholds()
        
        if self.enable_dynamic_weights:
            weights = self._compute_dynamic_weights(profile)
        else:
            weights = (self.w_direct, self.w_indirect, self.w_global)
        
        final_score = profile.trust_vector.get_weighted_score(*weights)
        risk_level, risk_score = profile.get_risk_level()
        
        return {
            'vehicle_id': vehicle_id,
            'trust_vector': profile.trust_vector.to_dict(),
            'final_score': final_score,
            'risk_level': risk_level,
            'risk_score': risk_score,
            'thresholds': {
                'suspicious': profile.suspicious_threshold,
                'anomaly': profile.anomaly_threshold
            },
            'weights': {
                'direct': weights[0],
                'indirect': weights[1],
                'global': weights[2]
            }
        }
    
    def _compute_dynamic_weights(
        self, 
        profile: VehicleTrustProfile
    ) -> Tuple[float, float, float]:
        """动态权重计算（优化版V4：更激进的权重调整）"""
        direct_trust = profile.trust_vector.direct_trust
        indirect_trust = profile.trust_vector.indirect_trust
        global_trust = profile.trust_vector.global_trust
        
        if global_trust > 0.6 and direct_trust < 0.3:
            return (0.75, 0.2, 0.05)
        
        if direct_trust < 0.3 and indirect_trust < 0.3:
            return (0.65, 0.3, 0.05)
        
        if profile.interaction_count < 10:
            return (0.6, 0.3, 0.1)
        
        return (self.w_direct, self.w_indirect, self.w_global)
    
    def get_trust_score(self, vehicle_id: str) -> float:
        """获取综合信任分数"""
        profile = self._get_profile(vehicle_id)
        return profile.trust_vector.get_weighted_score(
            self.w_direct, self.w_indirect, self.w_global
        )
    
    def get_trust_history(self, vehicle_id: str) -> List[float]:
        """获取信任历史序列（用于 LSTM）"""
        profile = self._get_profile(vehicle_id)
        return profile.trust_history.copy()
    
    def get_all_profiles(self) -> Dict[str, VehicleTrustProfile]:
        """获取所有车辆档案"""
        return self._profiles.copy()
    
    def export_for_prediction(self, vehicle_id: str) -> Optional[np.ndarray]:
        """导出特征用于预测模型"""
        profile = self._get_profile(vehicle_id)
        if len(profile.trust_history) < 5:
            return None
        
        return np.array(profile.trust_history[-20:], dtype=np.float32)
