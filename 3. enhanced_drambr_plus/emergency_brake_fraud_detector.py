# -*- coding: utf-8 -*-
"""
紧急刹车欺诈检测模块 (Emergency Brake Fraud Detector)

专门针对"紧急刹车欺诈"攻击的检测模块
- 物理一致性验证：速度、加速度、位置轨迹
- 多车交叉验证：邻近车辆的观测一致性
- 时空关联分析：刹车事件的时空合理性
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BrakeEvent:
    """刹车事件"""
    vehicle_id: str
    timestamp: float
    position: np.ndarray
    velocity: float
    acceleration: float
    brake_intensity: float
    is_emergency: bool
    reported_reason: str = ""


@dataclass
class PhysicsValidation:
    """物理一致性验证结果"""
    is_valid: bool
    velocity_consistent: bool
    acceleration_consistent: bool
    trajectory_consistent: bool
    inconsistency_score: float
    details: Dict = field(default_factory=dict)


class EmergencyBrakeFraudDetector:
    """紧急刹车欺诈检测器"""
    
    def __init__(
        self,
        max_deceleration: float = 9.8,
        min_emergency_deceleration: float = 4.0,
        position_tolerance: float = 2.0,
        velocity_tolerance: float = 1.5,
        history_window: int = 50
    ):
        self.max_deceleration = max_deceleration
        self.min_emergency_deceleration = min_emergency_deceleration
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.history_window = history_window
        
        self._vehicle_trajectories: Dict[str, deque] = {}
        self._brake_events: Dict[str, List[BrakeEvent]] = defaultdict(list)
        self._fraud_count: Dict[str, int] = defaultdict(int)
        self._total_brake_events: Dict[str, int] = defaultdict(int)
    
    def _get_trajectory(self, vehicle_id: str) -> deque:
        if vehicle_id not in self._vehicle_trajectories:
            self._vehicle_trajectories[vehicle_id] = deque(maxlen=self.history_window)
        return self._vehicle_trajectories[vehicle_id]
    
    def update_trajectory(
        self, 
        vehicle_id: str, 
        position: np.ndarray, 
        velocity: float,
        timestamp: float
    ):
        trajectory = self._get_trajectory(vehicle_id)
        trajectory.append({
            'position': position,
            'velocity': velocity,
            'timestamp': timestamp
        })
    
    def validate_physics(self, brake_event: BrakeEvent) -> PhysicsValidation:
        """物理一致性验证"""
        trajectory = self._get_trajectory(brake_event.vehicle_id)
        
        if len(trajectory) < 2:
            return PhysicsValidation(
                is_valid=True, velocity_consistent=True,
                acceleration_consistent=True, trajectory_consistent=True,
                inconsistency_score=0.0
            )
        
        recent_states = list(trajectory)[-5:]
        
        acc_valid = abs(brake_event.acceleration) <= self.max_deceleration
        if brake_event.is_emergency:
            acc_valid = acc_valid and abs(brake_event.acceleration) >= self.min_emergency_deceleration
        
        prev_velocity = recent_states[-1]['velocity']
        dt = brake_event.timestamp - recent_states[-1]['timestamp']
        
        if dt > 0:
            expected_velocity = prev_velocity + brake_event.acceleration * dt
            velocity_error = abs(expected_velocity - brake_event.velocity)
            vel_consistent = velocity_error <= self.velocity_tolerance
        else:
            vel_consistent = True
            velocity_error = 0.0
        
        prev_pos = recent_states[-1]['position']
        if dt > 0:
            avg_velocity = (brake_event.velocity + recent_states[-1]['velocity']) / 2
            expected_displacement = avg_velocity * dt
            actual_displacement = np.linalg.norm(brake_event.position - prev_pos)
            position_error = abs(actual_displacement - expected_displacement)
            traj_consistent = position_error <= self.position_tolerance
        else:
            traj_consistent = True
            position_error = 0.0
        
        inconsistency_score = 0.0
        if not acc_valid:
            inconsistency_score += 0.4
        if not vel_consistent:
            inconsistency_score += 0.3 * min(velocity_error / self.velocity_tolerance, 1.0)
        if not traj_consistent:
            inconsistency_score += 0.3 * min(position_error / self.position_tolerance, 1.0)
        
        return PhysicsValidation(
            is_valid=inconsistency_score < 0.5,
            velocity_consistent=vel_consistent,
            acceleration_consistent=acc_valid,
            trajectory_consistent=traj_consistent,
            inconsistency_score=float(np.clip(inconsistency_score, 0.0, 1.0)),
            details={'velocity_error': velocity_error, 'position_error': position_error}
        )
    
    def cross_validate_with_neighbors(
        self,
        brake_event: BrakeEvent,
        neighbor_observations: List[Dict]
    ) -> Tuple[bool, float]:
        """多车交叉验证"""
        if not neighbor_observations:
            return True, 1.0
        
        consistent_count = 0
        for obs in neighbor_observations:
            velocity_diff = abs(obs.get('observed_velocity', brake_event.velocity) - brake_event.velocity)
            velocity_consistent = velocity_diff <= self.velocity_tolerance * 1.5
            
            if 'observed_position' in obs:
                position_diff = np.linalg.norm(obs['observed_position'] - brake_event.position)
                position_consistent = position_diff <= self.position_tolerance * 2
            else:
                position_consistent = True
            
            if velocity_consistent and position_consistent:
                consistent_count += 1
        
        consistency_score = consistent_count / len(neighbor_observations)
        return consistency_score >= 0.5, consistency_score
    
    def detect_fraud(
        self,
        brake_event: BrakeEvent,
        neighbor_observations: Optional[List[Dict]] = None
    ) -> Dict:
        """综合欺诈检测"""
        self._total_brake_events[brake_event.vehicle_id] += 1
        
        physics_result = self.validate_physics(brake_event)
        
        if neighbor_observations:
            cross_valid, cross_score = self.cross_validate_with_neighbors(
                brake_event, neighbor_observations
            )
        else:
            cross_valid, cross_score = True, 1.0
        
        history_score = self._analyze_brake_history(brake_event.vehicle_id)
        
        fraud_score = (
            0.5 * physics_result.inconsistency_score +
            0.3 * (1.0 - cross_score) +
            0.2 * history_score
        )
        
        is_fraud = fraud_score > 0.45
        
        reasons = []
        if not physics_result.is_valid:
            reasons.append("物理不一致")
        if not cross_valid:
            reasons.append("邻居观测不一致")
        if history_score > 0.5:
            reasons.append("历史欺诈记录")
        
        if is_fraud:
            self._fraud_count[brake_event.vehicle_id] += 1
        
        self._brake_events[brake_event.vehicle_id].append(brake_event)
        self.update_trajectory(
            brake_event.vehicle_id,
            brake_event.position,
            brake_event.velocity,
            brake_event.timestamp
        )
        
        return {
            'vehicle_id': brake_event.vehicle_id,
            'is_fraud': is_fraud,
            'fraud_score': float(fraud_score),
            'physics_validation': physics_result,
            'cross_validation_score': float(cross_score),
            'history_score': float(history_score),
            'reason': "; ".join(reasons) if reasons else "正常",
            'timestamp': brake_event.timestamp
        }
    
    def _analyze_brake_history(self, vehicle_id: str) -> float:
        """分析车辆的刹车历史行为"""
        events = self._brake_events.get(vehicle_id, [])
        if len(events) < 3:
            return 0.0
        
        recent_events = events[-20:]
        emergency_count = sum(1 for e in recent_events if e.is_emergency)
        emergency_rate = emergency_count / len(recent_events)
        
        frequency_risk = min(emergency_rate / 0.3, 1.0) * 0.5 if emergency_rate > 0.3 else 0.0
        fraud_rate = self._fraud_count[vehicle_id] / max(1, self._total_brake_events[vehicle_id])
        
        return float(np.clip(0.6 * fraud_rate + 0.4 * frequency_risk, 0.0, 1.0))
    
    def get_vehicle_fraud_rate(self, vehicle_id: str) -> float:
        total = self._total_brake_events.get(vehicle_id, 0)
        return self._fraud_count.get(vehicle_id, 0) / total if total > 0 else 0.0
    
    def get_statistics(self) -> Dict:
        total_events = sum(self._total_brake_events.values())
        total_frauds = sum(self._fraud_count.values())
        
        return {
            'total_brake_events': total_events,
            'total_fraud_detected': total_frauds,
            'overall_fraud_rate': total_frauds / max(1, total_events),
            'tracked_vehicles': len(self._vehicle_trajectories),
            'vehicles_with_fraud': sum(1 for c in self._fraud_count.values() if c > 0)
        }
