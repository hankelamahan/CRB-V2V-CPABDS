# -*- coding: utf-8 -*-
"""
预测性信誉模块 (Predictive Reputation Module)

基于 LSTM 的信誉预测器，用于提前识别"首次作恶"行为
- 使用车辆的行为序列预测下一个时间窗口的信誉值
- 预测值与实际值的显著偏差触发提前预警
- 与 RSU 端异常检测形成联动
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# 尝试导入深度学习框架
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    pass


@dataclass
class PredictionResult:
    """预测结果"""
    vehicle_id: str
    predicted_score: float
    actual_score: float
    deviation: float
    is_anomalous: bool
    confidence: float
    timestamp: float = field(default_factory=time.time)


class SimpleLSTMPredictor(nn.Module if HAS_TORCH else object):
    """轻量级 LSTM 信誉预测器"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 2):
        if not HAS_TORCH:
            return
        
        super(SimpleLSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        if not HAS_TORCH:
            return None
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


class FallbackPredictor:
    """回退预测器（无 PyTorch 时使用移动平均）"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
    
    def predict(self, sequence: np.ndarray) -> float:
        """简单移动平均预测"""
        if len(sequence) < 2:
            return 0.5
        
        recent = sequence[-self.window_size:]
        trend = np.mean(np.diff(recent)) if len(recent) > 1 else 0.0
        prediction = recent[-1] + trend
        return float(np.clip(prediction, 0.0, 1.0))


class PredictiveReputationEngine:
    """预测性信誉引擎"""
    
    def __init__(
        self,
        sequence_length: int = 20,
        deviation_threshold: float = 0.15,
        confidence_threshold: float = 0.7,
        enable_training: bool = False
    ):
        self.sequence_length = sequence_length
        self.deviation_threshold = deviation_threshold
        self.confidence_threshold = confidence_threshold
        self.enable_training = enable_training
        
        # 初始化预测器
        if HAS_TORCH:
            self.predictor = SimpleLSTMPredictor()
            self.predictor.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.predictor.to(self.device)
        else:
            self.predictor = FallbackPredictor()
        
        # 存储每个车辆的预测历史
        self._prediction_history: Dict[str, deque] = {}
        self._sequence_buffer: Dict[str, deque] = {}
    
    def _get_sequence_buffer(self, vehicle_id: str) -> deque:
        """获取车辆的序列缓冲区"""
        if vehicle_id not in self._sequence_buffer:
            self._sequence_buffer[vehicle_id] = deque(maxlen=self.sequence_length)
        return self._sequence_buffer[vehicle_id]
    
    def _get_prediction_history(self, vehicle_id: str) -> deque:
        """获取车辆的预测历史"""
        if vehicle_id not in self._prediction_history:
            self._prediction_history[vehicle_id] = deque(maxlen=50)
        return self._prediction_history[vehicle_id]
    
    def add_observation(self, vehicle_id: str, trust_score: float):
        """添加观测值到序列"""
        buffer = self._get_sequence_buffer(vehicle_id)
        buffer.append(trust_score)
    
    def predict_next(self, vehicle_id: str) -> Optional[Tuple[float, float]]:
        """
        预测下一个时间窗口的信誉值
        
        返回:
            (predicted_score, confidence) 或 None（序列不足）
        """
        buffer = self._get_sequence_buffer(vehicle_id)
        
        if len(buffer) < 5:
            return None
        
        sequence = np.array(list(buffer), dtype=np.float32)
        
        if HAS_TORCH:
            with torch.no_grad():
                x = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1).to(self.device)
                prediction = self.predictor(x).item()
                
                # 计算置信度（基于序列稳定性）
                variance = np.var(sequence[-10:])
                confidence = 1.0 / (1.0 + variance * 10)
        else:
            prediction = self.predictor.predict(sequence)
            variance = np.var(sequence[-5:])
            confidence = 1.0 / (1.0 + variance * 10)
        
        return prediction, confidence
    
    def check_deviation(
        self, 
        vehicle_id: str, 
        actual_score: float
    ) -> Optional[PredictionResult]:
        """
        检查预测偏差并生成预警
        
        参数:
            vehicle_id: 车辆ID
            actual_score: 实际观测到的信誉值
        
        返回:
            PredictionResult 或 None（无预测）
        """
        prediction_result = self.predict_next(vehicle_id)
        
        if prediction_result is None:
            self.add_observation(vehicle_id, actual_score)
            return None
        
        predicted_score, confidence = prediction_result
        deviation = abs(predicted_score - actual_score)
        
        # 判断是否异常（预测值高但实际值骤降）
        is_anomalous = (
            deviation > self.deviation_threshold and
            confidence > self.confidence_threshold and
            predicted_score > 0.5 and
            actual_score < 0.4
        )
        
        result = PredictionResult(
            vehicle_id=vehicle_id,
            predicted_score=predicted_score,
            actual_score=actual_score,
            deviation=deviation,
            is_anomalous=is_anomalous,
            confidence=confidence
        )
        
        # 记录预测历史
        history = self._get_prediction_history(vehicle_id)
        history.append(result)
        
        # 更新序列
        self.add_observation(vehicle_id, actual_score)
        
        return result
    
    def get_early_warning_score(self, vehicle_id: str) -> float:
        """
        获取提前预警分数（0-1，越高越危险）
        
        基于最近的预测偏差历史
        """
        history = self._get_prediction_history(vehicle_id)
        
        if len(history) < 3:
            return 0.0
        
        recent = list(history)[-5:]
        recent_deviations = [r.deviation for r in recent]
        recent_anomalies = sum(1 for r in recent if r.is_anomalous)
        
        peak_deviation = max(recent_deviations)
        avg_deviation = np.mean(recent_deviations)
        effective_deviation = 0.7 * peak_deviation + 0.3 * avg_deviation
        scaled_deviation = float(np.clip(
            effective_deviation / (2 * self.deviation_threshold), 0.0, 1.0
        ))
        anomaly_rate = recent_anomalies / len(recent)
        
        warning_score = 0.55 * scaled_deviation + 0.45 * anomaly_rate
        return float(np.clip(warning_score, 0.0, 1.0))
    
    def get_statistics(self) -> Dict:
        """获取预测引擎统计信息"""
        total_predictions = sum(len(h) for h in self._prediction_history.values())
        total_anomalies = sum(
            sum(1 for r in h if r.is_anomalous) 
            for h in self._prediction_history.values()
        )
        
        return {
            'has_torch': HAS_TORCH,
            'tracked_vehicles': len(self._sequence_buffer),
            'total_predictions': total_predictions,
            'total_anomalies': total_anomalies,
            'anomaly_rate': total_anomalies / max(1, total_predictions)
        }


class ReputationTrendAnalyzer:
    """信誉趋势分析器（辅助预测）"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
    
    def analyze_trend(self, sequence: List[float]) -> Dict:
        """
        分析信誉趋势
        
        返回:
            {
                'trend': 'rising' | 'falling' | 'stable',
                'slope': float,
                'volatility': float,
                'risk_signal': bool
            }
        """
        if len(sequence) < 3:
            return {
                'trend': 'stable',
                'slope': 0.0,
                'volatility': 0.0,
                'risk_signal': False
            }
        
        recent = np.array(sequence[-self.window_size:])
        
        # 计算趋势斜率
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        # 计算波动性
        volatility = np.std(recent)
        
        # 判断趋势
        if slope > 0.02:
            trend = 'rising'
        elif slope < -0.02:
            trend = 'falling'
        else:
            trend = 'stable'
        
        # 风险信号：快速下降 + 高波动
        risk_signal = (slope < -0.05 and volatility > 0.15)
        
        return {
            'trend': trend,
            'slope': float(slope),
            'volatility': float(volatility),
            'risk_signal': risk_signal
        }
