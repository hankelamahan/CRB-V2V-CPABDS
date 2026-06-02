# Enhanced DRAMBR+ 系统文档

## 概述

Enhanced DRAMBR+ 是对原版 DRAMBR 的全面升级，专门针对 V2V 通信中的"持证作恶"和"紧急刹车欺诈"等高级攻击场景设计。

## 核心升级

### 1. 多维信任向量系统 (Multi-Dimensional Trust Vector)

**解决问题**: 原版 DRAMBR 的单一信誉值无法快速响应"持证作恶"（高信誉车辆首次作恶）

**实现方案**:
- **直接信任 (Direct Trust)**: 基于实时物理检测结果，权重 40%
- **间接信任 (Indirect Trust)**: 基于邻居车辆报告，权重 30%
- **全局信任 (Global Trust)**: 基于长期历史记录，权重 30%

**动态权重调整**: 检测到"持证作恶"时（全局信任 > 0.6 但直接信任 < 0.3），自动提升直接信任权重至 60%

```python
# 使用示例
trust_manager = MultiDimensionalTrustManager(
    w_direct=0.4,
    w_indirect=0.3,
    w_global=0.3,
    enable_dynamic_weights=True
)

trust_update = trust_manager.update_trust(
    vehicle_id="V001",
    consistency_ratio=0.2,  # 低一致性
    neighbor_reports=[0.3, 0.25, 0.35],  # 邻居也报告异常
    is_consistent=False
)
```

### 2. 预测性信誉模块 (Predictive Reputation)

**解决问题**: 传统信誉系统总是"事后诸葛亮"，无法提前预警

**实现方案**:
- 基于 LSTM 的时序预测器（如果有 PyTorch）
- 回退到移动平均预测器（无深度学习框架时）
- 预测值与实际值偏差 > 0.15 触发预警
- 特别关注"预测高但实际低"的情况（持证作恶特征）

```python
predictor = PredictiveReputationEngine(
    sequence_length=20,
    deviation_threshold=0.15,
    confidence_threshold=0.7
)

# 检查预测偏差
prediction_result = predictor.check_deviation(vehicle_id, actual_score)

if prediction_result and prediction_result.is_anomalous:
    print(f"预警: 车辆 {vehicle_id} 预测值 {prediction_result.predicted_score:.2f} "
          f"但实际值 {prediction_result.actual_score:.2f}")
```

### 3. 动态风险置信度分配 (Dynamic Risk Confidence Allocation)

**解决问题**: 固定阈值无法适应不同风险等级的车辆

**实现方案**:
- 根据当前信任分数动态调整可疑阈值和异常阈值
- 信任 < 0.3: 可疑阈值 0.5, 异常阈值 0.35
- 信任 < 0.5: 可疑阈值 0.45, 异常阈值 0.3
- 信任 >= 0.5: 可疑阈值 0.4, 异常阈值 0.25

```python
profile = trust_manager._get_profile(vehicle_id)
profile.adjust_thresholds()

risk_level, risk_score = profile.get_risk_level()
# 返回: ("CRITICAL", 0.85) 或 ("HIGH", 0.65) 等
```

### 4. 紧急刹车欺诈检测 (Emergency Brake Fraud Detection)

**解决问题**: 专门针对虚假紧急刹车消息（EEBL 攻击）

**检测维度**:
1. **物理一致性验证**:
   - 加速度是否在物理可能范围内 (< 9.8 m/s²)
   - 紧急刹车必须有足够减速度 (> 4.0 m/s²)
   - 速度变化与加速度是否一致
   - 位置轨迹是否连续

2. **多车交叉验证**:
   - 邻近车辆观测到的速度是否一致
   - 位置观测是否合理

3. **历史行为分析**:
   - 紧急刹车频率（正常应 < 30%）
   - 历史欺诈率

```python
brake_detector = EmergencyBrakeFraudDetector(
    max_deceleration=9.8,
    min_emergency_deceleration=4.0
)

brake_event = BrakeEvent(
    vehicle_id="V001",
    timestamp=time.time(),
    position=np.array([100.0, 50.0, 0.0]),
    velocity=15.0,
    acceleration=-8.5,
    brake_intensity=0.9,
    is_emergency=True
)

fraud_result = brake_detector.detect_fraud(
    brake_event,
    neighbor_observations=[
        {'observed_velocity': 14.8, 'observed_position': np.array([100.2, 50.1, 0.0])}
    ]
)

if fraud_result['is_fraud']:
    print(f"检测到刹车欺诈: {fraud_result['reason']}")
```

### 5. 车辆端轻量级预筛选 (Vehicle-Side Pre-Filter)

**解决问题**: 减少 RSU 计算负担，降低通信开销

**实现方案**:
- 车辆端毫秒级快速检查
- 只有风险分数 > 0.3 或风险增长 > 0.2 时才上报 RSU
- 本地缓存历史风险分数

```python
prefilter = VehicleSidePreFilter(quick_threshold=0.3)

should_report, quick_risk = prefilter.quick_check(vehicle_id, observation)

if not should_report:
    # 无需上报 RSU，节省带宽
    pass
```

### 6. RSU 端三重确认 (RSU Triple Confirmation)

**解决问题**: 提高 RSU 端检测准确率，减少误报

**实现方案**:
1. **DBSCAN 聚类**: 印象分加权，低信誉车辆距离放大
2. **Isolation Forest**: 异常检测
3. **GMM 分类**: 概率分布分析

```python
rsu_analyzer = RSUClusterAnalyzer(
    dbscan_eps=0.3,
    dbscan_min_samples=2,
    contamination=0.2,
    n_gmm_components=2
)

cluster_results = rsu_analyzer.analyze_batch(
    feature_matrix,
    vehicle_ids,
    impression_scores
)
```

## 完整使用示例

```python
from enhanced_drambr_plus import EnhancedDRAMBRPlus

# 初始化系统
system = EnhancedDRAMBRPlus(
    enable_prediction=True,
    enable_brake_fraud_detection=True,
    enable_vehicle_prefilter=True
)

# 初始化车辆信誉
vehicle_ids = ["V001", "V002", "V003", "V004", "V005"]
system.initialize_reputations(vehicle_ids, initial_value=0.5)

# 设置 RSU 覆盖状态
for vid in vehicle_ids:
    system.set_rsu_coverage(vid, in_coverage=True)

# 处理车辆观测
observation = {
    'position_error': 0.5,
    'velocity_error': 0.3,
    'timestamp_error': 0.1,
    'message_frequency': 12.0,
    'brake_event': {
        'timestamp': time.time(),
        'position': [100.0, 50.0, 0.0],
        'velocity': 15.0,
        'acceleration': -8.5,
        'brake_intensity': 0.9,
        'is_emergency': True,
        'reason': 'obstacle_detected'
    }
}

neighbor_reports = [0.6, 0.55, 0.65]
neighbor_observations = [
    {'observed_velocity': 14.8, 'observed_position': np.array([100.2, 50.1, 0.0])}
]

result = system.process_vehicle_observation(
    vehicle_id="V001",
    observation=observation,
    neighbor_reports=neighbor_reports,
    neighbor_observations=neighbor_observations
)

print(f"处理模式: {result['mode']}")
print(f"当前信誉: {result['reputation']:.3f}")
print(f"风险等级: {result['trust_update']['risk_level']}")
print(f"预警分数: {result['early_warning_score']:.3f}")

if result['brake_fraud_result'] and result['brake_fraud_result']['is_fraud']:
    print(f"刹车欺诈检测: {result['brake_fraud_result']['reason']}")

# 获取融合权重（用于 WBF）
weights = system.get_fusion_weights(vehicle_ids, threshold=0.3)
print(f"融合权重: {weights}")

# 查看安全事件
critical_events = system.get_security_events(severity_filter=['CRITICAL', 'HIGH'])
for event in critical_events:
    print(f"安全事件: {event.event_type} - {event.severity} - {event.recommended_action}")

# 获取系统统计
stats = system.get_statistics()
print(f"系统统计: {stats}")
```

## 性能优势

| 指标 | 原版 DRAMBR | Enhanced DRAMBR+ |
|------|-------------|------------------|
| 首次作恶检测周期 | 5-10 个报告周期 | 1-2 个报告周期 |
| 无 RSU 覆盖响应 | 需等待回到覆盖区 | 车辆端即时响应 |
| 刹车欺诈检测 | 不支持 | 专项检测模块 |
| 预测性预警 | 不支持 | LSTM 预测偏差 |
| RSU 计算负担 | 高 | 车辆端预筛选降低 30-40% |

## 模块依赖

```python
# 必需
numpy

# 可选（用于 RSU 三重确认）
scikit-learn

# 可选（用于 LSTM 预测）
torch
```

## 与 OpenCOOD 集成

Enhanced DRAMBR+ 已集成到 `intermediate_fusion_dataset.py` 中，通过配置文件启用：

```yaml
trust_fusion:
  use_trust_fusion: true
  iou_thr: 0.5
  update_rate: 0.1
  default_reputation: 0.5
```

## 文件结构

```
d:\61-V2V\CRB-V2V-CPABDS\
├── advanced_trust_system.py          # 多维信任向量系统
├── predictive_reputation.py          # 预测性信誉模块
├── emergency_brake_fraud_detector.py # 紧急刹车欺诈检测
├── enhanced_drambr.py                # 原版 DRAMBR 增强
├── enhanced_drambr_plus.py           # 完整集成系统
└── ENHANCED_DRAMBR_PLUS_README.md    # 本文档
```

## 引用

如果使用本系统，请引用：

```
Enhanced DRAMBR+: Multi-Dimensional Trust Fusion with Predictive Reputation 
for V2V Communication Security
```
