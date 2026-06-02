# -*- coding: utf-8 -*-
"""Enhanced DRAMBR+ 可视化演示脚本 - 修复中文字体问题"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from enhanced_drambr_plus import EnhancedDRAMBRPlus
from visualizer import EnhancedDRAMBRVisualizer

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def print_detailed_data(visualizer, system):
    """输出详细的数据统计"""
    history = visualizer.history
    time_steps = history['time_steps']
    
    # 1. 信誉值演化数据
    print("\n[1] 信誉值演化数据")
    print("-" * 80)
    print(f"{'时间步':<8}", end="")
    for vid in sorted(history['vehicles'].keys()):
        print(f"{vid:<12}", end="")
    print()
    print("-" * 80)
    
    for i, step in enumerate(time_steps):
        if i % 2 == 0:
            print(f"{step:<8}", end="")
            for vid in sorted(history['vehicles'].keys()):
                rep = history['vehicles'][vid]['reputation'][i]
                print(f"{rep:<12.4f}", end="")
            print()
    
    # 2. 最终信誉值
    print("\n[2] 最终信誉值")
    print("-" * 80)
    print(f"{'车辆':<10} {'信誉值':<10} {'状态':<10}")
    print("-" * 80)
    for vid in sorted(history['vehicles'].keys()):
        final_rep = history['vehicles'][vid]['reputation'][-1]
        status = "安全" if final_rep >= 0.6 else "警告" if final_rep >= 0.3 else "危险"
        print(f"{vid:<10} {final_rep:<10.4f} {status:<10}")
    
    # 3. 多维信任向量数据（V001为例）
    print("\n[3] 多维信任向量演化 (以V001为例)")
    print("-" * 80)
    print(f"{'时间步':<8} {'直接信任':<12} {'间接信任':<12} {'全局信任':<12}")
    print("-" * 80)
    vid = "V001"
    if vid in history['vehicles']:
        for i, step in enumerate(time_steps):
            if i % 2 == 0:
                direct = history['vehicles'][vid]['trust_vector']['direct'][i]
                indirect = history['vehicles'][vid]['trust_vector']['indirect'][i]
                global_t = history['vehicles'][vid]['trust_vector']['global'][i]
                print(f"{step:<8} {direct:<12.4f} {indirect:<12.4f} {global_t:<12.4f}")
    
    # 4. 风险等级统计
    print("\n[4] 风险等级分布")
    print("-" * 80)
    risk_stats = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0, 'N/A': 0}
    for vid, data in history['vehicles'].items():
        for risk in data['risk_level']:
            risk_stats[risk] = risk_stats.get(risk, 0) + 1
    
    total = sum(risk_stats.values())
    for risk, count in sorted(risk_stats.items()):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{risk:<12}: {count:>4} 次 ({percentage:>5.1f}%)")
    
    # 5. 预警分数统计
    print("\n[5] 预警分数统计")
    print("-" * 80)
    print(f"{'车辆':<10} {'平均预警':<12} {'最大预警':<12} {'最小预警':<12}")
    print("-" * 80)
    for vid in sorted(history['vehicles'].keys()):
        warnings = history['vehicles'][vid]['early_warning']
        avg_warn = np.mean(warnings) if warnings else 0
        max_warn = np.max(warnings) if warnings else 0
        min_warn = np.min(warnings) if warnings else 0
        print(f"{vid:<10} {avg_warn:<12.4f} {max_warn:<12.4f} {min_warn:<12.4f}")
    
    # 6. 刹车欺诈检测数据
    print("\n[6] 刹车欺诈检测数据")
    print("-" * 80)
    has_brake_data = False
    for vid, data in history['vehicles'].items():
        if any(s > 0 for s in data['brake_fraud_score']):
            has_brake_data = True
            scores = [s for s in data['brake_fraud_score'] if s > 0]
            if scores:
                print(f"{vid}: 检测到 {len(scores)} 次刹车事件")
                print(f"  平均欺诈分数: {np.mean(scores):.4f}")
                print(f"  最大欺诈分数: {np.max(scores):.4f}")
    
    if not has_brake_data:
        print("本次模拟中未检测到刹车欺诈事件")
    
    # 7. 安全事件统计
    print("\n[7] 安全事件统计")
    print("-" * 80)
    events = system.get_security_events(limit=100)
    if events:
        event_types = {}
        event_severity = {}
        for e in events:
            event_types[e.event_type] = event_types.get(e.event_type, 0) + 1
            event_severity[e.severity] = event_severity.get(e.severity, 0) + 1
        
        print("按类型分类:")
        for etype, count in sorted(event_types.items()):
            print(f"  {etype:<20}: {count:>4} 次")
        
        print("\n按严重程度分类:")
        for severity, count in sorted(event_severity.items()):
            print(f"  {severity:<20}: {count:>4} 次")
    else:
        print("无安全事件记录")
    
    # 8. 系统统计信息
    print("\n[8] 系统统计信息")
    print("-" * 80)
    stats = system.get_statistics()
    print(f"系统: {stats['system']}")
    print(f"总交互次数: {stats['total_interactions']}")
    print(f"追踪车辆数: {stats['tracked_vehicles']}")
    print(f"平均信誉值: {stats['avg_reputation']:.4f}")
    print(f"信誉值标准差: {stats['std_reputation']:.4f}")
    print(f"安全事件数: {stats['security_events']}")
    print(f"离线缓冲区大小: {stats['offline_buffer_size']}")
    
    print("\n启用模块:")
    print(f"  预测模块: {'ON' if stats['modules']['prediction'] else 'OFF'}")
    print(f"  刹车欺诈检测: {'ON' if stats['modules']['brake_fraud'] else 'OFF'}")
    print(f"  车辆端预筛选: {'ON' if stats['modules']['prefilter'] else 'OFF'}")
    
    if 'prediction' in stats:
        print(f"\n预测模块统计:")
        print(f"  总预测: {stats['prediction']['total_predictions']}")
        if 'anomalies_detected' in stats['prediction']:
            print(f"  检测到异常: {stats['prediction']['anomalies_detected']}")
    
    if 'brake_fraud' in stats:
        print(f"\n刹车欺诈统计:")
        print(f"  总刹车事件: {stats['brake_fraud']['total_brake_events']}")
        print(f"  检测到欺诈: {stats['brake_fraud']['total_fraud_detected']}")
    
    # 9. WBF融合权重
    print("\n[9] WBF融合权重")
    print("-" * 80)
    vehicle_ids = list(history['vehicles'].keys())
    weights = system.get_fusion_weights(vehicle_ids, threshold=0.3)
    print(f"{'车辆':<10} {'融合权重':<12} {'状态':<10}")
    print("-" * 80)
    for vid, weight in zip(sorted(vehicle_ids), weights):
        status = "有效" if weight >= 0.3 else "排除"
        print(f"{vid:<10} {weight:<12.4f} {status:<10}")
    
    # 10. 数据摘要
    print("\n[10] 数据摘要")
    print("-" * 80)
    print(f"模拟时间步: {len(time_steps)}")
    print(f"参与车辆数: {len(history['vehicles'])}")
    print(f"总数据点: {len(time_steps) * len(history['vehicles'])}")
    
    high_risk_vehicles = [vid for vid, data in history['vehicles'].items() 
                         if data['reputation'][-1] < 0.3]
    if high_risk_vehicles:
        print(f"\n高风险车辆: {', '.join(high_risk_vehicles)}")
    else:
        print(f"\n高风险车辆: 无")


def run_demo():
    """运行可视化演示"""
    print("=" * 80)
    print("Enhanced DRAMBR+ 高级可视化演示")
    print("=" * 80)
    
    system = EnhancedDRAMBRPlus(
        enable_prediction=True,
        enable_brake_fraud_detection=True,
        enable_vehicle_prefilter=False
    )
    
    vehicle_ids = ["V001", "V002", "V003", "V004", "V005"]
    system.initialize_reputations(vehicle_ids, initial_value=0.5)
    
    for vid in vehicle_ids:
        system.set_rsu_coverage(vid, in_coverage=True)
    
    visualizer = EnhancedDRAMBRVisualizer(system)
    
    print("\n运行模拟场景...")
    
    for step in range(20):
        vehicle_results = {}
        
        for vid in vehicle_ids:
            if vid == "V002" and step >= 15:
                observation = {
                    'position_error': np.random.uniform(2.0, 4.0),
                    'velocity_error': 0.05,
                    'timestamp_error': 0.02,
                    'message_frequency': 10.0
                }
                neighbor_reports = [0.2, 0.15, 0.25]
                neighbor_observations = None
            elif vid == "V003" and step >= 10:
                observation = {
                    'position_error': 0.05,
                    'velocity_error': 0.05,
                    'timestamp_error': 0.02,
                    'message_frequency': 10.0,
                    'brake_event': {
                        'timestamp': time.time(),
                        'position': np.array([100.0 + step * 2, 50.0, 0.0]),
                        'velocity': 25.0,
                        'acceleration': -1.5,
                        'brake_intensity': 0.9,
                        'is_emergency': True,
                        'reason': 'fake_obstacle'
                    }
                }
                neighbor_reports = [0.5, 0.52, 0.48]
                neighbor_observations = [
                    {'observed_velocity': 19.8, 
                     'observed_position': np.array([100.2 + step * 2, 50.1, 0.0])}
                ]
            else:
                observation = {
                    'position_error': 0.05,
                    'velocity_error': 0.05,
                    'timestamp_error': 0.02,
                    'message_frequency': 10.0
                }
                neighbor_reports = [0.5, 0.52, 0.48]
                neighbor_observations = None
            
            result = system.process_vehicle_observation(
                vehicle_id=vid,
                observation=observation,
                neighbor_reports=neighbor_reports,
                neighbor_observations=neighbor_observations
            )
            
            vehicle_results[vid] = result
        
        visualizer.record_step(step, vehicle_results)
        
        if step % 5 == 0:
            print(f"  步骤 {step}: 已处理")
    
    print("\n生成可视化...")
    visualizer.plot_dashboard('enhanced_drambr_plus_v3_dashboard.png')
    
    print("\n" + "=" * 80)
    print("详细数据输出")
    print("=" * 80)
    
    print_detailed_data(visualizer, system)
    
    print("\n" + "=" * 80)
    print("可视化完成！")
    print("=" * 80)
    print("\n生成文件: enhanced_drambr_plus_v3_dashboard.png")
    print("\n仪表板包含:")
    print("  1. 信誉值演化轨迹")
    print("  2. 多维信任向量演化")
    print("  3. 风险等级时间线")
    print("  4. 预测性预警分数热力图")
    print("  5. 紧急刹车欺诈检测")
    print("  6. 安全事件统计")
    print("  7. 最终信誉值分布")
    print("  8. 系统统计信息")
    print("  9. WBF 融合权重")


if __name__ == "__main__":
    run_demo()
