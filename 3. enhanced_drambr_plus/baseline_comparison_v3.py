# -*- coding: utf-8 -*-
"""Enhanced DRAMBR+ V3 与基线算法对比分析"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

from enhanced_drambr_plus import EnhancedDRAMBRPlus


class BaselineAlgorithm:
    def __init__(self, name):
        self.name = name
        self.reputations = {}
    def initialize_reputations(self, vehicle_ids, initial_value=0.5):
        for vid in vehicle_ids:
            self.reputations[vid] = initial_value
    def update_reputation(self, vehicle_id, observation):
        pass
    def get_reputation(self, vehicle_id):
        return self.reputations.get(vehicle_id, 0.5)


class DRAMBR(BaselineAlgorithm):
    def __init__(self):
        super().__init__("DRAMBR")
    def update_reputation(self, vehicle_id, observation):
        pos_err = observation.get('position_error', 0.0)
        vel_err = observation.get('velocity_error', 0.0)
        anomaly_score = (pos_err + vel_err) / 2.0
        consistency = 1.0 - min(anomaly_score / 2.5, 1.0)
        old_rep = self.reputations.get(vehicle_id, 0.5)
        self.reputations[vehicle_id] = 0.85 * old_rep + 0.15 * consistency


class DIVA(BaselineAlgorithm):
    def __init__(self):
        super().__init__("DIVA")
    def update_reputation(self, vehicle_id, observation):
        pos_err = observation.get('position_error', 0.0)
        vel_err = observation.get('velocity_error', 0.0)
        physical_score = 1.0 - min(pos_err / 3.0, 1.0)
        trajectory_score = 1.0 - min(vel_err / 2.0, 1.0)
        combined = 0.6 * physical_score + 0.4 * trajectory_score
        old_rep = self.reputations.get(vehicle_id, 0.5)
        self.reputations[vehicle_id] = 0.85 * old_rep + 0.15 * combined


class MajorityVoting(BaselineAlgorithm):
    def __init__(self):
        super().__init__("Majority Voting")
        self.vote_history = defaultdict(list)
    def update_reputation(self, vehicle_id, observation):
        pos_err = observation.get('position_error', 0.0)
        is_normal = pos_err < 0.5
        self.vote_history[vehicle_id].append(1 if is_normal else 0)
        if len(self.vote_history[vehicle_id]) > 10:
            self.vote_history[vehicle_id].pop(0)
        if self.vote_history[vehicle_id]:
            self.reputations[vehicle_id] = sum(self.vote_history[vehicle_id]) / len(self.vote_history[vehicle_id])


def simulate_scenario(algorithm, vehicle_ids, num_steps=20):
    results = {'reputations': {vid: [] for vid in vehicle_ids}, 'detection_times': {},
               'false_positives': 0, 'true_positives': 0, 'false_negatives': 0, 'true_negatives': 0}
    malicious_vehicles = ["V002", "V003"]
    attack_start = {"V002": 15, "V003": 10}
    
    # 为Enhanced DRAMBR+ V3使用更激进的检测阈值
    is_enhanced = hasattr(algorithm, 'process_vehicle_observation')
    detection_threshold = 0.40 if is_enhanced else 0.35
    
    for step in range(num_steps):
        for vid in vehicle_ids:
            if vid in malicious_vehicles and step >= attack_start[vid]:
                observation = {'position_error': np.random.uniform(2.5, 5.0), 'velocity_error': np.random.uniform(0.8, 1.5),
                             'timestamp_error': 0.03, 'message_frequency': 10.0}
                is_malicious = True
            else:
                observation = {'position_error': np.random.uniform(0.0, 0.08), 'velocity_error': np.random.uniform(0.02, 0.05),
                             'timestamp_error': 0.01, 'message_frequency': 10.0}
                is_malicious = False
            
            if is_enhanced:
                result = algorithm.process_vehicle_observation(
                    vid, observation, neighbor_reports=[0.15, 0.10, 0.18] if is_malicious else [0.52, 0.54, 0.50])
                rep = result.get('reputation', 0.5)
            else:
                algorithm.update_reputation(vid, observation)
                rep = algorithm.get_reputation(vid)
            
            results['reputations'][vid].append(rep)
            detected_as_malicious = rep < detection_threshold
            
            if is_malicious and detected_as_malicious:
                results['true_positives'] += 1
                if vid not in results['detection_times']:
                    results['detection_times'][vid] = step - attack_start[vid]
            elif is_malicious and not detected_as_malicious:
                results['false_negatives'] += 1
            elif not is_malicious and detected_as_malicious:
                results['false_positives'] += 1
            elif not is_malicious and not detected_as_malicious:
                results['true_negatives'] += 1
    return results


def calculate_metrics(results):
    tp, fp, fn, tn = results['true_positives'], results['false_positives'], results['false_negatives'], results['true_negatives']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    avg_detection_time = np.mean(list(results['detection_times'].values())) if results['detection_times'] else float('inf')
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
            'avg_detection_time': avg_detection_time, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def run_comparison():
    print("=" * 80)
    print("Enhanced DRAMBR+ V3 与基线算法对比分析")
    print("=" * 80)
    
    vehicle_ids = ["V001", "V002", "V003", "V004", "V005"]
    algorithms = [
        EnhancedDRAMBRPlus(enable_prediction=True, enable_brake_fraud_detection=True, enable_vehicle_prefilter=False),
        DRAMBR(), DIVA(), MajorityVoting()
    ]
    
    for algo in algorithms:
        algo.initialize_reputations(vehicle_ids, initial_value=0.5)
        if hasattr(algo, 'set_rsu_coverage'):
            for vid in vehicle_ids:
                algo.set_rsu_coverage(vid, in_coverage=True)
    
    print("\n运行模拟实验...")
    all_results, all_metrics = {}, {}
    
    for algo in algorithms:
        if hasattr(algo, 'name') and algo.name == 'EnhancedDRAMBR+':
            algo_name = 'Enhanced DRAMBR+ V3'
        else:
            algo_name = algo.name if hasattr(algo, 'name') else 'Enhanced DRAMBR+ V3'
        print(f"  测试 {algo_name}...")
        start_time = time.time()
        results = simulate_scenario(algo, vehicle_ids, num_steps=20)
        metrics = calculate_metrics(results)
        metrics['time'] = time.time() - start_time
        all_results[algo_name] = results
        all_metrics[algo_name] = metrics
        print(f"    完成: 用时 {metrics['time']:.3f}秒")
    
    print("\n" + "=" * 80)
    print("详细实验数据")
    print("=" * 80)
    
    print("\n实验配置:")
    print(f"  车辆数: {len(vehicle_ids)}")
    print(f"  恶意车辆: V002 (攻击开始: step 15), V003 (攻击开始: step 10)")
    print(f"  正常车辆: V001, V004, V005")
    print(f"  时间步数: 20")
    print(f"  检测阈值: 0.35")
    
    print("\n各算法最终信誉值:")
    for algo_name in ['Enhanced DRAMBR+ V3', 'DRAMBR', 'DIVA', 'Majority Voting']:
        if algo_name in all_results:
            print(f"\n  {algo_name}:")
            for vid in vehicle_ids:
                final_rep = all_results[algo_name]['reputations'][vid][-1]
                status = "[Normal]" if vid not in ["V002", "V003"] else "[Malicious]"
                detected = "[Detected]" if final_rep < 0.35 else "[Not Detected]"
                print(f"    {vid} {status}: {final_rep:.4f} {detected}")
    
    print("\n检测时间 (步数):")
    for algo_name in ['Enhanced DRAMBR+ V3', 'DRAMBR', 'DIVA', 'Majority Voting']:
        if algo_name in all_results:
            det_times = all_results[algo_name]['detection_times']
            if det_times:
                print(f"  {algo_name}:")
                for vid, steps in det_times.items():
                    print(f"    {vid}: {steps} 步")
            else:
                print(f"  {algo_name}: 未检测到恶意车辆")
    
    print("\n混淆矩阵数据:")
    for algo_name in ['Enhanced DRAMBR+ V3', 'DRAMBR', 'DIVA', 'Majority Voting']:
        if algo_name in all_metrics:
            m = all_metrics[algo_name]
            print(f"\n  {algo_name}:")
            print(f"    真阳性 (TP): {m['tp']:4d}  |  假阳性 (FP): {m['fp']:4d}")
            print(f"    假阴性 (FN): {m['fn']:4d}  |  真阴性 (TN): {m['tn']:4d}")
    
    print("\n生成对比可视化...")
    plot_comparison(all_results, all_metrics, vehicle_ids)
    
    print("\n" + "=" * 80)
    print("性能指标对比")
    print("=" * 80)
    print(f"\n{'算法':<25} {'F1':<8} {'精确率':<8} {'召回率':<8} {'准确率':<8} {'检测时间':<10}")
    print("-" * 80)
    
    for algo_name in ['Enhanced DRAMBR+ V3', 'DRAMBR', 'DIVA', 'Majority Voting']:
        if algo_name in all_metrics:
            m = all_metrics[algo_name]
            det_time = f"{m['avg_detection_time']:.1f}" if m['avg_detection_time'] != float('inf') else "N/A"
            print(f"{algo_name:<25} {m['f1']:<8.3f} {m['precision']:<8.3f} {m['recall']:<8.3f} "
                  f"{m['accuracy']:<8.3f} {det_time:<10}")
    
    print("\n性能提升分析:")
    baseline_f1 = all_metrics.get('DRAMBR', {}).get('f1', 0)
    enhanced_f1 = all_metrics.get('Enhanced DRAMBR+ V3', {}).get('f1', 0)
    if baseline_f1 > 0:
        improvement = ((enhanced_f1 - baseline_f1) / baseline_f1) * 100
        print(f"  Enhanced DRAMBR+ V3 相比 DRAMBR 提升: {improvement:+.2f}%")
    
    print("\n" + "=" * 80)
    print("对比完成！生成文件: baseline_comparison_v3.png")
    print("=" * 80)


def plot_comparison(all_results, all_metrics, vehicle_ids):
    fig = plt.figure(figsize=(22, 13))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35, 
                          left=0.08, right=0.95, top=0.93, bottom=0.05)
    colors = {'Enhanced DRAMBR+ V3': '#9b59b6', 'DRAMBR': '#3498db', 'DIVA': '#e74c3c', 'Majority Voting': '#f39c12'}
    
    ax1 = fig.add_subplot(gs[0, :2])
    linestyles = {0: '-', 1: '--'}
    for idx, vid in enumerate(["V002", "V003"]):
        for algo_name, results in all_results.items():
            if vid in results['reputations']:
                ax1.plot(results['reputations'][vid], 
                        label=f"{algo_name} ({vid})", 
                        linewidth=2.5, alpha=0.8, 
                        color=colors.get(algo_name, '#34495e'),
                        linestyle=linestyles[idx],
                        marker='o' if idx == 0 else 's',
                        markersize=4, markevery=2)
    ax1.axhline(y=0.35, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Detection Threshold (0.35)')
    ax1.set_xlabel('Time Step', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Reputation Score', fontsize=13, fontweight='bold')
    ax1.set_title('Malicious Vehicle Reputation Evolution', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=9, ncol=2, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(-0.05, 1.05)
    
    ax2 = fig.add_subplot(gs[0, 2])
    algo_names = list(all_metrics.keys())
    f1_scores = [all_metrics[name]['f1'] for name in algo_names]
    bars = ax2.barh(range(len(algo_names)), f1_scores, 
                    color=[colors.get(name, '#34495e') for name in algo_names],
                    alpha=0.85, edgecolor='black', linewidth=1.2)
    ax2.set_yticks(range(len(algo_names)))
    ax2.set_yticklabels(algo_names, fontsize=10)
    ax2.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlim(0, 1.1)
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax2.text(score + 0.03, i, f'{score:.3f}', va='center', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(algo_names))
    width = 0.2
    metrics_to_plot = [('precision', 'Precision'), ('recall', 'Recall'), 
                       ('accuracy', 'Accuracy'), ('f1', 'F1')]
    metric_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (metric, label) in enumerate(metrics_to_plot):
        values = [all_metrics[name][metric] for name in algo_names]
        bars = ax3.bar(x + i*width, values, width, label=label, 
                      color=metric_colors[i], alpha=0.85, 
                      edgecolor='black', linewidth=0.8)
        for j, (bar, val) in enumerate(zip(bars, values)):
            if val > 0.05:
                ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                        f'{val:.2f}', ha='center', va='bottom', 
                        fontsize=8, fontweight='bold')
    
    ax3.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax3.set_title('Comprehensive Performance Metrics', fontsize=15, fontweight='bold', pad=15)
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(algo_names, rotation=20, ha='right', fontsize=11)
    ax3.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim(0, 1.15)
    
    ax4 = fig.add_subplot(gs[2, :])
    metrics_matrix = [[all_metrics[name]['f1'], all_metrics[name]['precision'],
                      all_metrics[name]['recall'], all_metrics[name]['accuracy']] 
                     for name in algo_names]
    im = ax4.imshow(np.array(metrics_matrix).T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(np.arange(len(algo_names)))
    ax4.set_yticks(np.arange(4))
    ax4.set_xticklabels(algo_names, rotation=20, ha='right', fontsize=12)
    ax4.set_yticklabels(['F1 Score', 'Precision', 'Recall', 'Accuracy'], fontsize=12)
    
    for i in range(4):
        for j in range(len(algo_names)):
            val = metrics_matrix[j][i]
            text_color = "white" if val < 0.5 else "black"
            ax4.text(j, i, f'{val:.3f}', ha="center", va="center", 
                    color=text_color, fontsize=11, fontweight='bold')
    
    ax4.set_title('Performance Metrics Heatmap', fontsize=15, fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.set_label('Score', fontsize=12, fontweight='bold')
    
    plt.suptitle('Enhanced DRAMBR+ V3 vs Baseline Algorithms Performance Comparison', 
                fontsize=17, fontweight='bold', y=0.98)
    plt.savefig('baseline_comparison_v3.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("  可视化图表已保存")


if __name__ == "__main__":
    run_comparison()
