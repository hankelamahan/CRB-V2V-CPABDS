# -*- coding: utf-8 -*-
"""Enhanced DRAMBR+ 可视化核心类"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("whitegrid")


class EnhancedDRAMBRVisualizer:
    """Enhanced DRAMBR+ 可视化器"""
    
    def __init__(self, system):
        self.system = system
        self.history = {'time_steps': [], 'vehicles': {}}
        
    def record_step(self, step, vehicle_results):
        """记录每一步的数据"""
        self.history['time_steps'].append(step)
        for vid, result in vehicle_results.items():
            if vid not in self.history['vehicles']:
                self.history['vehicles'][vid] = {
                    'reputation': [], 'trust_vector': {'direct': [], 'indirect': [], 'global': []},
                    'risk_level': [], 'early_warning': [], 'brake_fraud_score': []
                }
            self.history['vehicles'][vid]['reputation'].append(result.get('reputation', 0.5))
            if 'trust_update' in result:
                tv = result['trust_update'].get('trust_vector', {})
                self.history['vehicles'][vid]['trust_vector']['direct'].append(tv.get('direct', 0.5))
                self.history['vehicles'][vid]['trust_vector']['indirect'].append(tv.get('indirect', 0.5))
                self.history['vehicles'][vid]['trust_vector']['global'].append(tv.get('global', 0.5))
                self.history['vehicles'][vid]['risk_level'].append(result['trust_update'].get('risk_level', 'N/A'))
            else:
                for key in ['direct', 'indirect', 'global']:
                    self.history['vehicles'][vid]['trust_vector'][key].append(0.5)
                self.history['vehicles'][vid]['risk_level'].append('N/A')
            self.history['vehicles'][vid]['early_warning'].append(result.get('early_warning_score', 0.0))
            if result.get('brake_fraud_result'):
                self.history['vehicles'][vid]['brake_fraud_score'].append(
                    result['brake_fraud_result'].get('fraud_score', 0.0))
            else:
                self.history['vehicles'][vid]['brake_fraud_score'].append(0.0)
    
    def plot_dashboard(self, save_path='dashboard.png'):
        """生成综合仪表板"""
        fig = plt.figure(figsize=(24, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
        time_steps = self.history['time_steps']
        
        self._plot_reputation(fig.add_subplot(gs[0, :2]), time_steps)
        self._plot_trust_vector(fig.add_subplot(gs[0, 2:]), time_steps)
        self._plot_risk_timeline(fig.add_subplot(gs[1, :2]), time_steps)
        self._plot_warning_heatmap(fig.add_subplot(gs[1, 2:]), time_steps)
        self._plot_brake_fraud(fig.add_subplot(gs[2, :2]), time_steps)
        self._plot_events(fig.add_subplot(gs[2, 2:]))
        self._plot_final_rep(fig.add_subplot(gs[3, 0]))
        self._plot_stats(fig.add_subplot(gs[3, 1:3]))
        self._plot_weights(fig.add_subplot(gs[3, 3]))
        
        plt.suptitle('Enhanced DRAMBR+ 综合仪表板', fontsize=20, fontweight='bold', y=0.995)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[保存] {save_path}")
        plt.close()
    
    def _plot_reputation(self, ax, ts):
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.history['vehicles'])))
        for i, (vid, data) in enumerate(self.history['vehicles'].items()):
            ax.plot(ts, data['reputation'], marker='o', label=vid, linewidth=2.5, 
                   markersize=5, color=colors[i], alpha=0.8)
        ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='安全')
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='危险')
        ax.set_xlabel('时间步', fontsize=12, fontweight='bold')
        ax.set_ylabel('信誉值', fontsize=12, fontweight='bold')
        ax.set_title('信誉值演化', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    def _plot_trust_vector(self, ax, ts):
        vids = list(self.history['vehicles'].keys())
        if vids:
            data = self.history['vehicles'][vids[0]]
            ax.plot(ts, data['trust_vector']['direct'], marker='o', label='直接', 
                   linewidth=2.5, color='#e74c3c')
            ax.plot(ts, data['trust_vector']['indirect'], marker='s', label='间接', 
                   linewidth=2.5, color='#3498db')
            ax.plot(ts, data['trust_vector']['global'], marker='^', label='全局', 
                   linewidth=2.5, color='#2ecc71')
        ax.set_xlabel('时间步', fontsize=12, fontweight='bold')
        ax.set_ylabel('信任值', fontsize=12, fontweight='bold')
        ax.set_title(f'多维信任向量 ({vids[0] if vids else "N/A"})', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    def _plot_risk_timeline(self, ax, ts):
        risk_colors = {'LOW': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e74c3c', 
                      'CRITICAL': '#c0392b', 'N/A': '#95a5a6'}
        for i, (vid, data) in enumerate(self.history['vehicles'].items()):
            for j, (step, risk) in enumerate(zip(ts, data['risk_level'])):
                ax.add_patch(Rectangle((step - 0.4, i - 0.3), 0.8, 0.6, 
                            facecolor=risk_colors.get(risk, '#95a5a6'), 
                            edgecolor='black', linewidth=0.5))
        ax.set_xlabel('时间步', fontsize=12, fontweight='bold')
        ax.set_ylabel('车辆', fontsize=12, fontweight='bold')
        ax.set_title('风险等级时间线', fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(self.history['vehicles'])))
        ax.set_yticklabels(list(self.history['vehicles'].keys()))
    
    def _plot_warning_heatmap(self, ax, ts):
        vids = list(self.history['vehicles'].keys())
        if not vids or not ts:
            ax.text(0.5, 0.5, '无预警数据', ha='center', va='center', fontsize=14,
                    transform=ax.transAxes)
            ax.axis('off')
            return
        
        matrix = np.zeros((len(vids), len(ts)))
        for i, vid in enumerate(vids):
            warnings = self.history['vehicles'][vid]['early_warning']
            n = min(len(warnings), len(ts))
            matrix[i, :n] = warnings[:n]
        
        data_max = float(np.max(matrix))
        if data_max <= 1e-6:
            ax.text(0.5, 0.5, '预警分数均为 0', ha='center', va='center', fontsize=14,
                    transform=ax.transAxes)
            ax.set_title('预警分数热力图', fontsize=14, fontweight='bold')
            ax.axis('off')
            return
        
        vmax = max(data_max * 1.05, 0.05)
        im = ax.imshow(
            matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax,
            extent=[ts[0] - 0.5, ts[-1] + 0.5, len(vids) - 0.5, -0.5]
        )
        tick_step = max(1, len(ts) // 8)
        ax.set_xticks(ts[::tick_step])
        ax.set_xlabel('时间步', fontsize=12, fontweight='bold')
        ax.set_ylabel('车辆', fontsize=12, fontweight='bold')
        ax.set_title('预警分数热力图', fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(vids)))
        ax.set_yticklabels(vids)
        plt.colorbar(im, ax=ax).set_label('预警分数', fontsize=10)
    
    def _plot_brake_fraud(self, ax, ts):
        for vid, data in self.history['vehicles'].items():
            if any(s > 0 for s in data['brake_fraud_score']):
                ax.plot(ts, data['brake_fraud_score'], marker='o', label=vid, linewidth=2.5)
        ax.axhline(y=0.45, color='red', linestyle='--', alpha=0.7, linewidth=2, label='阈值')
        ax.set_xlabel('时间步', fontsize=12, fontweight='bold')
        ax.set_ylabel('欺诈分数', fontsize=12, fontweight='bold')
        ax.set_title('刹车欺诈检测', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    def _plot_events(self, ax):
        events = self.system.get_security_events(limit=100)
        if not events:
            ax.text(0.5, 0.5, '无安全事件', ha='center', va='center', fontsize=16, 
                   transform=ax.transAxes)
            ax.axis('off')
            return
        types = {}
        for e in events:
            types[e.event_type] = types.get(e.event_type, 0) + 1
        ax.barh(list(types.keys()), list(types.values()), 
               color=['#e74c3c', '#f39c12', '#3498db'][:len(types)])
        ax.set_xlabel('数量', fontsize=12, fontweight='bold')
        ax.set_title('安全事件', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_final_rep(self, ax):
        reps = [d['reputation'][-1] if d['reputation'] else 0.5 
               for d in self.history['vehicles'].values()]
        vids = list(self.history['vehicles'].keys())
        colors = ['#2ecc71' if r >= 0.6 else '#f39c12' if r >= 0.3 else '#e74c3c' for r in reps]
        ax.barh(vids, reps, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('信誉值', fontsize=12, fontweight='bold')
        ax.set_title('最终信誉', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_stats(self, ax):
        stats = self.system.get_statistics()
        ax.axis('off')
        text = f"""系统: {stats['system']}
交互: {stats['total_interactions']}  车辆: {stats['tracked_vehicles']}
平均: {stats['avg_reputation']:.3f}  标准差: {stats['std_reputation']:.3f}
事件: {stats['security_events']}

模块: 预测{'ON' if stats['modules']['prediction'] else 'OFF'} 刹车{'ON' if stats['modules']['brake_fraud'] else 'OFF'}"""
        if 'brake_fraud' in stats:
            text += f"\n\n刹车: {stats['brake_fraud']['total_brake_events']}事件 {stats['brake_fraud']['total_fraud_detected']}检测"
        ax.text(0.1, 0.95, text, transform=ax.transAxes, fontsize=11, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='sans-serif')
        ax.set_title('统计', fontsize=12, fontweight='bold')
    
    def _plot_weights(self, ax):
        vids = list(self.history['vehicles'].keys())
        weights = self.system.get_fusion_weights(vids, threshold=0.3)
        colors = ['#2ecc71' if w >= 0.6 else '#f39c12' if w >= 0.3 else '#e74c3c' for w in weights]
        ax.bar(range(len(vids)), weights, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('权重', fontsize=12, fontweight='bold')
        ax.set_title('融合权重', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(vids)))
        ax.set_xticklabels(vids, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
