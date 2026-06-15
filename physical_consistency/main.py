import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LSTM_enhance.physics_lstm import PhysicsPredictor

from data import DataGenerator
from imm_manager import IMMManager
from intermediate_fusion_manager import IntermediateFusionManager
from visualizer import VehicleMonitor

data_gen = DataGenerator(num_vehicles=20)
imm_manager = IMMManager()
fusion_manager = IntermediateFusionManager(lstm_imm_weight=0.4)
monitor = VehicleMonitor(save_path="results")

_physics_model = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "physics_lstm.pth",
)
physics_predictor = PhysicsPredictor(
    model_path=_physics_model,
    seq_len=10,
    input_dim=4,
    max_position_error=10.0,
)

all_scores = {}
all_reputations = {}

for t in range(100):
    print(f"\n=== Time {t} ===")
    msgs = data_gen.step(t)

    temp_votes = {}
    scores_dict = {}

    for msg in msgs:
        vid = msg["vehicle_id"]
        pos = msg["pos"]
        vel = msg["vel"]
        z = np.array([pos[0], pos[1]])

        physics_predictor.update_history(vid, [pos[0], pos[1], vel[0], vel[1]])
        lstm_anomaly = physics_predictor.compute_anomaly_score(vid, [pos[0], pos[1]])

        residual, mu = imm_manager.step(vid, z)
        scores = fusion_manager.compute_all_scores(residual, vid, vel, lstm_anomaly_score=lstm_anomaly)
        scores_dict[vid] = scores
        temp_votes[vid] = fusion_manager.get_vote(scores["fused"])

    for msg in msgs:
        vid = msg["vehicle_id"]
        pos = msg["pos"]
        vel = msg["vel"]
        z = np.array([pos[0], pos[1]])

        lstm_anomaly = physics_predictor.compute_anomaly_score(vid, [pos[0], pos[1]])

        residual, mu = imm_manager.step(vid, z)
        scores = fusion_manager.compute_all_scores(residual, vid, vel, lstm_anomaly_score=lstm_anomaly)
        scores_dict[vid] = scores

        fusion_manager.update_neighbor_votes(vid, temp_votes[vid])
        reputation = fusion_manager.update_reputation(vid, scores["fused"])
        all_reputations[vid] = reputation

        print(
            f"{vid}: "
            f"imm_phy={scores['imm_physical']:.2f}, "
            f"lstm_anom={scores['lstm_anomaly']:.2f}, "
            f"phy={scores['physical']:.2f}, "
            f"traj={scores['trajectory']:.2f}, "
            f"rsu={scores['rsu']:.2f}, "
            f"fused={scores['fused']:.2f}, "
            f"rep={reputation:.2f}, "
            f"vote={fusion_manager.get_vote(scores['fused'])}"
        )

    monitor.update(t, msgs, scores_dict, all_reputations, data_gen.attack_vehicles)

print("\n" + "=" * 60)
print("模拟完成，生成可视化报告...")
print("=" * 60)

monitor.plot_scores_evolution()
monitor.plot_trajectory_map()
monitor.plot_detection_performance()
print("\n生成动画中...")
monitor.create_animation(interval=200)
monitor.generate_report()
print("\n所有结果已保存到 'results' 文件夹！")
