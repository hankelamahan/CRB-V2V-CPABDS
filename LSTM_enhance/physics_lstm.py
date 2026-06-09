"""
物理轨迹预测LSTM
功能：基于历史运动状态（位置+速度）预测下一时刻位置，计算异常分数
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import os

# ----------------------------- 模型定义 -----------------------------
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, output_dim=2):
        """
        input_dim: [x, y, vx, vy] 或 [x, y, vx, vy, ax, ay]
        output_dim: 预测下一时刻的 [x, y]
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ----------------------------- 预测器封装 -----------------------------
class PhysicsPredictor:
    def __init__(self, model_path=None, seq_len=10, input_dim=4, max_position_error=10.0, device=None):
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.max_error = max_position_error   # 用于归一化异常分数
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrajectoryLSTM(input_dim=input_dim, output_dim=2).to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.history = {}          # vehicle_id -> deque of state vectors

    def update_history(self, vehicle_id, state):
        """
        state: list or np.ndarray, 长度为 input_dim，例如 [x, y, vx, vy]
        """
        if vehicle_id not in self.history:
            self.history[vehicle_id] = deque(maxlen=self.seq_len + 5)
        self.history[vehicle_id].append(np.array(state, dtype=np.float32))

    def predict_next_position(self, vehicle_id):
        """返回预测的 [x, y] 或 None"""
        if vehicle_id not in self.history or len(self.history[vehicle_id]) < self.seq_len:
            return None
        seq = list(self.history[vehicle_id])[-self.seq_len:]  # list of arrays
        inp = np.stack(seq, axis=0)                # (seq_len, input_dim)
        inp_tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, seq_len, input_dim)
        with torch.no_grad():
            pred = self.model(inp_tensor).cpu().numpy()[0]
        return pred   # [x, y]

    def compute_anomaly_score(self, vehicle_id, actual_position):
        """返回异常分数 (0~1)，越高表示越异常"""
        pred = self.predict_next_position(vehicle_id)
        if pred is None:
            return 0.0
        error = np.linalg.norm(np.array(actual_position) - pred)
        score = min(1.0, error / self.max_error)
        return score


# ----------------------------- 训练函数 -----------------------------
def train_physics_lstm(trajectory_list, seq_len=10, input_dim=4, epochs=50, save_path="physics_lstm.pth"):
    """
    trajectory_list: list of np.ndarray, 每个数组形状 (T, input_dim)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryLSTM(input_dim=input_dim, output_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X, y = [], []
    for traj in trajectory_list:
        if len(traj) < seq_len + 1:
            continue
        for i in range(len(traj) - seq_len):
            X.append(traj[i:i+seq_len])
            y.append(traj[i+seq_len, :2])   # 真实下一时刻的 [x, y]
    if len(X) == 0:
        print("错误：没有足够长的轨迹用于训练。")
        return None
    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)   # (N, seq_len, input_dim)
    y = torch.tensor(np.array(y), dtype=torch.float32).to(device)   # (N, 2)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至 {save_path}")
    return model


# ----------------------------- 使用示例 -----------------------------
if __name__ == "__main__":
    # 生成模拟轨迹：匀速直线运动
    def gen_straight_trajectory(length=100, speed_x=0.5, speed_y=0.2):
        x = np.arange(length) * speed_x
        y = np.arange(length) * speed_y
        vx = np.full(length, speed_x)
        vy = np.full(length, speed_y)
        return np.stack([x, y, vx, vy], axis=1)

    # 生成100条正常轨迹用于训练
    train_trajs = [gen_straight_trajectory() for _ in range(80)]
    val_trajs   = [gen_straight_trajectory() for _ in range(20)]
    train_physics_lstm(train_trajs, epochs=30, save_path="physics_lstm.pth")

    # 测试预测器
    predictor = PhysicsPredictor(model_path="physics_lstm.pth", seq_len=10)
    # 模拟一段正常轨迹
    test_traj = gen_straight_trajectory(length=30)
    for t in range(len(test_traj)):
        state = test_traj[t]   # [x,y,vx,vy]
        predictor.update_history("car1", state)
        if t >= 10:
            actual_pos = state[:2]
            anomaly = predictor.compute_anomaly_score("car1", actual_pos)
            print(f"Step {t}, 异常分数: {anomaly:.4f}")