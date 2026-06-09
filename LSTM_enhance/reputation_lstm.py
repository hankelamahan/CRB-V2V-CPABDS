"""
信誉预测LSTM模型
功能：基于历史信誉序列预测下一个信誉值，检测异常波动（持证作恶预警）
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import os

# ----------------------------- 模型定义 -----------------------------
class ReputationLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])    # 取最后一个时间步
        return self.sigmoid(out)


# ----------------------------- 预测器封装 -----------------------------
class ReputationPredictor:
    def __init__(self, model_path=None, seq_len=10, threshold=0.15, device=None):
        self.seq_len = seq_len
        self.threshold = threshold
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ReputationLSTM().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.history = {}          # vehicle_id -> deque of reputation scores

    def update_history(self, vehicle_id, reputation):
        """添加新的信誉观测"""
        if vehicle_id not in self.history:
            self.history[vehicle_id] = deque(maxlen=self.seq_len + 5)
        self.history[vehicle_id].append(reputation)

    def predict_next(self, vehicle_id):
        """预测下一个信誉值，历史不足返回None"""
        if vehicle_id not in self.history or len(self.history[vehicle_id]) < self.seq_len:
            return None
        seq = list(self.history[vehicle_id])[-self.seq_len:]
        inp = torch.tensor(seq, dtype=torch.float32).view(1, self.seq_len, 1).to(self.device)
        with torch.no_grad():
            pred = self.model(inp).item()
        return pred

    def check_anomaly(self, vehicle_id, actual_reputation):
        """返回 (是否异常, 偏差值)"""
        pred = self.predict_next(vehicle_id)
        if pred is None:
            return False, 0.0
        deviation = abs(actual_reputation - pred)
        # 持证作恶典型模式：预测高但实际骤降
        if pred > 0.6 and actual_reputation < 0.4:
            is_anomaly = deviation > (self.threshold * 0.8)
        else:
            is_anomaly = deviation > self.threshold
        return is_anomaly, deviation

    def get_early_warning_score(self, vehicle_id):
        """基于最近5步波动性计算预警分 (0~1)"""
        if vehicle_id not in self.history or len(self.history[vehicle_id]) < 5:
            return 0.0
        recent = list(self.history[vehicle_id])[-5:]
        diffs = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        if not diffs:
            return 0.0
        volatility = np.mean(diffs)
        return min(1.0, volatility * 5.0)


# ----------------------------- 训练函数 -----------------------------
def train_reputation_lstm(reputation_sequences, seq_len=10, epochs=50, save_path="reputation_lstm.pth"):
    """
    reputation_sequences: list of list, 每个子列表是一辆车的信誉历史
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReputationLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X, y = [], []
    for seq in reputation_sequences:
        if len(seq) < seq_len + 1:
            continue
        for i in range(len(seq) - seq_len):
            X.append(seq[i:i+seq_len])
            y.append(seq[i+seq_len])
    X = torch.tensor(X, dtype=torch.float32).view(-1, seq_len, 1).to(device)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

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
    # 生成模拟训练数据
    normal_seq = [0.7 + np.random.rand()*0.2 for _ in range(60)]
    dropping_seq = [0.7] + [0.7 - i*0.05 for i in range(1,20)] + [0.2 + np.random.rand()*0.1 for _ in range(40)]
    train_reputation_lstm([normal_seq, dropping_seq], epochs=30)

    # 测试预测器
    predictor = ReputationPredictor(model_path="reputation_lstm.pth", seq_len=10)
    for rep in dropping_seq[:30]:
        predictor.update_history("car_1", rep)
        is_anomaly, dev = predictor.check_anomaly("car_1", rep)
        if is_anomaly:
            print(f"异常! rep={rep:.3f}, deviation={dev:.3f}")