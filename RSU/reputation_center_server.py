import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, List
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LSTM_enhance.reputation_lstm import ReputationPredictor

app = FastAPI()

# ---------- 数据库初始化 ----------
def init_db():
    conn = sqlite3.connect("reputation_center.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vehicle_reputation
                 (vehicle_id TEXT PRIMARY KEY,
                  reputation REAL DEFAULT 0.5,
                  pass_count INTEGER DEFAULT 0,
                  fail_count INTEGER DEFAULT 0,
                  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reputation_lstm.pth")
predictor = ReputationPredictor(model_path=_model_path, seq_len=10, threshold=0.15)

# ---------- 数据模型 ----------
class ReportItem(BaseModel):
    vehicle_id: str
    reporter_id: str
    verification_result: bool
    phy_score: float = 0.0
    traj_score: float = 0.0
    consensus_score: float = 0.0

class ReputationUpdateRequest(BaseModel):
    reports: List[ReportItem]

# ---------- 信誉更新逻辑 ----------
def update_reputation(vehicle_id: str, is_verified: bool):
    conn = sqlite3.connect("reputation_center.db")
    c = conn.cursor()
    c.execute("SELECT reputation, pass_count, fail_count FROM vehicle_reputation WHERE vehicle_id = ?", (vehicle_id,))
    row = c.fetchone()
    if row is None:
        old_rep = 0.5
        pass_cnt = fail_cnt = 0
    else:
        old_rep, pass_cnt, fail_cnt = row

    if is_verified:
        delta = 0.05
        pass_cnt += 1
    else:
        delta = -0.1
        fail_cnt += 1

    new_rep = max(0.0, min(1.0, old_rep + delta))

    # LSTM 异常检测：先更新历史，再检查当前值是否异常
    predictor.update_history(vehicle_id, new_rep)
    is_anomaly, _ = predictor.check_anomaly(vehicle_id, new_rep)
    if is_anomaly:
        # 持证作恶惩罚：额外扣减并重新 clamp
        new_rep = max(0.0, new_rep - 0.05)

    c.execute(
        "REPLACE INTO vehicle_reputation (vehicle_id, reputation, pass_count, fail_count, last_updated) "
        "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (vehicle_id, new_rep, pass_cnt, fail_cnt),
    )
    conn.commit()
    conn.close()
    return new_rep

# ---------- API 端点 ----------
@app.post("/report_batch")
async def report_batch(req: ReputationUpdateRequest):
    """接收多车上报的批量验证结果，更新信誉值"""
    updated = {}
    for report in req.reports:
        new_rep = update_reputation(report.vehicle_id, report.verification_result)
        updated[report.vehicle_id] = new_rep
    return {"status": "ok", "updated": updated}

@app.get("/reputation/{vehicle_id}")
async def get_reputation(vehicle_id: str):
    conn = sqlite3.connect("reputation_center.db")
    c = conn.cursor()
    c.execute("SELECT reputation FROM vehicle_reputation WHERE vehicle_id = ?", (vehicle_id,))
    row = c.fetchone()
    conn.close()
    if row is None:
        return {"reputation": 0.5}
    return {"reputation": row[0]}

@app.get("/all_reputations")
async def get_all_reputations():
    conn = sqlite3.connect("reputation_center.db")
    c = conn.cursor()
    c.execute("SELECT vehicle_id, reputation FROM vehicle_reputation")
    rows = c.fetchall()
    conn.close()
    return {vid: rep for vid, rep in rows}

@app.get("/early_warning/{vehicle_id}")
async def early_warning(vehicle_id: str):
    """返回基于LSTM的早期预警分数（0~1），越高表示行为越异常"""
    score = predictor.get_early_warning_score(vehicle_id)
    return {"vehicle_id": vehicle_id, "early_warning_score": score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
