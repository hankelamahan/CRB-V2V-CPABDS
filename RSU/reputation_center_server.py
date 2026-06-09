import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, List
import json

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

# ---------- 数据模型 ----------
class ReportItem(BaseModel):
    vehicle_id: str          # 被报告的车辆ID
    reporter_id: str         # 上报车辆ID
    verification_result: bool  # True=通过验证，False=失败
    phy_score: float = 0.0  # 可选：DIVA物理分数
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

    new_rep = old_rep + delta
    new_rep = max(0.0, min(1.0, new_rep))

    c.execute("REPLACE INTO vehicle_reputation (vehicle_id, reputation, pass_count, fail_count, last_updated) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
              (vehicle_id, new_rep, pass_cnt, fail_cnt))
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)