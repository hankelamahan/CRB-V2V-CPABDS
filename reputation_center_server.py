import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Dict, List, Optional
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
    vehicle_id: str
    reporter_id: str
    verification_result: bool
    phy_score: float = 0.0
    traj_score: float = 0.0
    consensus_score: float = 0.0

class ReputationUpdateRequest(BaseModel):
    reports: List[ReportItem]

class FusedBoxReport(BaseModel):
    reporter_id: str
    fused_boxes: List[List[float]]
    fused_scores: List[float]
    fused_labels: List[int]
    cav_detections: Dict[str, Dict[str, List]]
    timestamp: Optional[float] = None

class BatchReputationRequest(BaseModel):
    vehicle_ids: List[str]

# ---------- 信誉更新逻辑 ----------
def update_reputation(vehicle_id: str, is_verified: bool, delta_override: Optional[float] = None):
    conn = sqlite3.connect("reputation_center.db")
    c = conn.cursor()
    c.execute("SELECT reputation, pass_count, fail_count FROM vehicle_reputation WHERE vehicle_id = ?", (vehicle_id,))
    row = c.fetchone()
    if row is None:
        old_rep = 0.5
        pass_cnt = fail_cnt = 0
    else:
        old_rep, pass_cnt, fail_cnt = row

    if delta_override is not None:
        delta = delta_override
        if delta > 0:
            pass_cnt += 1
        else:
            fail_cnt += 1
    elif is_verified:
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

def _compute_detection_consistency(fused_boxes, cav_detections):
    """根据融合结果和各车检测计算一致性评分"""
    consistency_scores = {}
    if not fused_boxes or not cav_detections:
        return consistency_scores
    
    for cav_id, detection in cav_detections.items():
        cav_boxes = detection.get('boxes', [])
        if not cav_boxes:
            consistency_scores[cav_id] = 0.5
            continue
        
        matched = 0
        for fused_box in fused_boxes:
            for cav_box in cav_boxes:
                iou = _compute_iou(fused_box[:4], cav_box[:4])
                if iou > 0.5:
                    matched += 1
                    break
        
        consistency = matched / max(len(cav_boxes), len(fused_boxes))
        consistency_scores[cav_id] = consistency
    
    return consistency_scores

def _compute_iou(box1, box2):
    """计算两个2D边界框的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

# ---------- API 端点 ----------
@app.post("/report_batch")
async def report_batch(req: ReputationUpdateRequest):
    """接收多车上报的批量验证结果，更新信誉值"""
    updated = {}
    for report in req.reports:
        new_rep = update_reputation(report.vehicle_id, report.verification_result)
        updated[report.vehicle_id] = new_rep
    return {"status": "ok", "updated": updated}

@app.post("/report_fused_boxes")
async def report_fused_boxes(report: FusedBoxReport):
    """接收融合检测结果，根据一致性更新各车信誉值"""
    consistency_scores = _compute_detection_consistency(
        report.fused_boxes, 
        report.cav_detections
    )
    
    updated = {}
    for cav_id, consistency in consistency_scores.items():
        delta = (consistency - 0.5) * 0.1
        new_rep = update_reputation(
            cav_id, 
            is_verified=(consistency >= 0.5),
            delta_override=delta
        )
        updated[cav_id] = new_rep
    
    return {
        "status": "ok", 
        "updated": updated,
        "consistency_scores": consistency_scores
    }

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

@app.post("/batch_reputations")
async def batch_reputations(req: BatchReputationRequest):
    """批量查询多个车辆的信誉值"""
    conn = sqlite3.connect("reputation_center.db")
    c = conn.cursor()
    
    reputations = {}
    for vehicle_id in req.vehicle_ids:
        c.execute("SELECT reputation FROM vehicle_reputation WHERE vehicle_id = ?", (vehicle_id,))
        row = c.fetchone()
        reputations[vehicle_id] = row[0] if row else 0.5
    
    conn.close()
    return {"reputations": reputations}

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