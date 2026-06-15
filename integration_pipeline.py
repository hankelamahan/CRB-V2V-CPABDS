# -*- coding: utf-8 -*-
"""
全链路连调脚本 (End-to-End Integration / Joint-Debug Script)
=============================================================

把仓库里的四个核心模块 + C4(OpenCOOD) 串成一条完整的防御流水线，方便整体连调：

    ┌─────────────────────────────────────────────────────────────────────┐
    │ 1. physical_consistency  : 数据生成 + IMM 残差 + 物理/轨迹/RSU 一致性  │
    │ 2. enhanced_drambr_plus   : 多维信任向量 + 预测性信誉 → 每车信誉        │
    │ 3. reputation             : 中心化信誉服务(可选HTTP) / 本地信誉管理      │
    │ 4. overlap_field_voting   : 信誉加权 WBF 重叠视场投票融合               │
    │ 5. C4-main (OpenCOOD)     : 连通性自检(可选)                            │
    └─────────────────────────────────────────────────────────────────────┘

设计目标:
  * **能跑通**：缺少可选依赖 (ensemble_boxes / fastapi 服务) 时自动降级，不中断。
  * **可自检**：开头打印每个模块/依赖的可用性 (连调自检表)。
  * **可评估**：用 physical_consistency 内置的恶意车真值，统计整条链路的
                恶意检测 精确率/召回率/准确率，验证“低信誉车被降权/丢弃”。

用法:
    python integration_pipeline.py                 # 默认 20 车 / 60 步 / 本地信誉
    python integration_pipeline.py --vehicles 30 --steps 100
    python integration_pipeline.py --use-server     # 接 reputation_center_server (需先启动)
    python integration_pipeline.py --no-prediction  # 关闭 LSTM 预测以提速
    python integration_pipeline.py --output out.json

依赖:
    必需: numpy
    可选: ensemble_boxes (真正的 WBF; 缺失时退化为信誉加权特征级融合)
          fastapi+uvicorn (启动中心化信誉服务; 仅 --use-server 时需要)
          requests (访问信誉服务)
          torch    (predictive_reputation 的 LSTM; 缺失时自动关闭预测)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Windows 控制台默认 GBK，无法输出 emoji/部分中文；强制切到 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

# --------------------------------------------------------------------------- #
# 路径装配：各模块内部都用裸 import (from data import ...)，需把目录加入 sys.path
# --------------------------------------------------------------------------- #
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PHYS_DIR = os.path.join(ROOT_DIR, "physical_consistency")
DRAMBR_DIR = os.path.join(ROOT_DIR, "3. enhanced_drambr_plus")
C4_DIR = os.path.join(ROOT_DIR, "C4-main")

for _p in (ROOT_DIR, PHYS_DIR, DRAMBR_DIR, C4_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
# 可用性探测：每个模块/依赖单独 try，失败只记录原因，不让整个脚本崩溃
# --------------------------------------------------------------------------- #
@dataclass
class Capability:
    name: str
    ok: bool = False
    detail: str = ""


CAPS: Dict[str, Capability] = {}


def _probe(name: str, importer):
    cap = Capability(name)
    try:
        importer()
        cap.ok = True
        cap.detail = "可用"
    except Exception as e:  # noqa: BLE001 - 连调脚本需吞掉一切导入异常
        cap.ok = False
        cap.detail = f"{type(e).__name__}: {e}"
    CAPS[name] = cap
    return cap


# ---- physical_consistency ----
_phys = {}


def _imp_phys():
    from data import DataGenerator
    from imm_manager import IMMManager
    from intermediate_fusion_manager import IntermediateFusionManager
    _phys["DataGenerator"] = DataGenerator
    _phys["IMMManager"] = IMMManager
    _phys["IntermediateFusionManager"] = IntermediateFusionManager


# ---- enhanced_drambr_plus ----
_drambr = {}


def _imp_drambr():
    from enhanced_drambr_plus import EnhancedDRAMBRPlus
    _drambr["EnhancedDRAMBRPlus"] = EnhancedDRAMBRPlus


# ---- overlap_field_voting ----
_ofv = {}


def _imp_ofv():
    # OverlapFieldVotingSystem 在文件顶层 import ensemble_boxes，缺失会直接抛错。
    from overlap_field_voting import OverlapFieldVotingSystem, ReputationManager
    _ofv["OverlapFieldVotingSystem"] = OverlapFieldVotingSystem
    _ofv["ReputationManager"] = ReputationManager


def _imp_ofv_repmgr_only():
    """ensemble_boxes 缺失时，至少把不依赖它的 ReputationManager 捞出来。"""
    import importlib.util
    src = os.path.join(ROOT_DIR, "overlap_field_voting.py")
    # 直接读源码里 ReputationManager 这一段太脆弱，改为：临时屏蔽 ensemble_boxes。
    import types
    stub = types.ModuleType("ensemble_boxes")
    stub.weighted_boxes_fusion = lambda *a, **k: ([], [], [])  # 退化占位
    sys.modules.setdefault("ensemble_boxes", stub)
    spec = importlib.util.spec_from_file_location("_ofv_fallback", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ofv["OverlapFieldVotingSystem"] = mod.OverlapFieldVotingSystem
    _ofv["ReputationManager"] = mod.ReputationManager
    _ofv["_degraded"] = True


# ---- reputation (HTTP client) ----
_rep = {}


def _imp_rep_client():
    from reputation_client import ReputationClient
    from reputation_client_adapter import ReputationClientAdapter
    _rep["ReputationClient"] = ReputationClient
    _rep["ReputationClientAdapter"] = ReputationClientAdapter


# ---- C4 / OpenCOOD (仅连通性自检) ----
def _imp_opencood():
    import opencood  # noqa: F401


def run_self_check(use_server: bool) -> None:
    """连调自检：打印每个模块/依赖的可用性。"""
    _probe("physical_consistency", _imp_phys)
    _probe("enhanced_drambr_plus", _imp_drambr)

    ofv_cap = _probe("overlap_field_voting (WBF)", _imp_ofv)
    if not ofv_cap.ok:
        # 尝试降级：保留信誉加权逻辑，放弃真正的 WBF
        deg = _probe("overlap_field_voting (degraded)", _imp_ofv_repmgr_only)
        if deg.ok:
            deg.detail = "ensemble_boxes 缺失 → 退化为特征级加权融合"

    _probe("opencood (C4-main)", _imp_opencood)

    if use_server:
        _probe("reputation_client", _imp_rep_client)

    print("\n" + "=" * 68)
    print("  连调自检 (Module / Dependency Self-Check)")
    print("=" * 68)
    for cap in CAPS.values():
        mark = "✅" if cap.ok else "❌"
        print(f"  {mark}  {cap.name:<34} {cap.detail}")
    print("=" * 68 + "\n")


# --------------------------------------------------------------------------- #
# 信誉后端抽象：统一 HTTP 中心化服务 与 本地信誉管理 两种实现
# --------------------------------------------------------------------------- #
class ReputationBackend:
    """统一接口: get(vid) / set(vid, score) / all()."""

    def get(self, vid: str) -> float: ...
    def set(self, vid: str, score: float) -> None: ...
    def all(self) -> Dict[str, float]: ...
    def close(self) -> None: ...


class LocalReputationBackend(ReputationBackend):
    """基于 overlap_field_voting.ReputationManager 的本地信誉表 (无网络)。"""

    def __init__(self, default=0.5, min_rep=0.3):
        RM = _ofv["ReputationManager"]
        self._mgr = RM(default_reputation=default, min_reputation=min_rep, update_rate=0.1)

    def get(self, vid):
        return self._mgr.get_trust_score(vid)

    def set(self, vid, score):
        self._mgr.set_trust_score(vid, score)

    def all(self):
        return self._mgr.reputation_cache.copy()

    def close(self):
        pass


class ServerReputationBackend(ReputationBackend):
    """基于 reputation_client 的中心化信誉服务后端 (HTTP)。"""

    def __init__(self, server_url="http://localhost:8888", ego_id="ego"):
        Client = _rep["ReputationClient"]
        Adapter = _rep["ReputationClientAdapter"]
        self._client = Client(server_url=server_url)
        self._adapter = Adapter(client=self._client, ego_id=ego_id)
        self._local_mirror: Dict[str, float] = {}

    def get(self, vid):
        return self._adapter.get_reputation(vid)

    def set(self, vid, score):
        # 中心化服务通过上报(report_verification)间接更新；这里同时维护本地镜像
        self._local_mirror[vid] = score
        try:
            self._client.report_verification(
                reporter_id="ego", target_id=vid, result=(score >= 0.5),
                phy_score=float(score),
            )
        except Exception:
            pass

    def all(self):
        return dict(self._local_mirror)

    def close(self):
        try:
            self._client.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# 流水线
# --------------------------------------------------------------------------- #
@dataclass
class StageStatus:
    ran: bool = False
    error: str = ""
    summary: Dict = field(default_factory=dict)


class IntegrationPipeline:
    """把四个模块按真值流串起来，逐步运行并采集每车最终信誉。"""

    def __init__(self, args):
        self.args = args
        self.stages: Dict[str, StageStatus] = {}

        # --- 1. physical_consistency ---
        IMMManager = _phys["IMMManager"]
        IFM = _phys["IntermediateFusionManager"]
        # 数据源: 真实数据集 (--dataset) 或合成 DataGenerator
        if args.dataset:
            from dataset_adapter import OPV2VDatasetSource
            self.data_gen = OPV2VDatasetSource(args.dataset)
            self.using_dataset = True
            if args.steps <= 0 or args.steps > self.data_gen.num_frames:
                args.steps = self.data_gen.num_frames
            print(f"📦 数据集: {self.data_gen.episode_dir}")
            print(f"   CAV={self.data_gen.cav_ids} | ego={self.data_gen.ego_cav_id} | "
                  f"攻击={self.data_gen.attack_label} | 真值恶意车={sorted(self.data_gen.attack_vehicles)} | "
                  f"帧数={self.data_gen.num_frames}")
        else:
            DataGenerator = _phys["DataGenerator"]
            self.data_gen = DataGenerator(num_vehicles=args.vehicles)
            self.using_dataset = False
        self.imm = IMMManager()
        self.fusion_mgr = IFM()
        self.attack_vehicles = set(self.data_gen.attack_vehicles)

        # --- 2. enhanced_drambr_plus (可选预测) ---
        self.drambr = self._build_drambr(enable_prediction=not args.no_prediction)
        all_ids = [v.vid for v in self.data_gen.vehicles]
        self.drambr.initialize_reputations(all_ids, initial_value=0.5)

        # --- 3. reputation backend ---
        self.rep_backend = self._build_rep_backend()

        # --- 4. overlap_field_voting ---
        OFVS = _ofv["OverlapFieldVotingSystem"]
        # min_reputation=0.0 让物理异常的幽灵车能跌破 0.3 排除线 (方案一/四)
        self.voter = OFVS(iou_thr=0.4, default_reputation=0.5, min_reputation=0.0)

        # 上一帧位置缓存 + 异常记忆，用于估计 position_error / velocity_error
        self._last_pos: Dict[str, np.ndarray] = {}
        self._last_vel: Dict[str, np.ndarray] = {}
        self._err_memory: Dict[str, float] = {}

    # ----- 构建器 -----
    def _build_drambr(self, enable_prediction: bool):
        EnhancedDRAMBRPlus = _drambr["EnhancedDRAMBRPlus"]
        try:
            return EnhancedDRAMBRPlus(
                enable_prediction=enable_prediction,
                enable_brake_fraud_detection=False,  # 本连调不构造刹车事件
                enable_vehicle_prefilter=False,      # 关闭预筛，确保每车都过完整信任更新
            )
        except Exception as e:  # 预测依赖 torch，失败则退化
            print(f"⚠️  预测性信誉初始化失败 ({e})，自动关闭预测重试。")
            return EnhancedDRAMBRPlus(
                enable_prediction=False,
                enable_brake_fraud_detection=False,
                enable_vehicle_prefilter=False,
            )

    def _build_rep_backend(self) -> ReputationBackend:
        if self.args.use_server and CAPS.get("reputation_client", Capability("")).ok:
            if self._server_reachable(self.args.server_url):
                try:
                    be = ServerReputationBackend(server_url=self.args.server_url)
                    print(f"🌐 使用中心化信誉服务: {self.args.server_url}")
                    return be
                except Exception as e:
                    print(f"⚠️  初始化信誉服务客户端失败 ({e})，回退到本地信誉表。")
            else:
                print(f"⚠️  信誉服务 {self.args.server_url} 不可达 "
                      f"(请先运行 reputation_center_server.py)，回退到本地信誉表。")
        elif self.args.use_server:
            print("⚠️  reputation_client 不可用，回退到本地信誉表。")
        return LocalReputationBackend()

    @staticmethod
    def _server_reachable(server_url: str, timeout: float = 0.5) -> bool:
        """真正发一次 HTTP 探活，避免 ReputationClient 吞异常导致的假在线。"""
        try:
            import requests
            resp = requests.get(f"{server_url.rstrip('/')}/all_reputations", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    # ----- 单步：physical_consistency → observation -----
    def _msg_to_observation(self, msg) -> Dict:
        """用 IMM 残差 + 帧间位置/速度跳变估计 position_error / velocity_error。"""
        vid = msg["vehicle_id"]
        z = np.asarray(msg["pos"][:2], dtype=float)
        residual, _mu = self.imm.step(vid, z)

        last_p = self._last_pos.get(vid)
        last_v = self._last_vel.get(vid)
        cur_v = np.asarray(msg["vel"][:2], dtype=float)

        # 位置误差：IMM 残差归一 + 帧间跳变 (幽灵车跳变 ~10-20m → 高误差)
        pos_jump = float(np.linalg.norm(z - last_p)) if last_p is not None else 0.0
        inst_pos_err = residual / 6.0 + max(0.0, pos_jump - 0.6) / 6.0
        # 真实数据集: 叠加“上报不一致度”(幽灵车瞬移) 作为物理一致性证据
        inst_pos_err = max(inst_pos_err, float(msg.get("report_inconsistency", 0.0)))
        # 异常记忆：一次跳变的影响衰减式保留数帧，避免被随后的“正常”帧瞬间洗白
        prev_mem = self._err_memory.get(vid, 0.0)
        mem = max(inst_pos_err, prev_mem * 0.7)
        self._err_memory[vid] = mem
        position_error = float(np.clip(mem, 0.0, 1.0))
        # 速度误差：速度变化幅度
        vel_jump = float(np.linalg.norm(cur_v - last_v)) if last_v is not None else 0.0
        velocity_error = float(np.clip(vel_jump / 3.0, 0.0, 1.0))

        self._last_pos[vid] = z
        self._last_vel[vid] = cur_v

        return {
            "position_error": position_error,
            "velocity_error": velocity_error,
            "timestamp_error": 0.0,
            "message_frequency": 10.0,
            "_residual": float(residual),
            "_pos": z,
            "_vel": cur_v,
        }

    # ----- 主循环 -----
    def run(self):
        n_steps = self.args.steps
        n_veh = len(self.data_gen.vehicles) if self.using_dataset else self.args.vehicles
        src = "真实数据集" if self.using_dataset else "合成数据"
        print(f"▶️  开始连调[{src}]: {n_veh} 车 / {n_steps} 步 / "
              f"恶意车 {len(self.attack_vehicles)} 辆 {sorted(self.attack_vehicles)}\n")

        st_phys = StageStatus(ran=True)
        st_drambr = StageStatus(ran=True)
        last_observations: Dict[str, Dict] = {}

        for t in range(n_steps):
            msgs = self.data_gen.step(t)

            # 收集本帧所有 LMDM consistency 作为邻居报告
            obs_map = {m["vehicle_id"]: self._msg_to_observation(m) for m in msgs}
            neighbor_pool = [
                1.0 - (o["position_error"] + o["velocity_error"]) / 2.0
                for o in obs_map.values()
            ]

            # ---- overlap_field_voting 物理一致性预检 (方案一/四) ----
            # 基于车辆ID维护历史轨迹，瞬移/速度异常即时强扣信誉。
            dt = 0.05 if self.using_dataset else 0.1
            kin = {m["vehicle_id"]: {"position": m["pos"], "velocity": m["vel"]}
                   for m in msgs}
            try:
                self.voter.precheck_physical(kin, dt=dt)
                # 数据集: CAV 自身轨迹平滑，但若它上报了瞬移的幽灵目标
                # (report_inconsistency 高)，则把“上报幽灵”的责任记到该 CAV 头上
                if self.using_dataset:
                    rm = self.voter.reputation_manager
                    for m in msgs:
                        ri = float(m.get("report_inconsistency", 0.0))
                        if ri > 0.3:
                            rm._penalize(m["vehicle_id"], rm.physical_penalty * ri)
            except Exception as e:  # noqa: BLE001
                st_phys.error = st_phys.error or f"physcheck: {type(e).__name__}: {e}"

            for m in msgs:
                vid = m["vehicle_id"]
                obs = obs_map[vid]

                # ---- physical_consistency: 中间层多源融合分数 ----
                try:
                    scores = self.fusion_mgr.compute_all_scores(
                        residual=obs["_residual"], vid=vid, vel=obs["_vel"]
                    )
                    self.fusion_mgr.update_neighbor_votes(
                        vid, self.fusion_mgr.get_vote(scores["fused"])
                    )
                    self.fusion_mgr.update_reputation(vid, scores["fused"])
                except Exception as e:  # noqa: BLE001
                    st_phys.error = st_phys.error or f"{type(e).__name__}: {e}"

                # ---- enhanced_drambr_plus: 多维信任 → 信誉 ----
                try:
                    # 用其它车的一致性作为间接信任来源
                    neighbors = [r for r in neighbor_pool][:8]
                    self.drambr.process_vehicle_observation(
                        vehicle_id=vid,
                        observation=obs,
                        neighbor_reports=neighbors,
                    )
                except Exception as e:  # noqa: BLE001
                    st_drambr.error = st_drambr.error or f"{type(e).__name__}: {e}"

            last_observations = obs_map

            if (t + 1) % max(1, n_steps // 5) == 0:
                avg = np.mean(list(self.drambr.get_all_reputations().values()) or [0.5])
                print(f"  step {t + 1:>3}/{n_steps}  平均信誉={avg:.3f}")

        st_phys.summary = {"final_reputation_mean":
                           float(np.mean(list(self.fusion_mgr.reputation.values()) or [0.5]))}
        st_drambr.summary = self.drambr.get_statistics()
        self.stages["physical_consistency"] = st_phys
        self.stages["enhanced_drambr_plus"] = st_drambr

        # ---- 3. 信誉融合 + 同步到信誉后端 ----
        # 三条独立的信誉信号:
        #   drambr_reps : enhanced_drambr_plus 的多维信任最终分
        #   phys_reps   : physical_consistency 中间层融合分演化出的信誉
        #   ofv_reps    : overlap_field_voting 基于ID物理一致性的信誉 (方案一/四)
        # 组合策略: mean(drambr, phys) 给平滑排序，再对 ofv 取 min ——
        #   任一“硬”物理异常检测器抓到瞬移幽灵车，就能一票把综合信誉拉到很低 (安全 AND)。
        drambr_reps = self.drambr.get_all_reputations()
        phys_reps = dict(self.fusion_mgr.reputation)
        ofv_reps = self.voter.get_all_reputations()
        all_ids = [v.vid for v in self.data_gen.vehicles]
        combined_reps = {
            vid: float(min(
                np.mean([drambr_reps.get(vid, 0.5), phys_reps.get(vid, 0.5)]),
                ofv_reps.get(vid, 0.5),
            ))
            for vid in all_ids
        }
        self.combined_reps = combined_reps
        self.drambr_reps = drambr_reps
        self.phys_reps = phys_reps
        self.ofv_reps = ofv_reps

        st_rep = StageStatus(ran=True)
        try:
            for vid, rep in combined_reps.items():
                self.rep_backend.set(vid, float(rep))
                self.voter.set_reputation(vid, float(rep))
            st_rep.summary = {
                "vehicles": len(combined_reps),
                "backend": type(self.rep_backend).__name__,
                "drambr_mean": round(float(np.mean(list(drambr_reps.values()) or [0.5])), 4),
                "physical_mean": round(float(np.mean(list(phys_reps.values()) or [0.5])), 4),
                "combined_mean": round(float(np.mean(list(combined_reps.values()) or [0.5])), 4),
            }
        except Exception as e:  # noqa: BLE001
            st_rep.error = f"{type(e).__name__}: {e}"
        self.stages["reputation"] = st_rep

        # ---- 4. overlap_field_voting: 信誉加权 WBF 融合 ----
        self.stages["overlap_field_voting"] = self._run_fusion(last_observations, combined_reps)

        # ---- 评估 ----
        evaluation = self._evaluate(combined_reps)
        return combined_reps, evaluation

    def _run_fusion(self, observations: Dict[str, Dict], reps: Dict[str, float]) -> StageStatus:
        """用最后一帧位置构造检测框，做信誉加权融合，统计恶意车贡献被抑制的程度。"""
        st = StageStatus(ran=True)
        try:
            detections = {}
            for vid, obs in observations.items():
                p = obs["_pos"]
                # 归一化到 [0,1] 的小框；恶意车的框已被 data 层注入跳变 → 自然离群
                cx, cy = (p[0] % 50) / 50.0, (p[1] % 50) / 50.0
                half = 0.03
                box = [max(0, cx - half), max(0, cy - half),
                       min(1, cx + half), min(1, cy + half)]
                detections[vid] = {
                    "boxes": np.array([box], dtype=float),
                    "scores": np.array([0.9], dtype=float),
                    "labels": np.array([1], dtype=int),
                }

            degraded = _ofv.get("_degraded", False)
            if degraded:
                # 退化路径：特征级加权融合 (不依赖 ensemble_boxes)
                feats = [np.array(d["boxes"][0]) for d in detections.values()]
                trust = [reps.get(v, 0.5) for v in detections]
                w = np.array([t if t >= 0.3 else 0.0 for t in trust])
                wsum = w.sum() or 1.0
                fused = sum(f * (wi / wsum) for f, wi in zip(feats, w))
                st.summary = {"mode": "degraded(feature-level)",
                              "n_inputs": len(feats),
                              "n_excluded_lowtrust": int((np.array(trust) < 0.3).sum()),
                              "fused_box": [round(float(x), 4) for x in fused]}
            else:
                fused_boxes, fused_scores, fused_labels = self.voter.fuse(detections)
                consistency = self.voter.update_reputations(
                    (fused_boxes, fused_scores, fused_labels), detections
                )
                excluded = [v for v in detections if reps.get(v, 0.5) < 0.3]
                st.summary = {"mode": "WBF",
                              "n_inputs": len(detections),
                              "n_fused_boxes": int(len(fused_boxes)),
                              "n_excluded_lowtrust": len(excluded),
                              "excluded_vehicles": sorted(excluded)}
        except Exception as e:  # noqa: BLE001
            st.error = f"{type(e).__name__}: {e}"
            st.summary = {"trace": traceback.format_exc(limit=2)}
        return st

    # ----- 评估：恶意检测指标 -----
    def _evaluate(self, reps: Dict[str, float], k_std: float = 1.0) -> Dict:
        all_ids = [v.vid for v in self.data_gen.vehicles]
        gt_mal = self.attack_vehicles
        rep_vals = np.array([reps.get(v, 0.5) for v in all_ids], dtype=float)

        # 自适应阈值：均值 - k*标准差，对整体信誉漂移不敏感
        mu, sigma = float(rep_vals.mean()), float(rep_vals.std())
        threshold = mu - k_std * sigma
        pred_mal = {vid for vid in all_ids if reps.get(vid, 0.5) < threshold}

        tp = len(pred_mal & gt_mal)
        fp = len(pred_mal - gt_mal)
        fn = len(gt_mal - pred_mal)
        tn = len(set(all_ids) - pred_mal - gt_mal)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / len(all_ids) if all_ids else 0.0

        normal_reps = [reps.get(v, 0.5) for v in all_ids if v not in gt_mal]
        mal_reps = [reps.get(v, 0.5) for v in all_ids if v in gt_mal]

        # 排名式指标 (不依赖绝对阈值): 信誉最低的 K 辆里命中多少真值恶意车
        k = len(gt_mal)
        ranked = sorted(all_ids, key=lambda v: reps.get(v, 0.5))
        bottom_k = set(ranked[:k]) if k else set()
        rank_recall_at_k = len(bottom_k & gt_mal) / k if k else 0.0

        # 分离度: 正常均值与恶意均值之间相对标准差的间隔
        sep_margin = ((np.mean(normal_reps) if normal_reps else 0.0)
                      - (np.mean(mal_reps) if mal_reps else 0.0))
        separation = sep_margin / sigma if sigma > 1e-9 else 0.0

        return {
            "threshold_mode": f"adaptive(mean-{k_std}*std)",
            "threshold": round(threshold, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "rank_recall_at_k": round(rank_recall_at_k, 4),
            "separation_in_std": round(float(separation), 3),
            "normal_rep_mean": round(float(np.mean(normal_reps)) if normal_reps else 0.0, 4),
            "malicious_rep_mean": round(float(np.mean(mal_reps)) if mal_reps else 0.0, 4),
            "ground_truth_malicious": sorted(gt_mal),
            "predicted_malicious": sorted(pred_mal),
            "bottom_k_by_reputation": sorted(bottom_k),
        }

    def close(self):
        self.rep_backend.close()


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def print_report(pipeline: "IntegrationPipeline", reps, evaluation):
    print("\n" + "=" * 68)
    print("  连调阶段结果 (Stage Results)")
    print("=" * 68)
    for name, st in pipeline.stages.items():
        mark = "✅" if (st.ran and not st.error) else ("⚠️" if st.ran else "❌")
        print(f"  {mark}  {name}")
        if st.error:
            print(f"        error: {st.error}")
        for k, v in st.summary.items():
            print(f"        {k}: {v}")

    print("\n" + "=" * 68)
    print("  恶意检测评估 (整条链路综合信誉)")
    print("=" * 68)
    print(f"  阈值策略        : {evaluation['threshold_mode']} = {evaluation['threshold']}")
    print(f"  正常车平均信誉  : {evaluation['normal_rep_mean']}")
    print(f"  恶意车平均信誉  : {evaluation['malicious_rep_mean']}")
    print(f"  分离度(σ)       : {evaluation['separation_in_std']}  (>1 表示两类可分)")
    print(f"  TP/FP/FN/TN     : "
          f"{evaluation['tp']}/{evaluation['fp']}/{evaluation['fn']}/{evaluation['tn']}")
    print(f"  Precision       : {evaluation['precision']}")
    print(f"  Recall          : {evaluation['recall']}")
    print(f"  F1              : {evaluation['f1']}")
    print(f"  Accuracy        : {evaluation['accuracy']}")
    print(f"  Rank-Recall@K   : {evaluation['rank_recall_at_k']}  "
          f"(信誉最低 {len(evaluation['ground_truth_malicious'])} 辆中的命中率)")
    print(f"  真值恶意车       : {evaluation['ground_truth_malicious']}")
    print(f"  预测恶意车       : {evaluation['predicted_malicious']}")
    print(f"  信誉最低 K 辆    : {evaluation['bottom_k_by_reputation']}")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="CRB-V2V 全链路连调脚本 (physical_consistency + DRAMBR+ + reputation + WBF)"
    )
    parser.add_argument("--dataset", default="",
                        help="OPV2V/CARLA 数据集 episode 目录 (如 ./episode_0000)；"
                             "不指定则用合成数据")
    parser.add_argument("--vehicles", type=int, default=20,
                        help="合成数据的车辆数量 (--dataset 时忽略)")
    parser.add_argument("--steps", type=int, default=60,
                        help="仿真/重放步数 (--dataset 时若超过帧数会自动截断)")
    parser.add_argument("--no-prediction", action="store_true", help="关闭 LSTM 预测性信誉")
    parser.add_argument("--use-server", action="store_true",
                        help="使用中心化信誉服务 (需先启动 reputation_center_server.py)")
    parser.add_argument("--server-url", default="http://localhost:8888")
    parser.add_argument("--output", default="", help="把结果写入 JSON 文件")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (复现实验)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    print("=" * 68)
    print("  CRB-V2V-CPABDS 全链路连调 (End-to-End Integration)")
    print("=" * 68)

    # 1) 自检
    run_self_check(use_server=args.use_server)

    # 关键模块缺失则直接退出
    for required in ("physical_consistency", "enhanced_drambr_plus"):
        if not CAPS[required].ok:
            print(f"❌ 关键模块 {required} 不可用，无法连调: {CAPS[required].detail}")
            sys.exit(1)
    if not _ofv:
        print("❌ overlap_field_voting 完全不可用 (含降级路径)，无法连调。")
        sys.exit(1)

    # 2) 运行流水线
    t0 = time.time()
    pipeline = IntegrationPipeline(args)
    try:
        reps, evaluation = pipeline.run()
    finally:
        pipeline.close()
    elapsed = time.time() - t0

    # 3) 报告
    print_report(pipeline, reps, evaluation)
    print(f"\n⏱️  连调总耗时: {elapsed:.2f}s")

    # 4) 落盘
    if args.output:
        payload = {
            "config": vars(args),
            "capabilities": {k: {"ok": c.ok, "detail": c.detail} for k, c in CAPS.items()},
            "stages": {k: {"ran": s.ran, "error": s.error, "summary": s.summary}
                       for k, s in pipeline.stages.items()},
            "final_reputations": {k: round(float(v), 4) for k, v in reps.items()},
            "evaluation": evaluation,
            "elapsed_sec": round(elapsed, 2),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已写入: {args.output}")

    print("\n🎉 连调完成！")


if __name__ == "__main__":
    main()
