"""
LSTM 增强模块 — 专项可视化仪表盘
运行: python LSTM_enhance/lstm_dashboard.py
输出: LSTM_enhance/dashboard_output/*.png  (6 张图)
不修改任何现有文件。
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "physical_consistency"))

from data import DataGenerator
from imm_manager import IMMManager
from intermediate_fusion_manager import IntermediateFusionManager

try:
    from LSTM_enhance.physics_lstm import PhysicsPredictor as _RealPredictor
    _TORCH_OK = True
except Exception:
    _RealPredictor = None
    _TORCH_OK = False

class _NumpyStub:
    def __init__(self, **_):
        self._h = {}
    def update_history(self, vid, state):
        self._h.setdefault(vid, []).append(state[:2])
    def compute_anomaly_score(self, vid, pos):
        pts = self._h.get(vid, [])
        if len(pts) < 3:
            return 0.0
        mean_pos = np.mean(pts[-5:], axis=0)
        return float(min(1.0, np.linalg.norm(np.array(pos) - mean_pos) / 10.0))

def _make_predictor(model_path):
    if _TORCH_OK:
        return _RealPredictor(model_path=model_path, seq_len=10, input_dim=4, max_position_error=10.0)
    return _NumpyStub()


_rc = {
    "font.family":        ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor":   "#0d1117",
    "axes.facecolor":     "#161b22",
    "axes.edgecolor":     "#30363d",
    "axes.labelcolor":    "#e6edf3",
    "xtick.color":        "#8b949e",
    "ytick.color":        "#8b949e",
    "text.color":         "#e6edf3",
    "grid.color":         "#21262d",
    "grid.linewidth":     0.6,
    "legend.facecolor":   "#161b22",
    "legend.edgecolor":   "#30363d",
}
plt.rcParams.update(_rc)
try:
    plt.rcParams["legend.labelcolor"] = "#e6edf3"
except KeyError:
    pass

ATTACK_COLOR = "#f85149"
NORMAL_COLOR = "#3fb950"
IMM_COLOR    = "#58a6ff"
LSTM_COLOR   = "#d2a8ff"
FUSED_COLOR  = "#ffa657"
THRESH_COLOR = "#f0f6fc"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_output")
os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  ->  {path}")


# ── simulation ───────────────────────────────────────────────────────────────

def run_simulation(num_vehicles=20, steps=100, lstm_imm_weight=0.4, seed=42):
    np.random.seed(seed)
    data_gen   = DataGenerator(num_vehicles=num_vehicles)
    imm_mgr    = IMMManager()
    fusion_mgr = IntermediateFusionManager(lstm_imm_weight=lstm_imm_weight)
    _pmodel = os.path.join(_ROOT, "physics_lstm.pth")
    physics = _make_predictor(_pmodel)
    all_vids = [v.vid for v in data_gen.vehicles]
    KEYS = ("imm_physical", "lstm_anomaly", "physical",
            "trajectory", "rsu", "fused", "reputation", "positions")
    hist = {k: {v: [] for v in all_vids} for k in KEYS}

    for t in range(steps):
        msgs = data_gen.step(t)
        temp_votes = {}
        for msg in msgs:
            vid, pos, vel = msg["vehicle_id"], msg["pos"], msg["vel"]
            physics.update_history(vid, [pos[0], pos[1], vel[0], vel[1]])
            la     = physics.compute_anomaly_score(vid, [pos[0], pos[1]])
            res, _ = imm_mgr.step(vid, np.array([pos[0], pos[1]]))
            sc     = fusion_mgr.compute_all_scores(res, vid, vel, lstm_anomaly_score=la)
            temp_votes[vid] = fusion_mgr.get_vote(sc["fused"])
        for msg in msgs:
            vid, pos, vel = msg["vehicle_id"], msg["pos"], msg["vel"]
            la     = physics.compute_anomaly_score(vid, [pos[0], pos[1]])
            res, _ = imm_mgr.step(vid, np.array([pos[0], pos[1]]))
            sc     = fusion_mgr.compute_all_scores(res, vid, vel, lstm_anomaly_score=la)
            fusion_mgr.update_neighbor_votes(vid, temp_votes[vid])
            rep    = fusion_mgr.update_reputation(vid, sc["fused"])
            for k in ("imm_physical", "lstm_anomaly", "physical",
                      "trajectory", "rsu", "fused"):
                hist[k][vid].append(sc[k])
            hist["reputation"][vid].append(rep)
            hist["positions"][vid].append((pos[0], pos[1]))

    return hist, data_gen.attack_vehicles, all_vids


# ── plot 1: IMM vs LSTM physical score time series ───────────────────────────

def plot_imm_vs_lstm(hist, attack_vehicles, all_vids, steps):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("物理一致性分数：IMM 基线  vs  IMM+LSTM 融合",
                 fontsize=14, fontweight="bold", y=0.99)
    time = list(range(steps))

    for ax, key, base_col, title in [
        (axes[0], "imm_physical", IMM_COLOR,  "IMM 原始物理分数"),
        (axes[1], "physical",     FUSED_COLOR, "IMM+LSTM 融合后物理分数"),
    ]:
        for vid in all_vids:
            is_atk = vid in attack_vehicles
            ax.plot(time, hist[key][vid],
                    color=ATTACK_COLOR if is_atk else base_col,
                    alpha=0.90 if is_atk else 0.28,
                    linewidth=2.0 if is_atk else 0.7,
                    zorder=3 if is_atk else 1)

        if key == "physical":
            for vid in attack_vehicles:
                anom   = np.array(hist["lstm_anomaly"][vid])
                spikes = np.where(anom > 0.5)[0]
                if len(spikes):
                    ax.scatter(np.array(time)[spikes],
                               np.array(hist[key][vid])[spikes],
                               marker="v", color=LSTM_COLOR, s=35,
                               zorder=5, label="_nolegend_")

        ax.axhline(0.6, color=THRESH_COLOR, linestyle="--", linewidth=1.0, alpha=0.55)
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel("分数", fontsize=10)
        ax.set_title(title, fontsize=11, pad=4)
        ax.grid(True)

    axes[1].set_xlabel("时间步", fontsize=10)
    fig.legend(handles=[
        Line2D([0],[0], color=ATTACK_COLOR, lw=2,  label="攻击车辆"),
        Line2D([0],[0], color=IMM_COLOR,    lw=1,  alpha=0.5, label="正常车辆"),
        Line2D([0],[0], color=THRESH_COLOR, lw=1,  linestyle="--", label="可信阈值 0.6"),
        Line2D([0],[0], color=LSTM_COLOR,   lw=0,  marker="v", markersize=7,
               label="LSTM 异常触发 (>0.5)"),
    ], loc="lower center", ncol=4, fontsize=9,
       bbox_to_anchor=(0.5, 0.005), framealpha=0.3)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "01_imm_vs_lstm.png")


# ── plot 2: reputation heatmap ────────────────────────────────────────────────

def plot_reputation_heatmap(hist, attack_vehicles, all_vids, steps):
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle("车辆信誉值演化热力图  (红=低  绿=高)", fontsize=14, fontweight="bold")

    sorted_vids = sorted(all_vids,
                         key=lambda v: (v not in attack_vehicles,
                                        hist["reputation"][v][-1]))
    matrix = np.array([hist["reputation"][v] for v in sorted_vids])

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0,
                   extent=[-0.5, steps - 0.5, len(sorted_vids) - 0.5, -0.5])

    ax.set_yticks(range(len(sorted_vids)))
    ax.set_yticklabels(
        [("! " if v in attack_vehicles else "  ") + v for v in sorted_vids],
        fontsize=7.5,
    )

    for i, v in enumerate(sorted_vids):
        if v in attack_vehicles:
            ax.add_patch(plt.Rectangle(
                (-0.5, i - 0.5), steps, 1,
                fill=False, edgecolor=ATTACK_COLOR, linewidth=1.3, zorder=4,
            ))

    n_normal = sum(v not in attack_vehicles for v in sorted_vids)
    ax.axhline(n_normal - 0.5, color=THRESH_COLOR, lw=1.5, linestyle="--", alpha=0.45)

    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label("信誉值", fontsize=10)
    ax.set_xlabel("时间步", fontsize=10)
    ax.set_title("! 标注行为攻击车辆，虚线分隔正常 / 攻击区域", fontsize=9, pad=6)
    plt.tight_layout()
    _save(fig, "02_reputation_heatmap.png")


# ── plot 3: score decomposition stacked area ─────────────────────────────────

def plot_score_decomposition(hist, attack_vehicles, all_vids, steps):
    atk_v = min(
        (v for v in all_vids if v in attack_vehicles),
        key=lambda v: hist["reputation"][v][-1], default=None,
    )
    nrm_v = max(
        (v for v in all_vids if v not in attack_vehicles),
        key=lambda v: hist["reputation"][v][-1], default=None,
    )
    if atk_v is None or nrm_v is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("融合分数来源分解（物理 / 轨迹 / RSU 贡献）",
                 fontsize=14, fontweight="bold")
    time = np.arange(steps)
    W = {"physical": 0.4, "trajectory": 0.3, "rsu": 0.3}

    for ax, vid, rep_col, title in [
        (axes[0], atk_v, ATTACK_COLOR, f"攻击车辆  {atk_v}"),
        (axes[1], nrm_v, NORMAL_COLOR, f"正常车辆  {nrm_v}"),
    ]:
        phy  = np.array(hist["physical"][vid])   * W["physical"]
        traj = np.array(hist["trajectory"][vid]) * W["trajectory"]
        rsu  = np.array(hist["rsu"][vid])        * W["rsu"]

        ax.stackplot(time, phy, traj, rsu,
                     labels=["物理 x0.4", "轨迹 x0.3", "RSU x0.3"],
                     colors=[IMM_COLOR, LSTM_COLOR, FUSED_COLOR], alpha=0.72)
        ax.plot(time, hist["fused"][vid],
                color=THRESH_COLOR, lw=2.2, label="融合总分", zorder=5)
        ax.plot(time, hist["reputation"][vid],
                color=rep_col, lw=1.8, linestyle="--", label="信誉值", zorder=5)
        ax.axhline(0.6, color=THRESH_COLOR, linestyle=":", lw=1.0, alpha=0.45)

        anom = np.array(hist["lstm_anomaly"][vid])
        for i in range(len(time) - 1):
            if anom[i] > 0.5:
                ax.axvspan(time[i], time[i + 1],
                           color=LSTM_COLOR, alpha=0.12, zorder=0)

        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("时间步", fontsize=10)
        ax.set_ylabel("加权后分数", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True)

    plt.tight_layout()
    _save(fig, "03_score_decomposition.png")


# ── plot 4: trajectory map coloured by live reputation ────────────────────────

def plot_trajectory_reputation(hist, attack_vehicles, all_vids):
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle("车辆运动轨迹（线段颜色 = 实时信誉，红低绿高）",
                 fontsize=14, fontweight="bold")

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for vid in all_vids:
        pts    = np.array(hist["positions"][vid])
        rep_s  = np.array(hist["reputation"][vid])
        is_atk = vid in attack_vehicles
        if len(pts) < 2:
            continue
        for i in range(len(pts) - 1):
            seg_r = rep_s[min(i, len(rep_s) - 1)]
            ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1],
                    color=cmap(norm(seg_r)),
                    lw=2.5 if is_atk else 1.2,
                    alpha=0.95 if is_atk else 0.50,
                    zorder=3 if is_atk else 1)
        ax.scatter(pts[-1, 0], pts[-1, 1],
                   color=cmap(norm(rep_s[-1])),
                   s=160 if is_atk else 70,
                   marker="X" if is_atk else "o",
                   edgecolors="white", linewidths=1.2, zorder=6)
        ax.text(pts[-1, 0] + 0.15, pts[-1, 1] + 0.15,
                vid.replace("car_", "c"), fontsize=6.5,
                color=ATTACK_COLOR if is_atk else "#8b949e", zorder=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01).set_label("信誉值", fontsize=10)
    ax.legend(handles=[
        Line2D([0],[0], marker="X", color="w", markerfacecolor=ATTACK_COLOR,
               markersize=11, label="攻击车辆终点"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=NORMAL_COLOR,
               markersize=9,  label="正常车辆终点"),
    ], fontsize=9, loc="upper left")
    ax.set_xlabel("X 坐标 (m)", fontsize=10)
    ax.set_ylabel("Y 坐标 (m)", fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    _save(fig, "04_trajectory_reputation.png")


# ── plot 5: radar chart ───────────────────────────────────────────────────────

def plot_radar(hist, attack_vehicles, all_vids):
    keys   = ["imm_physical", "physical", "trajectory", "rsu", "fused", "reputation"]
    labels = ["IMM物理", "融合物理", "轨迹", "RSU", "融合总分", "信誉"]
    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    ac     = angles + angles[:1]

    def grp_means(vids):
        return [np.mean([np.mean(hist[k][v]) for v in vids]) for k in keys]

    atk_vids = [v for v in all_vids if v     in attack_vehicles]
    nrm_vids = [v for v in all_vids if v not in attack_vehicles]
    atk_m = grp_means(atk_vids) + grp_means(atk_vids)[:1]
    nrm_m = grp_means(nrm_vids) + grp_means(nrm_vids)[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle("攻击 vs 正常车辆 — 多维分数侧写（雷达图）",
                 fontsize=13, fontweight="bold", y=1.02)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=11)
    ax.set_ylim(0, 1)

    for r in [0.2, 0.4, 0.6, 0.8]:
        ax.plot(ac, [r] * (N + 1), color="#30363d", lw=0.7)

    ax.plot(ac, atk_m, color=ATTACK_COLOR, lw=2.3, zorder=4)
    ax.fill(ac, atk_m, color=ATTACK_COLOR, alpha=0.22)
    ax.plot(ac, nrm_m, color=NORMAL_COLOR, lw=2.3, zorder=4)
    ax.fill(ac, nrm_m, color=NORMAL_COLOR, alpha=0.22)

    ax.set_facecolor("#161b22")
    ax.spines["polar"].set_color("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.legend(handles=[
        Line2D([0],[0], color=ATTACK_COLOR, lw=2.2, label="攻击车辆（均值）"),
        Line2D([0],[0], color=NORMAL_COLOR, lw=2.2, label="正常车辆（均值）"),
    ], loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    plt.tight_layout()
    _save(fig, "05_radar_profile.png")


# ── plot 6: ROC + metric bar chart (IMM-only vs IMM+LSTM) ─────────────────────

def _roc(final_reps, attack_vehicles, all_vids):
    thresholds = np.linspace(0, 1, 300)
    tprs, fprs = [], []
    for thr in thresholds:
        tp = fp = tn = fn = 0
        for v in all_vids:
            pa = final_reps[v] < thr
            ra = v in attack_vehicles
            if   ra and pa:  tp += 1
            elif ra:         fn += 1
            elif pa:         fp += 1
            else:            tn += 1
        tprs.append(tp / max(tp + fn, 1))
        fprs.append(fp / max(fp + tn, 1))
    return np.array(fprs), np.array(tprs)


def _metrics(final_reps, attack_vehicles, all_vids, thr=0.6):
    tp = fp = tn = fn = 0
    for v in all_vids:
        pa = final_reps[v] < thr
        ra = v in attack_vehicles
        if   ra and pa:  tp += 1
        elif ra:         fn += 1
        elif pa:         fp += 1
        else:            tn += 1
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return acc, prec, rec, f1


def plot_roc_and_metrics(hist_imm, hist_fused, attack_vehicles, all_vids):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("检测性能对比：IMM-only  vs  IMM+LSTM",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    for hist, color, label in [
        (hist_imm,   IMM_COLOR,   "IMM-only"),
        (hist_fused, FUSED_COLOR, "IMM + LSTM (w=0.4)"),
    ]:
        final = {v: hist["reputation"][v][-1] for v in all_vids}
        fprs, tprs = _roc(final, attack_vehicles, all_vids)
        auc = abs(float(np.trapezoid(tprs, fprs) if hasattr(np, "trapezoid") else np.trapz(tprs, fprs)))
        ax.plot(fprs, tprs, color=color, lw=2.3,
                label=f"{label}   AUC={auc:.3f}")

    ax.plot([0, 1], [0, 1], color="#30363d", linestyle="--", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("假正率 (FPR)", fontsize=10)
    ax.set_ylabel("真正率 (TPR)", fontsize=10)
    ax.set_title("ROC 曲线", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True)

    ax = axes[1]
    mnames = ["准确率", "精确率", "召回率", "F1"]
    x  = np.arange(len(mnames))
    BW = 0.32
    for i, (hist, color, label) in enumerate([
        (hist_imm,   IMM_COLOR,   "IMM-only"),
        (hist_fused, FUSED_COLOR, "IMM+LSTM"),
    ]):
        final = {v: hist["reputation"][v][-1] for v in all_vids}
        vals  = list(_metrics(final, attack_vehicles, all_vids))
        bars  = ax.bar(x + (i - 0.5) * BW, vals, BW,
                       color=color, alpha=0.82, label=label)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.013, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9,
                    color="#e6edf3", fontweight="bold")

    ax.set_ylim(0, 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels(mnames, fontsize=10)
    ax.set_ylabel("分数", fontsize=10)
    ax.set_title("阈值 0.6 下的检测指标对比", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")
    plt.tight_layout()
    _save(fig, "06_roc_and_metrics.png")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    STEPS = 100
    SEED  = 42

    print("\n全部完成。")
    hist_fused, attack_vehicles, all_vids = run_simulation(
        steps=STEPS, lstm_imm_weight=0.4, seed=SEED)

    print("\n全部完成。")
    hist_imm, _, _ = run_simulation(
        steps=STEPS, lstm_imm_weight=0.0, seed=SEED)

    print("\n全部完成。")
    plot_imm_vs_lstm(hist_fused, attack_vehicles, all_vids, STEPS)
    plot_reputation_heatmap(hist_fused, attack_vehicles, all_vids, STEPS)
    plot_score_decomposition(hist_fused, attack_vehicles, all_vids, STEPS)
    plot_trajectory_reputation(hist_fused, attack_vehicles, all_vids)
    plot_radar(hist_fused, attack_vehicles, all_vids)
    plot_roc_and_metrics(hist_imm, hist_fused, attack_vehicles, all_vids)

    print("\n全部完成。")
