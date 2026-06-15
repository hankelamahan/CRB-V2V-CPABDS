import numpy as np
from utils import physical_score, trajectory_score, rsu_score


class IntermediateFusionManager:
    def __init__(self, lstm_imm_weight: float = 0.4):
        self.reputation = {}
        self.vel_history = {}
        self.neighbor_votes = {}
        self.beta = 0.5
        # Weight for LSTM anomaly score when synthesising physical consistency.
        # IMM contributes (1 - lstm_imm_weight), LSTM contributes lstm_imm_weight.
        # Set to 0.0 to run IMM-only (backward-compatible fallback).
        self.lstm_imm_weight = float(np.clip(lstm_imm_weight, 0.0, 1.0))
        self.weights = {
            "physical": 0.4,
            "trajectory": 0.3,
            "rsu": 0.3,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_vel_history(self, vid):
        if vid not in self.vel_history:
            self.vel_history[vid] = []
        return self.vel_history[vid]

    def update_vel_history(self, vid, vel):
        history = self.get_vel_history(vid)
        history.append(vel)
        if len(history) > 5:
            history.pop(0)
        self.vel_history[vid] = history

    def get_neighbor_votes(self, vid):
        if vid not in self.neighbor_votes:
            self.neighbor_votes[vid] = []
        return self.neighbor_votes[vid]

    def update_neighbor_votes(self, vid, vote):
        votes = self.get_neighbor_votes(vid)
        votes.append(vote)
        if len(votes) > 10:
            votes.pop(0)
        self.neighbor_votes[vid] = votes

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _fuse_physical_scores(self, imm_phy_score: float, lstm_anomaly_score: float) -> float:
        """Combine IMM-based physical score with LSTM anomaly score.

        lstm_anomaly_score is an *anomaly* value (high = bad), so it is
        converted to a consistency score via (1 - lstm_anomaly_score) before
        blending.  When lstm_imm_weight == 0 the result degrades to the
        pure IMM score.
        """
        lstm_consistency = 1.0 - float(np.clip(lstm_anomaly_score, 0.0, 1.0))
        w = self.lstm_imm_weight
        return float(np.clip((1.0 - w) * imm_phy_score + w * lstm_consistency, 0.0, 1.0))

    def fuse_scores(self, phy_score, traj_score, rsu_score_val):
        fused = (
            self.weights["physical"] * phy_score
            + self.weights["trajectory"] * traj_score
            + self.weights["rsu"] * rsu_score_val
        )
        return float(np.clip(fused, 0.0, 1.0))

    def compute_all_scores(self, residual, vid, vel, lstm_anomaly_score: float = 0.0):
        """Compute all per-vehicle scores and return them as a dict.

        Parameters
        ----------
        residual : float
            IMM fused residual for this vehicle/step.
        vid : str
            Vehicle identifier.
        vel : array-like
            Current velocity observation used to update trajectory history.
        lstm_anomaly_score : float, optional
            Anomaly score from PhysicsPredictor (0–1, higher = more anomalous).
            Defaults to 0 so callers that have not wired up the LSTM yet are
            unaffected.
        """
        imm_phy = physical_score(residual)
        combined_phy = self._fuse_physical_scores(imm_phy, lstm_anomaly_score)

        self.update_vel_history(vid, vel)
        traj = trajectory_score(self.get_vel_history(vid))

        rsu_val = rsu_score(self.get_neighbor_votes(vid))

        fused = self.fuse_scores(combined_phy, traj, rsu_val)
        return {
            "physical": combined_phy,
            "imm_physical": imm_phy,
            "lstm_anomaly": lstm_anomaly_score,
            "trajectory": traj,
            "rsu": rsu_val,
            "fused": fused,
        }

    # ------------------------------------------------------------------
    # Reputation
    # ------------------------------------------------------------------

    def update_reputation(self, vid, fused_score):
        if vid not in self.reputation:
            self.reputation[vid] = 0.5
        self.reputation[vid] = self.reputation[vid] + self.beta * (
            fused_score - self.reputation[vid]
        )
        self.reputation[vid] = float(np.clip(self.reputation[vid], 0.0, 1.0))
        return self.reputation[vid]

    def get_vote(self, fused_score):
        return -1 if fused_score < 0.6 else 1
