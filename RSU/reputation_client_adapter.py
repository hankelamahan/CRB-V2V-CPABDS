"""
轻量级 ReputationClientAdapter，用于测试环境。

与 intermediate_fusion_dataset.py 中的版本功能完全相同，
但不依赖 PyTorch 和 opencood，可以在测试脚本中独立导入。
"""

from reputation_client import ReputationClient


class ReputationClientAdapter:
    """
    Bridges ReputationClient (HTTP-backed) with the interface expected by
    IntermediateFusionDataset: get per-vehicle reputation scores and report
    fused detection results back to the server.

    Ego vehicle always returns 1.0 without a network round-trip.
    numpy arrays are serialised to plain lists before being handed to the
    HTTP client.
    """

    def __init__(self, client: ReputationClient, ego_id: str = "ego"):
        self._client = client
        self._ego_id = str(ego_id)

    def get_reputation(self, vehicle_id: str) -> float:
        if str(vehicle_id) == self._ego_id:
            return 1.0
        return self._client.get_reputation(str(vehicle_id))

    def get_batch_reputations(self, vehicle_ids) -> dict:
        non_ego = [str(v) for v in vehicle_ids if str(v) != self._ego_id]
        result = self._client.get_batch_reputations(non_ego)
        for vid in vehicle_ids:
            if str(vid) == self._ego_id:
                result[str(vid)] = 1.0
        return result

    def report_fused_boxes(
        self,
        fused_boxes,
        fused_scores,
        fused_labels,
        cav_detections: dict,
    ) -> dict:
        """Upload fused boxes and per-CAV detections; return updated reputations."""
        def _to_list(x):
            return x.tolist() if hasattr(x, "tolist") else list(x)

        serialised_detections = {}
        for cav_id, det in cav_detections.items():
            serialised_detections[str(cav_id)] = {
                "boxes":  _to_list(det["boxes"]),
                "scores": _to_list(det["scores"]),
                "labels": _to_list(det["labels"]),
            }

        return self._client.report_fused_boxes(
            reporter_id=self._ego_id,
            fused_boxes=_to_list(fused_boxes),
            fused_scores=_to_list(fused_scores),
            fused_labels=_to_list(fused_labels),
            cav_detections=serialised_detections,
        )
