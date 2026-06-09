import requests
import threading
import time

class ReputationClient:
    def __init__(self, server_url="http://localhost:8888"):
        self.server_url = server_url
        self.cache = {}          # 本地缓存
        self.cache_lock = threading.Lock()
        # 可以启动后台线程定期拉取全量信誉
        self._stop = False
        self._thread = threading.Thread(target=self._periodic_pull, daemon=True)
        self._thread.start()

    def get_reputation(self, vehicle_id):
        """先查缓存，若没有则从服务器获取"""
        with self.cache_lock:
            if vehicle_id in self.cache:
                return self.cache[vehicle_id]
        # 缓存未命中，同步请求
        try:
            resp = requests.get(f"{self.server_url}/reputation/{vehicle_id}", timeout=0.1)
            if resp.status_code == 200:
                rep = resp.json()["reputation"]
                with self.cache_lock:
                    self.cache[vehicle_id] = rep
                return rep
        except:
            pass
        return 0.5

    def report_verification(self, reporter_id, target_id, result, phy_score=0, traj_score=0, cons_score=0):
        """上报验证结果到服务器"""
        data = {
            "reports": [{
                "vehicle_id": target_id,
                "reporter_id": reporter_id,
                "verification_result": result,
                "phy_score": phy_score,
                "traj_score": traj_score,
                "consensus_score": cons_score
            }]
        }
        try:
            requests.post(f"{self.server_url}/report_batch", json=data, timeout=0.2)
        except:
            pass

    def _periodic_pull(self):
        """每10秒拉取一次全量信誉（可选）"""
        while not self._stop:
            time.sleep(10)
            try:
                resp = requests.get(f"{self.server_url}/all_reputations", timeout=0.5)
                if resp.status_code == 200:
                    all_reps = resp.json()
                    with self.cache_lock:
                        self.cache.update(all_reps)
            except:
                pass
import time
import threading
import requests
from typing import Dict, List, Optional

from local_cache import VehicleReputationCache


class ReputationClient:
    """
    Client for the centralized reputation server.

    Uses VehicleReputationCache as the local LRU cache with TTL.  Cache misses
    trigger a synchronous GET /reputation/{id} call.  A background thread
    periodically refreshes all known entries via GET /all_reputations.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8888",
        cache_capacity: int = 100,
        cache_ttl: int = 60,
        pull_interval: int = 10,
        request_timeout: float = 0.2,
    ):
        self.server_url = server_url.rstrip("/")
        self.request_timeout = request_timeout

        self._cache = VehicleReputationCache(
            capacity=cache_capacity,
            ttl=cache_ttl,
            server_sync_callback=self._fetch_single,
        )
        self._cache_lock = threading.Lock()

        self._stop = False
        self._pull_interval = pull_interval
        self._bg_thread = threading.Thread(target=self._periodic_pull, daemon=True)
        self._bg_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_reputation(self, vehicle_id: str) -> float:
        """Return reputation from local cache, fetching from server on miss."""
        vehicle_id = str(vehicle_id)
        with self._cache_lock:
            return self._cache.get(vehicle_id)

    def get_batch_reputations(self, vehicle_ids: List[str]) -> Dict[str, float]:
        """
        Fetch reputations for multiple vehicles in one request.
        Hits the cache first; only queries server for the missing ones.
        """
        vehicle_ids = [str(v) for v in vehicle_ids]
        result: Dict[str, float] = {}
        missing: List[str] = []

        with self._cache_lock:
            cached_all = self._cache.get_all()

        for vid in vehicle_ids:
            if vid in cached_all:
                result[vid] = cached_all[vid]
            else:
                missing.append(vid)

        if missing:
            fetched = self._fetch_batch(missing)
            with self._cache_lock:
                for vid, rep in fetched.items():
                    self._cache.update(vid, rep)
            result.update(fetched)

        for vid in vehicle_ids:
            result.setdefault(vid, 0.5)

        return result

    def report_verification(
        self,
        reporter_id: str,
        target_id: str,
        result: bool,
        phy_score: float = 0.0,
        traj_score: float = 0.0,
        cons_score: float = 0.0,
    ) -> None:
        """Report a verification outcome for a single vehicle."""
        payload = {
            "reports": [
                {
                    "vehicle_id": str(target_id),
                    "reporter_id": str(reporter_id),
                    "verification_result": result,
                    "phy_score": phy_score,
                    "traj_score": traj_score,
                    "consensus_score": cons_score,
                }
            ]
        }
        self._post("/report_batch", payload)

    def report_fused_boxes(
        self,
        reporter_id: str,
        fused_boxes: List[List[float]],
        fused_scores: List[float],
        fused_labels: List[int],
        cav_detections: Dict[str, Dict],
        timestamp: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Upload fused detection results to the server so it can derive
        per-CAV consistency scores and update their reputation.

        Returns the server's updated reputation map, or an empty dict on error.
        """
        payload = {
            "reporter_id": str(reporter_id),
            "fused_boxes": fused_boxes,
            "fused_scores": fused_scores,
            "fused_labels": fused_labels,
            "cav_detections": cav_detections,
            "timestamp": timestamp if timestamp is not None else time.time(),
        }
        response = self._post("/report_fused_boxes", payload)
        if response is None:
            return {}

        updated: Dict[str, float] = response.get("updated", {})
        with self._cache_lock:
            for vid, rep in updated.items():
                self._cache.update(vid, rep)
        return updated

    def stop(self) -> None:
        """Stop the background refresh thread."""
        self._stop = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_single(self, vehicle_id: str) -> float:
        """Callback used by VehicleReputationCache on a cache miss."""
        try:
            resp = requests.get(
                f"{self.server_url}/reputation/{vehicle_id}",
                timeout=self.request_timeout,
            )
            if resp.status_code == 200:
                return float(resp.json().get("reputation", 0.5))
        except Exception:
            pass
        return 0.5

    def _fetch_batch(self, vehicle_ids: List[str]) -> Dict[str, float]:
        try:
            resp = requests.post(
                f"{self.server_url}/batch_reputations",
                json={"vehicle_ids": vehicle_ids},
                timeout=self.request_timeout,
            )
            if resp.status_code == 200:
                return {str(k): float(v) for k, v in resp.json().get("reputations", {}).items()}
        except Exception:
            pass
        return {}

    def _post(self, path: str, payload: dict) -> Optional[dict]:
        try:
            resp = requests.post(
                f"{self.server_url}{path}",
                json=payload,
                timeout=self.request_timeout,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def _periodic_pull(self) -> None:
        while not self._stop:
            time.sleep(self._pull_interval)
            try:
                resp = requests.get(
                    f"{self.server_url}/all_reputations",
                    timeout=0.5,
                )
                if resp.status_code == 200:
                    all_reps: Dict[str, float] = resp.json()
                    with self._cache_lock:
                        self._cache.batch_update(all_reps)
            except Exception:
                pass
