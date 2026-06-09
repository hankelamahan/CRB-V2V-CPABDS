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