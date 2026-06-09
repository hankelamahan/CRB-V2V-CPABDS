"""
为基于中心化信誉的V2V协同感知系统设计的本地信誉缓存模块。
通过最近最少使用策略管理本地缓存的（车辆ID，信誉值）对，
用于减少与中心信誉服务器的通信延迟和负载。
"""

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Callable, Any, Tuple


class VehicleReputationCache:
    """
    车辆信誉本地缓存类。
    使用LRU策略管理缓存条目。
    """

    def __init__(self, capacity: int = 100, ttl: int = 300, server_sync_callback: Optional[Callable[[str], float]] = None):
        """
        初始化缓存。

        :param capacity: 最大缓存车辆数（LRU容量），默认100。
        :param ttl:     缓存条目的生存时间（秒），超时后认为数据“陈旧”，默认300秒（5分钟）。
        :param server_sync_callback: 用于向中心服务器查询最新信誉的回调函数。
                                    函数签名应为 func(vehicle_id: str) -> float。
                                    若未提供，则缓存未命中时返回默认值。
        """
        self.capacity = capacity
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()  # {vehicle_id: (reputation, timestamp)}
        self._server_sync_callback = server_sync_callback

    def _is_valid(self, timestamp: float) -> bool:
        """检查一个时间戳是否在TTL有效期内。"""
        return (time.time() - timestamp) < self.ttl

    def get(self, vehicle_id: str) -> float:
        """
        获取指定车辆的信誉值。

        :param vehicle_id: 目标车辆的ID。
        :return: 该车辆的信誉值。
        """
        # 1. 如果存在于缓存且未过期，直接返回
        if vehicle_id in self._cache:
            reputation, timestamp = self._cache[vehicle_id]
            if self._is_valid(timestamp):
                # LRU: 将访问的项移到末尾，表示最近使用过
                self._cache.move_to_end(vehicle_id)
                return reputation
            else:
                # 数据已过期，主动删除
                del self._cache[vehicle_id]

        # 2. 缓存未命中或已过期，从中心服务器获取
        if self._server_sync_callback:
            try:
                fresh_reputation = self._server_sync_callback(vehicle_id)
                # 更新缓存
                self.update(vehicle_id, fresh_reputation)
                return fresh_reputation
            except Exception as e:
                # 在实际应用中，此处应有更完善的错误处理和日志记录
                print(f"从中心服务器获取信誉失败 ({vehicle_id}): {e}")
                return 0.5  # 发生错误时返回一个中性/默认信誉值
        else:
            # 没有提供服务器同步回调，返回默认值
            return 0.5

    def update(self, vehicle_id: str, reputation: float):
        """
        更新或添加一个车辆的信誉值到缓存。

        :param vehicle_id: 车辆ID。
        :param reputation: 新的信誉值。
        """
        if vehicle_id in self._cache:
            # 如果已存在，更新其值并将它移到末尾
            self._cache[vehicle_id] = (reputation, time.time())
            self._cache.move_to_end(vehicle_id)
        else:
            # 如果是新条目，直接添加
            self._cache[vehicle_id] = (reputation, time.time())
            # 如果超过容量，删除最久未使用的项（第一个元素）
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)

    def batch_update(self, updates: Dict[str, float]):
        """
        批量更新缓存。

        :param updates: 一个字典，键为vehicle_id，值为新的信誉值。
        """
        for vid, rep in updates.items():
            self.update(vid, rep)

    def sync_from_server(self, vehicle_ids: List[str]):
        """
        主动从中心服务器同步一批车辆的信誉。

        :param vehicle_ids: 需要同步的车辆ID列表。
        """
        if not self._server_sync_callback:
            print("警告：未设置服务器同步回调函数，无法主动同步。")
            return

        for vid in vehicle_ids:
            # 直接调用get方法，它会自动处理缓存未命中和服务器同步
            self.get(vid)

    def remove(self, vehicle_id: str) -> bool:
        """
        从缓存中移除指定车辆。

        :param vehicle_id: 车辆ID。
        :return: 如果成功移除则返回True，否则返回False。
        """
        if vehicle_id in self._cache:
            del self._cache[vehicle_id]
            return True
        return False

    def clear(self):
        """清空所有缓存数据。"""
        self._cache.clear()

    def size(self) -> int:
        """返回当前缓存的车辆数量。"""
        return len(self._cache)

    def get_all(self) -> Dict[str, float]:
        """
        返回当前缓存中所有有效的（未过期的）车辆信誉值。
        :return: 一个字典，键为vehicle_id，值为信誉值。
        """
        valid_entries = {}
        current_time = time.time()
        # 使用list()来避免在迭代时修改字典
        for vid, (rep, ts) in list(self._cache.items()):
            if current_time - ts < self.ttl:
                valid_entries[vid] = rep
            else:
                # 顺便清理过期的条目
                del self._cache[vid]
        return valid_entries

    def stats(self) -> Dict[str, Any]:
        """
        返回缓存的统计信息。
        :return: 一个包含缓存容量、当前大小、TTL等信息的字典。
        """
        return {
            "capacity": self.capacity,
            "size": self.size(),
            "ttl": self.ttl,
            "hit_rate": "N/A (需外部计算)",  # 简单的命中率需要调用方自己记录
        }

# ================= 使用示例 =================
if __name__ == "__main__":
    # 模拟一个从中心服务器获取信誉的函数
    def fetch_reputation_from_server(vid: str) -> float:
        # 这里应实现真正的网络请求
        print(f"  -> [网络请求] 正在从中心服务器获取车辆 {vid} 的信誉...")
        # 模拟返回一个随机信誉值
        import random
        return round(random.uniform(0, 1), 2)

    # 1. 创建缓存实例：容量为3，TTL为10秒，并传入服务器回调函数
    cache = VehicleReputationCache(capacity=3, ttl=10, server_sync_callback=fetch_reputation_from_server)

    # 2. 获取信誉（首次访问，缓存未命中，将触发服务器查询）
    print("\n--- 首次访问，缓存未命中，触发服务器查询 ---")
    rep1 = cache.get("Vehicle_A")
    print(f"Vehicle_A 的信誉是: {rep1}")

    # 3. 再次获取同一车辆的信誉（缓存命中）
    print("\n--- 再次访问，缓存命中，无网络请求 ---")
    rep1_again = cache.get("Vehicle_A")
    print(f"再次获取 Vehicle_A 的信誉: {rep1_again}")

    # 4. 批量更新/添加多个车辆
    print("\n--- 批量更新缓存 ---")
    cache.batch_update({
        "Vehicle_B": 0.95,
        "Vehicle_C": 0.80,
        "Vehicle_D": 0.60
    })

    # 5. 打印当前缓存内容
    print("\n--- 当前缓存状态 ---")
    for vid, rep in cache.get_all().items():
        print(f"  车辆 {vid}: 信誉 {rep}")

    # 6. 由于容量限制，再次添加新车辆，会自动淘汰最久未使用的Vehicle_A
    print("\n--- 容量限制演示：添加新车辆 'Vehicle_E' ---")
    cache.update("Vehicle_E", 0.55)
    print("缓存已满，最久未使用的Vehicle_A被淘汰。")
    for vid, rep in cache.get_all().items():
        print(f"  车辆 {vid}: 信誉 {rep}")

    # 7. 演示TTL过期：等待TTL超时后，缓存的项将失效
    print(f"\n--- TTL过期演示：等待 {cache.ttl} 秒后，缓存将过期 ---")
    time.sleep(cache.ttl + 1)
    print("TTL已超时，再次获取Vehicle_B，将触发服务器查询。")
    repB = cache.get("Vehicle_B")
    print(f"Vehicle_B 的信誉是: {repB}")

    # 8. 查看统计信息
    print("\n--- 缓存统计 ---")
    print(cache.stats())