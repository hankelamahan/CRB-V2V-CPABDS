"""
完整的信誉中心系统测试脚本

测试内容:
1. 服务器基础端点 (GET /reputation, POST /batch_reputations)
2. ReputationClient 本地缓存和网络请求
3. 融合框上报和信誉更新 (POST /report_fused_boxes)
4. ReputationClientAdapter 的集成功能

运行前请确保服务器已启动: python reputation_center_server.py
"""

import time
import requests
import numpy as np
from reputation_client import ReputationClient


BASE_URL = "http://localhost:8888"
COLORS = {
    'green': '\033[92m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'blue': '\033[94m',
    'end': '\033[0m'
}


def print_section(title):
    print(f"\n{COLORS['blue']}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{COLORS['end']}\n")


def print_success(msg):
    print(f"{COLORS['green']}[OK] {msg}{COLORS['end']}")


def print_info(msg):
    print(f"{COLORS['yellow']}[>>] {msg}{COLORS['end']}")


def print_error(msg):
    print(f"{COLORS['red']}[!!] {msg}{COLORS['end']}")


def test_server_health():
    """测试服务器是否正常运行"""
    print_section("测试 1: 服务器健康检查")
    try:
        resp = requests.get(f"{BASE_URL}/all_reputations", timeout=1)
        if resp.status_code == 200:
            print_success("服务器正在运行")
            return True
        else:
            print_error(f"服务器返回错误状态码: {resp.status_code}")
            return False
    except Exception as e:
        print_error(f"无法连接到服务器: {e}")
        print_info("请先运行: python reputation_center_server.py")
        return False


def test_single_reputation_query():
    """测试单个车辆信誉查询"""
    print_section("测试 2: 单个车辆信誉查询 GET /reputation/{id}")
    
    vehicle_id = "test_vehicle_001"
    resp = requests.get(f"{BASE_URL}/reputation/{vehicle_id}")
    
    if resp.status_code == 200:
        rep = resp.json()["reputation"]
        print_success(f"查询成功: {vehicle_id} 的信誉值为 {rep}")
        assert rep == 0.5, "新车辆的初始信誉应为 0.5"
        print_info("新车辆初始信誉验证通过 (0.5)")
        return True
    else:
        print_error(f"查询失败: {resp.status_code}")
        return False


def test_batch_reputation_query():
    """测试批量信誉查询"""
    print_section("测试 3: 批量信誉查询 POST /batch_reputations")
    
    vehicle_ids = ["veh_001", "veh_002", "veh_003"]
    resp = requests.post(f"{BASE_URL}/batch_reputations", json={
        "vehicle_ids": vehicle_ids
    })
    
    if resp.status_code == 200:
        reps = resp.json()["reputations"]
        print_success(f"批量查询成功，获取了 {len(reps)} 个车辆的信誉值")
        for vid, rep in reps.items():
            print_info(f"  {vid}: {rep}")
        return True
    else:
        print_error(f"批量查询失败: {resp.status_code}")
        return False


def test_report_batch():
    """测试批量验证结果上报"""
    print_section("测试 4: 批量验证结果上报 POST /report_batch")
    
    payload = {
        "reports": [
            {
                "vehicle_id": "veh_good",
                "reporter_id": "ego",
                "verification_result": True,
                "phy_score": 0.9
            },
            {
                "vehicle_id": "veh_bad",
                "reporter_id": "ego",
                "verification_result": False,
                "phy_score": 0.2
            }
        ]
    }
    
    resp = requests.post(f"{BASE_URL}/report_batch", json=payload)
    
    if resp.status_code == 200:
        updated = resp.json()["updated"]
        print_success("验证结果上报成功")
        print_info(f"veh_good 新信誉: {updated.get('veh_good', 'N/A')} (期望 > 0.5)")
        print_info(f"veh_bad 新信誉: {updated.get('veh_bad', 'N/A')} (期望 < 0.5)")
        return True
    else:
        print_error(f"上报失败: {resp.status_code}")
        return False


def test_fused_boxes_report():
    """测试融合框上报和一致性评分"""
    print_section("测试 5: 融合框上报 POST /report_fused_boxes")
    
    payload = {
        "reporter_id": "ego",
        "fused_boxes": [
            [0.1, 0.1, 0.4, 0.4],
            [0.5, 0.5, 0.8, 0.8]
        ],
        "fused_scores": [0.9, 0.85],
        "fused_labels": [1, 1],
        "cav_detections": {
            "veh_consistent": {
                "boxes": [[0.1, 0.1, 0.4, 0.4], [0.52, 0.52, 0.78, 0.78]],
                "scores": [0.88, 0.84],
                "labels": [1, 1]
            },
            "veh_inconsistent": {
                "boxes": [[0.9, 0.9, 1.0, 1.0]],
                "scores": [0.7],
                "labels": [1]
            }
        },
        "timestamp": time.time()
    }
    
    resp = requests.post(f"{BASE_URL}/report_fused_boxes", json=payload)
    
    if resp.status_code == 200:
        result = resp.json()
        updated = result["updated"]
        consistency = result["consistency_scores"]
        
        print_success("融合框上报成功")
        print_info("一致性评分:")
        for vid, score in consistency.items():
            print_info(f"  {vid}: {score:.3f}")
        print_info("更新后的信誉值:")
        for vid, rep in updated.items():
            print_info(f"  {vid}: {rep:.3f}")
        
        if consistency["veh_consistent"] > 0.5:
            print_success("一致性评分验证通过 (veh_consistent > 0.5)")
        else:
            print_error(f"veh_consistent 一致性得分 {consistency['veh_consistent']:.3f} 未超过 0.5")
        
        if consistency["veh_inconsistent"] < 0.5:
            print_success("不一致性评分验证通过 (veh_inconsistent < 0.5)")
        else:
            print_error(f"veh_inconsistent 一致性得分 {consistency['veh_inconsistent']:.3f} 未低于 0.5")
        
        return consistency["veh_consistent"] > 0.5 and consistency["veh_inconsistent"] < 0.5
    else:
        print_error(f"上报失败: {resp.status_code}")
        return False


def test_reputation_client():
    """测试 ReputationClient 类"""
    print_section("测试 6: ReputationClient 类功能")
    
    client = ReputationClient(
        server_url=BASE_URL,
        cache_capacity=50,
        cache_ttl=30,
        pull_interval=5
    )
    
    try:
        vehicle_id = "client_test_001"
        
        print_info("第一次查询 (缓存未命中，走网络)")
        rep1 = client.get_reputation(vehicle_id)
        print_success(f"获取信誉: {rep1}")
        
        print_info("第二次查询 (缓存命中，无网络请求)")
        rep2 = client.get_reputation(vehicle_id)
        print_success(f"缓存命中: {rep2}")
        assert rep1 == rep2, "缓存值应与第一次查询一致"
        
        print_info("批量查询测试")
        batch_ids = ["batch_001", "batch_002", "batch_003"]
        batch_reps = client.get_batch_reputations(batch_ids)
        print_success(f"批量获取 {len(batch_reps)} 个信誉值")
        for vid, rep in batch_reps.items():
            print_info(f"  {vid}: {rep}")
        
        print_info("测试验证结果上报")
        client.report_verification(
            reporter_id="ego",
            target_id="report_test_001",
            result=True,
            phy_score=0.95
        )
        print_success("验证结果上报完成")
        
        print_info("测试融合框上报")
        updated = client.report_fused_boxes(
            reporter_id="ego",
            fused_boxes=[[0.1, 0.1, 0.3, 0.3]],
            fused_scores=[0.9],
            fused_labels=[1],
            cav_detections={
                "fusion_test_001": {
                    "boxes": [[0.1, 0.1, 0.3, 0.3]],
                    "scores": [0.88],
                    "labels": [1]
                }
            }
        )
        print_success(f"融合框上报完成，更新了 {len(updated)} 个车辆的信誉")
        
        client.stop()
        print_success("ReputationClient 所有功能测试通过")
        return True
        
    except Exception as e:
        print_error(f"ReputationClient 测试失败: {e}")
        client.stop()
        return False


def test_reputation_client_adapter():
    """测试 ReputationClientAdapter"""
    print_section("测试 7: ReputationClientAdapter 集成测试")
    
    try:
        from reputation_client_adapter import ReputationClientAdapter
        
        client = ReputationClient(server_url=BASE_URL)
        adapter = ReputationClientAdapter(client=client, ego_id="test_ego")
        
        print_info("测试单个信誉查询")
        rep = adapter.get_reputation("adapter_test_001")
        print_success(f"获取信誉: {rep}")
        
        print_info("测试 ego 车辆特殊处理")
        ego_rep = adapter.get_reputation("test_ego")
        assert ego_rep == 1.0, "ego 车辆信誉应始终为 1.0"
        print_success(f"ego 信誉验证通过: {ego_rep}")
        
        print_info("测试批量查询")
        batch_reps = adapter.get_batch_reputations(["v1", "v2", "test_ego"])
        print_success(f"批量查询完成，包含 ego: {batch_reps.get('test_ego')}")
        
        print_info("测试融合框上报 (numpy 数组)")
        fused_boxes = np.array([[0.2, 0.2, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]])
        fused_scores = np.array([0.92, 0.87])
        fused_labels = np.array([1, 1])
        cav_detections = {
            "adapter_cav_001": {
                "boxes": np.array([[0.21, 0.21, 0.49, 0.49]]),
                "scores": np.array([0.90]),
                "labels": np.array([1])
            }
        }
        
        updated = adapter.report_fused_boxes(
            fused_boxes=fused_boxes,
            fused_scores=fused_scores,
            fused_labels=fused_labels,
            cav_detections=cav_detections
        )
        print_success(f"融合框上报完成: {updated}")
        
        client.stop()
        print_success("ReputationClientAdapter 所有功能测试通过")
        return True
        
    except ImportError as e:
        print_error(f"无法导入 ReputationClientAdapter: {e}")
        print_info("请确保 reputation_client_adapter.py 在 Python 路径中")
        return False
    except Exception as e:
        print_error(f"ReputationClientAdapter 测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print(f"\n{COLORS['blue']}{'='*60}")
    print("  信誉中心系统完整测试")
    print(f"{'='*60}{COLORS['end']}")
    
    tests = [
        ("服务器健康检查", test_server_health),
        ("单个信誉查询", test_single_reputation_query),
        ("批量信誉查询", test_batch_reputation_query),
        ("批量验证上报", test_report_batch),
        ("融合框上报", test_fused_boxes_report),
        ("ReputationClient", test_reputation_client),
        ("ReputationClientAdapter", test_reputation_client_adapter),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if not result and test_name == "服务器健康检查":
                print_error("服务器未运行，停止后续测试")
                break
        except Exception as e:
            print_error(f"{test_name} 异常: {e}")
            results.append((test_name, False))
    
    print_section("测试总结")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "通过" if result else "失败"
        color = COLORS['green'] if result else COLORS['red']
        print(f"{color}{status:^6}{COLORS['end']} {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print(f"{COLORS['green']}\n[SUCCESS] 所有测试通过！系统工作正常。{COLORS['end']}")
    else:
        print(f"{COLORS['yellow']}\n[WARNING] 部分测试失败，请检查日志。{COLORS['end']}")


if __name__ == "__main__":
    run_all_tests()
