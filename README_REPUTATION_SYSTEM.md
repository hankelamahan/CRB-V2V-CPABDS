# 中心化信誉系统 - 使用文档

## 1. 系统概述

本系统为 V2V 协同感知提供中心化信誉管理服务，通过 HTTP API 实现车辆信誉的集中存储、查询和更新。客户端采用 LRU 缓存策略降低网络延迟，支持融合检测结果的自动上报与 IoU 一致性评估。

**核心特性**:
- 中心化存储：SQLite 持久化所有车辆信誉值
- 智能缓存：LRU + TTL 双重策略，缓存命中时零网络开销
- 自动评估：基于 IoU 计算检测一致性，动态调整信誉
- 无缝集成：与 `intermediate_fusion_dataset.py` 深度集成，每帧自动上报

---

## 2. 架构与组件

### 2.1 文件清单

| 文件 | 职责 | 关键依赖 |
|------|------|---------|
| `reputation_center_server.py` | HTTP 服务器，持久化信誉 | fastapi, uvicorn, sqlite3 |
| `reputation_client.py` | HTTP 客户端，封装缓存 | requests, local_cache |
| `local_cache.py` | LRU + TTL 缓存实现 | typing, collections |
| `reputation_client_adapter.py` | 轻量适配器（测试用） | reputation_client |
| `intermediate_fusion_dataset.py` | 数据集集成（训练用） | torch, opencood |
| `test_reputation_system.py` | 7 个测试用例 | requests, numpy |

### 2.2 服务器端 API (`reputation_center_server.py`)

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/reputation/{vehicle_id}` | 单车信誉查询 |
| POST | `/batch_reputations` | 批量信誉查询 |
| GET | `/all_reputations` | 全量信誉拉取 |
| POST | `/report_batch` | 批量验证结果上报 |
| POST | `/report_fused_boxes` | 融合检测上报（自动计算 IoU 一致性） |

### 2.3 客户端核心方法 (`reputation_client.py`)

```python
client = ReputationClient(server_url="http://localhost:8888")

rep  = client.get_reputation("veh_001")                     # 单车查询（优先缓存）
reps = client.get_batch_reputations(["veh_001", "veh_002"]) # 批量查询

client.report_verification(reporter_id="ego", target_id="veh_001", result=True)

client.report_fused_boxes(
    reporter_id="ego",
    fused_boxes=[[0.1, 0.1, 0.4, 0.4]],
    fused_scores=[0.9],
    fused_labels=[1],
    cav_detections={"veh_001": {"boxes": [...], "scores": [...], "labels": [...]}}
)
```

### 2.4 缓存配置 (`local_cache.py`)

```python
VehicleReputationCache(
    capacity=100,              # 最大缓存条目数（LRU 淘汰）
    ttl=60,                    # 缓存生存时间（秒）
    server_sync_callback=...   # 缓存未命中时的回调函数
)
```

---

## 3. 快速开始

### 3.1 安装依赖

```bash
pip install fastapi uvicorn pydantic requests
```

**Python 版本要求**: 3.7+

### 3.2 启动服务器

```bash
python reputation_center_server.py
```

启动成功后输出：

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888 (Press CTRL+C to quit)
```

服务器自动创建 `reputation_center.db` 数据库。

### 3.3 运行测试

```bash
conda activate opencood
python test_reputation_system.py
# 预期: 7/7 测试通过
```

### 3.4 配置训练 YAML

```yaml
trust_fusion:
  use_trust_fusion: true
  reputation_server_url: "http://localhost:8888"
  cache_capacity: 100
  cache_ttl: 60

ego_id: "ego_vehicle_001"   # ego 车辆固定返回信誉 1.0
```

---

## 4. 工作流程

每一帧的处理流程如下：

```
[数据集] 获取各 CAV 信誉值（缓存优先，未命中则请求服务器）
    ↓
[数据集] 收集各 CAV 原始检测框 → self.current_frame_detections
    ↓
[数据集] 执行检测融合 → fused_boxes, fused_scores
    ↓
[客户端] POST /report_fused_boxes 上报融合结果
    ↓
[服务器] 计算每辆 CAV 的 IoU 一致性评分
         delta = (consistency - 0.5) × 0.1
         new_rep = clip(old_rep + delta, 0.0, 1.0)
    ↓
[服务器] 写入数据库，返回更新后的信誉值
    ↓
[客户端] 更新本地缓存，下一帧立即生效
```

**一致性评分逻辑**：对每辆 CAV，统计其检测框与融合框之间 IoU > 0.5 的匹配比例作为一致性分数。分数越高信誉增长越多，反之下降。

| 一致性分数 | delta | 信誉变化 |
|-----------|-------|---------|
| 1.0 | +0.05 | 上升 |
| 0.5 | ±0.00 | 不变 |
| 0.0 | -0.05 | 下降 |

---

## 5. 数据管理

### 查看信誉数据库

```bash
# 查看全部记录
sqlite3 reputation_center.db "SELECT * FROM vehicle_reputation;"

# 按信誉降序排列
sqlite3 reputation_center.db "SELECT * FROM vehicle_reputation ORDER BY reputation DESC;"

# 统计摘要
sqlite3 reputation_center.db "SELECT COUNT(*), AVG(reputation), MIN(reputation), MAX(reputation) FROM vehicle_reputation;"
```

**字段说明**:

```
vehicle_id | reputation | pass_count | fail_count | last_updated
veh_001    | 0.65       | 3          | 0          | 2026-06-07 10:30:45
veh_002    | 0.42       | 1          | 2          | 2026-06-07 10:30:45
```

### 重置数据库

```bash
# 删除文件（服务器重启后自动重建）
del reputation_center.db

# 或仅清空数据
sqlite3 reputation_center.db "DELETE FROM vehicle_reputation;"
```

---

## 6. 调优参数

### 修改信誉更新幅度

在 `reputation_center_server.py` 中修改：

```python
if is_verified:
    delta = 0.05   # 验证通过时信誉增量（加大则收敛更快）
else:
    delta = -0.1   # 验证失败时信誉减量（绝对值加大则惩罚更重）
```

### 修改 IoU 匹配阈值

```python
if iou > 0.5:   # 降低到 0.3 宽松匹配，升高到 0.7 严格匹配
    matched += 1
```

### 多机训练共享信誉

1. 服务器机器确保监听 `host="0.0.0.0"`
2. 其他机器 YAML 指向服务器 IP：

```yaml
trust_fusion:
  reputation_server_url: "http://192.168.1.100:8888"
```

---

## 7. 故障排查

| 问题 | 原因 | 解决方法 |
|------|------|---------|
| `No module named 'fastapi'` | 缺少依赖 | `pip install fastapi uvicorn pydantic` |
| `ConnectTimeout` | 服务器未启动或防火墙 | 确认服务器运行，检查 8888 端口 |
| 测试 7 失败 `No module named 'opencood'` | 从错误路径导入 | 确认测试从 `reputation_client_adapter` 导入 |
| `TypeError: 'type' object is not subscriptable` | Python 3.7 不支持 `list[str]` 语法 | 使用 `typing.List[str]`（已修复） |
| 训练时服务器挂了 | 网络异常 | 客户端自动返回默认值 0.5，训练不中断 |

---

## 8. FAQ

**Q: ego 车辆为什么信誉固定是 1.0？**
A: ego 作为融合中心，其检测结果是评估其他 CAV 的基准，不参与一致性评估。

**Q: 信誉值会超出 [0, 1] 范围吗？**
A: 不会，服务器端强制 `clip(0.0, 1.0)`。

**Q: 缓存过期后会丢失数据吗？**
A: 不会，过期只触发重新从服务器拉取，数据始终持久化在数据库中。

**Q: 后台线程有什么作用？**
A: 每 10 秒全量同步一次服务器数据，确保长期运行时缓存与服务器数据保持一致。

**Q: 批量查询和单次查询有什么区别？**
A: 批量查询一次 HTTP 请求获取多个信誉值，延迟更低，每帧建议使用批量查询获取所有参与 CAV 的信誉。
