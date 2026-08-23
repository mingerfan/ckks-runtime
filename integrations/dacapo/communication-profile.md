# Placement 通信 Profile

这份 JSON 只服务于 Dacapo placement，时间单位为微秒，速率单位为
bytes/us。它不进入 RuntimePlan V1，也不改变 Runtime 的传输语义。

```json
{
  "format_version": 1,
  "coefficient_bytes": 4,
  "links": {
    "host_device": {
      "startup_latency_us": 8,
      "max_rate_bytes_per_us": 12000,
      "saturation_bytes": 1048576
    },
    "intra_rank": {
      "startup_latency_us": 5,
      "max_rate_bytes_per_us": 16000,
      "saturation_bytes": 524288,
      "rate_points": [
        {"payload_bytes": 4096, "rate_bytes_per_us": 2000},
        {"payload_bytes": 1048576, "rate_bytes_per_us": 12000}
      ]
    },
    "inter_rank": {
      "startup_latency_us": 20,
      "max_rate_bytes_per_us": 3000,
      "saturation_bytes": 4194304
    }
  }
}
```

`host_device` 用于同 rank 的 Host/GPU 传输，`intra_rank` 用于同 rank 的
不同 GPU，`inter_rank` 用于不同 rank。当前不区分传输方向和具体 GPU 对。

## V2 有序规则

`format_version: 2` 可以在旧 `links` 兜底模型之上增加拓扑和有序的
point-to-point 规则：

```json
{
  "format_version": 2,
  "coefficient_bytes": 8,
  "topology": {"rank_to_node": [0, 1]},
  "links": {"host_device": {}, "intra_rank": {}, "inter_rank": {}},
  "rules": [
    {
      "id": "special-pair",
      "from": {"kind": "device", "rank": 0, "device": 3},
      "to": {"kind": "device", "rank": 1, "device": 2},
      "direction": "both",
      "transport": "nccl",
      "cost": {
        "startup_latency_us": 8,
        "max_rate_bytes_per_us": 50000,
        "saturation_bytes": 1048576
      }
    }
  ]
}
```

规则数组采用 first-match 语义：从前到后检查，第一条匹配规则生效，后面的
规则不会覆盖它。`rank`、`device` 和 `node` 可以使用非负整数或 `"*"`；
`kind` 可以是 `host` 或 `device`。`direction` 默认为 `forward`，也可以写
`both` 表示反向同样匹配。规则的 `cost` 是完整链路模型，不做字段级合并。

`rank_to_node` 的下标是 MPI rank，值是逻辑节点编号。它用于区分不同 rank
但位于同一节点的进程间通信与真正的跨节点通信。它的长度必须与 placement
传入的 `device-counts` rank 数一致。

V1 profile 无需修改，仍按原来的三类 `links` 工作。V2 中如果规则没有匹配，
则回退到存在的旧 `links` 模型；如果既没有匹配规则也没有兜底模型，placement
直接报错。Profile 只影响编译期 placement，RuntimePlan V1 不携带这些 cost 字段。

## Payload 估算

placement 从 CKKS 类型和 OperatorSpec 估算传输量：

```text
payload_bytes = tensor_elements
              * components
              * (level + 1)
              * poly_degree
              * coefficient_bytes
```

`level + 1` 是当前 Runtime level 语义下仍然活跃的 RNS limb 数。动态 tensor
shape、超出 OperatorSpec 模数链的 level 和整数溢出都会直接报错。

## 速率和耗时

存在 `rate_points` 时，payload 小于第一个点时从原点线性插值；位于两个点
之间时做普通线性插值；大于最后一个点时使用最后一个点的速率。点位 payload
必须严格递增，速率必须单调不减，而且不能超过 `max_rate_bytes_per_us`。

没有 `rate_points` 时使用饱和曲线：

```text
rate(bytes) = max_rate * bytes / (bytes + saturation_bytes)
```

两种情况最终都使用：

```text
cost_us = startup_latency_us + ceil(payload_bytes / rate(payload_bytes))
```

如果 placement 不传 communication profile，则继续使用旧的固定
`intra-rank-communication-cost` 和 `inter-rank-communication-cost`，用于兼容
已有命令和测试。

## 当前边界

这个首版模型只修正“所有 value 传输成本都一样”的问题，仍不建模链路争用、
双向带宽差异、通信计算重叠、Host staging、显存峰值、密钥复制和具体 GPU
之间的异构拓扑。这里的默认数值是联调用的粗估值，不是硬件实测结果。
