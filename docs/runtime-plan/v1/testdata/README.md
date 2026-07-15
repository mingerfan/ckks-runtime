# RuntimePlan V1 测试数据

`valid/` 是 reader、verifier 和 Runtime 必须接受的冻结样例；`invalid/` 每份只引入一个错误，必须被拒绝。

合法样例覆盖 inline Encode、bundle Encode、同一 content 多次编码、Host compute、Host decrypt/re-encrypt Boot、Device native Boot、显式 Transfer/Replicate、Rotate、Relinearize 和 Rescale。

`bundles/v005-demo/` 使用冻结的纯 `blobs` manifest。`generate.py` 先写最终 OperatorSpec 和 manifest 字节，再计算原始字节 SHA-256 写入计划。重新生成命令：

```bash
python3 docs/runtime-plan/v1/testdata/generate.py
```

非法样例依次覆盖版本、整数类型、重复 ValueId、通信列表、Boot profile/目标、先用后定义、未知字段、Place、external input、OperatorSpec/manifest 摘要、Encode 阶段、Rotate 规范化和未使用 ValueDesc。
