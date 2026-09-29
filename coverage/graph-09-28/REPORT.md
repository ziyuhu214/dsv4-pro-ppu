# DeepSeek-V4-Pro (vLLM 0.24.0 + plugin-fl + FlagGems, dp2xtp16 EP32) GPU 算子覆盖率

分子按 Gems + Triton 口径，含框架原有及编译器生成的 Triton；不表示全部由 FlagGems 实现或插件新增替换。

| 范围 | 已覆盖/总数 |
|---|---:|
| GPU 关联 ATen | 13/16（81.25%） |
| 独立 GPU kernel | 35/57（61.40%） |
| 合计 | **48/73（65.75%）** |

精确符号（分组前）：48/73（65.75%）。已审核分组 0 个。
排除无 GPU 关联 ATen 47 项，其中原已覆盖 0 项；分子分母同步筛选。
原始 GPU kernel 符号 75 个，均可通过映射表追溯；未因关联失败遗漏 GPU kernel。

`operators.csv` 是四列分母，`covered.csv` 是四列分子（上传覆盖），两者配套使用。
`coverage.csv` 含 covered、原因和实现来源；`kernel_mapping.csv` 保留全部原始 kernel 符号。
分组后的 kernel_name 是审核的组名，不能当原始符号精确匹配。

这是类型覆盖率，不是调用次数、GPU 时间或吞吐比例。cpu_op 不能用来直接判断设备。
无 GPU 关联不等于 CPU-only；graph 内缺失 ATen 不补造名称。
Gems 日志可能包含初始化及预热；实际注册映射与本次观察集合取交集，不推断每个同名调用均被替换。
来源未确认时如实保留未知；metadata 只证明 Triton 类型，源目录名字不能证明实现归属。

采集范围与模式：`summary.json` 的 capture；完整证据与输入哈希：`operator_evidence.json`、`verification.json`。
