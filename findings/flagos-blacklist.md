# FlagGems blacklist（K2 修复）实测在生效 —— 已排除

**状态：已排除，不是本轮差距的原因**

## 背景

2026-09-03 在 DeepSeek-V4-Flash TP8 上测到：FlagGems 用 Triton 覆盖了
`copy_` / `clamp*` / `cat*` / `arange*` 这类极廉价的算子，**host 侧每次调用贵 5-10 倍**。
实测 `cpu_op` 合计 T-Head 4.712 ms/step vs FL 18.393 ms/step，
其中 `aten::copy_` 0.852 ms/56 calls vs 4.510 ms/54 calls。

修复是 plugin-fl `3c78240`：在 `thead` 平台上把 13 个廉价算子加进 blacklist，
让它们回落到原生实现。

## 本轮验证：两条独立证据

### 1. 运行时实测（import 后直接读配置）

```
platform    : thead
config path : /workspace/vllm-plugin-FL/vllm_fl/dispatch/config/thead.yaml
count       : 13
MATCH yaml  : True
```

这是实际 import 跑出来的，不是读代码推的。

选路路径：`get_platform_name()` → `current_platform.vendor_name` → `"thead"` → `thead.yaml`
（`dispatch/config/utils.py:89-113` 的 `get_config_path()`：`_CONFIG_DIR / f"{platform}.yaml"`）。
**文件名与 vendor 精确匹配**，不是 DeepGEMM 那个 device-name bug 的形状。

blacklist 13 项（`dispatch/config/thead.yaml:101-114`）：

```
copy_, clamp, clamp_tensor, clamp_, clamp_tensor_, clamp_max, clamp_max_,
clamp_min, clamp_min_, cat, cat_out, arange, arange_start
```

### 2. trace 侧独立印证（算子覆盖率的副产品）

两轮覆盖率抓取里，**未覆盖的 ATen 恰好就是 blacklist 里出现过的那几个**：

| 轮次 | 未覆盖 ATen | 是否在 blacklist |
|---|---|---|
| graph 09-28 | `aten.clamp`、`aten.clamp_`、`aten.copy_` | 全部 True |
| eager 09-29 | `aten.clamp`、`aten.clamp_`、`aten.copy_`、`aten.mm` | 前三个 True，`mm` 不在表里 |

`clamp` / `clamp_` / `copy_` 在覆盖率口径下算「未覆盖」，
正是因为它们被 blacklist 挡掉、**没有走 FlagGems 的 Triton 实现**，落回了 `at::native`。
graph 轮能看到它们链到 `at::native::vectorized_elementwise_kernel<...launch_clamp_scalar...>`
和 `at::native::unrolled_elementwise_kernel<...direct_copy_kernel_cuda...>`。

**两轮都成立，这个对应关系是稳的。**

## 一个坑

判断 blacklist 是否生效时，**`vllm_fl/utils.py:142-151` 的优先级-3 配置加载被
`except Exception: pass` 包着** —— 这是典型的静默失败形状，一开始怀疑它吞了异常。
**实测证伪**：配置确实加载了，13 项在生效。代码形状可疑 ≠ 实际有问题。

另一个坑：`flag_gems.enable(unused=...)` 匹配的是**函数 `__name__`**，不是 aten op 名。
所以 blacklist 里写的名字要对应 FlagGems 实现函数名。`VLLM_FL_FLAGOS_BLACKLIST` 环境变量会覆盖 yaml。

## 结论

K2 已修且在生效，**不是本轮 9-14% 差距的原因**。

但要注意 K2 只挑了 host 侧最重的 13 个，而那轮是 Flash TP8。
**Pro TP16 的算子分布可能不同 —— 还有很多没进 blacklist 的 FlagGems 覆盖算子，
这条仍是开放假设**（见 `../profile/target-decode.md` 的待查项）。
本轮的覆盖率数据可以用来列出候选：`../coverage/eager-09-29/coverage.csv` 里
`covered=true` 且 `coverage_reason` 含 `flaggems_execution_log_intersection` 的 24 个 ATen。
