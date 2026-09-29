# FlagGems blacklist（K2 修复）实测在生效

## 背景

2026-09-03 在 DeepSeek-V4-Flash TP8 上测到：FlagGems 用 Triton 覆盖了
`copy_` / `clamp*` / `cat*` / `arange*` 这类极廉价的算子，**host 侧每次调用贵 5-10 倍**。
实测 `cpu_op` 合计 T-Head 4.712 ms/step vs FL 18.393 ms/step，
其中 `aten::copy_` 0.852 ms/56 calls vs 4.510 ms/54 calls。

修复是 plugin-fl `3c78240`：在 `thead` 平台上把 13 个廉价算子加进 blacklist，
让它们回落到原生实现。

## 验证：两条独立证据

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

## 结论

K2 已修且在生效 —— blacklist 13 项确实挡住了 FlagGems 对这些算子的覆盖。

K2 的 13 项是按 Flash TP8 上 host 侧最重的算子挑的。Pro TP16 下仍由 FlagGems 覆盖、
未进 blacklist 的算子可以从 `../coverage/eager-09-29/coverage.csv` 列出：
`covered=true` 且 `coverage_reason` 含 `flaggems_execution_log_intersection` 的 24 个 ATen。
