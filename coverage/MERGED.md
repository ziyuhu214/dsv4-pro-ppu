# graph + eager 合并总表

`merged-graph-eager-coverage.csv` —— 把 `vllm-gpu-coverage` skill 的两轮
（graph 09-28、eager 09-29）合成一张表，去重后逐算子一行。

> 口径同 skill，**不是** `FlagOS覆盖率计算逻辑说明` 的口径。
> 两者分母定义不同（见 `spec-run-INCOMPLETE-graph-1024x1024/README.md`），不可混用。

## 结果

| 范围 | 行数 | 已覆盖 | 比例 |
|---|---:|---:|---:|
| GPU 关联 ATen | 31 | 27 | 87.10% |
| 独立 GPU kernel | 47 | 21 | 44.68% |
| **合计** | **78** | **48** | **61.54%** |

对照两轮各自的结果：

| | 分母 | 已覆盖 | 比例 |
|---|---:|---:|---:|
| graph 轮 | 73 | 48 | 65.75% |
| eager 轮 | 67 | 45 | 67.16% |
| **合并** | **78** | **48** | **61.54%** |

**合并后比例低于任一单轮**，这是对的，不是算错：合并把两轮各自没看到的算子都并进了分母
（graph 少了 eager 那 22 行、eager 少了 graph 那 11 行），而分子只有两轮都算覆盖的并集。
分母涨得比分子多，比例必然下降。

## 去重规则

两轮的原始 kernel 符号并集是 **85 个**（graph 75、eager 74、共有 64）。
同一个 kernel 在两轮里可能被归到不同类别，直接并集会重复计数。

### 关键冲突：17 个 kernel 在 graph 轮是独立项、在 eager 轮是 ATen 的实现

graph 模式下 99.0% 的 kernel event 是 graph replay、没有 ATen parent，
这些 kernel 只能算「独立 kernel」；eager 打开归因后它们露出了真实的 ATen 上游。
反方向（eager 独立、graph 有 ATen parent）是 **0 个**，
所以 **eager 的归因严格优于 graph，冲突一律以 eager 为准**。

涉及的 17 个：

```
_gather_flaggems_jit_function        _repeat_flaggems_jit_function
add_func_kernel_rank_2               add_func_tensor_scalar_kernel_rank_1
embedding_kernel                     full_func_scalar_kernel_rank_1
masked_fill_kernel_kernel_rank_2     mul_kernel
mul_scalar_kernel                    sigmoid_forward_kernel_rank_1
softplus_forward_kernel_rank_1       sqrt_func_kernel_rank_1
sum_dim_kernel_inner                 topk_single_stage_kernel
reduction_dtypeFP32xFP32xFP32_align4x4_...
void at::native::elementwise_kernel<128, 4, ...>
void at::native::vectorized_elementwise_kernel<4, ...launch_clamp_scalar...>
```

### 最终规则

1. **ATen 行** = 两轮 ATen 名称的并集，逐 API 名一行（§5.1.1 口径）
2. **独立 kernel 行** = 在**两轮里都**没有 ATen parent 的 kernel
3. **覆盖判定** = 任一轮判定为覆盖即覆盖，`coverage_reason` 取并集

### 校验

```
原始 kernel 符号并集            85
  其中支撑某个 ATen 行           38
  两轮都无 ATen parent（独立）   47
  38 + 47 = 85                  ✓ 无遗漏
两类别交集                      空  ✓ 无 kernel 既算 ATen 实现又算独立项
重复 operator_id                无  ✓
covered 行都有 reason           ✓
uncovered 行都无 reason         ✓
```

### 一个不是重复的情况

4 个 kernel 各支撑 2 个 ATen 行：

| kernel | 支撑的 ATen |
|---|---|
| `void at::native::vectorized_elementwise_kernel<4, ...launch_clamp_scalar...>` | `aten.clamp`、`aten.clamp_` |
| `true_div_func_kernel_rank_2` | `aten.div`、`aten.div_` |
| `mul_scalar_kernel` | `aten.mul`、`aten.mul_` |
| `zeros_kernel` | `aten.zero_`、`aten.zeros` |

都是 in-place / out-of-place 成对的 ATen API 共用一个 kernel 实现。
**分母数的是算子不是 kernel**，按 §5.1.1「应按照 ATen API 名称进行统计」，
这是 2 个算子而不是重复计数。已确认没有任何 kernel 同时出现在「ATen 实现」和「独立 kernel」
两个类别里 —— 那才会是真的重复。

## 各轮独有的行

| | 行数 |
|---|---:|
| 两轮都出现 | 45 |
| 仅 graph 轮 | 11 |
| 仅 eager 轮 | 22 |

两轮压测形状不同（graph 是 case1 1024/1024 并发 64；eager 是并发 8 × 96 token），
所以各有对方没触发到的算子。CSV 里的 `seen_in_graph_round` / `seen_in_eager_round`
两列标了每行的来源，可据此筛回任一单轮。

## 未覆盖的 30 行

**ATen 4 个**：`aten.clamp`、`aten.clamp_`、`aten.copy_`（命中 plugin-fl
`thead.yaml` 的 `flagos_blacklist`，落回 `at::native`）、`aten.mm`（走 vendor GEMM）。

**独立 kernel 26 个**：comm（pccl 3 个）、cutlass deep_gemm、flash attention、
tilelang 的 `mhc_*`/`hc_*`、vendor `gemm_ktype0_*` 等非 Triton 实现。

## 列说明

| 列 | 含义 |
|---|---|
| `operator_id` | ATen 行是 API 名；独立 kernel 行是 `kernel::<符号>` |
| `operator_kind` | `aten` / `kernel` |
| `covered` / `coverage_reason` / `implementation_source` | 任一轮的判定与理由并集 |
| `seen_in_graph_round` / `seen_in_eager_round` | 该行在哪一轮出现过 |
| `gpu_kernel_count` | 该算子对应的 kernel 符号数 |
| `graph_calls` / `graph_kernel_ms` | 该算子在 graph 轮 rank0 的调用数与 kernel 时间 |
| `eager_calls` / `eager_kernel_ms` | 同上，eager 轮 |
| `gpu_kernels` | 全部原始 kernel 符号，` \| ` 分隔，保留完整名便于核查 |

调用数与时间**仅供参考，不参与覆盖率计算** —— 覆盖率按算子种类计，
不按调用次数或执行时长加权。两轮压测形状不同，两组时间之间也不可直接相比。
