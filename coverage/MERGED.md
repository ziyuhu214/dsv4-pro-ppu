# graph + eager 合并总表（已归一化特化变体）

`merged-coverage-normalized.csv` —— `vllm-gpu-coverage` skill 两轮
（graph 09-28、eager 09-29）合成一张表，**逐算子一行**，
特化变体（tile / dtype / rank / 模板参数）已按 §5.1.2 / §5.1.5 / §5.1.6 合并。

> 口径同 skill，**不是** `FlagOS覆盖率计算逻辑说明` 的口径。
> 两者分母定义不同（见 `spec-run-INCOMPLETE-graph-1024x1024/README.md`），不可混用。

## 结果

| 范围 | 行数 | 已覆盖 | 比例 |
|---|---:|---:|---:|
| GPU 关联 ATen | 31 | 27 | 87.10% |
| 独立 GPU kernel | 39 | 21 | 53.85% |
| **合计** | **70** | **48** | **68.57%** |

## 两类去重，缺一不可

### 一、跨轮归类冲突（17 个 kernel）

两轮原始 kernel 符号并集 **85 个**（graph 75、eager 74、共有 64）。
graph 模式下 99.0% 的 kernel event 是 graph replay、没有 ATen parent，
这些 kernel 在 graph 轮只能算「独立 kernel」；eager 打开归因后露出了真实上游。

**17 个 kernel 在 graph 轮是独立项、在 eager 轮是 ATen 的实现。**
反方向 0 个 —— eager 的归因严格优于 graph，**冲突一律以 eager 为准**。

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

规则：ATen 行 = 两轮 API 名并集；独立 kernel 行 = **两轮都**无 ATen parent 的 kernel。

### 二、同一 kernel 的不同形状 / 特化（47 → 39 行）

规范要求「同一个 kernel 仅因设备编号、Rank、模板参数或数据类型等因素生成了不同的特化版本，
则应在归一化后合并统计」。4 组被合并：

| 归一化后 | 折入行数 | 原始差异 |
|---|---:|---|
| `cutlass::deep_gemm::GemmKernel<GemmType3,num_groups=12>` | 4 | GemmShape tile 16x128x128 / 16x128x256 / 64x128x128(×2)，N=6144/7168 |
| `cutlass::gemm::kernel::GemmWithEpilogueVisitor` | 3 | tile 16x64x128 / 32x128x64 / 64x128x64 |
| `gemm_ktype0_aiu1_mtype1` | 3 | 只差 `_dtype...__tile32x32x16...` / `_tile64x256x32...` / `_tile64x64x16...` |
| `cutlass::deep_gemm::GemmKernel<GemmType0,num_groups=1>` | 2 | tile 32x64x256 / 64x64x128 |

**4 组全部 `covered=false`**，所以合并合法 —— skill 自身规则
（`coverage.py:247`）拒绝合并 covered 与 uncovered 混合的组，本例不触发。
**分子完全不变（48）**，这次归一化只缩分母。

#### 刻意不合并的（功能不同，§5.1.2「具有不同功能的 kernel 应分别计数」）

| 保持独立 | 理由 |
|---|---|
| `cutlass::deep_gemm::GemmKernel` 的 `GemmType0,num_groups=1` vs `GemmType3,num_groups=12` | 前者是 dense GEMM、后者是 MoE grouped GEMM。plugin-fl 有各自独立的入口（`_int8_grouped_nopad_impl` / `_int8_grouped_masked_impl` 等），是不同功能不是特化 |
| `pcclKernel_AllGather` / `Broadcast` / `Reduce` / `ReduceScatter` / `twoShotAllReduceKernel` | 5 个不同的集合通信操作。§5.1.4：同一通信 API 下不同功能的 kernel 分别计数 |
| `flash::flash_fwd_splitkv_mla_combine_kernel` vs `flash::flash_sparse_decode_fwd_kernel` | 两个不同的 attention kernel |

各自的 dtype / 模板特化仍在组内合并（如 pccl 的 `int8_t` 与 `__ppu_bfloat16`）。

### 三、看起来像重复但不是的

4 个 kernel 各支撑 2 个 ATen 行：

| kernel | 支撑的 ATen |
|---|---|
| `void at::native::vectorized_elementwise_kernel<4,...launch_clamp_scalar...>` | `aten.clamp`、`aten.clamp_` |
| `true_div_func_kernel_rank_2` | `aten.div`、`aten.div_` |
| `mul_scalar_kernel` | `aten.mul`、`aten.mul_` |
| `zeros_kernel` | `aten.zero_`、`aten.zeros` |

in-place / out-of-place 成对的 API 共用一个实现。**分母数的是算子不是 kernel**，
按 §5.1.1 按 ATen API 名统计，这是 2 个算子而非重复。
已确认没有任何 kernel 同时落在「ATen 实现」和「独立 kernel」两个类别 —— 那才是真重复。

## 校验

```
原始 kernel 符号并集              85
CSV 中引用到的符号                85      ✓ 无遗漏、无凭空多出
出现在 >1 行的符号                 4      全部是 ATen-vs-ATen 共用实现，无跨类别
重复 operator_id                  无      ✓
重复 (kind, name)                 无      ✓
合并行内 covered 状态不混          ✓
```

## 与各单轮的对照

| | 分母 | 已覆盖 | 比例 |
|---|---:|---:|---:|
| graph 轮（skill 原始产出） | 73 | 48 | 65.75% |
| eager 轮（skill 原始产出） | 67 | 45 | 67.16% |
| 合并、未归一化特化 | 78 | 48 | 61.54% |
| **合并 + 归一化特化** | **70** | **48** | **68.57%** |

分子一路是 48（eager 单轮 45 是因为它少看到 3 个算子）。
分母 78 → 70 就是这次归一化的效果。

## 各轮独有的行

| | 行数 |
|---|---:|
| 两轮都出现 | 47 |
| 仅 graph 轮 | 5 |
| 仅 eager 轮 | 18 |

两轮压测形状不同（graph 是 case1 1024/1024 并发 64；eager 是并发 8 × 96 token），
各有对方没触发到的算子。`seen_in_graph_round` / `seen_in_eager_round` 两列可筛回单轮。

## 未覆盖的 22 行

**ATen 4 个**：`aten.clamp`、`aten.clamp_`、`aten.copy_`（命中 plugin-fl
`thead.yaml` 的 `flagos_blacklist`，落回 `at::native`）、`aten.mm`（走 vendor GEMM）。

**独立 kernel 18 个**：comm（pccl 5 个）、cutlass deep_gemm（2 个）、
GemmWithEpilogueVisitor、vendor `gemm_ktype0_*`、flash attention（2 个）、
tilelang 的 `mhc_*` / `hc_*` 等非 Triton 实现。

## 列说明

| 列 | 含义 |
|---|---|
| `operator_id` | ATen 行是 API 名；独立 kernel 行是 `kernel::<family 的 sha256 前 16 位>` |
| `operator_name` | 归一化后的算子名 |
| `variant_count` | 该行折入了几个原始 kernel 符号（1 = 未发生特化合并） |
| `normalization_basis` | 该行的归一化依据（规范条款） |
| `covered` / `coverage_reason` / `implementation_source` | 任一轮判定为覆盖即覆盖，理由取并集 |
| `seen_in_graph_round` / `seen_in_eager_round` | 该行在哪一轮出现过 |
| `gpu_kernel_count` | 该算子对应的原始 kernel 符号数 |
| `graph_calls` / `graph_kernel_ms` | graph 轮 rank0 的调用数与 kernel 时间（组内求和） |
| `eager_calls` / `eager_kernel_ms` | 同上，eager 轮 |
| `raw_kernel_symbols` | 全部原始 kernel 符号，` \| ` 分隔，完整保留便于核查（§5.1.6） |

调用数与时间**不参与覆盖率计算** —— 覆盖率按算子种类计，不按调用次数或执行时长加权。
两轮压测形状不同，两组时间之间也不可直接相比。

未归一化的那版保留为 `merged-graph-eager-coverage.csv`（78 行），便于对照本次合并做了什么。
