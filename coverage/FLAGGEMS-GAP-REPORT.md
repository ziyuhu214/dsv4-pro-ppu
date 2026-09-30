# merged-coverage-normalized.csv 未覆盖算子：FlagGems 缺口分类与实测

对象：`coverage/merged-coverage-normalized.csv` 70 行算子中 `covered=false` 的 **22 个**
（ATen 4 个 + 独立 kernel 18 个）。目标是区分「FlagGems 里没有」与「FlagGems 有但性能差」，
并对可测的做 kernel 级 A/B。

硬件 PPU-ZW810E 单卡；FlagGems v5.3.4；vLLM 0.24.0 + plugin-fl。

---

## 结论摘要

22 个未覆盖算子里，**真正「FlagGems 没有实现」的只有 5 个**（全部是通信算子，不属于
FlagGems 职责范围）。其余 17 个 FlagGems 都有实现，未覆盖的原因分三类：

| 类别 | 数量 | 原因 | 是否性能问题 |
|---|---|---|---|
| A 完全没有 | 5 | 通信域，FlagGems 不覆盖；FlagCX 未安装 | — |
| B 有源码但派发路径未启用 | 13 | `apply_gems_patches_to_vllm()` 在本栈从未被调用 | 未生效，无法比较 |
| C 有实现且正确，被主动禁用 | 3 | `thead.yaml` 的 `flagos_blacklist` | 是，host 派发开销 10–18x |
| D 已注册但实测跑了原生 kernel | 1 | 未查明 | 待定 |

最关键的一条：**B 类的 13 个算子不是性能问题，而是接线问题**。FlagGems 为 vLLM 准备的
14 个库算子 + 7 个模块方法补丁全部挂在 `apply_gems_patches_to_vllm()` 上，而这个函数在
`vllm-plugin-FL` 和 `vllm-0.24.0` 整个已安装代码树里**没有任何调用点**（已全树 grep 确认）。
`flag_gems.enable()` 只做 ATen 注册，不碰这张表。所以 MLA decode、paged MQA logits、
cutlass_scaled_mm、hc_head_fused_kernel 这些 FlagGems 明明写好的实现，在本栈里一次都没跑过。

---

## A 类：FlagGems 确实没有（5 个）

`pcclKernel_AllGather`、`pcclKernel_Broadcast`、`pcclKernel_Reduce`、
`pcclKernel_ReduceScatter`、`pcclKernel_twoShotAllReduceKernel`

FlagGems 不含通信算子实现，这部分归 FlagCX，而 FlagCX 在本环境未安装，通信实际走 pccl。
按 FlagOS 规范 §5.1.4 通信算子恒计入分母，因此这 5 个是结构性缺口，无对照实现可测。

## B 类：有源码，但派发路径在本栈未启用（13 个）

| 算子（trace 中的 kernel） | FlagGems 对应实现 | 本栈为何未生效 |
|---|---|---|
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | `fused/mhc/mhc_pre.py` | 无注册项，且不在补丁表 |
| `mhc_post_tilelang_kernel` | `fused/mhc/mhc_post.py` | 同上 |
| `mhc_fused_tilelang_kernel` | `fused/mhc/` | 同上 |
| `hc_prenorm_gemm_tilelang_kernel` | `fused/mhc/mhc_pre.py` | 同上 |
| `hc_head_fuse_tilelang_kernel` | `fused/mhc/hc_head_fused_kernel.py` | **在补丁表内**，但补丁从未应用 |
| `cutlass::gemm::kernel::GemmWithEpilogueVisitor` | `fused/cutlass_scaled_mm.py` | **在补丁表内**，但补丁从未应用 |
| `cutlass::gemm::kernel::Sm80PagedMqaLogits` | `fused/bf16_paged_mqa_logits.py`、`fused/fp8_fp4_mqa_logits.py` | 不在补丁表，无入口 |
| `flash::flash_fwd_splitkv_mla_combine_kernel` | `fused/flash_mla.py` | 经 `TritonMLAImpl._forward_decode` 模块补丁，未应用 |
| `flash::flash_sparse_decode_fwd_kernel` | `fused/flashmla_sparse.py`、`fused/DSA/sparse_mla.py` | 不在补丁表，无入口 |
| `get_mla_metadata_kernel` | `fused/flash_mla_with_kvcache.py` | 不在补丁表，无入口 |
| `cutlass::deep_gemm::GemmKernel<GemmType0,num_groups=1>` | `ops/group_gemm.py` | deep_gemm 调用点无 FlagGems 入口 |
| `cutlass::deep_gemm::GemmKernel<GemmType3,num_groups=12>` | `ops/group_gemm.py` | 同上（MoE 分组 GEMM，12 专家/卡） |
| `gemm_ktype0_aiu1_mtype1` | `ops/mm.py` | 见 D 类 |

这 13 个里唯一已经拿到 kernel 级对照数据的是 mHC 系列（下文）。其余需要先把补丁接上才有
可比对象，目前不可测。

## C 类：FlagGems 有实现、正确，但被主动禁用（3 个）

`aten.clamp`、`aten.clamp_`、`aten.copy_`。

证据链是闭合的：serve 进程注册了 854 个 ATen key，而裸进程 `enable()` 注册 868 个，
差的 14 个恰好是 `arange`×3、`cat`×2、`clamp` 族×8、`copy_`——与
`vllm_fl/dispatch/config/thead.yaml` 的 `flagos_blacklist`（13 个函数名覆盖 14 个注册项）
完全吻合。这是本项目此前自己加的优化，不是 FlagGems 缺失。

实测复现了当初的判断，但把它拆成了两个量（`gems_k`/`nat_k` 为 profiler 统计的 device
kernel 时间，`gems_w`/`nat_w` 为含 host 派发的单次调用墙钟，单位 µs）：

| 算子 / 形状 | gems_k | nat_k | kernel 比 | gems_w | nat_w | 墙钟比 |
|---|---|---|---|---|---|---|
| clamp int32[64,256] | 1.45 | 1.63 | **0.89x** | 93.91 | 8.37 | 11.22x |
| clamp int32[64,8192] | 2.09 | 2.24 | **0.93x** | 90.17 | 6.44 | 14.00x |
| clamp int32[64,16384] | 2.61 | 3.13 | **0.83x** | 89.34 | 6.48 | 13.79x |
| clamp_ int32[64,8192] | 2.09 | 2.27 | **0.92x** | 70.64 | 5.38 | 13.13x |
| clamp_ int32[64,16384] | 2.82 | 3.19 | **0.88x** | 70.32 | 5.20 | 13.52x |
| clamp bf16[4096,2048] | 10.84 | 8.31 | 1.30x | 91.49 | 8.74 | 10.47x |
| copy_ bf16[64,16,8080] | 13.87 | —(1) | n/a | 100.23 | 5.51 | 18.19x |
| copy_ int32[64]→int64 | 0.79 | 2.05 | **0.39x** | 73.86 | 4.52 | 16.34x |

(1) 原生 `copy_` 走 DtoD Memcpy 搬运引擎而非 kernel，profiler 的 kernel 时间为 0，
只能用墙钟比较；FlagGems 用一个 Triton kernel 替掉了硬件 memcpy。

读法：**在 device 上 FlagGems 这几个 kernel 并不慢，整数路径甚至普遍更快
（0.83–0.93x，`copy_` 的 int32→int64 快 2.6x）。差距全部在 host 派发**——Python +
Triton JIT cache 查找 + launch，每次调用 70–100 µs，而原生只要 4–9 µs。所以这是
per-call 固定开销问题，不是 kernel 质量问题。形状越小越吃亏，这也解释了为什么被黑名单选中的
都是廉价 elementwise / 搬运算子。

一个直接推论：这笔开销在 CUDA graph 模式下会被捕获后 replay 掉，只在 eager 路径上全额付出。
`aten.clamp` 在 graph 轮 4888 次调用、eager 轮 4512 次调用，量级不小。

正确性：13 个用例在两侧输入逐位相同（输入在 CPU 固定种子生成后拷到设备——启用 FlagGems 时
`torch.randn` 本身也会被派发到 Gems RNG，若在设备上生成，两次运行的输入数据就不一致，
早先一版比较因此完全失效）。结果 **13/13 完全一致（max_abs_diff = 0）**。

## D 类：已注册但实测跑了原生 kernel（1 个）

`aten.mm`。`mm`/`mm.out` 在 serve 的 854 个注册项里**在**，也**不在**黑名单（plugin-fl 自己的
单元测试还专门断言 `"mm" not in flagos_blacklist`）。但 trace 里 `aten.mm` 归因到的
唯一 kernel 是原生
`gemm_ktype0_aiu1_mtype1_dtypeBF16xBF16xFP32xFP32xFP32_tile16x64x16x16x128x4_layout0x1x0x0…`
（bf16 输入、fp32 输出、B 转置、M 很小）。

排查过并已排除的解释：FlagGems 无 `.so`（`USE_C_EXTENSION` 默认 0 且全栈未设置），
所以 `cpp_patched_ops` 为空，不是 C++ 抢占；`_thead` 后端没有 exclude YAML；
`mm` 的 config 项没有 condition 函数；Gems `mm` 对 bf16→fp32、`mm.out`、B 转置三种变体
都能正常派发到 `mm_kernel_general`，不存在回退。**这些 mm 调用的来源尚未查明**，
需要带 stack trace 重新 profile 才能定位，此处如实标为未决。

顺带一提 `aten.linear` 是 covered=true 且跑的是 FlagGems 的 `linear_kernel`，
说明 GEMM 类并非整体失效。

---

## 附：GEMM 类 A/B（FlagGems Triton vs 厂商原生）

虽然 `mm` 归到 D 类，但既然它是唯一「已注册却没跑上」的算子，顺手测了 FlagGems 的 GEMM
相对厂商原生的真实水位，用于判断「接上去到底值不值」：

| 算子 / 形状 | gems_k | nat_k | 比值 |
|---|---|---|---|
| mm bf16[64,7168]×[7168,2048] | 38.34 | 48.27 | **0.79x（Gems 更快）** |
| mm bf16[1024,7168]×[7168,2048] | 348.43 | 280.44 | 1.24x |
| mm bf16[4096,7168]×[7168,2048] | 1103.85 | 966.42 | 1.14x |
| mm bf16[4096,7168]×[7168,7168] | 3750.38 | 3155.38 | 1.19x |
| addmm bf16[4096,7168]×[7168,2048]+bias | 1657.50 | 972.71 | **1.70x** |
| bmm bf16[1,4096,7168]×[1,7168,2048] | 1110.28 | 971.72 | 1.14x |

结论：大 M 下 FlagGems GEMM 比厂商原生慢 14–24%，小 M（64）反而快 21%。
`addmm` 的 1.70x 是因为 FlagGems 的 `addmm_kernel` 不融合 bias，而原生走
`fusion5` epilogue 一次算完——这是可优化的具体点。数值上 6 个用例两侧逐位一致。

## 附：mHC 系列 A/B（vLLM tilelang vs FlagGems Triton）

B 类里唯一已有对照数据的一组，摘自先前 kernel 级测量（`ratio>1` 表示 FlagGems 更快）：

| tokens | mhc_post 比值 | mhc_pre 比值 |
|---|---|---|
| 1 | 0.51 | 0.56 |
| 8 | 0.50 | 0.50 |
| 32 | 0.49 | 0.51 |
| 64 | 0.51 | 0.49 |
| 128 | 0.50 | 0.49 |
| 256 | 0.50 | 0.49 |
| 1024 | 0.55 | **1.82** |
| 4096 | 0.50 | **2.18** |

`mhc_post`：两侧逐位相同（max_abs_diff = 0），tilelang 全程快约 2x，FlagGems 接进来没有收益。
`mhc_pre`：1024 tokens 是分水岭——`vllm/model_executor/kernels/mhc/tilelang.py:43`
的 `x.shape[0] >= 1024` 分支切到 `hc_prenorm_gemm_block_m_tilelang`，之后 FlagGems 快
1.82x / 2.18x。FlagGems 该路径把 prenorm GEMM 做成 bf16 cuBLAS
（`fused/mhc/mhc_pre.py:656-670`），tilelang 用 TF32，这解释了数值分歧随 token 数放大
（sinkhorn 会把差异放大成 assignment 翻转），因此不能只看速度就替换。

---

## 可执行的结论

1. **优先级最高不是调优，是接线。** B 类 13 个算子里，`hc_head_fused_kernel` 和
   `cutlass_scaled_mm` 已经在 FlagGems 的补丁表里，只差一次
   `apply_gems_patches_to_vllm()` 调用；MLA decode 的三个模块补丁同理。
   接上之后这些算子才可能进分子，也才有性能可比。
2. **C 类不要动。** FlagGems 的 kernel 本身不慢（整数路径更快），慢的是 per-call 派发。
   在 eager 路径上黑名单是正确选择；若要回收这部分覆盖率，方向是降低 Triton 派发开销
   （或只在 graph 模式下放开），而不是改 kernel。
3. **`addmm` 的 bias 不融合**是 FlagGems 侧一个明确可提的性能问题（1.70x）。
4. **A 类需要 FlagCX**，否则通信部分的分母无法回收。
5. `aten.mm` 的来源未查明，需要带 stack trace 的 profile；在此之前不把它归入任何性能结论。

## 复现

| 脚本 | 用途 |
|---|---|
| `bench/gems_dispatch.py` | 用 profiler 确认每个算子实际落到哪个 kernel（`GEMS=1`/`GEMS=0` 各跑一次） |
| `bench/gems_bench2.py` | device kernel 时间 + 墙钟双指标 A/B |
| `bench/gems_verify.py` | 两侧输入逐位相同的正确性比对 + registrar key 集合导出 |

均为单卡、`GEMS=1` 与 `GEMS=0` 分两个进程跑（ATen 注册是全局的，同一进程无法干净地来回切）。
