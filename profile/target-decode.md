# 目标栈 decode profile（单侧，**无基线对比**）

## ⚠️ 这是单侧数据，不能用来归因差距

只 profile 了目标栈。**基线栈从未被 profile 过。**
一个单侧 profile 可以说明「目标栈的时间花在哪」，
**不能说明「目标栈比基线慢在哪」** —— 后者需要两侧同条件的 profile 对比。

如果要做两侧对比：基线是 vLLM 0.20.1，`--profiler-config` 那套语法不适用，
需要用 `VLLM_TORCH_PROFILER_DIR`，**但这条机制未经验证，用之前先验**。

## 抓取条件

| | |
|---|---|
| 轮次 | 2026-09-28 18:20 |
| 栈 | vLLM 0.24.0 + plugin-fl + FlagGems |
| util | 0.90 |
| 形状 | case1（1024 prefill / 1024 decode，并发 64，256 请求） |
| 窗口 | 1856 token ≈ 29 个 decode step，取其中 25 个纯 decode step |
| rank | rank0（16 个 local rank 各一份 trace） |
| 体积 | 244 MB / 17 文件（16 worker + 1 async_llm 前端） |

交叉校验：profile 算出的 wall 74.99 ms/step，与实测 `Median ITL 78.73 ms` 一致（差 4.7%）。

## 子系统拆解（rank0，25 个纯 decode step 的均值）

```
wall 74.99 ms/step，GPU kernel 合计 79.89 ms/step

子系统                        ms/step    占比   calls/step
comm                          30.910    38.7%      230.6
dense GEMM (vendor/acext)     13.829    17.3%      513.3
deep_gemm GEMM (MoE+dense)    13.587    17.0%      171.4
mHC                            5.241     6.6%      343.8
ATTN sparse decode             3.924     4.9%       57.1
copy/alloc                     3.707     4.6%     2798.8
other                          3.562     4.5%     1345.2
KV                             1.667     2.1%      288.7
indexer                        1.605     2.0%      170.3
quant                          1.508     1.9%      485.4
```

## 并发与空闲

```
kernel 时间求和        88.511 ms/step
GPU busy union         81.845 ms/step   ← 用这个，不是求和
span                   83.876 ms/step
GPU idle                2.032 ms/step  (2.4%)
concurrency             1.08x
```

**看 busy union 而不是 kernel time 求和** —— 多流并发下求和会重复计时。
GPU idle 只有 2.4%，说明 decode 阶段 GPU 基本吃满，瓶颈不在 host 侧调度。

## 观察

1. **comm 占 38.7%**，`pcclKernel_twoShotAllReduceKernel` 单个 kernel 就 418.24 ms / 3198 calls
   （占全部 kernel 时间的 19.2%）。加上 ReduceScatter(226.50 ms) 和 AllGather(210.75 ms)，
   三个 comm kernel 合计 855 ms = 39.3%。
2. **`copy/alloc` 每步 2798.8 次调用**但只占 4.6% 时间 —— 次数极多、单次极短。
3. **`zeros_kernel` 47892 calls / 49.94 ms** 是调用次数最多的单个 kernel。
4. concurrency 仅 1.08x，说明多流重叠很少。

## 未解决

**差距根因没有定位。** 已排除和待查：

- ❌ **已排除**：K2 blacklist 失效 —— 实测 blacklist 13 项在生效，见 `../findings/flagos-blacklist.md`
- ❌ **已排除**：DeepGEMM 配置的 device-name 匹配 bug —— 7 个 PPU-ZW810E 文件正常加载
- ⬜ **待查**：DeepGEMM 调优表 `num_groups=12` 零命中 —— 见 `../findings/deepgemm-num-groups.md`
- ⬜ **待查**：comm 占 38.7% 是否高于基线（**需要基线 profile 才能回答**）
- ⬜ **待查**：case4 额外那约 5pp 差距

## 数据未附

trace 原始文件（244 MB graph 轮 + 987 MB eager 轮）没有放进仓库，体积过大。
`analyze_timeline.py` / `decode_breakdown.py` 的输出摘要即上表。

注意 `analyze_timeline.py` 有硬编码 `iters = 12`（无 step 自动检测），
用在非 12 step 的 trace 上会把 per-step 数值放大，必须传 `--iters <实际步数>`。
`decode_breakdown.py` 靠 `new_0` annotation 检测纯 decode step，是可信的那个工具。
