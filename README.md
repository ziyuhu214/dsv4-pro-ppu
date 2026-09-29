# DeepSeek-V4-Pro on PPU：基线栈 vs 目标栈

在 2 节点 × 16 张 PPU-ZW810E（world=32）上，对 DeepSeek-V4-Pro W8A8 比较两套推理栈，
并用 `vllm-gpu-coverage` skill 抓目标栈的 GPU 算子覆盖率。

| | 基线栈 | 目标栈 |
|---|---|---|
| vLLM | T-Head 0.20.1 | 官方 0.24.0 |
| kernel provider | T-Head 自带 | plugin-fl + FlagGems v5.3.4 |
| 并行 | dp=2 × tp=16 + EP，ep_size=32 | 同 |

## 性能：目标栈慢 9-14%

| case | prefill/decode | Total tok/s 差距 | Median ITL | Peak Output |
|---|---|---:|---:|---:|
| 1 | 1024/1024 | −9.1% | +9.2% | −13.3% |
| 2 | 4096/1024 | −11.2% | +17.9% | −14.3% |
| 3 | 16384/1024 | −9.4% | +16.4% | −15.4% |
| 4 | 65536/1024 | **−14.3%** | +16.2% | −19.4% |

并发 64、256 请求，4 轮取中位数（首轮作 warmup 丢弃；case4 只跑 1 轮）。
两侧 `gpu_memory_utilization` 有差异（case1-3 目标 0.85 / 基线 0.90），
**已验证不影响结论** —— case1 两侧 KV 余量都有 2 倍以上而仍差 −9.1%，
且缺口幅度不随 KV 压力变化。三条证据与全部指标见 [`bench/RESULTS.md`](bench/RESULTS.md)。

同 util=0.90 下两侧 KV 仍差 464,711 vs 422,138 = **9.2%** —— 两栈的真实差异。

## 算子覆盖率：48/73（graph）→ 45/67（eager）

| 范围 | graph 轮 | eager 轮 |
|---|---:|---:|
| GPU 关联 ATen | 13/16（81.25%） | **24/28（85.71%）** |
| 独立 GPU kernel | 35/57（61.40%） | 21/39（53.85%） |
| 合计 | **48/73（65.75%）** | **45/67（67.16%）** |

graph 模式下 99.0% 的 kernel event 是 CUDA graph replay、没有 ATen parent，
ATen 分母塌到 16；`--enforce-eager` 后恢复到 28。

**口径**：这是**类型**覆盖率，不是调用次数、GPU 时间或吞吐比例；
分子含框架原有 Triton 与 TorchInductor 生成的，**不等于插件贡献率**。
eager 轮只用于覆盖率，其吞吐（~1.5 s/decode step）不是性能数据。

未覆盖的 ATen 是 `clamp` / `clamp_` / `copy_`（命中 plugin-fl blacklist）与 `aten.mm`（vendor GEMM）。
详见 [`coverage/COVERAGE.md`](coverage/COVERAGE.md)。

## 目标栈 decode 时间分布

rank0，25 个纯 decode step，case1 形状。wall 74.99 ms/step，GPU busy union 81.845 ms/step，
GPU idle 2.4%，concurrency 1.08x。

| 子系统 | ms/step | 占比 |
|---|---:|---:|
| comm | 30.910 | 38.7% |
| dense GEMM (vendor/acext) | 13.829 | 17.3% |
| deep_gemm GEMM (MoE+dense) | 13.587 | 17.0% |
| mHC | 5.241 | 6.6% |
| ATTN sparse decode | 3.924 | 4.9% |
| copy/alloc | 3.707 | 4.6% |
| 其他 | 3.562 | 4.5% |
| KV | 1.667 | 2.1% |
| indexer | 1.605 | 2.0% |
| quant | 1.508 | 1.9% |

单侧数据（仅目标栈），详见 [`profile/target-decode.md`](profile/target-decode.md)。

## 目录

```
bench/                基线与目标栈压测原始数据
  RESULTS.md          全部指标 + util 差异为何不影响结论
  baseline/case{1..4}/
  target/case{1..4}/
coverage/             算子覆盖率
  COVERAGE.md         两轮结果、口径、Gems 日志贡献 0 的对照实验
  graph-09-28/        含 summary.json / coverage.csv / verification.json
  eager-09-29/
  eager-09-29-control-no-gems/    对照组
  gems-oplists/       FlagGems 执行日志（DP0 侧）
profile/
  target-decode.md    目标栈 decode 子系统拆解
findings/
  deepgemm-num-groups.md   DeepGEMM 调优表对 num_groups=12 零命中
  flagos-blacklist.md      K2 blacklist 实测在生效
plugins/
  README.md           两个插件的改动整理（仅整理，未提交上游）
```

## 环境要点

- `gpu_memory_utilization` 是**总**显存的比例，不是空闲显存
  （`gpu_worker.py:514-517` 的 `cg_util_delta = cudagraph_memory_estimate / total_mem`）。
- `o_groups` 把 TP 上限卡在 16，所以 tp=32 不可行；32 卡只能 dp=2 × tp=16。
- 目标栈要求 `/workspace/vllm-0.24.0/vllm/` 下**没有 `.so`**：
  `_C_ops_registry.py:168-172` 一旦 `import vllm._C` 成功就提前 return，
  跳过 105 个 schema 的注册 —— 插件本身就是 kernel provider。
- 384 experts / ep_size=32 = 12 experts/rank。
