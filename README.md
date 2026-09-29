# DeepSeek-V4-Pro on PPU：基线栈 vs 目标栈

在 2 节点 × 16 张 PPU-ZW810E（world=32）上，对 DeepSeek-V4-Pro W8A8 比较两套推理栈，
并用 `vllm-gpu-coverage` skill 抓目标栈的 GPU 算子覆盖率。

| | 基线栈 | 目标栈 |
|---|---|---|
| vLLM | T-Head 0.20.1 | 官方 0.24.0 |
| kernel provider | T-Head 自带 | plugin-fl + FlagGems v5.3.4 |
| 并行 | dp=2 × tp=16 + EP，ep_size=32 | 同 |

## 结论摘要

### 性能：目标栈慢 9-14%

| case | prefill/decode | Total tok/s 差距 | util 对齐？ |
|---|---|---:|:---:|
| 1 | 1024/1024 | −9.1% | ❌ 0.90 vs 0.85 |
| 2 | 4096/1024 | −11.2% | ❌ 0.90 vs 0.85 |
| 3 | 16384/1024 | −9.4% | ❌ 0.90 vs 0.85 |
| 4 | 65536/1024 | **−14.3%** | ✅ 两侧 0.90 |

**case1-3 的数字带 util 混淆**（目标栈跑在 0.85，基线 0.90，KV 差 37%），需要重跑。
只有 case4 是干净的。详见 [`bench/RESULTS.md`](bench/RESULTS.md)。

### 算子覆盖率：48/73（graph）→ 45/67（eager）

| 范围 | graph 轮 | eager 轮 |
|---|---:|---:|
| GPU 关联 ATen | 13/16（81.25%） | **24/28（85.71%）** |
| 独立 GPU kernel | 35/57（61.40%） | 21/39（53.85%） |
| 合计 | **48/73（65.75%）** | **45/67（67.16%）** |

graph 模式下 99.0% 的 kernel event 是 CUDA graph replay、没有 ATen parent，
ATen 分母塌到 16；`--enforce-eager` 后恢复到 28。详见 [`coverage/COVERAGE.md`](coverage/COVERAGE.md)。

### 差距根因：**未定位**

只有单侧（目标栈）profile，无法归因两栈差异。已排除两条、仍有两条待查。
详见 [`profile/target-decode.md`](profile/target-decode.md)。

## 必读的限制

1. **差距根因没查清。** 本仓库不含任何两侧 profile 对比。
   要做需要另跑一轮基线 profile（0.20.1 要用 `VLLM_TORCH_PROFILER_DIR`，**这条机制未验证**）。
2. **case1-3 的 util 不一致**，那三个百分比不能当作干净结论。
3. **覆盖率是类型覆盖率**，不是调用次数/GPU 时间/吞吐比例；
   分子含框架原有 Triton 和 TorchInductor 生成的，**不能当插件贡献率**。
4. **eager 轮的吞吐（~1.5 s/decode step，graph 模式的 19 倍）不是性能数据**，
   只用于覆盖率，不可进任何性能对比。
5. case4 只跑了 1 轮，而那 1 轮本身是 warmup 轮（正常口径会丢弃第 1 轮）。

## 目录

```
bench/                基线与目标栈压测原始数据
  RESULTS.md          对比表 + util 混淆说明
  baseline/case{1..4}/
  target/case{1..4}/
coverage/             算子覆盖率
  COVERAGE.md         两轮结果、口径、Gems 日志贡献 0 的对照实验
  graph-09-28/        含 summary.json / coverage.csv / verification.json
  eager-09-29/
  eager-09-29-control-no-gems/    对照组
  gems-oplists/       FlagGems 执行日志（DP0 侧）
profile/
  target-decode.md    目标栈 decode 子系统拆解（单侧）
findings/
  deepgemm-num-groups.md   num_groups=12 零命中（待查）
  flagos-blacklist.md      K2 blacklist 实测生效（已排除）
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
- 两节点用外部 LB 式 DP（`--data-parallel-rank`），**握手有 5 分钟硬超时**，
  所以两侧必须约定整点启动。
- 容器 PID 1 是 `sleep infinity`，**没有 init reaper**，僵尸进程会堆积；
  判断进程死活不能用 `[ -d /proc/$PID ]` 或 `ps -p`（僵尸仍保留 `/proc/<pid>`），
  要读 `/proc/<pid>/stat` 的 state 字段并把 `Z` 当作已死。

## 已知的流程缺陷

约定模式只负责**起**服务，**没有任何一侧负责停**。
09-28 那轮两个节点各自漏了 17 小时（每卡 ~90 GB、util 0%），是系统性缺陷不是疏忽。
计划在 launcher 里加可选的 `MAX_ALIVE_SEC` 看门狗，**尚未实现**。
