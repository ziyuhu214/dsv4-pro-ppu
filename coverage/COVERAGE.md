# GPU 算子覆盖率（vllm-gpu-coverage skill）

用 `vllm-gpu-coverage` skill 对目标栈（vLLM 0.24.0 + plugin-fl + FlagGems v5.3.4）抓的两轮。

## 结果

| 范围 | graph 轮 09-28 | eager 轮 09-29 |
|---|---:|---:|
| GPU 关联 ATen | 13/16（81.25%） | **24/28（85.71%）** |
| 独立 GPU kernel | 35/57（61.40%） | 21/39（53.85%） |
| **合计** | **48/73（65.75%）** | **45/67（67.16%）** |

两轮 `verification.json` 的 `checks_passed` 都是 `true`。

## 抓取条件

| | graph 轮 09-28 | eager 轮 09-29 |
|---|---|---|
| 形状 | case1（1024/1024、并发 64、256 请求） | 8 并发 × 96 token |
| 窗口 | 1856 token ≈ 29 step | 168 token / 8 seq ≈ 21 step |
| trace | 244 MB / 16 worker | 987 MB / 16 worker |

eager 每个算子都是真 launch 而非 replay，所以 trace 大 4 倍、
decode 约 **1.5 s/step**（graph 模式 Median ITL 78.73 ms 的 ~19 倍）。
**eager 轮只用于算子覆盖率，其吞吐不是性能数据。**

## 为什么要抓两轮

graph 轮的 ATen 分母塌了：

```
                       graph 09-28   eager 09-29
kernel 符号                    75            74
kernel event              156286        141264
  其中 graph replay      154778(99.0%)    0(0.0%)
有 ATen parent 的符号          18            35
可关联的 ATen 基名             16            28
```

**99.0% 的 kernel event 在 CUDA graph 里，没有 ATen parent。**
按 skill 的 `references/accounting.md:10`「graph launch 不暴露内部 ATen，不把整个 graph 归到一个外层 API」，
这些 kernel 的上游算子无法归因，ATen 分母只剩 graph 外的 1.0%（16 项）。
skill 自带的参考案例 Qwen3.5-35B ATen 分母是 34 —— 那次应该是 eager 抓的。

`--enforce-eager` 后 ATen 分母 16 → 28，这是 eager 轮唯一的实际收益。

**kernel 分母 57 → 39 不是丢了东西**：更多 kernel 找到了 ATen parent，
按 `coverage.py:117` 的 `counted_as_kernel = not links[name] or name in fused_names`，
它们从「独立 kernel」归到了「ATen 的实现」。总符号 75 vs 74 基本不变，合计分母 73 → 67 同理。

## FlagGems 执行日志对分子的贡献 = 0（对照实验）

eager 轮额外挂了 `--worker-extension-cls` 拿注册表快照、并喂了 FlagGems 的 oplist 当 `--log`。
**实测对覆盖率没有任何提升**：

```
带 --audit --log ：ATen 24/28  kernel 21/39  combined 45/67（67.16%）
不带（仅 provenance）：ATen 24/28  kernel 21/39  combined 45/67（67.16%）
```

`flaggems_execution_log_intersection` 确实进了 24 个 ATen 的 `coverage_reason`，
但这 24 个**全部同时**有 `correlated_triton_implementation`。

```
只靠 Gems 日志覆盖、没有 Triton 关联的算子：0 个
```

所以 audit 表 + oplist 的作用是**多一条独立证据链（佐证）**，不是增加覆盖。
对照组结果存在 `eager-09-29-control-no-gems/summary.json`。

### FlagGems 日志在哪

一开始以为日志缺失。实际不缺，只是没进 stdout：
`flag_gems/logging_utils.py:59-80` 的 `setup_flaggems_logging` 建 `FileHandler(mode="w")`、
`setLevel(DEBUG)`、`propagate=False`，写到 `FLAGGEMS_ENABLE_OPLIST_PATH`
（默认 `/tmp/flaggems_enable_oplist.txt`），formatter 是
`"[%(levelname)s] %(name)s.%(funcName)s: %(message)s"` ——
**正好是 `coverage.py:93` 的 `r'\[DEBUG\]\s+(flag_gems\.[^:\s]+):'` 能吃的格式**，
而且和注册表的 `f"{module}.{implementation}"` key 对得上。

`enable()` 在 `vllm_fl/worker/worker.py:236-266` 的 `WorkerFL.__init__` 内部调用，
所以裸 import `flag_gems` 时测到的「logger 无 handler、effective level 30」**不能用来描述 serve 进程**。

`worker.py:248` 的 `should_record = (rank == 0)` 里的 `rank` 是 **per-DP-replica** 的，
所以两个节点各写一份，head 那份覆盖 DP0、worker 那份覆盖 DP1，**要并集**。

不要为了把日志转到 stdout 去加 `VLLM_LOGGING_CONFIG_PATH`：
`once` 去重过滤器挂在 FileHandler 上（`logging_utils.py:72-73`），不在 logger 上，
额外的 StreamHandler 继承不到去重，**eager 下会变成每次调用一行 × 16 rank**。

oplist 见 `gems-oplists/`（head 侧 DP0：09-28 graph 轮 63 key，09-29 eager 轮 62 key，
差 1 条是两轮压测覆盖的分支不同，不是 DP 差异）。

## 未覆盖的项

### ATen（eager 轮 4 项）

| 算子 | 情况 |
|---|---|
| `aten.clamp` | 命中 plugin-fl `dispatch/config/thead.yaml:101-114` 的 `flagos_blacklist` |
| `aten.clamp_` | 同上 |
| `aten.copy_` | 同上 |
| `aten.mm` | **新增**，graph 轮不在分母里（被 graph 吞了） |

前三项两轮都精确命中 blacklist 13 项名单，**这个对应关系是稳的**，
独立印证了 blacklist 在生效（见 `../findings/flagos-blacklist.md`）。

`aten.mm` 链到 4 个 kernel，全是 vendor GEMM，无 Triton 证据也不在 Gems 日志里：

```
744 calls  13.94 ms  gemm_ktype0_aiu1_mtype1_dtypeBF16xBF16xFP32xFP32xFP32_tile16x64x16x16x...
720 calls  14.81 ms  gemm_ktype0_aiu1_mtype1_...
720 calls   5.84 ms  gemm_ktype0_aiu1_mtype1_...tile32x128x32x32...
720 calls   2.51 ms  reduction_dtypeFP32xFP32xFP32_align4x4_...
```

`implementation_source = 尚未确认`。

### 独立 kernel（graph 轮 22 项占 82.6% GPU 时间）

graph 轮的时间权重（**单独口径，不是 skill 指标，不要混用**）：

```
covered   50 符号   379.42 ms  17.4%   调用 113074  72.4%
uncovered 25 符号  1798.89 ms  82.6%   调用  43212  27.6%
```

未覆盖的那 25 个符号占 82.6% GPU 时间，因为 comm（pccl 三个 kernel 合计 855 ms、39.3%）、
deep_gemm cutlass、flash attention、tilelang 的 `mhc_*`/`hc_*` 全是非 Triton 实现。

**tilelang 和 vendor GEMM 是 plugin 提供的实现，按 Gems+Triton 口径算「未覆盖」，
这不代表插件没贡献，只是口径外。**

## 口径注意事项

1. **这是类型覆盖率**，不是调用次数、GPU 时间或吞吐比例。
2. **`covered` 的分子含框架原有 Triton 和 TorchInductor 生成的**，
   按 skill `accounting.md:36`「默认 Gems + Triton 覆盖包含框架原有 Triton；
   插件贡献还需选路/日志/基线证据」——**不能当作插件贡献率报出去**。
3. **Triton 证据是「同名精确匹配 cache 里的 `triton_version` metadata」**，
   只证明 kernel 是 Triton 编译的，不证明归属、也不证明本轮每次调用都走它。
   两轮的 cache 文件 mtime 都早于抓取时间（全 warm cache、零重编译）。
4. graph 轮的 trace 形状是 case1（1024/1024、并发 64、256 请求），
   eager 轮是 8 并发 × 96 token。**都不是 skill 参考案例的 4096/1024/128**，
   所以 65.75% / 67.16% 不能直接和 Qwen3.5-35B 的 83.75% 比。

## eager 轮的额外数据

`missing_parent` 15721/141264 events = **11.1%**（`ambiguous = 0`），18 个符号，
全是 `mhc_pre_big_fuse_with_norm_tilelang_kernel`(2928)、`mhc_fused_tilelang_kernel`(2904)、
`_save_partial_states_kernel`(2184)、`flash_sparse_decode_fwd_kernel`(1464) 这类
**不经 ATen dispatch、直接 launch 的插件 kernel** ——
按 `accounting.md:9` 保留为独立项，不猜关联。graph 轮这个数只有 286，
因为那时它们都藏在 graph 里没被单独看见。

16 个 audit json 全部 `cudagraph_mode=NONE`、`registered_aten=854`，0 个 SKIPPED。

## 复现

```bash
# graph 轮
for f in /tmp/prof_tgt/dp0_pp0_tp*_rank*.pt.trace.json.gz; do
  r=$(basename "$f" | sed -E 's/.*_rank([0-9]+)\..*/\1/')
  python scripts/scan_trace.py "$f" /tmp/cov_scans/rank$r.json
done
python scripts/provenance.py --scans /tmp/cov_scans/rank*.json \
  --cache /root/.triton --cache /tmp/torchinductor_root --out /tmp/cov_prov.json
python scripts/coverage.py --scans /tmp/cov_scans/rank*.json \
  --provenance /tmp/cov_prov.json --model "..." --out /tmp/cov_out

# eager 轮：加 --audit / --log / --allow-no-graph
#   eager 无 graph event，coverage.py:63 会直接抛错，必须给 --allow-no-graph
python scripts/coverage.py --scans /tmp/cov_scans_eager/rank*.json \
  --provenance /tmp/cov_prov_eager.json \
  --audit /tmp/prof_eager/worker_audit_rank*.json \
  --log /tmp/flaggems_oplist_eager.txt \
  --allow-no-graph --model "..." --out /tmp/cov_out_eager
```

注意排除 `*async_llm*.pt.trace.json.gz`（前端 trace，不是 worker）。
