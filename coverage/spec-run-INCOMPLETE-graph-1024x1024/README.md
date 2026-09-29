# ⚠️ 不合规范的中间产物 —— 不可用于覆盖率判定

这里的 `operator_list.csv` 是用 **plugin-fl 官方工具**
（`tools/operator_profile/extract_operator_shapes.py`，上游 PR #422，已合入 `main`）
在**已有的 09-28 graph 轮 trace** 上离线跑出来的，格式正确、`summary.json.conservation`
全部为 `true`。

**但它不满足 `../FlagOS覆盖率计算逻辑说明-模型端到端优化.md` 的口径，不能用来算覆盖率。**
三条硬性不符：

## 1. 负载配置不对（规范 §2.1(2)）

| | 规范要求 | 本 trace |
|---|---|---|
| 输入长度 | 4096 | **1024** |
| 输出长度 | 256 | **1024** |
| 并发 | 64 | 64 ✓ |

规范明确「所有测试应采用以下统一负载」，本 trace 是 case1 形状，**不可比**。

## 2. 缺少基线算子集合 D（规范 §3.1.2、§4.2）

PPU 属国产芯片平台，需要产出 D / E / F 三组：

- **D**：国产芯片基线（原始 vLLM、不加载任何 FlagOS 组件）—— **不存在，从未采集**
- **E**：国产芯片 + FlagOS 全量算子集合 —— 本目录的 `operator_list.csv` 勉强算（但负载不对）
- **F**：从 E 中提取的 FlagOS 算子集合 —— 未产出

覆盖率公式 $|F \cap D| / |D|$ 的分母就是 D。官方工具的
`generate_flagos_coverage.py` 也把 `--baseline` 设为**必填**，缺它直接报错：

```
generate_flagos_coverage.py: error: the following arguments are required: --baseline
```

规范 §3.1.2 说「如果该平台支持原生 vLLM，则必须提供 D」。
本项目的基线栈是 T-Head vLLM 0.20.1，不加载任何 FlagOS 组件
（`VLLM_PLUGINS=""`、`PYTHONPATH=/workspace/_thead_baseline`），符合 D 的定义。

## 3. 归因率过低，与参考案例差距悬殊

```
kernel_mapping_event_count_by_status:
  missing_external_id    : 155119   (99.25%)
  operator_shape_matched :   1167   ( 0.75%)
```

94 行里 **71 行是 `unattributed`、`operator_name = null`**，只有 19 行 ATen、
3 行 communication、1 行 triton_compiled。

规范 FAQ 引用的参考案例（qwen3.6-35b-a3b）是**插件自动识别 44/66**。
本 trace 的可归因比例远低于此。差异来源尚未确认，候选：
本轮 `enable_prefix_caching=True`（官方示例用 `--no-enable-prefix-caching`）、
未限制 `cudagraph_capture_sizes`（官方示例限定 `[1,2,4,8,16,32,64]`）、
以及本轮 profiler 未设 `ignore_frontend:true` / `torch_profiler_dump_cuda_time_total:false`。

## 与本仓库早先那份覆盖率数据的关系

`../graph-09-28/` 和 `../eager-09-29/` 是用 `vllm-gpu-coverage` skill 算的
（48/73 = 65.75% 和 45/67 = 67.16%）。**那两个数字与本规范口径不同，不能混用**：

| | skill 口径 | 本规范口径 |
|---|---|---|
| 分母 | 目标栈自身观察到的算子 | **基线栈 D 的算子** |
| 分子判定 | 任何有 Triton 证据的 kernel 都计入 | 只认 FlagGems / FlagTree-Triton / FlagCX-comm |
| 抓取模式 | graph 和 eager 各一轮 | **只认 graph**（§2.1(1)） |

另外本环境的两个结构性事实会显著压低规范口径下的分子：

- **FlagTree 未安装**（`triton 3.6.0+v0.1.0.ppu2.1.0`，backends `['amd','nvidia','ppu']`，
  是 PPU 厂商 patch 版而非 FlagTree）→ §6 第 2 类分子为 0
- **FlagCX 未安装**（`FLAGCX_PATH` 未设），comm 实走 pccl
  （`pcclKernel_twoShotAllReduceKernel` 等 3 个，占 39.3% GPU 时间）
  → §6 第 3 类分子为 0，但这 3 个**必须计入分母**（§5.1.4）

所以规范口径下分子**只有 FlagGems 一类**。
（注：官方工具 README 写的是「当前策略把观察到的每个 Triton 操作都计入分子」，
比规范 §6 的文字更宽松。两者不一致，以哪个为准需要确认。）

## 要出合规结果，需要

1. 两轮 graph 模式、**4096 输入 / 256 输出 / 并发 64** 的采集：
   - D：基线栈（T-Head 0.20.1，不加载 FlagOS）
   - E：目标栈（plugin-fl + FlagGems）
2. 两轮都用 `tools/operator_profile/serve.sh` + `profile.sh`，产出各自的 `operator_list.csv`
3. `generate_flagos_coverage.py --baseline <D> --plugin <E> --flaggems-oplist <E 的 oplist>`

两轮都要 32 卡、都要和对端约定整点启动。
