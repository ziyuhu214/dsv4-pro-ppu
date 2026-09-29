# 两个插件的代码改动整理（**仅整理，未提交**）

按要求只整理成可提交形态，**没有向任何上游仓库提交或推送**。

## 现状盘点

| 仓库 | 本地分支 | fork 远端 | 上游 | 状态 |
|---|---|---|---|---|
| vllm-plugin-FL | `feat/deepseek-v4-flash-ppu-perf` | `mine` = ziyuhu214/vllm-plugin-FL | `origin` = flagos-ai/vllm-plugin-FL | 13 个 commit 已在 fork；**2 个文件未提交** |
| FlagGems v5.3.4 | `fix/deepseek-v4-ppu-fixes` | `mine` = ziyuhu214/FlagGems | `origin` = flagos-ai/FlagGems | 6 个 commit 已推到 fork，无未提交改动 |

FlagGems 本地 HEAD (`c993b3c8`) 与 `mine/fix/deepseek-v4-ppu-fixes` 完全一致 —— 已同步，无待整理项。

---

## A. vllm-plugin-FL：2 个未提交文件

`git diff --stat`：

```
 vllm_fl/platform.py      | 35 +++++++++++++++++++++++++++++++++--
 vllm_fl/worker/worker.py | 29 ++++++++++++++++++++++++++++-
 2 files changed, 61 insertions(+), 3 deletions(-)
```

两处都是 **ray 多节点下的设备可见性问题**，与本轮 Pro 压测/覆盖率工作无直接因果，
是更早排查 ray 路径时留下的。建议拆成两个独立 commit 提交。

### A1. `vllm_fl/platform.py` — 补 `device_control_env_var` 与 `ray_noset_device_env_vars`

原代码是一行 TODO 注释：

```python
### TODO(lms): dispatch device_control_env_var
# device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
```

改动新增 `device_control_env_var_dict`（仿照已有的 `dist_backend_dict`，按 `device_type` 分派），
并据此设置两个类属性。

**为什么必须两个都设**：`ray_executor` 从 `current_platform.ray_noset_device_env_vars`
读取 NOSET 列表；基类默认是空列表，ray 因此从不知道该放手，会把可见设备变量
掩到每个 actor 一张卡。`device_count()` 随后返回 1，所有 `local_rank > 0` 的 worker
以 `CUDA error: invalid device ordinal` 死掉。

**保守取舍**：只列名称可核实的 `cuda` 和 ray 的 npu/ascend。
`musa` / `ptpu` 故意不列 —— ray 没有对应的 accelerator 模块，没有 NOSET 变量可设，
猜一个名字比回落到现状更糟。不在表里的 device_type 回落基类占位值，行为不变。

建议 commit message：

```
fix(platform): dispatch device_control_env_var and ray_noset_device_env_vars

PlatformFL left device_control_env_var at the base-class placeholder and
ray_noset_device_env_vars at an empty list. ray_executor populates the actor
runtime_env from the latter, so an empty list means ray masks the visible-devices
variable down to one card per actor; device_count() then returns 1 and every
worker with local_rank > 0 fails with "CUDA error: invalid device ordinal".

Dispatch both on device_type, mirroring the existing dist_backend_dict. Only
device types whose env-var names are verifiable are mapped (cuda, npu); musa and
ptpu fall back to the base-class behaviour rather than guessing a name.
```

### A2. `vllm_fl/worker/worker.py` — 按 ray 分配的物理 GPU 映射绑定设备

原代码直接用 `local_rank` 当设备序号：

```python
self.device = torch.device(f"{current_platform.device_type}:{self.local_rank}")
```

改动先发布 ray 发现的逻辑→物理映射（`set_assigned_physical_gpu_ids`），
再经 `logical_device_id_to_visible_device_id()` 翻译后绑定。

**为什么需要**：`WorkerFL` 是 `WorkerBase` 的平行实现，不是 core `Worker` 的子类，
所以 core 在 `gpu_worker.py:276-313` 免费做的这两件事这里都不会发生。
直接用 `local_rank` 只在「分到的 GPU 恰好连续且从 0 开始」时正确 ——
整节点独占时成立,而 ray 只分一个子集时（例如两个实例共享一个节点）
worker 会**静默绑到错误的设备上**。

`device_type` 保留在 f-string 里而不是像 core 那样硬编码 `"cuda"`，因为本插件还服务 npu/musa。

建议 commit message：

```
fix(worker): bind to ray-assigned physical GPU instead of local_rank

WorkerFL is a parallel implementation of WorkerBase rather than a subclass of
core's Worker, so the logical->physical device translation core performs in
gpu_worker.py:276-313 never happened here: local_rank was used directly as the
device ordinal. That is correct only while the assigned GPUs are contiguous and
start at 0 -- true for a whole-node allocation, false as soon as ray hands out a
subset, where the worker silently binds to the wrong device.

Publish the mapping via set_assigned_physical_gpu_ids, then translate through
logical_device_id_to_visible_device_id. device_type is kept in the f-string
rather than hardcoded to "cuda" since this plugin also serves npu/musa.
```

完整 diff 见 `vllm-plugin-FL-uncommitted.diff`。

---

## B. 与本轮工作直接相关的已提交改动

这些已在 fork 上，列出来是为了让本仓库的结论可追溯到代码。**不是本次新写的。**

### vllm-plugin-FL

| commit | 与本仓库结论的关系 |
|---|---|
| `85350aa` fix(ops): actually load the deepgemm tuned configs | 修的是 device-name 匹配 bug（153 条 DSv4 MoE 调优配置静默不加载）。**本轮已验证这个修复在生效**：7 个 PPU-ZW810E 文件正常加载。但另一个问题仍在，见 `../findings/deepgemm-num-groups.md` |
| `3c78240` perf(dispatch): blacklist 13 cheap FlagGems ops on thead | 即 K2 修复。**本轮从 trace 侧独立印证在生效**，见 `../findings/flagos-blacklist.md` |
| `150afcb` perf(ops): route mHC prenorm GEMM to PPU deep_gemm | profile 里 `hc_prenorm_gemm_tilelang_kernel` 74.17 ms/3172 calls 对应这条路径 |
| `3474000` feat(worker): enable CUDA graph capture on thead PPU + fix graph memory accounting | 与 `gpu_worker.py:514-517` 的 `cg_util_delta = cudagraph_memory_estimate / total_mem` 相关，是 util 语义（占**总**显存而非空闲显存）的来源 |

### FlagGems

| commit | 关系 |
|---|---|
| `c993b3c8` fix(ops): unregister aten::empty override (upstream PR #5438) | — |
| `ae6fd75e` perf(fused): num_real_heads fast path in dsv4 qnorm insert kernel | profile 里 `fused_qnorm_rope_kv_insert_kernel` 12.71 ms/1586 calls |
| `efd9e35e` fix(fused): software E4M3FN encode for triton without fp8e4nv (PPU/SM80) | PPU 无 fp8e4nv，kv-cache fp8 依赖这条 |
| `26ea167e` fix(fused): int64 scale offsets in cp_gather_indexer_quant_cache | — |
| `a597ec4c` fix(utils): raise sqlite busy timeout for shared tuning-cache DB | 32 卡共享调优 DB 时的并发问题 |
| `8d3179b0` docs(ops): warn against removing empty()'s zero-store kernel | — |

---

## C. 一个可提给 skill 作者的改进（不属于两个插件）

`vllm-gpu-coverage` 的 `coverage_worker_audit.py` 在 `flag_gems.current_work_registrar`
为 `None` 时会抛 `AttributeError`，把 16 个 worker 在 init 期全部打死，
且 traceback 指向 audit 模块而不是真正原因（`fl_envs.USE_FLAGGEMS` 被关）。

`flag_gems/__init__.py:68` 初始化为 `None`，只在 `enable()`/`only_enable()` 里赋值,
而 `worker.py:236` 用 `if fl_envs.USE_FLAGGEMS:` 门着 `enable()`。
同一个 `None` 也会让 `all_registered_keys()` / `all_registered_ops()` 报错
（`__init__.py:1174-1179` 在 `None` 上取 `get_all_ops`/`get_all_keys`）。

本轮用的副本已加 guard：退化成写一个 `*.SKIPPED.json` 并打一行日志，而不是整轮起不来。
见 `coverage-worker-audit-none-guard.diff`。

**这个修改没有提交到任何地方**，skill 是 zip 分发的，不是 git 仓库。
