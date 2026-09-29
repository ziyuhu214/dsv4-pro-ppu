# DeepGEMM 调优表对 DeepSeek-V4-Pro 零命中（`num_groups=12`）

## 结论

plugin-fl 的 DeepGEMM 调优配置表里，**没有任何一条能匹配 Pro 在 32 卡下的 MoE 形状**。
Pro 的 MoE GEMM 全部走未调优路径。

## 证据

### 1. `num_groups=12` 在全部 1048 条配置里出现 0 次

384 experts / 32 rank(ep_size=32) = **12 experts/rank**，即 `num_groups=12`。

`vllm_fl/ops/ppu_deepgemm_configs/` 下 12 个 JSON、共 1048 条记录，`num_groups` 分布：

```
{1: 489, 2: 42, 3: 47, 4: 42, 5: 48, 10: 54, 11: 49,
 16: 42, 17: 36, 20: 46, 21: 54, 32: 50, 33: 49}
```

**12 不在其中。**

### 2. 22 个调优过的 (N,K) 组合里没有 Pro 的形状

Pro 的形状（取自模型 `config.json`：`moe_intermediate_size=3072`、`hidden_size=7168`）：

```
(N=3072, K=7168)   (N=6144, K=7168)   (N=7168, K=3072)
```

这三个在 22 个调优过的 (N,K) 对里**一个都没有**。

### 3. 没有 DeepSeek-V4-Pro 的配置文件

配置目录里不存在针对 Pro 的文件。

### 4. 查表逻辑对 N/K/num_groups 要求精确匹配

`vllm_fl/ops/ppu_deep_gemm.py:107-134`：

```python
@functools.cache
def get_deep_gemm_config(M, N, K, num_groups):
    ...
```

`find_closest` **只对 M 做近似**，N / K / num_groups 必须精确命中。
所以上面三条任意一条成立就足够导致 miss，三条同时成立。

### 5. 直接调用查表函数验证

用真实形状直接调 `get_deep_gemm_config`，`num_groups` 从 1 遍历到 33，**全部返回 `None`**。

## 与另一个已修 bug 的关系

plugin-fl `85350aa` 修的是**配置文件的 device-name 匹配** bug（153 条 DSv4 MoE 调优配置
因为设备名不匹配而静默不加载）。**那个修复在本轮是生效的**：
7 个 `PPU-ZW810E` 文件正常加载，`ZW810`（无 E）的被正确跳过。

所以这里说的零命中**不是** device-name bug 的残留，是**调优表本身没有 Pro 的形状**。
两个是独立问题。
