# FlagOS覆盖率计算逻辑说明\-模型端到端优化

# FlagOS 算子覆盖率测试与统计规范

## 1\. 背景与目的

为客观评估 FlagOS 在不同硬件平台和推理、训练场景下对算子的支持与接管情况，需要建立一套统一、可复现且可对比的算子覆盖率测试与统计方法。

本规范面向**英伟达平台及国产芯片平台**，对测试场景、数据采集方式、算子分类方法、归一化规则、覆盖率计算口径以及融合算子的覆盖关系作出统一规定。按照本规范产出的结果，可用于评估 FlagOS 的算子适配范围，并为后续的算子适配、功能优化和跨平台能力对比提供依据。

本规范定义的覆盖率用于衡量针对指定模型的FlagOS 对算子种类的覆盖程度，不代表算子调用次数覆盖率、执行时间覆盖率或性能收益比例。

## 2\. 测试场景

### 2\.1 推理测试场景

#### （1） 抓取模式

考虑到实际推理优化通常在完整计算图中进行，本规范统一采用 Graph 模式抓取算子，以保证测试结果能够反映真实优化场景下的算子执行情况。

#### （2） 负载配置

为确保不同平台和不同软件配置下的测试结果具有可比性，所有测试应采用以下统一负载：

- 输入长度设置为 4096 tokens。

- 输出长度设置为 256 tokens。

- 并发数设置为 64。

除硬件平台及待验证的软件组件外，模型版本、精度配置、推理参数和请求数据等其他测试条件均应保持一致。



### 2\.2 训练测试场景

#### （1） 抓取模式

训练的config中开启以下配置：

```YAML
train:
  system:
    profile: true
    use_pytorch_profiler: true
    profile_ranks: [0]          # 全局 rank；空列表表示所有 rank
    profile_step_start: 5
    profile_step_end: 10
    pytorch_profiler_collect_shapes: true
    pytorch_profiler_collect_memory: false
    pytorch_profiler_collect_callstack: false
  model:
    train_iters: 15
```

#### （2） 负载配置

为确保不同平台的测试结果具有可比性，所有测试应采用以下统一负载：

- NVIDIA 基线模型配置（Model Config）保持不变；

- NVIDIA 基线负载配置保持不变，包括 Sequence Length、分辨率等；

- 允许调整并行策略与优化策略，各平台应充分启用可用优化手段。



## 3\. 数据产出要求

### 3\.1 推理数据产出要求：

#### 3\.1\.1 英伟达平台

英伟达平台需要产出 A、B、C 三组算子数据。

##### **A：英伟达基线算子集合**

A 表示英伟达平台上的基线算子集合。采集该组数据时，应使用原始 vLLM 执行推理，并且不得加载任何 FlagOS 组件。

该集合用于表示未启用 FlagOS 时，在当前模型和负载下实际参与计算及通信的全部算子。

##### **B：英伟达 \+ FlagOS 场景全量算子集合**

B 表示英伟达平台启用 FlagOS 后的全量算子集合。采集该组数据时，应加载 `vllm-plugin-FL`，并开启当前场景下所有应当启用且实际可用的 FlagOS 算子及后端能力。

该集合应包含启用 FlagOS 后，当前推理任务实际执行的全部计算算子和通信算子，而不仅限于由 FlagOS 提供或接管的算子。

##### **C：英伟达 FlagOS 算子集合**

C 表示从 B 中识别并提取出的、由 FlagOS 提供或接管的算子集合。该集合仅包括以下三类算子：

1. 由 FlagGems 提供或替换的算子；

2. 明确使用 FlagTree 后端执行的 Triton 算子；

3. 由 FlagCX 接管的通信算子。

#### 3\.1\.2 国产芯片平台

国产芯片平台需要产出 E、F 两组算子数据。如果平台支持原生 vLLM，还必须产出 D 组基线数据。【原则：有D用D】

##### **D：国产芯片基线算子集合（可选）**

D 表示国产芯片平台上的基线算子集合。采集该组数据时，应使用该平台支持的原始 vLLM 执行推理，并且不得加载任何 FlagOS 组件。

**如果该平台支持原生 vLLM，则必须提供 D。**如果该平台确实无法运行原生 vLLM，可以不提供 D，但应在测试报告中说明具体原因及相关限制。

##### **E：国产芯片 \+ FlagOS 场景全量算子集合**

E 表示国产芯片平台启用 FlagOS 后的全量算子集合。采集该组数据时，应加载 `vllm-plugin-FL`，并开启当前场景下所有应当启用且实际可用的 FlagOS 算子及后端能力。

该集合应包含当前推理任务实际执行的全部计算算子和通信算子。

##### **F：国产芯片 FlagOS 算子集合**

F 表示从 E 中识别并提取出的、由 FlagOS 提供或接管的算子集合。该集合仅包括以下三类算子：

1. 由 FlagGems 提供或替换的算子；

2. 明确使用 FlagTree 后端执行的 Triton 算子；

3. 由 FlagCX 接管的通信算子。



### 3\.2 训练数据产出要求

#### 3\.2\.1 英伟达平台

|Level|场景|备注|
|---|---|---|
|A: 英伟达基线|te\_fl\_prefer: vendor<br>enable\_flag\_gems: false|该集合用于表示未启用 FlagOS 后端时，在当前模型和负载下实际参与计算及通信的全部算子。|
|B: 英伟达 \+ flagos|te\_fl\_prefer: flagos<br>enable\_flag\_gems: false|该集合应包含启用 FlagOS 后，当前训练任务实际执行的全部计算算子和通信算子，而不仅限于由 FlagOS 提供或接管的算子。|
|C: 英伟达 \+ flagos \+ 全量gems算子|te\_fl\_prefer: flagos<br>enable\_flag\_gems: true|该集合应包含启用 FlagOS \+ 全量Gems算子 后，当前训练任务实际执行的全部计算算子和通信算子，而不仅限于由 FlagOS 提供或接管的算子。|

最终需要从 C 中产出由 FlagOS 提供或接管的算子集合，该集合包括以下三类：

1. 由 FlagGems 提供或替换的算子；

2. 明确使用 FlagTree 后端执行的 Triton 算子；

3. 由 FlagCX 接管的通信算子。



#### 3\.2\.2 国产芯片平台

|Level|场景|备注|
|---|---|---|
|D: vendor基线|te\_fl\_prefer: vendor<br>enable\_flag\_gems: false|该集合用于表示未启用 FlagOS 后端时，在当前模型和负载下实际参与计算及通信的全部算子。|
|E: vendor \+ flagos|te\_fl\_prefer: flagos<br>enable\_flag\_gems: false|该集合应包含启用 FlagOS 后，当前训练任务实际执行的全部计算算子和通信算子，而不仅限于由 FlagOS 提供或接管的算子。|
|F: vendor \+ flagos \+ 全量gems算子|te\_fl\_prefer: flagos<br>enable\_flag\_gems: true|该集合应包含启用 FlagOS \+ 全量Gems算子 后，当前训练任务实际执行的全部计算算子和通信算子，而不仅限于由 FlagOS 提供或接管的算子。|

最终需要从 F 中产出由 FlagOS 提供或接管的算子集合，该集合包括以下三类：

1. 由 FlagGems 提供或替换的算子；

2. 明确使用 FlagTree 后端执行的 Triton 算子；

3. 由 FlagCX 接管的通信算子。



### 3\.3 算子统计表的性能占比



生成各场景的算子统计表时，除算子名称、类型及对应 kernel 外，需要记录执行时间占比，以便识别值得优先优化的算子。输出格式可参考 [Qwen3\.6\-35B\-A3B 基线 ](https://github.com/cyber-pioneer/op_profile/blob/main/qwen3.6_35b_a3b_native_graph_4096_256/results/operator_list.csv)[`operator_list.csv`](https://github.com/cyber-pioneer/op_profile/blob/main/qwen3.6_35b_a3b_native_graph_4096_256/results/operator_list.csv)，在每个 kernel 对应的行提供 `kernel_time_percent(%)`：



$\text{kernel\_time\_percent(\%)}=100\times\frac{\text{该行 kernel 在统计窗口内的 GPU 执行时长之和}}{\text{同一统计窗口、同一统计 rank 的全部 GPU kernel 执行时长之和}}$



统计窗口仅包含实际执行的请求或训练步骤，不包含预热、Graph 捕获等准备阶段；分母包含该窗口内所有 GPU kernel 事件，包括无法归因到算子的事件，但不包含 CPU 时间、GPU memcpy/memset 或端到端等待时间。分子与分母必须使用相同的窗口和 rank 范围。无法归因的 kernel 仍应保留在表中并计入分母，不得因缺少算子名称或输入 shape 而丢弃。



`kernel_time_percent(%)` 是**行级 kernel 时间占比**，不等同于算子级占比。同一 `operator_id` 对应多个 kernel 时，应先合计这些 kernel 的原始 GPU 时长，再除以上述分母，得到该算子的时间占比；不得直接相加已经四舍五入的展示值。表头标明 `%`，通常填写不带百分号的数值，例如 `1.23` 表示 `1.23%`；保留小数点后两位，正值低于 `0.01%` 标记为 `<0.01`。没有有效计时数据时应标记为不可用并说明原因，不得写成 `0`。



## 4\. 覆盖率计算方法

覆盖率应按照归一化并去重后的算子集合计算，不得按照算子调用次数或执行时长进行加权。

### 4\.1 英伟达平台

英伟达平台的 FlagOS 算子覆盖率按照以下公式计算：

$\text{英伟达 FlagOS 覆盖率}=\frac{|C\cap A|}{|A|}$

其中：

- A 表示经过归一化和去重后的英伟达基线算子集合。

- C 表示经过归一化和去重后的英伟达 FlagOS 算子集合。

- `|A|` 表示英伟达基线算子集合中的算子总数。

- `|C∩A|` 表示英伟达 FlagOS 算子集合与英伟达基线算子集合取交集后的算子数。

### 4\.2 国产芯片平台

如果国产芯片平台支持原生 vLLM，并且能够提供 D，则覆盖率按照以下公式计算：

$\text{国产芯片 FlagOS 覆盖率}=\frac{|F\cap D|}{|D|}$

其中：

- D 表示经过归一化和去重后的国产芯片基线算子集合。

- F 表示经过归一化和去重后的国产芯片 FlagOS 算子集合。

- `|D|` 表示国产芯片基线算子集合中的算子总数。

- `|F∩D|` 表示国产芯片 FlagOS 算子集合与国产芯片基线算子集合取交集后的算子数。

如果国产芯片平台不支持原生 vLLM，因而无法提供 D，则应使用同一模型、同一负载条件下的英伟达 \+ FlagOS 场景全量算子集合 B 作为参考分母：

$\text{国产芯片 FlagOS 覆盖率}=\frac{|F\cap B|}{|B|}$

采用 B 作为参考分母时，测试报告中必须明确说明未采用国产芯片原生基线的原因，避免将该结果误解为严格意义上的平台内基线覆盖率。

## 5\. 分母统计规则



### 5\.1 推理

覆盖率分母表示当前统计口径下实际参与计算的全部算子，其中既包括计算算子，也包括通信算子。所有算子均应按照所属类型进行归一化和去重。

#### 5\.1\.1 ATen 类型算子

对于由 ATen API 发起的底层 kernel，应按照 ATen API 名称进行统计。

例如，同一次运行中可能出现多个由 `aten::add` 发起的特化 kernel：

```Python
# vLLM
aten::add,aten,"void at::native::vectorized_elementwise_kernel<2, ...>"
aten::add,aten,"void at::native::vectorized_elementwise_kernel<4, ...>"
```

上述记录归一化后均应记为：

```Plain Text
aten::add
```

因此，这些记录合计只统计为一个算子。



#### 5\.1\.2 Custom 类型算子

Custom 算子应按照 `kernel_name` 进行统计，具有不同功能的 kernel 应分别计数。

例如：

```Python
# vLLM
vllm::qwen_gdn_attention_core,custom,_causal_conv1d_fwd_kernel
vllm::qwen_gdn_attention_core,custom,_causal_conv1d_update_kernel
vllm::qwen_gdn_attention_core,custom,_fused_post_conv_kernel

# Megatron
FusedAttnFunc,cuDNN,cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x256_4x1x1_kernel0_0
_Linear,cuBLASLt,backend,nvjet_tst_40x64_64x16_4x2_v_bz_TNN
```

由于上述几条记录对应不同的 `kernel_name`，因此应统计为不同的算子。



如果同一个 kernel 仅因设备编号、Rank、模板参数或数据类型等因素生成了不同的特化版本，则应在归一化后合并统计。例如：

```Plain Text
silu_and_mul_kernel_kernel_rank_1
silu_and_mul_kernel_kernel_rank_2
```

上述记录归一化后均应记为：

```Plain Text
silu_and_mul_kernel
```

因此，最终只统计为一个算子。



#### 5\.1\.3 Torch Compile 类型算子

Torch Compile 算子应使用 `compile_module` 和 `kernel_name` 的组合作为唯一标识：

```Plain Text
compile_module + kernel_name
```

同一个 `compile_module` 生成的不同融合 kernel 不得合并，每个不同的 `kernel_name` 均应独立计数。

例如：

```Python
# vLLM
vllm.model_executor.layers.vocab_parallel_embedding.get_masked_input_and_mask,torch_compile,triton_poi_fused___and____or___add_bitwise_not_ge_lt_mul_sub_0
vllm.model_executor.layers.vocab_parallel_embedding.get_masked_input_and_mask,torch_compile,triton_poi_fused___and____or___add_ge_lt_mul_sub_0
vllm.model_executor.layers.vocab_parallel_embedding.get_masked_input_and_mask,torch_compile,triton_poi_fused___and____or___bitwise_not_ge_lt_1
```

虽然上述记录来自同一个 `compile_module`，但其 `kernel_name` 不同，因此应统计为三个算子。



#### 5\.1\.4 Communication 类型算子

通信算子属于推理过程中实际参与执行的算子，因此必须计入分母，不得因其不属于普通计算算子而将其排除。

通信算子应根据通信 API 与 `kernel_name` 的组合进行识别。同一个通信 API 下，不同功能的实际 kernel 应分别计数；同一个 kernel 因设备、Rank、数据类型或模板参数产生的特化版本应合并统计。

例如：

```Plain Text
_C_custom_ar::all_reduce,communication,"void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>(...)"
```

该通信算子必须计入分母。



#### 5\.1\.5 其他 Triton 类型算子

对于未归入 ATen、Custom 或 Torch Compile 类型的 Triton kernel，应按照归一化后的 `kernel_name` 进行统计。

例如：

```Plain Text
unattributed,_triton_mrope_forward
```

每个具有不同功能的 Triton kernel 应独立计数。同一个 kernel 因 Rank、设备编号或编译特化产生的不同版本，应在归一化后合并统计。

#### 5\.1\.6 其他非 Triton 类型算子

对于无法归入上述类型的非 Triton kernel，也应按照归一化后的 `kernel_name` 进行统计，并遵循以下规则：

1. 不同功能的 kernel 应分别计数。

2. 同一个 kernel 因 Rank、设备编号、数据类型或模板参数产生的特化版本应合并统计。

3. 对于无法稳定识别 `kernel_name` 的记录，应单独标记为 `unattributed`，不得直接忽略。

为避免多个无法识别的 kernel 被错误合并，在条件允许的情况下，还应保留必要的原始标识或来源信息，以支持后续核查。



### 5\.2 训练

覆盖率分母表示当前统计口径下实际参与计算的全部算子，其中既包括计算算子，也包括通信算子。所有算子均应按照所属类型进行归一化和去重。

#### 5\.2\.1 ATen 类型算子

对于由 ATen API 发起的底层 kernel，应按照 ATen API 名称进行统计。

例如，同一次运行中可能出现多个由 `aten::add` 发起的特化 kernel：

|**execution\_operator**|operator\_kind|**kernel\_name**|
|---|---|---|
|aten::add|aten|void at::native::**elementwise\_kernel**\<128, 2, at::native::gpu\_kernel\_impl\_nocast\<at::native::CUDAFunctor\_add\<float\> \>\(at::TensorIteratorBase\&, at::native::CUDAFunctor\_add\<float\> const\&\)::\{lambda\(int\)\#1\}\>\(int, at::native::gpu\_kernel\_impl\_nocast\<at::native::CUDAFunctor\_add\<float\> \>\(at::TensorIteratorBase\&, at::native::CUDAFunctor\_add\<float\> const\&\)::\{lambda\(int\)\#1\}\)|
|aten::add|aten|void at::native::**unrolled\_elementwise\_kernel**\<at::native::CUDAFunctor\_add\<float\>, std::array\<char\*, 3ul\>, 4, TrivialOffsetCalculator\<2, unsigned int\>, TrivialOffsetCalculator\<1, unsigned int\>, at::native::memory::LoadWithCast\<2\>, at::native::memory::StoreWithCast\<1\> \>\(int, at::native::CUDAFunctor\_add\<float\>, std::array\<char\*, 3ul\>, TrivialOffsetCalculator\<2, unsigned int\>, TrivialOffsetCalculator\<1, unsigned int\>, at::native::memory::LoadWithCast\<2\>, at::native::memory::StoreWithCast\<1\>\)|

上述记录归一化后均应记为：

```Plain Text
aten::add
```

因此，这些记录合计只统计为一个算子。



#### 5\.2\.2 Custom 类型算子/三方库算子

Custom 算子应按照 `kernel_name` 进行统计，具有不同功能的 kernel 应分别计数。

例如：

|**custom\_operator**|**execution\_operator**|**operator\_kind**|**kernel\_name**|
|---|---|---|---|
|\_Linear|cuBLASLt|backend|nvjet\_tst\_48x64\_64x15\_4x2\_h\_bz\_TNN|
|FusedAttnFunc|cuDNN|backend|cudnn\_generated\_fort\_native\_sdpa\_sm90\_flash\_fprop\_wgmma\_f16\_knob\_7\_64x128x256\_4x1x1\_kernel0\_0|
|RouterGatingLinearFunction|CUTLASS|backend|void cutlass::Kernel2\<cutlass\_80\_wmma\_tensorop\_bf16\_s161616gemm\_bf16\_32x32\_64x1\_tn\_align2\>\(cutlass\_80\_wmma\_tensorop\_bf16\_s161616gemm\_bf16\_32x32\_64x1\_tn\_align2::Params\)|

由于上述几条记录对应不同的 `kernel_name`，因此应统计为不同的算子。



如果同一个 kernel 仅因设备编号、Rank、模板参数或数据类型等因素生成了不同的特化版本，则应在归一化后合并统计。例如：

```Plain Text
silu_and_mul_kernel_kernel_rank_1
silu_and_mul_kernel_kernel_rank_2
```

上述记录归一化后均应记为：

```Plain Text
silu_and_mul_kernel
```

因此，最终只统计为一个算子。



#### 5\.2\.3 Torch Compile / Torch JIT 类型算子

Torch Compile 算子使用  `kernel_name` 的组合作为唯一标识，每个不同的 `kernel_name` 均应独立计数。

例如：

```Python
# Megatron
triton_per_fused_add_sum_0
triton_poi_fused_add_0
```

上述记录 `kernel_name` 不同，因此应统计为三个算子。



#### 5\.2\.4 Communication 类型算子

通信算子属于训练过程中实际参与执行的算子，因此必须计入分母，不得因其不属于普通计算算子而将其排除。

通信算子应根据通信 API 与 `kernel_name` 的组合进行识别。同一个通信 API 下，不同功能的实际 kernel 应分别计数；同一个 kernel 因设备、Rank、数据类型或模板参数产生的特化版本应合并统计。

例如：

|c10d::allreduce\_|nccl:all\_reduce|communication|ncclDevKernel\_AllReduce\_Sum\_u64\_RING\_LL\(ncclDevKernelArgsStorage\<4096ul\>\)|
|---|---|---|---|

该通信算子必须计入分母。



## 6\. 分子统计规则

覆盖率分子仅允许包含由 FlagOS 提供或明确接管的算子，具体包括以下三类：

1. FlagGems 算子；

2. 使用 FlagTree 后端执行的 Triton 算子；

3. 由 FlagCX 接管的通信算子。

对于非 FlagGems 生成的 Triton 融合算子，只有在能够确认其已经接入并实际使用 FlagTree 后端时，才允许计入分子。例如：

```Plain Text
triton_poi_fused___and____or___add_bitwise_not_ge_lt_mul_sub_0
```

不能仅凭该算子属于 Triton kernel，就将其计入 FlagOS 覆盖算子集合。

通信算子虽然必须计入分母，但只有在能够确认其已经由 FlagCX 接管时，才允许计入分子。仍由 NCCL、平台原生通信库或其他非 FlagCX 后端执行的通信算子，不得计入分子。

所有计入分子的算子均应保留可核查的来源证据，例如 FlagGems 替换日志、FlagTree 后端执行标识、FlagCX 接管日志或运行时采集信息。

## 7\. 分子与分母的交集要求

`/tmp/flaggems_enable_oplist.txt` 日志记录的是被替换的上层算子，而这些上层算子内部还可能调用其他底层算子。部分日志中的上层算子未必能够在基线抓取结果中被直接观测到。

因此，不得直接使用 FlagOS 日志中的算子数量作为覆盖率分子。有效 FlagOS 算子集合必须按照以下公式计算：

$S_{\text{有效}}=S_{\text{FlagOS}}\cap S_{\text{分母}}$

其中：

- `S有效` 表示最终用于计算覆盖率的有效 FlagOS 算子集合。

- $S_{\text{FlagOS}}$表示经过归一化和去重后的 FlagOS 算子集合。

- `S分母` 表示经过归一化和去重后的分母算子集合。

采用交集后的算子集合计算分子，可以保证：

1. 分子中的每个算子都能够在分母中找到对应项。

2. 覆盖率不会因分子与分母统计口径不一致而超过 100%。

3. 不会同时统计上层替换算子及其内部调用的底层算子，从而避免重复计数。

例如，FlagGems 日志中可能记录 60 余个算子，但基线抓取结果中只有 40 余个算子。在这种情况下，必须以归一化后与基线集合的交集作为有效分子，不得直接使用日志中的 60 余个算子计算覆盖率。

## 8\. 融合算子的覆盖关系

如果 FlagOS 使用一个融合算子替换了分母中的多个融合前算子，覆盖率统计应反映该融合算子实际替换的原始算子，而不能简单地将融合 kernel 只计算为一个覆盖项。

例如，某个 FlagOS 融合 kernel 实际替换了以下三个算子：

```Plain Text
aten::add
aten::mul
aten::silu
```

在证据充分的情况下，该融合 kernel 可以视为覆盖了三个分母算子。

假设融合 kernel 为 K，其能够提供证据的融合前算子集合为 M\(K\)，分母算子集合为 `S分母`，则该融合 kernel 的有效覆盖集合可以表示为：

$S_{\text{融合}}(K)=M(K)\cap S_{\text{分母}}$



如果存在多个融合 kernel，则所有融合 kernel 的有效覆盖集合应先取并集，再进行去重：



$S_{\text{融合}}=\bigcup_{K}S_{\text{融合}}(K)$

采用融合映射统计时，必须同时满足以下条件：

1. 必须提供融合 kernel 与融合前算子之间明确且可核查的映射关系。

2. 被映射的融合前算子必须实际存在于当前分母集合中。

3. 同一个分母算子在最终分子中只能计入一次。

4. 不得同时将融合 kernel 及其覆盖的融合前算子重复计入分子。

5. 如果无法提供可靠的映射证据，则只能按照能够与分母直接匹配的算子计算覆盖数量。

建议在最终结果中单独提供融合映射明细，至少包括融合 kernel、被替换的原始算子、映射证据来源以及最终计数结果。

## 9\. 最终定义

设：

- `S分母` 表示经过归一化和去重后的分母算子集合。

- `SFlagOS` 表示经过归一化和去重后的 FlagOS 算子集合。

- `S融合` 表示根据有效映射关系得到的融合前算子覆盖集合。

最终有效覆盖算子集合按照以下公式计算：

$S_{\text{已覆盖}}=\left(S_{\text{FlagOS}}\cap S_{\text{分母}}\right)\cup S_{\text{融合}}$



在计算并集时，同一个分母算子只能保留一次。已经按照融合前算子进行映射统计的融合 kernel，不得再次作为独立覆盖项计入。



FlagOS 算子覆盖率最终定义如下：



$\text{FlagOS 算子覆盖率}=\frac{|S_{\text{已覆盖}}|}{|S_{\text{分母}}|}$

该指标用于衡量 FlagOS 对算子种类的覆盖程度，不代表算子调用次数覆盖率、执行时间覆盖率或性能收益比例，因此不得直接用于推导性能提升幅度。

## 10\. 算子抓取工具

### 10\.1 英伟达平台

英伟达平台原则上应使用 `vllm-plugin-FL` 内置的算子抓取工具。该工具复用了 vLLM 原生提供的 `/start_profile` 和 `/stop_profile` 接口。

当前相关实现可参考：

[vllm\-plugin\-FL PR \#422](https://github.com/flagos-ai/vllm-plugin-FL/pull/422)

该工具的使用方法、输出格式、算子分类逻辑和归一化规则仍需进一步完善。在工具能力完善前，应对采集结果进行人工抽查，确保其符合本规范定义的统计口径。

### 10\.2 国产芯片平台

如果国产芯片平台支持 `vllm-plugin-FL` 内置抓取工具，应优先使用该工具进行数据采集。

如果该工具暂不支持目标平台，则应使用对应国产平台提供的性能分析工具或 Profile 工具。无论采用何种采集工具，最终的算子分类、归一化、去重、分子识别以及覆盖率计算都必须遵循本规范，不得因工具差异改变统计口径。

## 11\. 结果交付要求

为保证测试结果可核查、可复现，最终测试报告至少应包含以下内容：

1. 测试平台、硬件型号、驱动版本及软件环境。

2. 模型名称、模型版本、精度配置及并行策略。

3. 芯片平台推理适用的原始算子集合及归一化后的算子集合。

4. 每条算子的类型、原始名称、归一化名称及归一化依据。

5. FlagGems、FlagTree 和 FlagCX 算子的识别证据。

6. 分子与分母的交集明细。

7. 融合 kernel 与融合前算子的映射明细及相关证据。

8. 分母算子数、有效分子算子数和最终覆盖率。

9. 未识别算子、`unattributed` 算子及其他异常数据的说明。

10. 如果国产芯片平台未提供 D，应说明，并明确标注使用 B 作为参考分母。



## FAQ

**Q：**插件工具计算flagos算子覆盖率时，能否覆盖所有算子？

**A：**插件工具计算flagos算子覆盖率时，可以自动匹配aten类型算子和同名kernel，其他类型算子需要手动判断。以[qwen3\.6\-35b\-a3b](https://github.com/cyber-pioneer/op_profile/blob/main/qwen3.6_35b_a3b_flagos_coverage/operator_flagos_coverage_summary.csv)为例，插件自动识别匹配算子数：44/66，无法自动匹配算子数：22/66。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ODdlMjU3NGQyYTEwNTVjNjE4NjRlZWU3MWM2NjYyYzJfOTEwMTE4NmU4ZDY5MmZhNzY5YjJhMGE3YzFlNWZiMzdfSUQ6NzY4ODk5MDIyODA2MTI3NzQzMV8xNzkwNjY1ODE5OjE3OTA3NTIyMTlfVjM)



