# ComfyUI-QwenImage21-Fun-PDD

**Qwen-Image 2.1 Fun-Acc 四步 PDD 的非官方 ComfyUI 原生适配节点。**

[English](README.md) · **简体中文** · [上游模型](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs) · [问题反馈](https://github.com/slmonker/ComfyUI-QwenImage21-Fun-PDD/issues)

> 本项目不是 Alibaba PAI / VideoX-Fun 官方插件，也不代表其维护团队。
> 当前为实验性适配。下方展示维护者提供的演示截图；自动化验证覆盖权重完整性、数值测试及缩小尺寸的原生模型 CPU/CUDA 前向，尚未独立复跑完整模型并完成与上游的受控对比。

## Demo 演示

**以下两张演示截图的本地测试环境（由维护者提供）：**

- **显卡：NVIDIA GeForce RTX 5090 D，32GB 显存**
- **系统内存：128GB DDR5**

截图中的耗时来自这套本地配置，不代表其他硬件环境下的运行表现。

### 工作流与生成效果

![ComfyUI PDD 工作流及科幻城市生成效果](assets/demo-workflow.png)

维护者提供的工作流截图，展示 PDD 节点、自定义采样路径及生成结果。

### 采样器对比

![维护者提供的自定义采样器与 40 步 KSampler 对比截图](assets/demo-sampler-comparison.png)

截图展示自定义采样器与设置为 40 步的 KSampler。画面中的耗时来自维护者的单次运行，**不代表受控基准测试或通用的速度、画质保证**；硬件、参数、缓存和输出尺寸均可能影响结果。

[下载示例工作流](#示例工作流)。

## 为什么需要专用节点？

`Qwen-Image-2.1-Fun-Acc-4Step.safetensors` 不只是普通 LoRA：

| 内容 | 本节点的处理 |
| --- | --- |
| 231 对低秩权重，rank=64、alpha=64 | 使用原生 LoRA 补丁映射，强度固定为 1 |
| 65 个完整归一化参数 | 替换原参数，而不是作为增量相加 |
| `proj_out.weight`，形状 `[4, 64, 4096]` | 按当前采样 sigma 切换四个输出头 |
| 固定四步 sigma | 输出配套 Euler SAMPLER 与 SIGMAS |

只给 `lora_up` / `lora_down` 补上 `.weight` 后缀，不能处理完整的四步推理逻辑。本节点直接读取原始 safetensors，不覆盖或转换源文件。

## 安装

在 **ComfyUI 根目录**执行：

```bash
git clone https://github.com/slmonker/ComfyUI-QwenImage21-Fun-PDD.git custom_nodes/ComfyUI-QwenImage21-Fun-PDD
```

然后重启 ComfyUI 并刷新页面。不需要额外安装 Python 包；使用 ComfyUI 环境已有的 PyTorch 和 safetensors。

从[上游模型页](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs)下载原始 `Qwen-Image-2.1-Fun-Acc-4Step.safetensors`，放入 ComfyUI 的 `models/loras` 或已注册的 LoRA 目录。模型权重不包含在本仓库中。

### 前提

- ComfyUI 必须已支持**原生 Qwen-Image 2.1**，不是旧 Qwen-Image / Edit 2509 / Edit 2511。
- 需要 `ModelPatcher.add_weight_wrapper`、`DIFFUSION_MODEL` wrapper，以及 Qwen-Image 2.1 融合 MLP 的 LoRA 分片映射。
- 本次测试使用的 ComfyUI commit：`1568e6cfd0`，相关核心文件无本地改动。更老版本不保证兼容。

## 节点与接线

搜索 **`PDD`**，添加 **`Qwen-Image 2.1 Fun PDD (4-Step)`**。

- 内部名称：`QwenImage21FunPDDLoader`
- 分类：`loaders / Qwen-Image 2.1`
- 输入：`model`、`lora_name`
- 输出：`MODEL`、`SAMPLER`、`SIGMAS`

```text
原生 Qwen-Image 2.1 MODEL
          │
          ▼
Qwen-Image 2.1 Fun PDD (4-Step)
          ├── model ──────► SamplerCustom.model
          ├── pdd_euler ──► SamplerCustom.sampler
          └── pdd_sigmas ─► SamplerCustom.sigmas

正面 / 负面条件 ────────────► SamplerCustom.positive / negative
潜空间图像 ────────────────► SamplerCustom.latent_image
SamplerCustom.output ──────► VAE Decode
```

**SamplerCustom 设置 `cfg=1`、`add_noise=true`。** 文本编码器、潜空间节点与 VAE 沿用现有 Qwen-Image 2.1 工作流。

也支持 `BasicGuider + RandomNoise + SamplerCustomAdvanced`：MODEL 接 BasicGuider，正面条件接其 conditioning，另外两个输出接高级自定义采样器的 sampler 与 sigmas。

### 示例工作流

| 用途 | 工作流 |
| --- | --- |
| 四步文生图（T2I） | [Fun-PDD-sampling-4steps-t2i.json](examples/Fun-PDD-sampling-4steps-t2i.json) |
| 四步图像编辑（Edit） | [qwenimage2.1-pdd-4steps-edit.json](examples/qwenimage2.1-pdd-4steps-edit.json) |

这两个示例由仓库维护者提供，按原文件收录。下载 JSON 后拖入 ComfyUI，按本机环境选择对应的模型、文本编码器、VAE 和 LoRA，并安装工作流使用的额外自定义节点。编辑工作流中的输入图片需要重新选择；模型权重和输入图片不随 JSON 分发。

收录时已检查 JSON 解析和节点连线，未发现明显凭据；这不等于已在其他环境完成出图验证，也不改变下文的兼容性边界。

### 注意

- 不要用普通 KSampler 设置四步来替代此接法。
- 不需要额外 Scheduler 或 Shift 节点；不要替换、裁剪本节点输出的 sigma 序列。
- 不要再次用普通 LoraLoader 加载同一个文件。
- 推荐从没有其他加速补丁的 BF16 基础模型开始，CFG=1。

固定 sigma：

```text
1.0
0.9169867038726807
0.7861579060554504
0.5494909882545471
0.0
```

## 实现和边界

通过模型克隆及 ModelPatcher 注册 LoRA、完整参数补丁和输出头 wrapper；不改 ComfyUI 核心文件，也不直接替换模型 forward。输出头选择使用 ContextVar，在一次 forward 内隔离状态，异常后清理。采样状态使用 FP32，时间输入沿用原生 Qwen-Image 2.1 的舍入逻辑；该模型克隆的前缀 KV 缓存关闭，以匹配上游示例。

## 局限性

### 1. 画质：上游模型已说明的限制

[上游模型说明](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs#limitations)指出，相比教师模型，加速模型存在以下画质取舍：

- **密集小字：**文字笔画可能失真，可读性明显下降。
- **图像编辑：**部分结果可能略显模糊、偏暗，精细细节的清晰度降低。

以上是上游加速模型报告的限制，不是本节点测试已经认定的实现缺陷。四步加速不等于保证与教师模型的多步结果画质一致，应根据实际用途检查输出。

### 2. 模型和采样支持范围有限

- 仅支持 **ComfyUI 原生 Qwen-Image 2.1** 基础模型上的 `qwenimage21_extracted_prefused_v1` 四输出头 Fun-Acc 文件。本节点不是通用 LoRA 转换器或加载器，不适用于旧 Qwen-Image / Edit 2509 / Edit 2511，也不接收 Diffusers pipeline 对象。
- 使用**固定四步 Euler 和配套 sigma 序列**。更改步数、替换或裁剪 sigma、额外添加 shift，或换成普通 KSampler，均不属于支持路径，并可能触发校验错误。需要截短采样序列的部分降噪、传统 img2img 接法不受该固定采样器支持。
- LoRA 强度固定为 **1**，没有强度滑块。预期使用 **CFG=1**；在通常的 CFG=1 采样中，负面条件不提供常规的负面提示词引导。其他 CFG 值未验证。
- 依赖上文列出的原生模型支持及 ModelPatcher API；较旧的 ComfyUI 或未来 API 变动可能需要适配更新。

### 3. 数值差异与未验证组合

ComfyUI 将 LoRA 更新合并进模型权重，上游 Diffusers 则计算独立的低秩分支。运算顺序、舍入方式和底模精度不同，都可能改变结果。**即使提示词和种子相同，也不承诺逐像素或逐位复现上游输出。**

建议从未叠加其他补丁的 BF16 底模开始。FP8/GGUF 及其他量化版本、动态显存与卸载路径、多设备分片、torch.compile、蒙版局部重绘，以及与其他 LoRA、加速插件或模型补丁的组合，均未完成全面验证。文生图示例引用了 INT8 底模，这只是维护者提供的一套配置，不代表普遍保证量化兼容性。普通指令式编辑演示也不等同于蒙版局部重绘支持验证。

### 4. 显存、内存与速度

四步采样减少的是所需模型评估次数，**不会免除加载基础模型、文本编码器和 VAE 的需求**，不能据此认为低显存设备一定能运行整条工作流。目前尚未确定最低显存／内存要求、最大分辨率和批量上限。

两张演示图的环境是 **RTX 5090 D 32GB 显存 + 128GB DDR5 内存**，这是演示机器配置，**不是最低配置要求**。截图耗时不是受控的端到端性能基准，不能用来承诺通用加速倍数。模型加载、文本编码、VAE 解码、分辨率、缓存和其他工作流节点都会影响总耗时。本节点为匹配上游示例，关闭了模型克隆的前缀 KV 缓存。

### 5. 示例依赖与验证边界

示例 JSON 需要相应的本地模型和额外自定义节点；编辑示例还需要在本机重新选择输入图片。这些依赖和图片不随工作流分发，不能保证导入任意安装环境后无需调整即可运行。

自动化检查覆盖权重映射、低秩计算、输出头切换、模型恢复，以及缩小尺寸的 CPU/CUDA 前向。截图是维护者提供的演示，**不等于独立复跑完整模型，也不是与上游进行的受控对比**。广泛提示词、分辨率和跨硬件的兼容性仍待验证。本项目是非官方实验性适配，不代表上游团队提供支持保证。

## 验证状态

见 [TESTING.md](TESTING.md)。本地完成的 7 项集成测试包括：

- 原始检查点全部 **528 个张量**与完整尺寸的 meta 模型对应；
- 融合 gate_up 两半的 LoRA 数值与归一化替换；
- 四个输出头选择与异常后的状态清理；
- FP32 Euler 与固定四步 sigma；
- 缩小尺寸的原生 Qwen-Image 2.1 CPU / CUDA BF16 前向；
- 克隆模型卸载后原模型恢复；
- 错误文件、形状、重复应用与越界文件名的拒绝。

上方截图为维护者提供的演示，不等同于独立的完整模型复现或受控性能基准；欢迎提交可复现的问题报告。

## 来源与许可说明

模型、PDD 方法和采样配置来自 [Alibaba PAI 模型仓库](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs)，上游实现见 [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)。本仓库附带的 `pdd_config.json` 保留上游配置；不包含模型权重或上游推理脚本。

模型与上游材料受其各自许可约束。本仓库暂未指定独立软件许可证；公开发布不替代上游授权，也不表示项目组认可本适配。
