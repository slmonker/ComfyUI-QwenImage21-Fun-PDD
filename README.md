# ComfyUI-QwenImage21-Fun-PDD

**Qwen-Image 2.1 Fun-Acc 四步 PDD 的非官方 ComfyUI 原生适配节点。**

[English](README.en.md) · [上游模型](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs) · [问题反馈](https://github.com/slmonker/ComfyUI-QwenImage21-Fun-PDD/issues)

> 本项目不是 Alibaba PAI / VideoX-Fun 官方插件，也不代表其维护团队。
> 当前为实验性适配：通过权重完整性检查、数值测试及缩小尺寸的原生模型 CPU/CUDA 前向测试，**尚未完成完整基础模型的端到端出图及与官方结果的视觉对比**。

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

原生 ComfyUI 合并 LoRA 权重，而上游 Diffusers 使用独立低秩分支，**不承诺逐像素或逐位复现上游输出**。FP8/GGUF、动态显存路径、多设备分片、torch.compile、局部重绘蒙版以及其他模型修改插件的组合，尚未完成验证。

## 验证状态

见 [TESTING.md](TESTING.md)。本地完成的 7 项集成测试包括：

- 原始检查点全部 **528 个张量**与完整尺寸的 meta 模型对应；
- 融合 gate_up 两半的 LoRA 数值与归一化替换；
- 四个输出头选择与异常后的状态清理；
- FP32 Euler 与固定四步 sigma；
- 缩小尺寸的原生 Qwen-Image 2.1 CPU / CUDA BF16 前向；
- 克隆模型卸载后原模型恢复；
- 错误文件、形状、重复应用与越界文件名的拒绝。

完整模型出图、性能基准和官方图像对比仍待验证，欢迎提交可复现的问题报告。

## 来源与许可说明

模型、PDD 方法和采样配置来自 [Alibaba PAI 模型仓库](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs)，上游实现见 [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)。本仓库附带的 `pdd_config.json` 保留上游配置；不包含模型权重或上游推理脚本。

模型与上游材料受其各自许可约束。本仓库暂未指定独立软件许可证；公开发布不替代上游授权，也不表示项目组认可本适配。
