# ComfyUI-QwenImage21-Fun-PDD

**Unofficial native ComfyUI adapter for Qwen-Image 2.1 Fun-Acc four-step PDD exports.**

[中文](README.md) · [Upstream model](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs)

> Experimental. Not an official Alibaba PAI / VideoX-Fun plugin. Checkpoint mapping, numerical tests and reduced-size native-model CPU/CUDA forwards have passed. Full-model image generation and visual comparison against upstream have **not** been completed.

## Install

From the ComfyUI root:

```bash
git clone https://github.com/slmonker/ComfyUI-QwenImage21-Fun-PDD.git custom_nodes/ComfyUI-QwenImage21-Fun-PDD
```

Restart ComfyUI and refresh the browser. No additional Python packages are required beyond the ComfyUI environment's PyTorch and safetensors.

Download the original `Qwen-Image-2.1-Fun-Acc-4Step.safetensors` from the upstream model repository and place it in a registered ComfyUI LoRA directory. Weights are not distributed here. Only the `qwenimage21_extracted_prefused_v1` four-head export is supported.

Requires native Qwen-Image 2.1 support, ModelPatcher weight wrappers, the DIFFUSION_MODEL wrapper extension, and fused MLP LoRA slice mapping. Tested against ComfyUI commit `1568e6cfd0` with the relevant core files unmodified.

## Use

Search for **PDD** and add **Qwen-Image 2.1 Fun PDD (4-Step)**.

- Class: `QwenImage21FunPDDLoader`
- Category: `loaders / Qwen-Image 2.1`
- Inputs: base `MODEL`, original LoRA filename
- Outputs: patched `MODEL`, fixed PDD Euler `SAMPLER`, fixed `SIGMAS`

Connect all three outputs to **SamplerCustom**, set **cfg=1, add_noise=true**, and connect the positive/negative conditioning and latent from your existing Qwen-Image 2.1 workflow. Decode its output using the existing VAE.

Alternatively use BasicGuider + RandomNoise + SamplerCustomAdvanced. The model output goes to BasicGuider; sampler and sigmas go directly to SamplerCustomAdvanced.

Do not also apply the same file through a regular LoRA loader. Do not replace the supplied schedule with a normal four-step KSampler or an additional shift/scheduler node.

## Example workflows

| Use case | Workflow |
| --- | --- |
| Four-step text-to-image (T2I) | [Fun-PDD-sampling-4steps-t2i.json](examples/Fun-PDD-sampling-4steps-t2i.json) |
| Four-step image editing | [qwenimage2.1-pdd-4steps-edit.json](examples/qwenimage2.1-pdd-4steps-edit.json) |

Both workflows were supplied by the repository maintainer and are included unchanged. Download and drag the JSON into ComfyUI, select the model, text encoder, VAE and LoRA available in your installation, and install any additional custom nodes used by the workflow. Reselect the input image for the editing workflow. Model weights and input images are not bundled.

JSON parsing and node-link consistency were checked when adding these files, with no obvious credentials found. This is not an end-to-end render test on other installations and does not extend the compatibility claims below.

## What it loads

- 231 LoRA pairs, rank=alpha=64, fixed strength 1.
- 65 complete normalization weights as replacement patches, not additive deltas.
- Four output heads from `proj_out.weight`, shape `[4, 64, 4096]`, selected by sigma.
- The fixed sigma sequence `[1, 0.9169867038726807, 0.7861579060554504, 0.5494909882545471, 0]`.

The adapter validates every checkpoint key and tensor shape, including the separate gate/projection halves of fused MLPs. It uses a model clone and ModelPatcher extensions, not core edits or direct forward monkey-patching. Head selection is scoped to each forward using ContextVar, with cleanup on exceptions. Euler state remains FP32; native Qwen-Image 2.1 owns timestep rounding. Prefix KV caching is disabled for the patched clone to match the upstream example.

## Limitations and tests

Start with an unmodified BF16 base model and CFG=1. This is not for legacy Qwen-Image, Edit 2509/2511, or third-party Diffusers pipeline objects.

ComfyUI merges LoRA weights whereas the upstream Diffusers implementation evaluates separate low-rank branches. Bitwise or pixel-identical upstream output is **not** promised. FP8/GGUF, dynamic VRAM paths, multi-device sharding, torch.compile, masked inpainting and combinations with other model patches remain unverified.

See [TESTING.md](TESTING.md) for reproducible tests and the precise validation scope. No complete-base-model render or performance benchmark is claimed.

## Provenance and licensing

The model, PDD method and bundled `pdd_config.json` originate from [Alibaba PAI](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs); see also [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun). This repository contains no model weights or copied upstream inference scripts.

Upstream materials retain their respective terms. No independent software license has been selected for this repository yet. Public availability does not replace upstream permissions or imply endorsement.
