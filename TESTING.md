# Testing / 测试

## Package checks — no ComfyUI installation needed

From this repository:

```bash
python -m unittest discover -s tests -p "test_package.py" -v
```

Checks source parsing, the bundled four-step configuration and LoRA/full-weight target counts.

## Native integration tests

Use the Python interpreter from an existing ComfyUI environment. Tests load the plugin directly from this repository; it need not already be installed in custom_nodes.

```bash
python tests/test_integration.py --comfy-root /path/to/ComfyUI
```

To also audit the original checkpoint against a full-size meta model:

```bash
python tests/test_integration.py --comfy-root /path/to/ComfyUI --checkpoint /path/to/Qwen-Image-2.1-Fun-Acc-4Step.safetensors
```

On Windows portable installations, substitute the portable environment's `python_embeded/python.exe` and quote paths containing spaces. `COMFYUI_PATH` may replace `--comfy-root`. Use `--skip-cuda` to skip the CUDA test.

No test downloads files. The checkpoint test is skipped if `--checkpoint` is omitted; the CUDA test is skipped if CUDA is unavailable or disabled. Synthetic tests instantiate reduced-size native Qwen-Image 2.1 models and do not load a full base checkpoint or text encoder.

## Observed local results

Seven integration tests passed with the original checkpoint supplied:

1. All **528 tensors** accounted for: 231 LoRA pairs + 65 normalization weights + one four-head tensor. Full architecture checked on the meta device; no full base weights loaded.
2. Fused gate_up halves matched the expected low-rank matrix updates; full normalization weights replaced the originals exactly in the test.
3. Four-head selection, repeated/out-of-order calls and exception cleanup passed.
4. Fixed-schedule Euler maintained FP32 state and matched the test reference within its numerical tolerance.
5. Reduced-size native CPU forwards produced finite outputs; original parameters and the original head were restored when returning to the unpatched model.
6. Missing keys, incompatible shapes, duplicate application and out-of-directory filename inputs were rejected.
7. Reduced-size native CUDA BF16 forwards produced finite FP32 denoised outputs for all four sigmas.

Environment: ComfyUI commit `1568e6cfd0`; CPU and NVIDIA GeForce RTX 5090 D. The relevant ComfyUI core files had no uncommitted changes at validation.

## Not yet validated

- Full-base-checkpoint end-to-end image generation or editing.
- Visual quality or bitwise parity against official Diffusers output.
- Dynamic VRAM, FP8/GGUF, multi-GPU, torch.compile, masked inpainting or arbitrary third-party patch combinations.

Please include the ComfyUI version, base-model type/precision, complete error traceback and a minimal workflow when filing an issue. Do not upload model weights, access tokens or private prompts.
