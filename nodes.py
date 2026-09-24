"""Native ComfyUI adapter for the Qwen-Image 2.1 Fun prefused PDD export."""

import json
import logging
from contextvars import ContextVar
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

import comfy.lora
import comfy.model_base
import comfy.model_management
import comfy.samplers
from comfy.k_diffusion.sampling import sample_euler
from comfy.patcher_extension import WrappersMP
import folder_paths


PATCH_KEY = "qwenimage21_fun_pdd"
HEAD_KEY = "diffusion_model.proj_out.weight"
CONFIG_PATH = Path(__file__).with_name("pdd_config.json")


def read_config():
    with CONFIG_PATH.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def build_patches(base_model, state, config):
    """Validate every tensor before handing patches to ComfyUI's native loader."""
    targets = config["lora_targets"].split(",")
    full_names = config["pdd_full_parameters"]
    expected = {"proj_out.weight", *full_names}
    for name in targets:
        expected.update((name + ".lora_down", name + ".lora_up"))
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise ValueError(f"Not the supported Fun PDD export: missing={missing[:5]}, extra={extra[:5]}")

    model_sd = base_model.state_dict()
    key_map = comfy.lora.model_lora_keys_unet(base_model, {})
    converted = {}
    mapping = {}
    rank = config["lora_rank"]
    for name in targets:
        if name not in key_map:
            raise ValueError(f"ComfyUI cannot map {name}; update native Qwen-Image 2.1 support.")
        destination = key_map[name]
        key = destination if isinstance(destination, str) else destination[0]
        shape = list(model_sd[key].shape)
        if not isinstance(destination, str):
            axis, start, length = destination[1]
            if start + length > shape[axis]:
                raise ValueError(f"Invalid fused-layer mapping: {name}")
            shape[axis] = length
        down, up = state[name + ".lora_down"], state[name + ".lora_up"]
        if len(shape) != 2 or tuple(down.shape) != (rank, shape[1]) or tuple(up.shape) != (shape[0], rank):
            raise ValueError(f"LoRA/base shape mismatch at {name}: down={tuple(down.shape)}, up={tuple(up.shape)}, base={shape}")
        converted[name + ".lora_down.weight"] = down
        converted[name + ".lora_up.weight"] = up
        converted[name + ".alpha"] = torch.tensor(config["lora_alpha"], dtype=torch.float32)
        mapping[name] = destination

    for name in full_names:
        key = "diffusion_model." + name
        if key not in model_sd or state[name].shape != model_sd[key].shape:
            raise ValueError(f"Full-parameter/base shape mismatch: {name}")
        module = name.removesuffix(".weight")
        converted[module + ".set_weight"] = state[name]
        mapping[module] = key

    heads = state["proj_out.weight"]
    if HEAD_KEY not in model_sd or tuple(heads.shape) != (4, *model_sd[HEAD_KEY].shape):
        raise ValueError(f"Expected four output heads compatible with this base, got {tuple(heads.shape)}")
    patches = comfy.lora.load_lora(converted, mapping, log_missing=True)
    if len(patches) != len(targets) + len(full_names):
        raise ValueError("ComfyUI did not load all PDD patches; no model was modified.")
    return patches, heads


class PDDHeadSelector:
    """A patcher-managed weight wrapper; step selection is scoped to one forward."""

    def __init__(self, heads, sigmas):
        self.heads = heads
        self.sigmas = tuple(sigmas[:-1])
        self.active_head = ContextVar("qwen21_pdd_head", default=None)

    def __call__(self, weight):
        index = self.active_head.get()
        if index is None:
            raise RuntimeError("PDD output head called outside its model wrapper.")
        # Keep the 2 MiB bank on CPU; use Comfy's cast path for the current head only.
        return comfy.model_management.cast_to_device(self.heads[index], weight.device, weight.dtype)

    def wrap(self, executor, x, timestep, *args, **kwargs):
        values = timestep.detach().float().reshape(-1)
        sigma = values[0].item()
        index = min(range(4), key=lambda i: abs(self.sigmas[i] - sigma))
        if abs(self.sigmas[index] - sigma) > 1e-6 or not torch.all(values == values[0]).item():
            raise ValueError("Fun PDD needs its exact four sigmas. Connect this node's SAMPLER and SIGMAS to SamplerCustom.")
        token = self.active_head.set(index)
        try:
            # Native QwenImage21 already implements the released timestep rounding.
            return executor(x, timestep, *args, **kwargs).float()
        finally:
            self.active_head.reset(token)


class PDDEulerSampler:
    def __init__(self, sigmas):
        self.sigmas = tuple(sigmas)

    def __call__(self, model, x, sigmas, extra_args=None, callback=None, disable=None):
        expected = torch.tensor(self.sigmas, dtype=torch.float32, device=sigmas.device)
        if sigmas.shape != expected.shape or not torch.allclose(sigmas.float(), expected, rtol=0, atol=1e-6):
            raise ValueError("Fun PDD uses a fixed four-step schedule; do not replace or slice its SIGMAS output.")
        return sample_euler(model, x.float(), sigmas.float(), extra_args=extra_args, callback=callback, disable=disable)


def patch_model(model, state, config):
    if not isinstance(model.model, comfy.model_base.QwenImage21):
        raise ValueError("Use a native ComfyUI Qwen-Image 2.1 base MODEL, not Qwen-Image/2509/2511 or a Diffusers pipeline.")
    if model.get_wrappers(WrappersMP.DIFFUSION_MODEL, PATCH_KEY):
        raise ValueError("Fun PDD is already applied. Connect the unpatched base model.")
    if HEAD_KEY in model.weight_wrapper_patches:
        raise ValueError("The output head already has a weight wrapper. Connect the unpatched base model.")
    patches, heads = build_patches(model.model, state, config)
    clone = model.clone()
    applied = clone.add_patches(patches, strength_patch=1.0, strength_model=1.0)
    if len(applied) != len(patches):
        raise ValueError("ComfyUI rejected some PDD patches; no model was returned.")
    selector = PDDHeadSelector(heads, config["pdd_sigmas"])
    clone.add_weight_wrapper(HEAD_KEY, selector)
    clone.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, PATCH_KEY, selector.wrap)
    # Match the official example's use_kv_cache=False without editing the shared model.
    options = clone.model_options["transformer_options"]
    options["qwen_image21_cache"] = {"device": "off"}
    logging.info("QwenImage21 Fun PDD: loaded %d LoRA pairs, %d full parameters and 4 output heads.",
                 len(config["lora_targets"].split(",")), len(config["pdd_full_parameters"]))
    sampler = comfy.samplers.KSAMPLER(PDDEulerSampler(config["pdd_sigmas"]))
    sigmas = torch.tensor(config["pdd_sigmas"], dtype=torch.float32)
    return clone, sampler, sigmas


class QwenImage21FunPDDLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
        }}

    RETURN_TYPES = ("MODEL", "SAMPLER", "SIGMAS")
    RETURN_NAMES = ("model", "pdd_euler", "pdd_sigmas")
    FUNCTION = "load"
    CATEGORY = "loaders/Qwen-Image 2.1"
    DESCRIPTION = "Loads the Fun-Acc-4Step prefused PDD export, including full norms and step-dependent heads. Use all outputs with SamplerCustom, CFG=1."

    def load(self, model, lora_name):
        if lora_name not in folder_paths.get_filename_list("loras"):
            raise ValueError("Choose a LoRA from ComfyUI's registered loras folders.")
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if Path(path).suffix.lower() != ".safetensors":
            raise ValueError("Fun PDD expects the original .safetensors checkpoint.")
        config = read_config()
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if metadata.get("format") != config["pdd_export_format"]:
            raise ValueError("Not a qwenimage21_extracted_prefused_v1 checkpoint. Select the original Fun-Acc-4Step file.")
        state = load_file(path, device="cpu")
        return patch_model(model, state, config)


NODE_CLASS_MAPPINGS = {"QwenImage21FunPDDLoader": QwenImage21FunPDDLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenImage21FunPDDLoader": "Qwen-Image 2.1 Fun PDD (4-Step)"}
