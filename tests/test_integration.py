import argparse
import importlib.util
import os
import sys
import unittest
from pathlib import Path

parser = argparse.ArgumentParser(description="Native ComfyUI PDD integration tests")
parser.add_argument("--comfy-root", default=os.environ.get("COMFYUI_PATH"))
parser.add_argument("--checkpoint", help="Optional original Fun-Acc-4Step safetensors")
parser.add_argument("--skip-cuda", action="store_true")
FLAGS = parser.parse_args()
if not FLAGS.comfy_root:
    parser.error("Provide --comfy-root or COMFYUI_PATH")
ROOT = Path(FLAGS.comfy_root).resolve()
if not (ROOT / "comfy" / "model_patcher.py").is_file():
    parser.error("--comfy-root must point to the ComfyUI root")
PLUGIN = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(FLAGS.checkpoint).resolve() if FLAGS.checkpoint else None
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import torch.nn.functional as F
import folder_paths
import comfy.lora
import comfy.model_base
import comfy.model_patcher
import comfy.supported_models
from comfy.patcher_extension import WrappersMP, add_wrapper_with_key

spec = importlib.util.spec_from_file_location("qwen21_fun_pdd", PLUGIN / "__init__.py", submodule_search_locations=[str(PLUGIN)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
nodes = sys.modules[spec.name + ".nodes"]
CONFIG = nodes.read_config()
CPU = torch.device("cpu")
torch.set_num_threads(4)


def make_model(meta=False, dtype=torch.float32):
    options = {"image_model": "qwen_image21", "dtype": dtype}
    if not meta:
        options.update(num_layers=32, num_attention_heads=2, attention_head_dim=8,
                       context_in_dim=16, axes_dims_rope=(2, 2, 4))
    cfg = comfy.supported_models.QwenImage21(options)
    base = comfy.model_base.QwenImage21(cfg, device=torch.device("meta") if meta else CPU)
    if not meta:
        for parameter in base.parameters():
            parameter.data.copy_(torch.randn_like(parameter) * 0.03)
    return comfy.model_patcher.ModelPatcher(base, CPU, CPU)


def make_state(model):
    sd = model.model.state_dict()
    mapping = comfy.lora.model_lora_keys_unet(model.model, {})
    state = {}
    for name in CONFIG["lora_targets"].split(","):
        target = mapping[name]
        key = target if isinstance(target, str) else target[0]
        out_dim, in_dim = sd[key].shape
        if not isinstance(target, str):
            out_dim = target[1][2]
        state[name + ".lora_down"] = torch.randn(64, in_dim) * 0.01
        state[name + ".lora_up"] = torch.randn(out_dim, 64) * 0.01
    for name in CONFIG["pdd_full_parameters"]:
        state[name] = torch.randn_like(sd["diffusion_model." + name]) * 0.03
    state["proj_out.weight"] = torch.randn(4, *sd[nodes.HEAD_KEY].shape) * 0.05
    return state


class PDDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    @unittest.skipIf(CHECKPOINT is None, "No original checkpoint supplied")
    def test_01_real_checkpoint_full_architecture_audit(self):
        model = make_model(meta=True)
        folder_paths.add_model_folder_path("loras", str(CHECKPOINT.parent))
        clone, sampler, sigmas = nodes.QwenImage21FunPDDLoader().load(model, CHECKPOINT.name)
        self.assertEqual(sum(map(len, clone.patches.values())), 296)
        self.assertEqual(len(clone.patches), 264)  # 32 pairs address halves of one gate_up.
        self.assertEqual(len(model.patches), 0)
        self.assertEqual(len(model.weight_wrapper_patches), 0)
        selector = clone.weight_wrapper_patches[nodes.HEAD_KEY][0]
        self.assertEqual(tuple(selector.heads.shape), (4, 64, 4096))
        self.assertEqual(selector.heads.dtype, torch.bfloat16)
        self.assertEqual(sigmas.tolist(), CONFIG["pdd_sigmas"])
        self.assertEqual(clone.model_options["transformer_options"]["qwen_image21_cache"], {"device": "off"})
        print("AUDIT: all 528 source tensors accounted for; 231 pairs + 65 full weights + 4 heads.")

    def test_02_fused_lora_math_and_norm_replacement(self):
        model = make_model()
        state = make_state(model)
        clone, _, _ = nodes.patch_model(model, state, CONFIG)
        name = "diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight"
        original = model.model.state_dict()[name].clone()
        actual = comfy.lora.calculate_weight(clone.patches[name], original.clone(), name)
        gate = "transformer_blocks.0.img_mlp.gate_layer"
        proj = "transformer_blocks.0.img_mlp.proj"
        expected = original + torch.cat([
            state[gate + ".lora_up"] @ state[gate + ".lora_down"],
            state[proj + ".lora_up"] @ state[proj + ".lora_down"],
        ])
        torch.testing.assert_close(actual, expected)
        name = "diffusion_model.txt_in.text_norm.weight"
        actual = comfy.lora.calculate_weight(clone.patches[name], model.model.state_dict()[name].clone(), name)
        torch.testing.assert_close(actual, state["txt_in.text_norm.weight"], rtol=0, atol=0)

    def test_03_head_order_and_exception_cleanup(self):
        heads = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        selector = nodes.PDDHeadSelector(heads, CONFIG["pdd_sigmas"])
        weight = torch.zeros(2, 3)
        x = torch.ones(1, 3)
        for index in (0, 1, 2, 3, 0, 3, 1):
            result = selector.wrap(lambda x, t: F.linear(x, selector(weight)), x,
                                   torch.tensor([CONFIG["pdd_sigmas"][index]]))
            torch.testing.assert_close(result, F.linear(x, heads[index]), rtol=0, atol=0)
            self.assertIsNone(selector.active_head.get())
        with self.assertRaisesRegex(RuntimeError, "outside"):
            selector(weight)
        def broken(x, t):
            raise RuntimeError("deliberate")
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            selector.wrap(broken, x, torch.tensor([1.0]))
        self.assertIsNone(selector.active_head.get())
        with self.assertRaisesRegex(ValueError, "exact four"):
            selector.wrap(broken, x, torch.tensor([0.75]))
        with self.assertRaisesRegex(ValueError, "exact four"):
            selector.wrap(broken, x, torch.tensor([1.0, CONFIG["pdd_sigmas"][1]]))

    def test_04_sampler_fp32_and_exact_schedule(self):
        sampler = nodes.PDDEulerSampler(CONFIG["pdd_sigmas"])
        sigmas = torch.tensor(CONFIG["pdd_sigmas"])
        visited = []
        def model(x, sigma):
            self.assertEqual(x.dtype, torch.float32)
            visited.append(sigma[0].item())
            velocity = (len(visited) * 0.125) * torch.ones_like(x)
            return x - sigma.reshape(-1, 1, 1, 1) * velocity
        x = torch.ones(1, 2, 2, 2, dtype=torch.bfloat16)
        output = sampler(model, x, sigmas, disable=True)
        expected = x.float()
        for i in range(4):
            expected = expected + (sigmas[i + 1] - sigmas[i]) * ((i + 1) * 0.125)
        torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(visited, CONFIG["pdd_sigmas"][:-1])
        with self.assertRaisesRegex(ValueError, "fixed four-step"):
            sampler(model, x, torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]), disable=True)

    def test_05_real_model_forward_and_clone_restore(self):
        model = make_model()
        state = make_state(model)
        original = {k: v.clone() for k, v in model.model.state_dict().items()}
        clone, _, _ = nodes.patch_model(model, state, CONFIG)
        clone.patch_model(device_to=CPU)
        clone.pre_run()
        opts = {"qwen_image21_cache": {"device": "off"}}
        for wrapper in clone.get_wrappers(WrappersMP.DIFFUSION_MODEL, nodes.PATCH_KEY):
            add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, nodes.PATCH_KEY, wrapper, opts)
        x = torch.randn(1, 64, 2, 2)
        context = torch.randn(1, 3, 16)
        outputs = []
        try:
            for sigma in CONFIG["pdd_sigmas"][:-1]:
                output = clone.model.apply_model(x, torch.tensor([sigma]), c_crossattn=context,
                                                transformer_options=opts)
                self.assertEqual(tuple(output.shape), tuple(x.shape))
                self.assertTrue(torch.isfinite(output).all())
                outputs.append(output)
            self.assertFalse(torch.equal(outputs[0], outputs[1]))
            self.assertIsNone(clone.weight_wrapper_patches[nodes.HEAD_KEY][0].active_head.get())
        finally:
            clone.cleanup()
            clone.unpatch_model(device_to=CPU)
        # Loading the original clone must remove the temporary weight wrapper too.
        model.patch_model(device_to=CPU)
        model.pre_run()
        try:
            for name, tensor in model.model.state_dict().items():
                torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
            output = model.model.apply_model(x, torch.tensor([1.0]), c_crossattn=context,
                                           transformer_options={"qwen_image21_cache": {"device": "off"}})
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(model.model.diffusion_model.proj_out.weight_function, [])
        finally:
            model.cleanup()
            model.unpatch_model(device_to=CPU)

    def test_06_malformed_and_duplicate_rejected(self):
        model = make_model()
        state = make_state(model)
        bad = dict(state)
        bad.pop("proj_out.weight")
        with self.assertRaisesRegex(ValueError, "missing"):
            nodes.patch_model(model, bad, CONFIG)
        bad = dict(state)
        bad["proj_out.weight"] = bad["proj_out.weight"][:1]
        with self.assertRaisesRegex(ValueError, "four output heads"):
            nodes.patch_model(model, bad, CONFIG)
        bad = dict(state)
        bad["img_in.lora_down"] = torch.empty(63, 64)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            nodes.patch_model(model, bad, CONFIG)
        clone, _, _ = nodes.patch_model(model, state, CONFIG)
        with self.assertRaisesRegex(ValueError, "already applied"):
            nodes.patch_model(clone, state, CONFIG)
        self.assertEqual(model.patches, {})
        with self.assertRaisesRegex(ValueError, "registered loras"):
            nodes.QwenImage21FunPDDLoader().load(model, "../outside.safetensors")


    @unittest.skipUnless(torch.cuda.is_available() and not FLAGS.skip_cuda, "CUDA unavailable or disabled")
    def test_07_cuda_bf16_native_forward(self):
        device = torch.device("cuda")
        model = make_model(dtype=torch.bfloat16)
        model.load_device = device
        state = {key: value.to(torch.bfloat16) for key, value in make_state(model).items()}
        clone, _, _ = nodes.patch_model(model, state, CONFIG)
        clone.patch_model(device_to=device)
        clone.pre_run()
        opts = {"qwen_image21_cache": {"device": "off"}}
        for wrapper in clone.get_wrappers(WrappersMP.DIFFUSION_MODEL, nodes.PATCH_KEY):
            add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, nodes.PATCH_KEY, wrapper, opts)
        x = torch.randn(1, 64, 2, 2, device=device)
        context = torch.randn(1, 3, 16, dtype=torch.bfloat16, device=device)
        try:
            for sigma in CONFIG["pdd_sigmas"][:-1]:
                output = clone.model.apply_model(x, torch.tensor([sigma], device=device), c_crossattn=context,
                                                transformer_options=opts)
                self.assertEqual(tuple(output.shape), tuple(x.shape))
                self.assertEqual(output.dtype, torch.float32)
                self.assertTrue(torch.isfinite(output).all())
            selector = clone.weight_wrapper_patches[nodes.HEAD_KEY][0]
            self.assertEqual(selector.heads.device.type, "cpu")
            self.assertIsNone(selector.active_head.get())
        finally:
            clone.cleanup()
            clone.unpatch_model(device_to=CPU)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
