import ast
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PackageTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in ROOT.rglob("*.py"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    def test_pdd_config(self):
        config = json.loads((ROOT / "pdd_config.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(config["pdd_export_format"], "qwenimage21_extracted_prefused_v1")
        self.assertEqual(config["pdd_num_steps"], 4)
        self.assertEqual(config["pdd_block_size"], 1)
        self.assertEqual(config["lora_rank"], 64)
        self.assertEqual(config["lora_alpha"], 64)
        self.assertEqual(config["pdd_sampling_precision"], "native_time_fp32_state")
        sigmas = config["pdd_sigmas"]
        self.assertEqual(sigmas, [1.0, 0.9169867038726807, 0.7861579060554504, 0.5494909882545471, 0.0])
        self.assertTrue(all(a > b for a, b in zip(sigmas, sigmas[1:])))
        targets = config["lora_targets"].split(",")
        self.assertEqual(len(targets), 231)
        self.assertEqual(len(set(targets)), len(targets))
        self.assertEqual(len(set(config["pdd_full_parameters"])), 65)


    def test_maintainer_example_workflows(self):
        names = ("Fun-PDD-sampling-4steps-t2i.json", "qwenimage2.1-pdd-4steps-edit.json")
        for name in names:
            with self.subTest(workflow=name):
                workflow = json.loads((ROOT / "examples" / name).read_text(encoding="utf-8-sig"))
                self.assertEqual(workflow["version"], 0.4)
                nodes = {node["id"]: node for node in workflow["nodes"]}
                self.assertEqual(len(nodes), len(workflow["nodes"]))
                self.assertTrue(any(node["type"] == "QwenImage21FunPDDLoader" for node in nodes.values()))
                link_ids = set()
                for link_id, src, src_slot, dst, dst_slot, _ in workflow["links"]:
                    self.assertNotIn(link_id, link_ids)
                    link_ids.add(link_id)
                    self.assertIn(src, nodes)
                    self.assertIn(dst, nodes)
                    self.assertIn(link_id, nodes[src]["outputs"][src_slot]["links"])
                    self.assertEqual(nodes[dst]["inputs"][dst_slot]["link"], link_id)


    def test_bilingual_readmes_and_demo_assets(self):
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
        self.assertIn("**Unofficial native ComfyUI adapter", english)
        self.assertIn("[简体中文](README.zh-CN.md)", english)
        self.assertIn("[English](README.md)", chinese)
        self.assertFalse((ROOT / "README.en.md").exists())
        self.assertIn("NVIDIA GeForce RTX 5090 D", english)
        self.assertIn("32GB VRAM", english)
        self.assertIn("128GB DDR5 RAM", english)
        self.assertIn("NVIDIA GeForce RTX 5090 D，32GB 显存", chinese)
        self.assertIn("系统内存：128GB DDR5", chinese)
        expected = {"assets/demo-workflow.png", "assets/demo-sampler-comparison.png"}
        for text in (english, chinese):
            images = set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text))
            self.assertEqual(images, expected)
            for path in images:
                self.assertTrue((ROOT / path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
                if "://" not in target and not target.startswith("#"):
                    self.assertTrue((ROOT / target.split("#", 1)[0]).is_file(), target)


if __name__ == "__main__":
    unittest.main()
