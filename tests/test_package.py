import ast
import json
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


if __name__ == "__main__":
    unittest.main()
