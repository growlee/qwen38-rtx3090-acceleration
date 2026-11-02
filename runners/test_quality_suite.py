import copy
import ast
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_suite


class QualitySuiteTests(unittest.TestCase):
    def test_contract_counts_ids_and_generation(self):
        suite = quality_suite.build_suite()
        self.assertEqual(suite["schema_version"], 1)
        self.assertEqual(suite["suite_id"], "qwen38-quality-60-v1")
        self.assertEqual(len(suite["cases"]), 60)
        self.assertEqual(Counter(c["category"] for c in suite["cases"]), {
            "coding": 30, "reasoning_ru": 12, "instruction": 12, "retrieval": 6,
        })
        ids = [c["case_id"] for c in suite["cases"]]
        self.assertEqual(len(ids), len(set(ids)))
        for case in suite["cases"]:
            generation = case["request"]["generation"]
            self.assertEqual(generation["temperature"], 0)
            self.assertIsInstance(generation["seed"], int)
            self.assertGreater(generation["max_tokens"], 0)
            self.assertLessEqual(generation["max_tokens"], 4096)

    def test_full_function_oracles_use_five_or_more_cases(self):
        function_cases = [c for c in quality_suite.SUITE["cases"] if c["check"]["type"] == "python_function_cases"]
        self.assertEqual(len(function_cases), 29)
        for case in function_cases:
            self.assertGreaterEqual(len(case["check"]["cases"]), 5, case["case_id"])
            namespace = {"__builtins__": {"min": min, "max": max, "len": len, "list": list, "set": set, "sorted": sorted, "sum": sum, "enumerate": enumerate, "isinstance": isinstance, "dict": dict, "str": str, "int": int, "type": type, "all": all, "any": any, "range": range, "ValueError": ValueError, "KeyError": KeyError, "IndexError": IndexError}}
            exec(case["reference_source"], namespace)  # trusted fixture source, never model output
            fn = namespace[case["check"]["function_name"]]
            for vector in case["check"]["cases"]:
                outcomes = {"expected", "expected_exception"}.intersection(vector)
                self.assertEqual(len(outcomes), 1, case["case_id"])
                args = copy.deepcopy(vector["args"])
                kwargs = copy.deepcopy(vector["kwargs"])
                if "expected_exception" in vector:
                    exception_types = {"ValueError": ValueError}
                    self.assertIn(vector["expected_exception"], exception_types, case["case_id"])
                    with self.assertRaises(exception_types[vector["expected_exception"]], msg=case["case_id"]):
                        fn(*args, **kwargs)
                else:
                    self.assertEqual(fn(*args, **kwargs), vector["expected"], case["case_id"])

    def test_embedded_legacy_six_are_exact(self):
        source = Path(r"E:\Projects\home_lab\projects\ai-server\benchmarks\suites\acceleration-quality-v1.json")
        original = json.loads(source.read_text(encoding="utf-8"))["cases"]
        embedded = quality_suite.build_suite()["cases"][:6]
        stripped = [{k: v for k, v in case.items() if k != "category"} for case in embedded]
        self.assertEqual(stripped, original)

    def test_suite_is_defensively_copied_and_json_serializable(self):
        first = quality_suite.build_suite()
        second = quality_suite.build_suite()
        first["cases"][0]["case_id"] = "mutated"
        self.assertNotEqual(first, second)
        payload = json.dumps(second, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(len(hashlib.sha256(payload).hexdigest()), 64)
        self.assertEqual(json.loads(payload.decode("utf-8")), second)

    def test_fixture_module_has_no_io_process_or_dynamic_execution(self):
        module_path = Path(quality_suite.__file__)
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        forbidden_names = {"open", "exec", "eval", "compile", "__import__"}
        forbidden_attributes = {"write", "write_text", "write_bytes", "unlink", "mkdir", "rmdir", "remove", "rename", "replace", "run", "Popen", "socket", "urlopen"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, forbidden_names)
            elif isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, forbidden_attributes)

        for case in quality_suite.SUITE["cases"]:
            prompt_chars = sum(len(message["content"]) for message in case["request"]["messages"])
            self.assertLessEqual(prompt_chars, 32_000, case["case_id"])


if __name__ == "__main__":
    unittest.main()
