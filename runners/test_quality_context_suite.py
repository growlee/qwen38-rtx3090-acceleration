import copy
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_context_suite as context_suite


class QualityContextSuiteTests(unittest.TestCase):
    def test_twelve_distinct_families_and_generation_contract(self):
        suite = context_suite.build_suite(7)
        self.assertEqual(suite["schema_version"], 1)
        self.assertEqual(len(suite["cases"]), 12)
        self.assertEqual(len(set(context_suite.FAMILY_IDS)), 12)
        self.assertEqual(Counter(c["fixture"]["task_type"] for c in suite["cases"]), {"fact": 3, "join": 3, "version": 3, "code": 3})
        self.assertEqual(len({context_suite.canonical_sha256(c["request"]) for c in suite["cases"]}), 12)
        for case in suite["cases"]:
            generation = case["request"]["generation"]
            self.assertEqual(generation["temperature"], 0)
            self.assertIs(type(generation["seed"]), int)
            expected_cap = 2048 if case["fixture"]["task_type"] == "code" else 256
            self.assertEqual(generation["max_tokens"], expected_cap)

    def test_build_is_deterministic_json_native_and_defensive(self):
        for family_id in context_suite.FAMILY_IDS:
            first = context_suite.build_case(family_id, 19)
            second = context_suite.build_case(family_id, 19)
            self.assertEqual(first, second)
            self.assertEqual(context_suite.canonical_sha256(first), context_suite.canonical_sha256(second))
            self.assertEqual(json.loads(json.dumps(first, ensure_ascii=False)), first)
            first["request"]["messages"][0]["content"] = "mutated"
            self.assertNotEqual(first, context_suite.build_case(family_id, 19))

    def test_only_padding_changes_across_lengths(self):
        for family_id in context_suite.FAMILY_IDS:
            short = context_suite.build_case(family_id, 1)
            long = context_suite.build_case(family_id, 37)
            self.assertEqual(short["check"], long["check"])
            self.assertEqual(short.get("reference_source"), long.get("reference_source"))
            self.assertEqual(short["request"]["generation"], long["request"]["generation"])
            short_question = short["request"]["messages"][1]["content"].split("\n\nВОПРОС:\n", 1)[1]
            long_question = long["request"]["messages"][1]["content"].split("\n\nВОПРОС:\n", 1)[1]
            self.assertEqual(short_question, long_question)
            spec = context_suite._SPECS[family_id]
            for record in (spec["anchor"], *spec.get("support", ())):
                self.assertEqual(short["request"]["messages"][1]["content"].count(record), 1)
                self.assertEqual(long["request"]["messages"][1]["content"].count(record), 1)

    def test_anchor_placement_and_support_distribution(self):
        for family_id in context_suite.FAMILY_IDS:
            padding_count = 90
            case = context_suite.build_case(family_id, padding_count)
            content = case["request"]["messages"][1]["content"].split("\n\nВОПРОС:\n", 1)[0]
            spec = context_suite._SPECS[family_id]
            ratio = content.index(spec["anchor"]) / len(content)
            if spec["position"] == "start":
                self.assertLess(ratio, 0.05)
            elif spec["position"] == "middle":
                self.assertGreater(ratio, 0.35)
                self.assertLess(ratio, 0.65)
            else:
                self.assertGreater(ratio, 0.90)
            supports = spec.get("support", ())
            if supports:
                support_ratios = [content.index(record) / len(content) for record in supports]
                self.assertTrue(any(value < ratio for value in support_ratios) or spec["position"] == "start")
                self.assertTrue(any(value > ratio for value in support_ratios) or spec["position"] == "end")
            self.assertEqual(case["fixture"]["padding_count"], padding_count)

    def test_three_trusted_code_oracles_and_sandbox_schema(self):
        code_cases = [context_suite.build_case(family_id, 0) for family_id in context_suite.FAMILY_IDS if "-code-" in family_id]
        self.assertEqual(len(code_cases), 3)
        self.assertEqual(len({case["check"]["function_name"] for case in code_cases}), 3)
        safe_builtins = {"max": max, "type": type, "int": int, "ValueError": ValueError}
        for case in code_cases:
            check = case["check"]
            self.assertEqual(check["type"], "python_function_cases")
            self.assertGreaterEqual(len(check["cases"]), 5)
            namespace = {"__builtins__": safe_builtins}
            exec(case["reference_source"], namespace)  # trusted authored oracle, never model output
            function = namespace[check["function_name"]]
            for vector in check["cases"]:
                outcomes = {"expected", "expected_exception"}.intersection(vector)
                self.assertEqual(len(outcomes), 1)
                args = copy.deepcopy(vector["args"])
                kwargs = copy.deepcopy(vector["kwargs"])
                if "expected_exception" in vector:
                    self.assertEqual(vector["expected_exception"], "ValueError")
                    with self.assertRaises(ValueError):
                        function(*args, **kwargs)
                else:
                    self.assertEqual(function(*args, **kwargs), vector["expected"])

    def test_bad_builder_inputs_are_rejected(self):
        for value in (-1, 1.5, True, "2"):
            with self.assertRaises(ValueError):
                context_suite.build_case(context_suite.FAMILY_IDS[0], value)
        with self.assertRaises(ValueError):
            context_suite.build_case("missing", 0)


if __name__ == "__main__":
    unittest.main()
