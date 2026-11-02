"""Deterministic, content-free quality evaluation for acceleration candidates.

Responses are accepted only as in-memory values (or from stdin by the CLI) and
are never written to the receipt.  This module intentionally does not create or
verify the external operator attestation consumed by ``promotion_evaluator``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, NoReturn


EVALUATOR_ID = "candidate-quality-evaluator-v1"
SUITE_SCHEMA_VERSION = 1
RECEIPT_SCHEMA_VERSION = 1
MAX_INPUT_BYTES = 16_777_216
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ALLOWED_ARMS = frozenset({"plain", "mtp", "dflash2", "dflash2-n4", "dflash2-n7"})
NOT_EVALUATED_REASONS = frozenset(
    {"responses_unavailable", "capability_unsupported", "evaluation_not_run"}
)


class QualityContractError(ValueError):
    """Stable error code that cannot contain response content."""


def _reject(code: str) -> NoReturn:
    raise QualityContractError(code)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        _reject(code)
    return value


def _safe_id(value: Any, code: str) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        _reject(code)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        _reject(code)


def _read_json(path: Path, error_prefix: str) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualityContractError(f"{error_prefix}_unreadable") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= MAX_INPUT_BYTES:
            _reject(f"{error_prefix}_file_invalid")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 65_536)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_INPUT_BYTES:
                _reject(f"{error_prefix}_file_invalid")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualityContractError(f"{error_prefix}_json_invalid") from exc
    if not isinstance(value, dict):
        _reject(f"{error_prefix}_json_invalid")
    return value


def _validate_generation(value: Any) -> None:
    if not isinstance(value, dict):
        _reject("generation_invalid")
    _exact_keys(value, {"temperature", "seed", "max_tokens"}, "generation_keys_invalid")
    temperature = value["temperature"]
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature != 0
    ):
        _reject("generation_temperature_invalid")
    if not isinstance(value["seed"], int) or isinstance(value["seed"], bool):
        _reject("generation_seed_invalid")
    if (
        not isinstance(value["max_tokens"], int)
        or isinstance(value["max_tokens"], bool)
        or not 1 <= value["max_tokens"] <= 4096
    ):
        _reject("generation_max_tokens_invalid")


def _validate_request(value: Any) -> None:
    if not isinstance(value, dict):
        _reject("request_invalid")
    allowed = {"messages", "generation", "tools", "tool_choice"}
    if not {"messages", "generation"}.issubset(value) or not set(value).issubset(allowed):
        _reject("request_keys_invalid")
    messages = value["messages"]
    if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
        _reject("messages_invalid")
    for message in messages:
        if not isinstance(message, dict):
            _reject("message_invalid")
        _exact_keys(message, {"role", "content"}, "message_keys_invalid")
        if message["role"] not in {"system", "user", "assistant"}:
            _reject("message_role_invalid")
        if (
            not isinstance(message["content"], str)
            or not message["content"]
            or len(message["content"]) > 131_072
        ):
            _reject("message_content_invalid")
    _validate_generation(value["generation"])
    if ("tools" in value) != ("tool_choice" in value):
        _reject("tool_request_incomplete")
    if "tools" in value:
        tools = value["tools"]
        if not isinstance(tools, list) or len(tools) != 1 or not isinstance(tools[0], dict):
            _reject("tools_invalid")
        tool = tools[0]
        _exact_keys(tool, {"type", "function"}, "tool_keys_invalid")
        if tool["type"] != "function" or not isinstance(tool["function"], dict):
            _reject("tool_invalid")
        _exact_keys(
            tool["function"], {"name", "description", "parameters"}, "tool_function_keys_invalid"
        )
        _safe_id(tool["function"]["name"], "tool_name_invalid")
        if not isinstance(tool["function"]["description"], str):
            _reject("tool_description_invalid")
        if not isinstance(tool["function"]["parameters"], dict):
            _reject("tool_parameters_invalid")
        if value["tool_choice"] != {"type": "function", "function": {"name": tool["function"]["name"]}}:
            _reject("tool_choice_invalid")


def _validate_check(value: Any) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        _reject("check_invalid")
    check_type = value["type"]
    expected_keys = {
        "exact": {"type", "expected", "casefold"},
        "json_shape": {"type", "schema", "expected"},
        "python_lambda_cases": {"type", "argument", "cases"},
        "text_constraints": {"type", "required_regex", "forbidden_regex", "max_lines"},
        "tool_call": {"type", "function_name", "arguments"},
    }.get(check_type)
    if expected_keys is None:
        _reject("check_type_unsupported")
    _exact_keys(value, expected_keys, "check_keys_invalid")
    if check_type == "exact":
        if not isinstance(value["expected"], str) or not isinstance(value["casefold"], bool):
            _reject("exact_check_invalid")
    elif check_type == "json_shape":
        if not isinstance(value["schema"], dict) or not isinstance(value["expected"], dict):
            _reject("json_check_invalid")
    elif check_type == "python_lambda_cases":
        _safe_id(value["argument"], "python_argument_invalid")
        cases = value["cases"]
        if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
            _reject("python_cases_invalid")
        for case in cases:
            if not isinstance(case, dict):
                _reject("python_case_invalid")
            _exact_keys(case, {"input", "expected"}, "python_case_keys_invalid")
            if not isinstance(case["input"], int) or not isinstance(case["expected"], int):
                _reject("python_case_value_invalid")
    elif check_type == "text_constraints":
        if (
            not isinstance(value["required_regex"], list)
            or not all(isinstance(item, str) and item for item in value["required_regex"])
            or not isinstance(value["forbidden_regex"], list)
            or not all(isinstance(item, str) and item for item in value["forbidden_regex"])
            or not isinstance(value["max_lines"], int)
            or isinstance(value["max_lines"], bool)
            or not 1 <= value["max_lines"] <= 32
        ):
            _reject("constraint_check_invalid")
        try:
            for pattern in value["required_regex"] + value["forbidden_regex"]:
                re.compile(pattern)
        except re.error as exc:
            raise QualityContractError("constraint_regex_invalid") from exc
    elif check_type == "tool_call":
        _safe_id(value["function_name"], "tool_check_name_invalid")
        if not isinstance(value["arguments"], dict):
            _reject("tool_check_arguments_invalid")


def load_suite(path: Path) -> tuple[dict[str, Any], str]:
    suite = _read_json(path, "suite")
    _exact_keys(suite, {"schema_version", "suite_id", "cases"}, "suite_keys_invalid")
    if suite["schema_version"] != SUITE_SCHEMA_VERSION:
        _reject("suite_version_unsupported")
    _safe_id(suite["suite_id"], "suite_id_invalid")
    cases = suite["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 100:
        _reject("suite_cases_invalid")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            _reject("suite_case_invalid")
        _exact_keys(case, {"case_id", "critical", "request", "check"}, "suite_case_keys_invalid")
        case_id = _safe_id(case["case_id"], "case_id_invalid")
        if case_id in seen:
            _reject("case_id_duplicate")
        seen.add(case_id)
        if not isinstance(case["critical"], bool):
            _reject("case_critical_invalid")
        _validate_request(case["request"])
        _validate_check(case["check"])
    return suite, _digest(suite)


def _json_type_matches(value: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    return False


def _matches_json_schema(value: Any, schema: Mapping[str, Any]) -> bool:
    """Validate the small, explicit JSON-schema subset used by the public suite."""

    allowed = {"type", "required", "properties", "additionalProperties", "items"}
    if not set(schema).issubset(allowed) or not isinstance(schema.get("type"), str):
        _reject("json_schema_contract_invalid")
    if not _json_type_matches(value, schema["type"]):
        return False
    if schema["type"] == "object":
        if not isinstance(value, dict):
            return False
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        if (
            not isinstance(required, list)
            or not all(isinstance(item, str) for item in required)
            or not isinstance(properties, dict)
            or not all(isinstance(key, str) and isinstance(child, dict) for key, child in properties.items())
        ):
            _reject("json_schema_contract_invalid")
        if not set(required).issubset(value):
            return False
        if schema.get("additionalProperties") is False and not set(value).issubset(properties):
            return False
        return all(key not in value or _matches_json_schema(value[key], child) for key, child in properties.items())
    if schema["type"] == "array":
        items = schema.get("items")
        if not isinstance(items, dict):
            _reject("json_schema_contract_invalid")
        return all(_matches_json_schema(item, items) for item in value)
    return True


def _eval_arithmetic(node: ast.AST, argument: str, argument_value: int) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        if abs(node.value) > 1_000_000:
            _reject("python_constant_out_of_range")
        return node.value
    if isinstance(node, ast.Name) and node.id == argument:
        return argument_value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_arithmetic(node.operand, argument, argument_value)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    ):
        left = _eval_arithmetic(node.left, argument, argument_value)
        right = _eval_arithmetic(node.right, argument, argument_value)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            _reject("python_division_by_zero")
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        return left % right
    _reject("python_expression_unsafe")


def _check_python_lambda(response: str, check: Mapping[str, Any]) -> bool:
    if len(response) > 4096 or "```" in response:
        return False
    try:
        parsed = ast.parse(response.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return False
    expression = parsed.body
    argument = check["argument"]
    if (
        not isinstance(expression, ast.Lambda)
        or len(expression.args.args) != 1
        or expression.args.args[0].arg != argument
        or expression.args.vararg is not None
        or expression.args.kwarg is not None
        or expression.args.kwonlyargs
        or expression.args.defaults
    ):
        return False
    try:
        return all(
            _eval_arithmetic(expression.body, argument, case["input"]) == case["expected"]
            for case in check["cases"]
        )
    except QualityContractError:
        return False


def _evaluate_case(case: Mapping[str, Any], response: Any) -> bool:
    check = case["check"]
    check_type = check["type"]
    if check_type == "tool_call":
        if not isinstance(response, dict):
            return False
        if set(response) != {"tool_calls"} or not isinstance(response["tool_calls"], list):
            return False
        calls = response["tool_calls"]
        if len(calls) != 1 or not isinstance(calls[0], dict):
            return False
        call = calls[0]
        if (
            not {"type", "function"}.issubset(call)
            or not set(call).issubset({"id", "type", "function"})
            or call["type"] != "function"
            or not isinstance(call["function"], dict)
            or ("id" in call and (not isinstance(call["id"], str) or not call["id"]))
        ):
            return False
        function = call["function"]
        if set(function) != {"name", "arguments"} or function["name"] != check["function_name"]:
            return False
        arguments = function["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return False
        return arguments == check["arguments"]
    if not isinstance(response, str):
        return False
    if check_type == "exact":
        actual = response.strip()
        expected = check["expected"].strip()
        if check["casefold"]:
            actual, expected = actual.casefold(), expected.casefold()
        return actual == expected
    if check_type == "json_shape":
        try:
            value = json.loads(response)
        except json.JSONDecodeError:
            return False
        return _matches_json_schema(value, check["schema"]) and value == check["expected"]
    if check_type == "python_lambda_cases":
        return _check_python_lambda(response, check)
    if check_type == "text_constraints":
        lines = response.strip().splitlines()
        if len(lines) > check["max_lines"]:
            return False
        return all(re.fullmatch(pattern, response.strip(), re.DOTALL) for pattern in check["required_regex"]) and not any(
            re.search(pattern, response, re.DOTALL) for pattern in check["forbidden_regex"]
        )
    _reject("check_type_unsupported")


def _receipt_core(
    *,
    arm: str,
    suite: Mapping[str, Any],
    suite_sha256: str,
    benchmark_report_sha256: str,
    case_results: list[dict[str, Any]],
    status: str,
    reason: str | None,
    baseline_receipt_content_sha256: str | None,
) -> dict[str, Any]:
    passed = sum(result["status"] == "pass" for result in case_results)
    evaluated = sum(result["status"] in {"pass", "fail"} for result in case_results)
    score = round(passed / evaluated, 6) if evaluated else None
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "evaluator_id": EVALUATOR_ID,
        "arm": arm,
        "suite_id": suite["suite_id"],
        "suite_canonical_sha256": suite_sha256,
        "benchmark_report_sha256": benchmark_report_sha256,
        "quality_status": status,
        "not_evaluated_reason": reason,
        "baseline_receipt_content_sha256": baseline_receipt_content_sha256,
        "cases": case_results,
        "aggregate": {
            "total_cases": len(suite["cases"]),
            "evaluated_cases": evaluated,
            "passed_cases": passed,
            "failed_cases": evaluated - passed,
            "score": score,
        },
        "privacy": {
            "prompts_recorded": False,
            "responses_recorded": False,
            "tokens_recorded": False,
            "endpoints_recorded": False,
            "secrets_recorded": False,
        },
        "external_attestation_created": False,
    }


def _validate_baseline_receipt(
    baseline: Mapping[str, Any], suite: Mapping[str, Any], suite_sha256: str
) -> str:
    required = {
        "schema_version", "evaluator_id", "arm", "suite_id", "suite_canonical_sha256",
        "benchmark_report_sha256", "quality_status", "not_evaluated_reason",
        "baseline_receipt_content_sha256", "cases", "aggregate", "privacy",
        "external_attestation_created", "content_sha256",
    }
    _exact_keys(baseline, required, "baseline_receipt_keys_invalid")
    claimed = _sha256(baseline["content_sha256"], "baseline_content_sha256_invalid")
    content = dict(baseline)
    content.pop("content_sha256")
    if _digest(content) != claimed:
        _reject("baseline_content_sha256_mismatch")
    if (
        baseline["schema_version"] != RECEIPT_SCHEMA_VERSION
        or baseline["evaluator_id"] != EVALUATOR_ID
        or baseline["arm"] not in {"plain", "mtp"}
        or baseline["suite_id"] != suite["suite_id"]
        or baseline["suite_canonical_sha256"] != suite_sha256
        or baseline["quality_status"] == "not_evaluated"
        or baseline["not_evaluated_reason"] is not None
        or baseline["baseline_receipt_content_sha256"] is not None
    ):
        _reject("baseline_receipt_incompatible")
    _sha256(baseline["benchmark_report_sha256"], "baseline_report_sha256_invalid")
    case_ids = [case["case_id"] for case in suite["cases"]]
    critical_by_id = {case["case_id"]: case["critical"] for case in suite["cases"]}
    cases = baseline["cases"]
    if (
        not isinstance(cases, list)
        or [result.get("case_id") for result in cases if isinstance(result, dict)] != case_ids
        or any(
            not isinstance(result, dict)
            or set(result) != {"case_id", "critical", "status"}
            or result["status"] not in {"pass", "fail"}
            or result["critical"] is not critical_by_id.get(result["case_id"])
            for result in cases
        )
    ):
        _reject("baseline_cases_invalid")
    passed = sum(result["status"] == "pass" for result in cases)
    expected_aggregate = {
        "total_cases": len(cases),
        "evaluated_cases": len(cases),
        "passed_cases": passed,
        "failed_cases": len(cases) - passed,
        "score": round(passed / len(cases), 6),
    }
    if baseline["aggregate"] != expected_aggregate:
        _reject("baseline_aggregate_invalid")
    expected_status = "pass" if passed == len(cases) else "fail"
    if baseline["quality_status"] != expected_status:
        _reject("baseline_quality_status_invalid")
    if baseline["privacy"] != {
        "prompts_recorded": False,
        "responses_recorded": False,
        "tokens_recorded": False,
        "endpoints_recorded": False,
        "secrets_recorded": False,
    } or baseline["external_attestation_created"] is not False:
        _reject("baseline_privacy_invalid")
    return claimed


def evaluate(
    suite: Mapping[str, Any],
    suite_sha256: str,
    benchmark_report_sha256: str,
    arm: str,
    responses: Mapping[str, Any] | None,
    *,
    baseline_receipt: Mapping[str, Any] | None = None,
    not_evaluated_reason: str | None = None,
) -> dict[str, Any]:
    """Evaluate transient responses and return a content-free receipt."""

    _sha256(suite_sha256, "suite_sha256_invalid")
    if _digest(suite) != suite_sha256:
        _reject("suite_sha256_mismatch")
    _sha256(benchmark_report_sha256, "benchmark_report_sha256_invalid")
    if arm not in ALLOWED_ARMS:
        _reject("arm_invalid")
    is_dflash = arm.startswith("dflash2")
    if baseline_receipt is not None and not is_dflash:
        _reject("baseline_requirement_invalid")
    if is_dflash and baseline_receipt is None and not_evaluated_reason is None:
        _reject("baseline_requirement_invalid")

    baseline_hash: str | None = None
    if baseline_receipt is not None:
        baseline_hash = _validate_baseline_receipt(baseline_receipt, suite, suite_sha256)

    if not_evaluated_reason is not None:
        if not_evaluated_reason not in NOT_EVALUATED_REASONS or responses is not None:
            _reject("not_evaluated_contract_invalid")
        case_results = [
            {"case_id": case["case_id"], "critical": case["critical"], "status": "not_evaluated"}
            for case in suite["cases"]
        ]
        receipt = _receipt_core(
            arm=arm,
            suite=suite,
            suite_sha256=suite_sha256,
            benchmark_report_sha256=benchmark_report_sha256,
            case_results=case_results,
            status="not_evaluated",
            reason=not_evaluated_reason,
            baseline_receipt_content_sha256=baseline_hash,
        )
        receipt["content_sha256"] = _digest(receipt)
        return receipt

    if not isinstance(responses, Mapping):
        _reject("responses_missing")
    expected_case_ids = [case["case_id"] for case in suite["cases"]]
    if set(responses) != set(expected_case_ids):
        _reject("responses_case_matrix_invalid")
    case_results = []
    for case in suite["cases"]:
        passed = _evaluate_case(case, responses[case["case_id"]])
        case_results.append(
            {"case_id": case["case_id"], "critical": case["critical"], "status": "pass" if passed else "fail"}
        )

    quality_status = "pass" if all(result["status"] == "pass" for result in case_results) else "fail"
    if baseline_receipt is not None:
        baseline_cases = {result["case_id"]: result for result in baseline_receipt["cases"]}
        baseline_passed = baseline_receipt["aggregate"]["passed_cases"]
        baseline_evaluated = baseline_receipt["aggregate"]["evaluated_cases"]
        candidate_passed = sum(result["status"] == "pass" for result in case_results)
        critical_regression = any(
            result["critical"]
            and baseline_cases[result["case_id"]]["status"] == "pass"
            and result["status"] == "fail"
            for result in case_results
        )
        score_not_lower = candidate_passed * baseline_evaluated >= baseline_passed * len(case_results)
        quality_status = "pass" if score_not_lower and not critical_regression else "fail"

    receipt = _receipt_core(
        arm=arm,
        suite=suite,
        suite_sha256=suite_sha256,
        benchmark_report_sha256=benchmark_report_sha256,
        case_results=case_results,
        status=quality_status,
        reason=None,
        baseline_receipt_content_sha256=baseline_hash,
    )
    receipt["content_sha256"] = _digest(receipt)
    return receipt


def write_immutable_receipt(receipt: Mapping[str, Any], output_dir: Path) -> Path:
    content_sha256 = _sha256(receipt.get("content_sha256"), "receipt_content_sha256_invalid")
    content = dict(receipt)
    content.pop("content_sha256")
    if _digest(content) != content_sha256:
        _reject("receipt_content_sha256_mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"candidate-quality-{content_sha256}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(target, flags, 0o600)
    try:
        payload = _canonical_bytes(receipt) + b"\n"
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written < 1:
                raise OSError("receipt_write_incomplete")
            offset += written
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        target.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    try:
        target.chmod(0o400)
        if os.name != "nt":
            directory_fd = os.open(output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


def _read_responses_from_stdin() -> Mapping[str, Any]:
    payload = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if not payload or len(payload) > MAX_INPUT_BYTES:
        _reject("responses_stdin_invalid")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualityContractError("responses_stdin_invalid") from exc
    if not isinstance(value, dict):
        _reject("responses_stdin_invalid")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--benchmark-report-sha256", required=True)
    parser.add_argument("--arm", choices=sorted(ALLOWED_ARMS), required=True)
    parser.add_argument("--baseline-receipt", type=Path)
    parser.add_argument("--not-evaluated", choices=sorted(NOT_EVALUATED_REASONS))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        suite, suite_sha256 = load_suite(args.suite)
        baseline = _read_json(args.baseline_receipt, "baseline") if args.baseline_receipt else None
        responses = None if args.not_evaluated else _read_responses_from_stdin()
        receipt = evaluate(
            suite,
            suite_sha256,
            args.benchmark_report_sha256,
            args.arm,
            responses,
            baseline_receipt=baseline,
            not_evaluated_reason=args.not_evaluated,
        )
        target = write_immutable_receipt(receipt, args.output_dir)
    except (OSError, QualityContractError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
