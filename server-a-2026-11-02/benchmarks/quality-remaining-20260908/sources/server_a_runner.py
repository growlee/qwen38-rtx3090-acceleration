#!/usr/bin/env python3
"""Single bounded Server A baseline/FreeToken run with fail-closed restore."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import socket
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


LOCK = Path("/srv/ai-server/state/acceleration-experiments/locks/host.lock")
OPERATOR_SOCKET = "/run/ai-server/supervisor/operator/supervisor.sock"
MODEL_DIR = Path("/srv/ai-server/staging/experiments/qwen38-nvfp4-freetoken/37b5130a2d2a1f7d4456ab3f8d05d0b2a45ea350/model")
MODEL_FILE = MODEL_DIR / "model.safetensors"
TOKENIZER_FILE = MODEL_DIR / "tokenizer.json"
SHA256SUMS = MODEL_DIR / "SHA256SUMS"
MODEL_SIZE = 19_709_883_808
MODEL_SHA256 = "bc642c1159ab9b0fcd2fdc65be4fee72fa560cf0570604741db50061ea063277"
TOKENIZER_SIZE = 19_989_325
TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
CANDIDATE_FILES = {
    ".gitattributes": 1_752,
    "README.md": 4_915,
    "chat_template.jinja": 8_952,
    "config.json": 15_095,
    "generation_config.json": 214,
    "model.safetensors": MODEL_SIZE,
    "preprocessor_config.json": 390,
    "processor_config.json": 1_191,
    "recipe.yaml": 239,
    "tokenizer.json": TOKENIZER_SIZE,
    "tokenizer_config.json": 1_166,
    "video_preprocessor_config.json": 385,
}
CURRENT_MODEL_SHA256 = "3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec"
CURRENT_MODEL_PATH = Path(f"/srv/ai-server/model-store/sha256/{CURRENT_MODEL_SHA256}/Qwen3.8-27B-Uncensored-Q4_K_M.gguf")
CURRENT_MODEL_DESTINATION = "/models/qwen/Qwen3.8-27B-Uncensored-Q4_K_M.gguf"
IMAGE = "local/ai-server-freetoken:9db1a39455a3fb107f3db83e381d10ceadfe5d99-ccfix1"
IMAGE_ID = "sha256:59c9b03387fa41a5580662ee2dbefd2458e04a8e820cefe06f489311cba5b160"
CONTAINER = "ai-exp-qwen38-nvfp4-freetoken"
MODEL_ID = "qwen3.8-27b-uncensored"
CLIENT = Path(__file__).with_name("benchmark_client.py")
BENCHMARK_CLIENT_SHA256 = "de6a783e295f4e12aee5a208fb4ca042d11220a49fb6ed8ff82e5ae1c671b24d"
DIAGNOSTIC_TAIL_CHARS = 8192
DIAGNOSTIC_LOG_LINES = 200
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
AUTHORIZATION_HEADER = re.compile(r"(?i)(authorization\s*:\s*)(?:bearer\s+)?\S+")
CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Za-z][A-Za-z0-9_-]*)(\s*=\s*)(?:'[^']*'|\"[^\"]*\"|[^\s]+)"
)
URL_USERINFO = re.compile(r"(?i)(https?://)[^\s/@:]+(?::[^\s/@]*)?@")


class RunError(RuntimeError):
    def __init__(self, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.details = details or {}


def sanitized_tail(value: bytes | str | None) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else value
    text = ANSI_ESCAPE.sub("", text)
    text = AUTHORIZATION_HEADER.sub(r"\1<redacted>", text)
    def redact_assignment(match: re.Match[str]) -> str:
        normalized = match.group(1).lower().replace("_", "").replace("-", "")
        if any(marker in normalized for marker in ("token", "password", "passwd", "secret", "apikey", "authorization")):
            return match.group(1) + match.group(2) + "<redacted>"
        return match.group(0)

    text = CREDENTIAL_ASSIGNMENT.sub(redact_assignment, text)
    text = URL_USERINFO.sub(r"\1<redacted>@", text)
    text = "".join(
        character if character in {"\n", "\t"} or (32 <= ord(character) < 127) or ord(character) >= 160 else "?"
        for character in text
    )
    return text[-DIAGNOSTIC_TAIL_CHARS:]


def command_diagnostic(value: subprocess.CompletedProcess[bytes]) -> dict[str, Any]:
    stdout = value.stdout or b""
    stderr = value.stderr or b""
    return {
        "returncode": value.returncode,
        "stdout_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_tail": sanitized_tail(stdout),
        "stderr_bytes": len(stderr),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_tail": sanitized_tail(stderr),
    }


def run(
    argv: list[str],
    *,
    timeout: float,
    input_bytes: bytes | None = None,
    check: bool = True,
    diagnostic_output: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    try:
        value = subprocess.run(
            argv,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, bytes) else (exc.stdout or "").encode("utf-8")
        stderr = exc.stderr if isinstance(exc.stderr, bytes) else (exc.stderr or "").encode("utf-8")
        partial = subprocess.CompletedProcess(argv, -1, stdout=stdout, stderr=stderr)
        details = command_diagnostic(partial) if diagnostic_output else {"returncode": -1}
        raise RunError("command_timeout", details) from exc
    if check and value.returncode != 0:
        details = command_diagnostic(value) if diagnostic_output else {"returncode": value.returncode}
        raise RunError("command_failed", details)
    return value


def operator(request: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(OPERATOR_SOCKET)
        client.sendall(json.dumps(request, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
        raw = b""
        while b"\n" not in raw:
            chunk = client.recv(65536)
            if not chunk or len(raw) + len(chunk) > 1024 * 1024:
                raise RunError("operator_response_invalid")
            raw += chunk
    except (OSError, TimeoutError) as exc:
        raise RunError("operator_outcome_unknown") from exc
    finally:
        client.close()
    try:
        value = json.loads(raw.split(b"\n", 1)[0])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunError("operator_response_invalid") from exc
    if not isinstance(value, dict) or value.get("ok") is not True:
        raise RunError("operator_command_failed")
    return value


class Operator:
    def __init__(self) -> None:
        self.sequence = 0

    def admission(self, verb: str, phase: str | None = None) -> dict[str, Any]:
        self.sequence += 1
        request: dict[str, Any] = {
            "schema": "ai-admission-operator-command/v1",
            "verb": verb,
            "request_id": f"q38-nvfp4-{os.getpid()}-{self.sequence}",
        }
        if phase is not None:
            request["phase"] = phase
        value = operator(request, 60.0)
        if value.get("schema") != "ai-admission-operator-response/v1" or value.get("request_id") != request["request_id"]:
            raise RunError("admission_response_invalid")
        return value

    def status(self) -> dict[str, Any]:
        return self.admission("status")

    def advance(self, phase: str) -> None:
        value = self.admission("advance-phase", phase)
        if value.get("phase") != phase:
            raise RunError("admission_transition_not_proven")

    def qwen_status(self) -> dict[str, Any]:
        return operator({"verb": "status", "profile": "qwen"}, 60.0)

    def qwen(self, verb: str) -> None:
        self.sequence += 1
        value = operator({
            "verb": verb,
            "profile": "qwen",
            "idempotency_key": f"q38-nvfp4-{os.getpid()}-{verb}-{self.sequence}",
        })
        expected = "ready" if verb == "start" else "stopped"
        if value.get("profile") != "qwen" or value.get("state") != expected:
            raise RunError(f"qwen_{verb}_not_proven")


def admission_safe(value: dict[str, Any], phase: str) -> bool:
    return (
        value.get("phase") == phase
        and value.get("capacity_mode") == "serialized-99"
        and value.get("restart_pending") is False
        and value.get("reconciliation_required") is False
        and value.get("active_ledger") == []
        and value.get("queue_depth") == 0
        and value.get("active_ledger_digest") == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    )


def qwen_state(value: dict[str, Any]) -> str:
    status = value.get("status")
    state = status.get("state") if isinstance(status, dict) else None
    if state == "ready" and value.get("active_profile") == "qwen" and value.get("global_breaker_open") is False and status.get("breaker_open") is False:
        return "ready"
    if state == "stopped" and value.get("active_profile") is None and value.get("global_breaker_open") is False and status.get("breaker_open") is False:
        return "stopped"
    raise RunError("qwen_state_unknown")


def exact_file(path: Path, size: int) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RunError("candidate_file_missing") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size != size or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o444:
        raise RunError("candidate_file_identity_invalid")


def verify_candidate() -> None:
    if (
        MODEL_DIR.is_symlink()
        or not MODEL_DIR.is_dir()
        or MODEL_DIR.stat().st_uid != 0
        or stat.S_IMODE(MODEL_DIR.stat().st_mode) != 0o555
    ):
        raise RunError("candidate_directory_identity_invalid")
    actual_names = {path.name for path in MODEL_DIR.iterdir()}
    if actual_names != set(CANDIDATE_FILES) | {"SHA256SUMS"}:
        raise RunError("candidate_file_set_invalid")
    for name, size in CANDIDATE_FILES.items():
        exact_file(MODEL_DIR / name, size)
    exact_file(SHA256SUMS, SHA256SUMS.stat().st_size if SHA256SUMS.exists() else -1)
    rows: dict[str, str] = {}
    for line in SHA256SUMS.read_text(encoding="ascii").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or not name or name in rows:
            raise RunError("candidate_checksum_receipt_invalid")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RunError("candidate_checksum_receipt_invalid")
        rows[name] = digest
    if set(rows) != set(CANDIDATE_FILES):
        raise RunError("candidate_checksum_receipt_invalid")
    if rows.get("model.safetensors") != MODEL_SHA256 or rows.get("tokenizer.json") != TOKENIZER_SHA256:
        raise RunError("candidate_checksum_receipt_invalid")


def verify_benchmark_client() -> None:
    try:
        info = CLIENT.lstat()
        raw = CLIENT.read_bytes()
    except OSError as exc:
        raise RunError("benchmark_client_identity_invalid") from exc
    if CLIENT.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise RunError("benchmark_client_identity_invalid")
    if hashlib.sha256(raw).hexdigest() != BENCHMARK_CLIENT_SHA256:
        raise RunError("benchmark_client_identity_invalid")


def verify_image() -> None:
    value = run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}} {{.Architecture}} {{.Os}}"], timeout=30).stdout.decode("ascii").strip()
    if value != f"{IMAGE_ID} amd64 linux":
        raise RunError("freetoken_image_identity_invalid")
    present = run(["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}"], timeout=15).stdout.decode("ascii").strip()
    if present:
        raise RunError("stale_candidate_container_present")


def verify_production_qwen_identity() -> None:
    raw = run(["docker", "inspect", "ai-server-qwen"], timeout=30).stdout
    try:
        values = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunError("production_qwen_identity_invalid") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise RunError("production_qwen_identity_invalid")
    mounts = values[0].get("Mounts")
    expected = [
        mount for mount in mounts if isinstance(mount, dict)
        and mount.get("Type") == "bind"
        and mount.get("Source") == str(CURRENT_MODEL_PATH)
        and mount.get("Destination") == CURRENT_MODEL_DESTINATION
        and mount.get("RW") is False
    ] if isinstance(mounts, list) else []
    if len(expected) != 1:
        raise RunError("production_qwen_model_mount_invalid")


def load_baseline_report(path: Path) -> tuple[dict[str, Any], str]:
    try:
        info = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise RunError("baseline_report_invalid") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or stat.S_IMODE(info.st_mode) != 0o600
        or len(raw) > 4 * 1024 * 1024
    ):
        raise RunError("baseline_report_invalid")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunError("baseline_report_invalid") from exc
    if not isinstance(value, dict):
        raise RunError("baseline_report_invalid")
    coordinates = value.get("coordinates")
    modes = value.get("modes")
    restore = value.get("restore")
    mode = modes.get("current_uncensored_plain") if isinstance(modes, dict) else None
    result = mode.get("result") if isinstance(mode, dict) else None
    if (
        value.get("schema") != "qwen38-uncensored-dflash2-one-shot-result/v1"
        or value.get("outcome") != "passed_restored"
        or value.get("content_recorded") is not False
        or value.get("secrets_recorded") is not False
        or not isinstance(coordinates, dict)
        or coordinates.get("target_sha256") != CURRENT_MODEL_SHA256
        or not isinstance(mode, dict)
        or mode.get("status") != "PASS"
        or not isinstance(result, dict)
        or result.get("warmup_count") != 1
        or result.get("measured_request_count") != 3
        or not isinstance(restore, dict)
        or restore.get("candidate_absent") is not True
        or restore.get("qwen_ready") is not True
        or restore.get("gateway_completions") != 2
        or restore.get("admission_public") is not True
    ):
        raise RunError("baseline_report_invalid")
    return result, hashlib.sha256(raw).hexdigest()


def gpu_rows() -> list[float]:
    raw = run([
        "nvidia-smi",
        "--query-gpu=memory.used,memory.free,power.draw,power.limit,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], timeout=2).stdout.decode("ascii").strip()
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 6:
        raise RunError("gpu_sample_invalid")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise RunError("gpu_sample_invalid") from exc
    if any(not math.isfinite(item) for item in values):
        raise RunError("gpu_sample_invalid")
    return values


def gpu_pids() -> set[int]:
    raw = run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], timeout=15).stdout.decode("ascii")
    try:
        return {int(line.strip()) for line in raw.splitlines() if line.strip()}
    except ValueError as exc:
        raise RunError("gpu_pid_sample_invalid") from exc


def wait_gpu_idle(timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if not gpu_pids():
            return
        if time.monotonic() >= deadline:
            raise RunError("gpu_not_idle")
        time.sleep(1.0)


class Sampler:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.samples: list[list[float]] = []
        self.error = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                self.samples.append(gpu_rows())
            except RunError:
                self.error = True
            self.stop_event.wait(max(0.0, 0.25 - (time.monotonic() - started)))

    def __enter__(self) -> "Sampler":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        self.thread.join(3.0)

    def report(self) -> dict[str, Any]:
        if self.error or not self.samples:
            raise RunError("gpu_sampling_failed")
        return {
            "sample_count": len(self.samples),
            "max_vram_used_mib": round(max(row[0] for row in self.samples), 3),
            "min_vram_free_mib": round(min(row[1] for row in self.samples), 3),
            "max_power_w": round(max(row[2] for row in self.samples), 3),
            "power_limit_w": round(max(row[3] for row in self.samples), 3),
            "max_temperature_c": round(max(row[4] for row in self.samples), 3),
            "max_gpu_utilization_percent": round(max(row[5] for row in self.samples), 3),
        }


def run_client(container: str, python: str, base_url: str, token_file: str | None, mode_id: str, timeout: float) -> dict[str, Any]:
    argv = ["docker", "exec", "-i", container, python, "-B", "-", "--base-url", base_url, "--model", MODEL_ID, "--mode-id", mode_id, "--timeout", str(timeout)]
    if token_file is not None:
        argv.extend(["--token-file", token_file])
    # benchmark_client.py never prints response text: on failure its stderr contains
    # only a bounded BenchmarkError code/traceback. Preserve that diagnostic so a
    # failed live run has a discriminating reason without retaining model content.
    value = run(
        argv,
        timeout=timeout + 30,
        input_bytes=CLIENT.read_bytes(),
        diagnostic_output=True,
    )
    try:
        result = json.loads(value.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunError("benchmark_result_invalid") from exc
    if not isinstance(result, dict) or result.get("measured_request_count") != 3 or result.get("warmup_count") != 1:
        raise RunError("benchmark_result_invalid")
    return result


RESTORE_SCRIPT = b"""import json,sys,urllib.request
ordinal=int(sys.argv[1]); marker='QWEN_RESTORE_OK_'+str(ordinal)
token=open('<redacted-secret-path>','r',encoding='ascii').read().strip()
body=json.dumps({'model':'qwen3.8-27b-uncensored','messages':[{'role':'user','content':'Return exactly: '+marker}],'temperature':0,'max_tokens':64,'stream':False},separators=(',',':')).encode()
request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=body,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'},method='POST')
with urllib.request.urlopen(request,timeout=180) as response: value=json.load(response)
choice=value['choices'][0]; content=choice['message']['content']
if choice.get('finish_reason')!='stop' or not isinstance(content,str) or not content.strip() or content.count(marker)!=1: raise SystemExit(22)
print('QWEN_COMPLETION_OK')
"""


def restore_completion(ordinal: int) -> None:
    value = run(["docker", "exec", "-i", "ai-server-gateway", "/usr/local/bin/python3", "-B", "-", str(ordinal)], timeout=240, input_bytes=RESTORE_SCRIPT)
    if value.stdout.decode("ascii").strip() != "QWEN_COMPLETION_OK":
        raise RunError("gateway_completion_failed")


def start_freetoken() -> None:
    argv = [
        "docker", "run", "-d", "--name", CONTAINER, "--restart", "no",
        "--gpus", "all", "--network", "none", "--read-only", "--user", "65532:65532",
        "--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--pids-limit", "1024",
        "--memory", "29g", "--memory-swap", "36g", "--shm-size", "1g",
        "--tmpfs", "/tmp:rw,nosuid,size=4g", "--tmpfs", "/cache:rw,nosuid,size=4g",
        "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1", "--env", "HOME=/cache", "--env", "XDG_CACHE_HOME=/cache",
        "--mount", f"type=bind,src={MODEL_DIR},dst=/models/target,readonly",
        IMAGE,
        "ft", "serve", "--model-path", "/models/target", "--served-model-name", MODEL_ID,
        "--host", "127.0.0.1", "--port", "1919", "--dtype", "bfloat16",
        "--max-running-requests", "1", "--max-seq-len-override", "4096", "--max-output-tokens", "128",
        "--memory-ratio", "0.90", "--cuda-graph-max-bs", "1", "--num-tokenizer", "0",
        "--max-prefill-length", "64", "--num-tokens", "4096", "--attention-backend", "triton",
        "--cache-type", "naive", "--sampling-defaults", "none", "--reasoning-parser", "off",
        "--nvfp4-backend", "triton", "--expert-load", "serial", "--cors-origins", "",
    ]
    run(argv, timeout=60, diagnostic_output=True)


def wait_freetoken_ready(timeout: float = 900.0) -> float:
    started = time.monotonic()
    deadline = started + timeout
    probe = "import urllib.request; urllib.request.urlopen('http://127.0.0.1:1919/health',timeout=5).read(1)"
    while True:
        state_raw = run(["docker", "inspect", CONTAINER, "--format", "{{json .State}}"], timeout=15).stdout
        try:
            state = json.loads(state_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunError("candidate_state_invalid") from exc
        if not state.get("Running"):
            raise RunError("candidate_exit_before_ready", {"exit_code": state.get("ExitCode"), "oom_killed": state.get("OOMKilled")})
        probe_result = run(["docker", "exec", CONTAINER, "/opt/python/bin/python3.12", "-B", "-c", probe], timeout=10, check=False)
        if probe_result.returncode == 0:
            return round(time.monotonic() - started, 3)
        if time.monotonic() >= deadline:
            raise RunError("candidate_readiness_timeout")
        time.sleep(5.0)


def capture_freetoken_failure(*, include_logs: bool = True) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {}
    try:
        state_result = run(
            ["docker", "inspect", CONTAINER, "--format", "{{json .State}}"],
            timeout=15,
            check=False,
        )
    except RunError as exc:
        diagnostic["state_query_error"] = {"code": exc.code, **exc.details}
    else:
        if state_result.returncode != 0:
            diagnostic["state_query_error"] = {"code": "command_failed", **command_diagnostic(state_result)}
        else:
            try:
                state = json.loads(state_result.stdout.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                diagnostic["state_query_error"] = {
                    "code": "candidate_state_invalid",
                    **command_diagnostic(state_result),
                }
            else:
                if isinstance(state, dict):
                    diagnostic["state"] = {
                        "error": sanitized_tail(str(state.get("Error") or "")),
                        "exit_code": state.get("ExitCode"),
                        "oom_killed": state.get("OOMKilled"),
                    }
                else:
                    diagnostic["state_query_error"] = {"code": "candidate_state_invalid"}

    if not include_logs:
        diagnostic["logs_omitted"] = "request_execution_started"
        return diagnostic

    try:
        logs_result = run(
            ["docker", "logs", "--tail", str(DIAGNOSTIC_LOG_LINES), CONTAINER],
            timeout=30,
            check=False,
        )
    except RunError as exc:
        diagnostic["logs_error"] = {"code": exc.code, **exc.details}
    else:
        diagnostic["logs"] = command_diagnostic(logs_result)
        diagnostic["logs"]["tail_line_limit"] = DIAGNOSTIC_LOG_LINES
    return diagnostic


def stop_freetoken() -> None:
    present = run(["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}"], timeout=15).stdout.decode("ascii").strip()
    if not present:
        return
    running = run(["docker", "inspect", CONTAINER, "--format", "{{.State.Running}}"], timeout=15).stdout.decode("ascii").strip()
    if running == "true":
        stopped = run(["docker", "stop", "--time", "60", CONTAINER], timeout=75, check=False)
        if stopped.returncode != 0:
            run(["docker", "rm", "-f", CONTAINER], timeout=30)
            return
    run(["docker", "rm", CONTAINER], timeout=30)


def kernel_fault_free(since_epoch: int) -> bool:
    value = run(["journalctl", "-k", f"--since=@{since_epoch}", "--no-pager", "--output=cat"], timeout=30, check=False)
    text = (value.stdout + value.stderr).decode("utf-8", "replace").lower()
    return not any(marker in text for marker in ("nvrm: xid", "cuda out of memory", "oom-kill", "out of memory: killed process"))


def write_report(path: Path, report: dict[str, Any]) -> None:
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-report", required=True, type=Path)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("root_required")
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise SystemExit("output_path_invalid")

    report: dict[str, Any] = {
        "schema": "qwen38-nvfp4-server-a-result/v1",
        "started_epoch": int(time.time()),
        "content_recorded": False,
        "secrets_recorded": False,
        "coordinates": {
            "current_gguf_sha256": CURRENT_MODEL_SHA256,
            "nvfp4_repository": "preetpatel/Qwen3.8-27B-Uncensored-NVFP4",
            "nvfp4_revision": "37b5130a2d2a1f7d4456ab3f8d05d0b2a45ea350",
            "nvfp4_model_sha256": MODEL_SHA256,
            "freetoken_commit": "9db1a39455a3fb107f3db83e381d10ceadfe5d99",
            "freetoken_image_id": IMAGE_ID,
        },
        "modes": {
            "current_llama_cpp": {"status": "PENDING"},
            "standard_nvfp4": {"status": "BLOCKED", "reason": "no_existing_compatible_standard_backend_image"},
            "nvfp4_dflash2": {"status": "BLOCKED", "reason": "exact_target_format_runtime_and_vram_incompatible"},
            "nvfp4_freetoken": {"status": "PENDING"},
        },
        "restore": {"candidate_absent": False, "gpu_idle": False, "qwen_ready": False, "gateway_completions": 0, "admission_public": False},
        "outcome": "running",
        "error_code": None,
        "cleanup_error_code": None,
    }
    operator_client = Operator()
    admission_mutated = False
    candidate_mutation_attempted = False
    candidate_request_attempted = False
    error: RunError | None = None
    cleanup_error: RunError | None = None
    since_epoch = int(time.time())

    descriptor = os.open(LOCK, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunError("host_lock_busy") from exc
        try:
            admission = operator_client.status()
            if not admission_safe(admission, "public"):
                raise RunError("admission_preflight_not_public_empty")
            verify_candidate()
            verify_benchmark_client()
            verify_image()
            verify_production_qwen_identity()
            baseline, baseline_report_sha256 = load_baseline_report(args.baseline_report)
            initial_qwen = qwen_state(operator_client.qwen_status())
            power_limit = gpu_rows()[3]
            if abs(power_limit - 300.0) > 0.05:
                raise RunError("power_limit_not_300w")
            if initial_qwen == "stopped":
                wait_gpu_idle()

            report["modes"]["current_llama_cpp"] = {
                "status": "PASS",
                "result": baseline,
                "source_report_sha256": baseline_report_sha256,
            }

            operator_client.advance("closed")
            admission_mutated = True
            if not admission_safe(operator_client.status(), "closed"):
                raise RunError("admission_close_not_proven")
            if qwen_state(operator_client.qwen_status()) == "ready":
                operator_client.qwen("stop")
            if qwen_state(operator_client.qwen_status()) != "stopped":
                raise RunError("qwen_stop_not_proven")
            wait_gpu_idle()

            with Sampler() as sampler:
                candidate_mutation_attempted = True
                start_freetoken()
                load_seconds = wait_freetoken_ready()
                candidate_request_attempted = True
                candidate = run_client(
                    CONTAINER, "/opt/python/bin/python3.12", "http://127.0.0.1:1919",
                    None, "nvfp4-freetoken", 600.0,
                )
            candidate["load_seconds"] = load_seconds
            candidate["resources"] = sampler.report()
            report["modes"]["nvfp4_freetoken"] = {"status": "PASS", "result": candidate}
        except RunError as exc:
            error = exc
            report["error_code"] = exc.code
            candidate_diagnostics = (
                capture_freetoken_failure(include_logs=not candidate_request_attempted)
                if candidate_mutation_attempted else None
            )
            if report["modes"]["current_llama_cpp"]["status"] == "PENDING":
                report["modes"]["current_llama_cpp"] = {"status": "FAIL", "reason": exc.code, **exc.details}
            elif report["modes"]["nvfp4_freetoken"]["status"] == "PENDING":
                report["modes"]["nvfp4_freetoken"] = {
                    "status": "FAIL", "reason": exc.code, **exc.details,
                    "candidate_diagnostics": candidate_diagnostics,
                }
        except Exception as exc:
            error = RunError("unexpected_runner_failure", {"exception_type": type(exc).__name__})
            report["error_code"] = error.code
            candidate_diagnostics = (
                capture_freetoken_failure(include_logs=not candidate_request_attempted)
                if candidate_mutation_attempted else None
            )
            if report["modes"]["current_llama_cpp"]["status"] == "PENDING":
                report["modes"]["current_llama_cpp"] = {"status": "FAIL", "reason": error.code, **error.details}
            elif report["modes"]["nvfp4_freetoken"]["status"] == "PENDING":
                report["modes"]["nvfp4_freetoken"] = {
                    "status": "FAIL", "reason": error.code, **error.details,
                    "candidate_diagnostics": candidate_diagnostics,
                }
        finally:
            try:
                if candidate_mutation_attempted:
                    stop_freetoken()
                remaining = run([
                    "docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}",
                ], timeout=15).stdout.decode("ascii").strip()
                if remaining:
                    raise RunError("candidate_cleanup_not_proven")
                report["restore"]["candidate_absent"] = True
                state = qwen_state(operator_client.qwen_status())
                if state == "stopped":
                    wait_gpu_idle()
                    report["restore"]["gpu_idle"] = True
                    operator_client.qwen("start")
                else:
                    report["restore"]["gpu_idle"] = False
                if qwen_state(operator_client.qwen_status()) != "ready":
                    raise RunError("qwen_restore_not_proven")
                verify_production_qwen_identity()
                report["restore"]["qwen_ready"] = True
                if admission_mutated:
                    operator_client.advance("smoke")
                    if not admission_safe(operator_client.status(), "smoke"):
                        raise RunError("admission_smoke_not_proven")
                restore_completion(1)
                report["restore"]["gateway_completions"] = 1
                restore_completion(2)
                report["restore"]["gateway_completions"] = 2
                if operator_client.status().get("phase") != "public":
                    operator_client.advance("public")
                if not admission_safe(operator_client.status(), "public"):
                    raise RunError("admission_public_not_proven")
                report["restore"]["admission_public"] = True
            except RunError as exc:
                cleanup_error = exc
                report["cleanup_error_code"] = exc.code
    finally:
        os.close(descriptor)

    report["kernel_fault_free"] = kernel_fault_free(since_epoch)
    report["ended_epoch"] = int(time.time())
    if cleanup_error is not None:
        report["outcome"] = "restore_unknown"
    elif error is not None:
        report["outcome"] = "failed_restored"
    elif report["kernel_fault_free"] is not True:
        report["outcome"] = "failed_restored"
        report["error_code"] = "kernel_fault_detected"
    else:
        report["outcome"] = "passed_restored"
    write_report(args.output, report)
    print(json.dumps({"outcome": report["outcome"], "report": str(args.output)}, sort_keys=True))
    return 0 if report["outcome"] == "passed_restored" else 1


if __name__ == "__main__":
    raise SystemExit(main())
