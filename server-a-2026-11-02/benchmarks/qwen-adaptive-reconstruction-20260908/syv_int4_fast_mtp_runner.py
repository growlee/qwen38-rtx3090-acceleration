#!/usr/bin/env python3
"""Build and benchmark one exact-target int4 fast-MTP Leminkozey candidate."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import shutil
import stat
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import server_a_runner as common
import syv_acceleration_runner as base


SOURCE_REPOSITORY = "leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound"
SOURCE_REVISION = "f958acb153eb8c7bbd25638149c8c75185559226"
SOURCE_LEAF = Path("/srv/ai-server/staging/experiments/qwen38-syv-uncensored/leminkozey") / SOURCE_REVISION
SOURCE_MODEL = SOURCE_LEAF / "model"
SOURCE_RECEIPT_SHA256 = "a1a385314549147c266eb404ab8d9255fc224b126040973797a95d34adfa621a"

SYV_REPOSITORY = "syv-ai/qwen38-27b-rtx3090"
SYV_REVISION = "69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c"
IMAGE = base.IMAGE
VARIANT_ID = "syv69ba4d0-exact-leminkozey-gptq-int4-mtp-adaptive-damp-vocab40960-k4"
DRAFT_TOKENS = 4
RETAIN_TRANSIENT = False
FOLLOW_ON_PEAK_BOUND_BYTES = 0
RESUME_CURRENT_K4 = False
RESUME_TREE_VALIDATED = False
RESUME_RETAINED_BYTES = 0
RESUME_TARGET_LEAF_REUSABLE = False
BUILD_OWNER_LABEL = "ai-server.experiment-owner"
BUILD_OWNER_TOKEN = hashlib.sha256(
    f"{os.getpid()}:{time.time_ns()}:{Path(__file__).resolve()}".encode("utf-8")
).hexdigest()
ATTEMPTED_BUILD_STEPS: set[str] = set()
TRANSIENT_STAGING_BASE = Path("/srv/ai-server/staging/experiments/qwen38-syv-uncensored")
WORK_ROOT = TRANSIENT_STAGING_BASE / "leminkozey-int4-fast-mtp-adaptive-work-r1" / SOURCE_REVISION
TARGET_LEAF = TRANSIENT_STAGING_BASE / "leminkozey-int4-fast-mtp-adaptive-r1" / SOURCE_REVISION
TARGET_MODEL = TARGET_LEAF / "model"
TARGET_RECEIPT = TARGET_LEAF / "COMPLETE.receipt"
BUILD_CANDIDATE_MODEL = WORK_ROOT / "candidate"
RECEIPT_SCHEMA = "qwen38-syv-exact-int4-fast-mtp-adaptive-candidate/v1"
CONTAINER = "ai-exp-syv-leminkozey-exact-int4-fast-mtp-adaptive"

BASELINE_REPORT = Path(
    "/srv/ai-server/benchmarks/experiments/syv-leminkozey-fast-mtp-int8-vocab40960-live03-c987880e.json"
)
BASELINE_REPORT_SHA256 = "3553ffd24706edeb1b388aa75855a288b444e0ee7f3fe401cccab53448a8d645"
BASELINE_DECODE_TPS = 94.782545
BASELINE_ACCEPTANCE = 0.363328
BASELINE_END_TO_END_TPS = 93.410569
BASELINE_TTFT_SECONDS = 0.139537
BASELINE_RESULT_KEYS = ("modes", "mtp_fast", "result")
K4_BASELINE_REPORT = BASELINE_REPORT
K4_BASELINE_REPORT_SHA256 = BASELINE_REPORT_SHA256
K4_BASELINE_DECODE_TPS = BASELINE_DECODE_TPS
K4_BASELINE_ACCEPTANCE = BASELINE_ACCEPTANCE
K4_BASELINE_END_TO_END_TPS = BASELINE_END_TO_END_TPS
K4_BASELINE_TTFT_SECONDS = BASELINE_TTFT_SECONDS

K3_BASELINE_REPORT = Path(
    "/srv/ai-server/benchmarks/experiments/"
    "syv-leminkozey-int4-fast-mtp-adaptive-live09r5-990f93f2.json"
)
K3_BASELINE_REPORT_SHA256 = "454ecd10b26edbfe26cd5e3cc73f5b09ae80b790c604986c858dbfe4892c64f6"
K3_BASELINE_DECODE_TPS = 103.615801
K3_BASELINE_ACCEPTANCE = 0.339793
K3_BASELINE_END_TO_END_TPS = 102.115791
K3_BASELINE_TTFT_SECONDS = 0.134937

RESERVE_BYTES = 150 * 1024**3
HARD_PEAK_CAP_BYTES = 20 * 1024**3
PREPARATION_BUDGET_SECONDS = 6 * 60 * 60
LIVE_BUDGET_SECONDS = 30 * 60
RESTORE_RESERVE_SECONDS = 90
CALIBRATION_RECORDS = 128
CALIBRATION_OUTPUT_TOKENS = 1024
DRAFT_VOCAB_SIZE = 40960
BUILDER_PATH = "/app/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
CDI_GPU_DEVICE = "nvidia.com/gpu=0"
NVIDIA_VISIBLE_DEVICES = "NVIDIA_VISIBLE_DEVICES=void"
NVIDIA_DRIVER_CAPABILITIES = "NVIDIA_DRIVER_CAPABILITIES=compute,utility"

CAL_MODEL = WORK_ROOT / "calibration-model"
CAL_DATA = WORK_ROOT / "calibration"
RUN_DIR = WORK_ROOT / "runs" / "eval"
LM4_DIR = WORK_ROOT / "lm4"
MTP_WORK_DIR = WORK_ROOT / "mtp-work"
INGRESS_DIR = WORK_ROOT / "ingress"
HESSIANS = RUN_DIR / "mtp_hessians.pt"
HESSIAN_DIAGNOSTICS = RUN_DIR / "gptq_hessian_diagnostics.jsonl"


def configure_draft_tokens(
    draft_tokens: int,
    *,
    retain_transient: bool = False,
    follow_on_peak_bound_bytes: int = 0,
    resume_current_k4: bool = False,
) -> None:
    """Select one supported runtime k without changing the sealed k4 default."""
    if draft_tokens not in {3, 4}:
        raise ValueError("unsupported_draft_tokens")
    if (
        not isinstance(follow_on_peak_bound_bytes, int)
        or isinstance(follow_on_peak_bound_bytes, bool)
        or follow_on_peak_bound_bytes < 0
    ):
        raise ValueError("invalid_follow_on_peak_bound_bytes")
    if not isinstance(resume_current_k4, bool):
        raise ValueError("invalid_resume_current_k4")
    if resume_current_k4 and draft_tokens != 4:
        raise ValueError("resume_current_k4_requires_k4")
    global DRAFT_TOKENS, RETAIN_TRANSIENT, FOLLOW_ON_PEAK_BOUND_BYTES, VARIANT_ID
    global WORK_ROOT, TARGET_LEAF, TARGET_MODEL, TARGET_RECEIPT, BUILD_CANDIDATE_MODEL, CONTAINER
    global CAL_MODEL, CAL_DATA, RUN_DIR, LM4_DIR, MTP_WORK_DIR, INGRESS_DIR, HESSIANS, HESSIAN_DIAGNOSTICS
    global BASELINE_REPORT, BASELINE_REPORT_SHA256, BASELINE_DECODE_TPS, BASELINE_ACCEPTANCE
    global BASELINE_END_TO_END_TPS, BASELINE_TTFT_SECONDS, BASELINE_RESULT_KEYS
    global RESUME_CURRENT_K4, RESUME_TREE_VALIDATED, RESUME_RETAINED_BYTES
    global RESUME_TARGET_LEAF_REUSABLE

    DRAFT_TOKENS = draft_tokens
    RETAIN_TRANSIENT = retain_transient
    FOLLOW_ON_PEAK_BOUND_BYTES = follow_on_peak_bound_bytes
    RESUME_CURRENT_K4 = resume_current_k4
    RESUME_TREE_VALIDATED = False
    RESUME_RETAINED_BYTES = 0
    RESUME_TARGET_LEAF_REUSABLE = False
    VARIANT_ID = (
        "syv69ba4d0-exact-leminkozey-gptq-int4-mtp-adaptive-damp-"
        f"vocab40960-k{draft_tokens}"
    )
    if draft_tokens == 3:
        slug = "leminkozey-int4-fast-mtp-adaptive-k3-r1"
        work_slug = "leminkozey-int4-fast-mtp-adaptive-k3-work-r1"
        CONTAINER = "ai-exp-syv-leminkozey-exact-int4-fast-mtp-adaptive-k3"
        BASELINE_REPORT = K3_BASELINE_REPORT
        BASELINE_REPORT_SHA256 = K3_BASELINE_REPORT_SHA256
        BASELINE_DECODE_TPS = K3_BASELINE_DECODE_TPS
        BASELINE_ACCEPTANCE = K3_BASELINE_ACCEPTANCE
        BASELINE_END_TO_END_TPS = K3_BASELINE_END_TO_END_TPS
        BASELINE_TTFT_SECONDS = K3_BASELINE_TTFT_SECONDS
        BASELINE_RESULT_KEYS = ("benchmark", "result")
    else:
        slug = "leminkozey-int4-fast-mtp-adaptive-r1"
        work_slug = "leminkozey-int4-fast-mtp-adaptive-work-r1"
        CONTAINER = "ai-exp-syv-leminkozey-exact-int4-fast-mtp-adaptive"
        BASELINE_REPORT = K4_BASELINE_REPORT
        BASELINE_REPORT_SHA256 = K4_BASELINE_REPORT_SHA256
        BASELINE_DECODE_TPS = K4_BASELINE_DECODE_TPS
        BASELINE_ACCEPTANCE = K4_BASELINE_ACCEPTANCE
        BASELINE_END_TO_END_TPS = K4_BASELINE_END_TO_END_TPS
        BASELINE_TTFT_SECONDS = K4_BASELINE_TTFT_SECONDS
        BASELINE_RESULT_KEYS = ("modes", "mtp_fast", "result")
    WORK_ROOT = TRANSIENT_STAGING_BASE / work_slug / SOURCE_REVISION
    TARGET_LEAF = TRANSIENT_STAGING_BASE / slug / SOURCE_REVISION
    TARGET_MODEL = TARGET_LEAF / "model"
    TARGET_RECEIPT = TARGET_LEAF / "COMPLETE.receipt"
    BUILD_CANDIDATE_MODEL = WORK_ROOT / "candidate"
    CAL_MODEL = WORK_ROOT / "calibration-model"
    CAL_DATA = WORK_ROOT / "calibration"
    RUN_DIR = WORK_ROOT / "runs" / "eval"
    LM4_DIR = WORK_ROOT / "lm4"
    MTP_WORK_DIR = WORK_ROOT / "mtp-work"
    INGRESS_DIR = WORK_ROOT / "ingress"
    HESSIANS = RUN_DIR / "mtp_hessians.pt"
    HESSIAN_DIAGNOSTICS = RUN_DIR / "gptq_hessian_diagnostics.jsonl"

PATCHED_GPTQ_UTILS = Path(__file__).resolve().with_name("syv_gptq_utils_adaptive.py")
PATCHED_REQUANT_MTP = Path(__file__).resolve().with_name("syv_requant_mtp_gptq_adaptive.py")
PATCHED_RECIPE_FILES = {
    "/app/drafter/gptq_utils.py": (
        PATCHED_GPTQ_UTILS,
        "936ddad9d726b75e0a170368a2c8620e356d65285685792665a13b56e336fea1",
    ),
    "/app/drafter/requant_mtp_gptq.py": (
        PATCHED_REQUANT_MTP,
        "51306be000fd3c0563b8fec6c5a2859c1b91f60dcc6ca44dcf6e2476566e7d76",
    ),
}
MTP_MODULES = (
    "mtp.fc",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
)

GENERIC_IDS_CANONICAL_SHA256 = "16048e758c7df4b5d5b4285c0d8f23f34f825c20d393350b60db80e2a3260dc6"
FAST_OVERLAY_REPOSITORY = "syvai/qwen3.8-27b-3090-fast-variant"
FAST_OVERLAY_FILE_SHA256 = {
    "config.json": "5a3e7312dee3a6c5909beb854b05ac818be836683730e4facba81f547b77fd62",
    "model-00007-of-00007.safetensors": "ac4511ddfbdcd8399dffee570c31708c13f974a18e46d3c337ca652a46a1a437",
    "model.safetensors.index.json": "0821f46e4a9888262e8f63d5a3c39e3be93e62fa3acd95febeb9a2c9bd9131a1",
    "model_extra_tensors.safetensors": "3643128012845562d066540990eceb012e6dc9d1ba794c668d774a91edcd9dbb",
}


def verify_fast_overlay_absent(candidate_file_sha256: dict[str, str]) -> list[str]:
    matching_files = sorted(
        name
        for name, expected_sha256 in FAST_OVERLAY_FILE_SHA256.items()
        if candidate_file_sha256.get(name) == expected_sha256
    )
    if len(matching_files) == len(FAST_OVERLAY_FILE_SHA256):
        raise common.RunError(
            "dbirks_fast_overlay_identity_detected",
            {"files": matching_files},
        )
    return matching_files


@dataclass(frozen=True)
class BackupBlob:
    source_name: str
    installed_name: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{SOURCE_REPOSITORY}/resolve/{SOURCE_REVISION}/{self.source_name}?download=true"

    @property
    def ingress(self) -> Path:
        return INGRESS_DIR / self.installed_name


BACKUP_BLOBS = (
    BackupBlob(
        "model-00006-of-00007.safetensors.bak_embed",
        "model-00006-of-00007.safetensors.bak_embed",
        2_542_807_272,
        "55a14ee79d3e5a65a8731d89426f4df477e8bdc7daa7976d41254d8afb9432f0",
    ),
    BackupBlob(
        "model-00007-of-00007.safetensors.bak",
        "model-00007-of-00007.safetensors.bak",
        2_542_796_896,
        "6866cf8adcccc4cc6a00e74bc025f1a774fb52103b70f2674d0288272951a733",
    ),
)

RESUME_PARTIAL_SIZE = 471_363_020
RESUME_PARTIAL_SHA256 = "ba07a9c6652763b0d5da663a18c32fc2412da4adf8f45b891706fa54ea86097d"

MTP_LINEAR_INPUTS = (10_240, 17_408, 5_120, 5_120, 5_120, 5_120, 5_120, 5_120)
INT4_TENSOR_DIMS = {
    "lm_head": (248_320, 5_120),
    "mtp.fc": (5_120, 10_240),
    "mtp.layers.0.mlp.down_proj": (5_120, 17_408),
    "mtp.layers.0.mlp.gate_proj": (17_408, 5_120),
    "mtp.layers.0.mlp.up_proj": (17_408, 5_120),
    "mtp.layers.0.self_attn.q_proj": (12_288, 5_120),
    "mtp.layers.0.self_attn.k_proj": (1_024, 5_120),
    "mtp.layers.0.self_attn.v_proj": (1_024, 5_120),
    "mtp.layers.0.self_attn.o_proj": (5_120, 6_144),
    "mtp.draft_lm_head": (40_960, 5_120),
}


def storage_components() -> dict[str, int]:
    """Conservative physical peak; retained intermediates are never discounted."""
    max_rows = CALIBRATION_RECORDS * (512 + CALIBRATION_OUTPUT_TOKENS)
    return {
        # Sequential acquisition includes either both retained blobs or the
        # first retained blob plus the second .partial, never a third copy.
        "hf_ingress_including_sequential_partial": sum(blob.size for blob in BACKUP_BLOBS),
        "calibration_prompts_and_generated_jsonl": 128 * 1024**2,
        "calibration_hidden_npy": max_rows * 5_120 * 2 + 4096,
        "calibration_tokens_seqs": max_rows * 4 + 16 * 1024**2,
        "mtp_hessians": sum(width * width * 4 for width in MTP_LINEAR_INPUTS) + 64 * 1024**2,
        "gptq_lm_head_output": 1024**3,
        "distinct_raw_extra": 900 * 1024**2,
        "int8_mtp_output_and_backups": 1536 * 1024**2,
        "draft_head_output_and_backup": 1280 * 1024**2,
        "final_int4_extra": 512 * 1024**2,
        "patched_recipe_and_hessian_diagnostics": 1024**2,
        # Retained writable trees and their no-overwrite finalization outputs
        # can coexist; hardlinks are excluded, distinct bytes are not.
        "mutable_worktrees_and_finalization_overlap": 4352 * 1024**2,
        "metadata_logs_and_allocator_slack": 512 * 1024**2,
    }


PINNED_PROMPTS = (
    "Write a complete Python module implementing an LRU cache with TTL expiry, thread safety, and type hints. Include a docstring for every method, a small usage example, and unit tests using pytest. Explain your design decisions at the end.",
    "Explain how a modern SSD's flash translation layer works: mapping tables, garbage collection, wear leveling, write amplification, and TRIM. Use headers and be thorough but precise.",
    "Skriv en grundig artikel på dansk om, hvordan man planlægger og gennemfører en to-ugers cykeltur langs den danske vestkyst. Inkluder pakkeliste, ruteforslag pr. dag, budget og sikkerhedsråd.",
    "Refactor this code for readability and performance, then explain each change:\n\n```python\ndef f(d):\n    r = []\n    for k in d.keys():\n        if d[k] != None:\n            if type(d[k]) == list:\n                for i in range(len(d[k])):\n                    if d[k][i] > 0:\n                        r.append((k, d[k][i]))\n            else:\n                if d[k] > 0:\n                    r.append((k, d[k]))\n    s = sorted(r, key=lambda x: x[1], reverse=True)\n    out = {}\n    for k, v in s:\n        if k in out:\n            out[k] = out[k] + [v]\n        else:\n            out[k] = [v]\n    return out\n```",
    "A train leaves city A at 9:00 traveling at 80 km/h toward city B, 420 km away. A second train leaves B at 10:30 toward A at 100 km/h. When and where do they meet? Generalize the formula and check it.",
    "Produce a JSON document describing a fictional e-commerce order system schema: 6 tables with columns, types, constraints, indexes and relations, followed by PostgreSQL DDL and three analytical queries.",
    "Write a detailed technical blog post comparing Rust ownership with C++ RAII and Go garbage collection, with code samples and a section on when each approach wins.",
    "Summarize the following text and then list the ten most important facts as bullet points:\n\nThe Danish energy island project in the North Sea was conceived as a hub that collects power from surrounding offshore wind farms and distributes it to several countries. Initially planned with a capacity of 3 GW and later 10 GW, the artificial island would sit roughly 80 km off the coast of Jutland.\n\nIn 2023 the tender was postponed after cost estimates rose sharply, and the government asked for a redesign that could reduce the state's financial exposure. Alternatives discussed included platform-based solutions instead of a reclaimed island, phased construction, and closer integration with Belgian and German grids. Critics argued that delays put Denmark's 2030 climate targets at risk, while proponents said a cheaper design would still deliver the same electricity.\n\nMeanwhile, the Baltic Sea energy island around Bornholm proceeded on a different model, using the existing island as the hub, with connections to Zealand and Germany. Both projects depend on new HVDC converter technology and on the availability of specialised installation vessels, whose booking schedules have become a bottleneck across Europe.",
)

PREPARATION_PEAK_BOUND_BYTES = sum(storage_components().values())
if PREPARATION_PEAK_BOUND_BYTES > HARD_PEAK_CAP_BYTES:
    raise RuntimeError("preparation_peak_bound_exceeds_20_gib")

PREPARE_SCRIPTS = (
    "/app/drafter/gen_data.py",
    "/app/drafter/capture.py",
    "/app/drafter/train_mtp.py",
    "/app/drafter/gptq_lm_head.py",
    "/app/drafter/gptq_utils.py",
    "/app/prepare/quant_mtp.py",
    "/app/prepare/build_draft_vocab.py",
    "/app/drafter/requant_mtp_gptq.py",
)
BUILD_STEPS = (
    "preflight-tokenizer", "seed", "gpu-preflight", "gen-data", "capture", "hessians", "gptq-lm-head",
    "quant-mtp-int8", "draft-vocab", "requant-mtp-int4", "verify",
)


def preparation_commands(captured: int) -> dict[str, list[str]]:
    if captured <= 20_000:
        raise ValueError("captured_rows_too_small")
    calib_rows = min(300_000, captured - 20_000)
    if calib_rows < 65_536:
        raise ValueError("lm_head_calibration_rows_below_minimum")
    return {
        "gen_data": [
            "/app/venv/bin/python", "-B", "/app/drafter/gen_data.py", "--limit", "128", "--chunk", "16",
        ],
        "capture": ["/app/venv/bin/python", "-B", "/app/drafter/capture.py", "--limit", "128"],
        "hessians": [
            "/app/venv/bin/python", "-B", "/app/drafter/train_mtp.py", "--model", "/model",
            "--data", "/app/drafter/data", "--out", "/work/run", "--eval-only", "1",
            "--val-frac", "1", "--max-seqs", "128", "--depths", "2",
            "--draft-ids", "/model/draft_vocab_ids.json", "--dump-hessians", "/work/run/mtp_hessians.pt",
        ],
        "gptq_lm_head": [
            "/app/venv/bin/python", "-B", "/app/drafter/gptq_lm_head.py",
            "/work/calibration-model", "/work/lm4",
            "--bits", "4", "--calib-rows", str(calib_rows),
        ],
        "quant_mtp_int8": [
            "/app/venv/bin/python", "-B", "/app/prepare/quant_mtp.py", "/work/mtp-work", "--bits", "8",
        ],
        "draft_vocab": [
            "/app/venv/bin/python", "-B", "/app/prepare/build_draft_vocab.py", "/work/mtp-work",
            "--ids", "/app/prepare/draft_vocab_ids.json",
        ],
        "requant_mtp_int4": [
            "/app/venv/bin/python", "-B", "/app/drafter/requant_mtp_gptq.py",
            "/work/mtp-work", "/work/candidate", "/work/runs/eval/mtp_hessians.pt",
            "--bits", "4", "--orig", "/work/calibration-model",
        ],
    }


def preparation_timeout(deadline: float, cap: float) -> float:
    return base.bounded_timeout(deadline, cap, "preparation_deadline_exhausted")


def sha256_file(path: Path, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            if deadline is not None:
                preparation_timeout(deadline, 1)
            digest.update(chunk)
    return digest.hexdigest()


def available_bytes(path: Path = Path("/srv/ai-server")) -> int:
    value = os.statvfs(path)
    return value.f_bavail * value.f_frsize


class StorageGuard:
    def __init__(self) -> None:
        self.available_before = available_bytes()
        self.samples: list[dict[str, Any]] = []
        self.remaining_preparation_peak_bound_bytes = PREPARATION_PEAK_BOUND_BYTES - RESUME_RETAINED_BYTES
        if self.remaining_preparation_peak_bound_bytes < 0:
            raise common.RunError("resume_storage_arithmetic_invalid")
        self.matrix_peak_bound_bytes = self.remaining_preparation_peak_bound_bytes + FOLLOW_ON_PEAK_BOUND_BYTES
        required = RESERVE_BYTES + self.matrix_peak_bound_bytes
        if self.available_before < required:
            raise common.RunError(
                "preparation_storage_budget_not_met",
                {
                    "available_bytes": self.available_before,
                    "required_bytes": required,
                    "preparation_peak_bound_bytes": PREPARATION_PEAK_BOUND_BYTES,
                    "resume_retained_bytes": RESUME_RETAINED_BYTES,
                    "remaining_preparation_peak_bound_bytes": self.remaining_preparation_peak_bound_bytes,
                    "follow_on_peak_bound_bytes": FOLLOW_ON_PEAK_BOUND_BYTES,
                    "matrix_peak_bound_bytes": self.matrix_peak_bound_bytes,
                },
            )

    def check(self, phase: str) -> dict[str, Any]:
        free = available_bytes()
        growth = max(0, self.available_before - free)
        sample = {"phase": phase, "available_bytes": free, "growth_bytes": growth}
        self.samples.append(sample)
        if free < RESERVE_BYTES:
            raise common.RunError("storage_reserve_crossed", sample)
        if growth > self.remaining_preparation_peak_bound_bytes:
            raise common.RunError("preparation_peak_bound_exceeded", sample)
        return sample


def verify_baseline() -> None:
    try:
        info = BASELINE_REPORT.lstat()
    except OSError as exc:
        raise common.RunError("baseline_report_identity_invalid") from exc
    if (
        BASELINE_REPORT.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_size > 1024 * 1024
        or sha256_file(BASELINE_REPORT) != BASELINE_REPORT_SHA256
    ):
        raise common.RunError("baseline_report_identity_invalid")
    try:
        value = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
        result = value
        for key in BASELINE_RESULT_KEYS:
            result = result[key]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise common.RunError("baseline_report_invalid") from exc
    if (
        result.get("metrics", {}).get("decode_tokens_per_second_median") != BASELINE_DECODE_TPS
        or result.get("metrics", {}).get("end_to_end_tokens_per_second_median") != BASELINE_END_TO_END_TPS
        or result.get("metrics", {}).get("ttft_seconds_median") != BASELINE_TTFT_SECONDS
        or result.get("spec_decode_summary", {}).get("acceptance_ratio") != BASELINE_ACCEPTANCE
    ):
        raise common.RunError("baseline_report_invalid")


def _validate_root_path(path: Path, *, mode: int, directory: bool = True) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise common.RunError("resume_tree_identity_invalid", {"path": str(path)}) from exc
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        stat.S_ISLNK(info.st_mode)
        or not expected_type(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or stat.S_IMODE(info.st_mode) != mode
    ):
        raise common.RunError("resume_tree_identity_invalid", {"path": str(path)})
    return info


def _validate_empty_resume_target(target_info: os.stat_result, work_info: os.stat_result) -> None:
    if (
        stat.S_ISLNK(target_info.st_mode)
        or not stat.S_ISDIR(target_info.st_mode)
        or target_info.st_uid != 0
        or target_info.st_gid != 0
        or stat.S_IMODE(target_info.st_mode) != 0o700
        or target_info.st_dev != work_info.st_dev
        or os.path.ismount(TARGET_LEAF)
    ):
        raise common.RunError("resume_tree_identity_invalid", {"path": str(TARGET_LEAF)})
    try:
        if any(TARGET_LEAF.iterdir()):
            raise common.RunError("resume_tree_identity_invalid", {"path": str(TARGET_LEAF)})
    except OSError as exc:
        raise common.RunError("resume_tree_identity_invalid", {"path": str(TARGET_LEAF)}) from exc


def validate_current_k4_resume_tree(deadline: float) -> int:
    global RESUME_TARGET_LEAF_REUSABLE
    if DRAFT_TOKENS != 4:
        raise common.RunError("resume_tree_identity_invalid", {"path": str(TARGET_LEAF)})
    work_info = _validate_root_path(WORK_ROOT, mode=0o700)
    try:
        target_info = TARGET_LEAF.lstat()
    except FileNotFoundError:
        RESUME_TARGET_LEAF_REUSABLE = False
    except OSError as exc:
        raise common.RunError("resume_tree_identity_invalid", {"path": str(TARGET_LEAF)}) from exc
    else:
        _validate_empty_resume_target(target_info, work_info)
        RESUME_TARGET_LEAF_REUSABLE = True
    _validate_root_path(INGRESS_DIR, mode=0o700)
    _validate_root_path(CAL_DATA, mode=0o700)
    runs_dir = RUN_DIR.parent
    _validate_root_path(runs_dir, mode=0o755)
    _validate_root_path(RUN_DIR, mode=0o700)
    try:
        if set(WORK_ROOT.iterdir()) != {INGRESS_DIR, CAL_DATA, runs_dir}:
            raise common.RunError("resume_tree_identity_invalid", {"path": str(WORK_ROOT)})
        if any(CAL_DATA.iterdir()) or set(runs_dir.iterdir()) != {RUN_DIR} or any(RUN_DIR.iterdir()):
            raise common.RunError("resume_tree_identity_invalid", {"path": str(WORK_ROOT)})
    except OSError as exc:
        raise common.RunError("resume_tree_identity_invalid", {"path": str(WORK_ROOT)}) from exc

    first, second = BACKUP_BLOBS
    final = first.ingress
    partial = second.ingress.with_name(second.ingress.name + ".partial")
    if set(INGRESS_DIR.iterdir()) != {final, partial}:
        raise common.RunError("resume_tree_identity_invalid", {"path": str(INGRESS_DIR)})
    final_info = _validate_root_path(final, mode=0o444, directory=False)
    partial_info = _validate_root_path(partial, mode=0o644, directory=False)
    if (
        final_info.st_size != first.size
        or sha256_file(final, deadline) != first.sha256
        or partial_info.st_size != RESUME_PARTIAL_SIZE
        or sha256_file(partial, deadline) != RESUME_PARTIAL_SHA256
    ):
        raise common.RunError("resume_tree_identity_invalid", {"path": str(INGRESS_DIR)})
    return first.size + RESUME_PARTIAL_SIZE


def verify_fresh_paths(deadline: float) -> None:
    global RESUME_TREE_VALIDATED, RESUME_RETAINED_BYTES, RESUME_TARGET_LEAF_REUSABLE
    if RESUME_CURRENT_K4:
        RESUME_TREE_VALIDATED = False
        RESUME_RETAINED_BYTES = 0
        RESUME_TARGET_LEAF_REUSABLE = False
        RESUME_RETAINED_BYTES = validate_current_k4_resume_tree(deadline)
        RESUME_TREE_VALIDATED = True
    else:
        for path in (WORK_ROOT, TARGET_LEAF):
            if path.exists() or path.is_symlink():
                raise common.RunError("fresh_candidate_path_required", {"path": str(path)})
    raw = common.run(
        ["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}"],
        timeout=preparation_timeout(deadline, 15),
    ).stdout.decode("ascii").strip()
    if raw:
        raise common.RunError("stale_candidate_container_present")
    for step in BUILD_STEPS:
        name = f"{CONTAINER}-build-{step}"
        raw = common.run(
            ["docker", "ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"],
            timeout=preparation_timeout(deadline, 15),
        ).stdout.decode("ascii").strip()
        if raw:
            raise common.RunError("stale_builder_container_present", {"step": step})


def verify_patched_recipe_files() -> dict[str, str]:
    digests: dict[str, str] = {}
    for destination, (source, expected_sha256) in PATCHED_RECIPE_FILES.items():
        try:
            info = source.lstat()
        except OSError as exc:
            raise common.RunError("patched_recipe_file_identity_invalid", {"path": str(source)}) from exc
        if source.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > 256 * 1024:
            raise common.RunError("patched_recipe_file_identity_invalid", {"path": str(source)})
        actual_sha256 = sha256_file(source)
        if actual_sha256 != expected_sha256:
            raise common.RunError(
                "patched_recipe_file_identity_invalid",
                {"path": str(source), "expected_sha256": expected_sha256, "actual_sha256": actual_sha256},
            )
        digests[destination] = actual_sha256
    return digests


def preflight(operator: common.Operator, deadline: float) -> tuple[StorageGuard, dict[str, Any]]:
    status = operator.status()
    phase = status.get("phase")
    if phase != "public" or not common.admission_safe(status, "public"):
        raise common.RunError("admission_preflight_not_public_empty")
    qwen = common.qwen_state(operator.qwen_status())
    if qwen not in {"ready", "stopped"}:
        raise common.RunError("qwen_preflight_state_unsupported", {"state": qwen})
    common.verify_production_qwen_identity()
    if abs(common.gpu_rows()[3] - 300.0) > 0.05:
        raise common.RunError("power_limit_not_300w")
    base.verify_dependencies()
    if base.verify_target() != SOURCE_RECEIPT_SHA256:
        raise common.RunError("source_receipt_identity_invalid")
    base.verify_image()
    verify_baseline()
    verify_fresh_paths(deadline)
    patched_recipe_digests = verify_patched_recipe_files()
    storage = StorageGuard()
    max_prompt_tokens = validate_prompt_token_bound(deadline)
    gateway_healthy = base.service_healthy("ai-server-gateway")
    open_webui_healthy = base.service_healthy("ai-server-open-webui")
    if not gateway_healthy or not open_webui_healthy:
        raise common.RunError("user_service_preflight_not_healthy")
    return storage, {
        "admission_phase": phase,
        "qwen_state": qwen,
        "gateway_healthy": gateway_healthy,
        "open_webui_healthy": open_webui_healthy,
        "power_limit_w": common.gpu_rows()[3],
        "max_calibration_prompt_tokens": max_prompt_tokens,
        "patched_recipe_digests": patched_recipe_digests,
    }


def download_blob(blob: BackupBlob, deadline: float) -> None:
    final = blob.ingress
    partial = final.with_name(final.name + ".partial")
    if RESUME_TREE_VALIDATED and final.exists():
        info = _validate_root_path(final, mode=0o444, directory=False)
        if info.st_size != blob.size or sha256_file(final, deadline) != blob.sha256:
            raise common.RunError("pinned_backup_identity_invalid", {"file": final.name})
        return
    resume_offset = 0
    if RESUME_TREE_VALIDATED and partial.exists():
        info = _validate_root_path(partial, mode=0o644, directory=False)
        resume_offset = info.st_size
        if (
            blob != BACKUP_BLOBS[1]
            or resume_offset != RESUME_PARTIAL_SIZE
            or sha256_file(partial, deadline) != RESUME_PARTIAL_SHA256
        ):
            raise common.RunError("resume_partial_identity_invalid", {"file": final.name})
    elif final.exists() or final.is_symlink() or partial.exists() or partial.is_symlink():
        raise common.RunError("fresh_backup_path_required", {"file": final.name})
    headers = {"User-Agent": "ai-server-pinned-artifact/1"}
    if resume_offset:
        headers["Range"] = f"bytes={resume_offset}-"
    request = urllib.request.Request(blob.url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=preparation_timeout(deadline, 120)) as response:
            if resume_offset:
                status = getattr(response, "status", None)
                if status is None:
                    status = response.getcode()
                expected_range = f"bytes {resume_offset}-{blob.size - 1}/{blob.size}"
                expected_length = str(blob.size - resume_offset)
                if (
                    status != 206
                    or response.headers.get("Content-Range") != expected_range
                    or response.headers.get("Content-Length") != expected_length
                ):
                    raise common.RunError("pinned_backup_range_contract_invalid", {"file": final.name})
            mode = "ab" if resume_offset else "xb"
            with partial.open(mode) as handle:
                received = 0
                while True:
                    read_timeout = preparation_timeout(deadline, 120)
                    if response.fp is not None:
                        try:
                            response.fp.raw._sock.settimeout(read_timeout)
                        except AttributeError as exc:
                            raise common.RunError("pinned_backup_read_timeout_unavailable", {"file": final.name}) from exc
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    if resume_offset + received + len(chunk) > blob.size:
                        raise common.RunError("pinned_backup_download_exceeds_manifest", {"file": final.name})
                    if available_bytes() - len(chunk) < RESERVE_BYTES:
                        raise common.RunError("pinned_backup_download_reserve_crossed", {"file": final.name})
                    handle.write(chunk)
                    received += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    except (OSError, urllib.error.URLError) as exc:
        raise common.RunError("pinned_backup_download_failed", {"file": final.name}) from exc
    preparation_timeout(deadline, 1)
    size = partial.stat().st_size
    if size != blob.size or sha256_file(partial, deadline) != blob.sha256:
        raise common.RunError("pinned_backup_identity_invalid", {"file": final.name, "size": size})
    rename_noreplace(partial, final)
    os.chmod(final, 0o444)


def rename_noreplace(source: Path, destination: Path) -> None:
    """Linux atomic rename that refuses an existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise common.RunError("rename_noreplace_unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace_flag = 1
    result = renameat2(
        at_fdcwd, os.fsencode(source), at_fdcwd, os.fsencode(destination), rename_noreplace_flag
    )
    if result != 0:
        error = ctypes.get_errno()
        raise common.RunError("rename_noreplace_failed", {"errno": error, "file": destination.name})


def write_calibration_prompts() -> str:
    records = calibration_records()

    if len(records) != CALIBRATION_RECORDS:
        raise common.RunError("calibration_record_count_invalid")
    path = CAL_DATA / "prompts.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def calibration_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(PINNED_PROMPTS):
        for variant in range(16):
            records.append(
                {
                    "id": f"real-{prompt_index:02d}-{variant:02d}",
                    "src": f"syv-pinned-real-{prompt_index:02d}",
                    "think": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{prompt}\n\nCalibration variant {variant + 1}/16. Produce a self-contained answer.",
                        }
                    ],
                }
            )
    return records


def builder_args(step: str, *, mounts: tuple[tuple[Path, str, bool], ...], gpu: bool = True) -> list[str]:
    argv = [
        "docker", "run", "--rm", "--pull", "never", "--name", f"{CONTAINER}-build-{step}",
        "--label", f"{BUILD_OWNER_LABEL}={BUILD_OWNER_TOKEN}",
        "--restart", "no", "--user", "0:0", "--network", "none", "--ipc", "private", "--read-only",
        "--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--pids-limit", "2048",
        "--shm-size", "16g", "--memory", "30g", "--memory-swap", "32g",
        "--tmpfs", "/tmp:rw,nosuid,exec,size=8g",
        "--env", "HOME=/tmp", "--env", "XDG_CACHE_HOME=/tmp", "--env", "HF_HUB_OFFLINE=1",
        "--env", "TRANSFORMERS_OFFLINE=1", "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", f"PATH={BUILDER_PATH}",
        "--entrypoint", "/usr/bin/env",
    ]
    if gpu:
        argv += [
            "--device", CDI_GPU_DEVICE,
            "--env", NVIDIA_VISIBLE_DEVICES,
            "--env", NVIDIA_DRIVER_CAPABILITIES,
        ]
    for source, destination, read_only in mounts:
        suffix = ",readonly" if read_only else ""
        argv += ["--mount", f"type=bind,src={source},dst={destination}{suffix}"]
    argv.append(IMAGE)
    return argv


def validate_prompt_token_bound(deadline: float) -> int:
    script = (
        "import json,shutil,subprocess,sys;"
        "assert shutil.which('ninja')=='/app/venv/bin/ninja';"
        "subprocess.run(['ninja','--version'],check=True,stdout=subprocess.DEVNULL);"
        "from transformers import AutoTokenizer;"
        "t=AutoTokenizer.from_pretrained('/model');r=json.load(sys.stdin);"
        "n=[len(t(t.apply_chat_template(x['messages'],tokenize=False,add_generation_prompt=True,"
        "enable_thinking=x['think']),add_special_tokens=False)['input_ids']) for x in r];"
        "print(max(n));assert len(n)==128 and max(n)<=512"
    )
    argv = builder_args("preflight-tokenizer", mounts=((SOURCE_MODEL, "/model", True),), gpu=False)
    image_index = argv.index(IMAGE)
    argv[image_index:image_index] = ["--interactive"]
    argv.extend(["/app/venv/bin/python", "-B", "-c", script])
    timeout = preparation_timeout(deadline, 300)
    ATTEMPTED_BUILD_STEPS.add("preflight-tokenizer")
    try:
        result = common.run(
            argv,
            timeout=timeout,
            input_bytes=json.dumps(calibration_records(), ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            diagnostic_output=True,
        )
    except BaseException:
        stop_build_container("preflight-tokenizer")
        raise
    prove_build_container_absent("preflight-tokenizer")
    try:
        value = int(result.stdout.decode("ascii").strip().splitlines()[-1])
    except (UnicodeDecodeError, ValueError, IndexError) as exc:
        raise common.RunError("calibration_token_bound_not_proven") from exc
    if value > 512:
        raise common.RunError("calibration_token_bound_exceeded", {"max_prompt_tokens": value})
    return value


def run_builder(
    step: str,
    command: list[str],
    *,
    mounts: tuple[tuple[Path, str, bool], ...],
    deadline: float,
    env: tuple[str, ...] = (),
    gpu: bool = True,
) -> bytes:
    argv = builder_args(step, mounts=mounts, gpu=gpu)
    image_index = argv.index(IMAGE)
    for assignment in env:
        argv[image_index:image_index] = ["--env", assignment]
        image_index += 2
    argv.extend(command)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise common.RunError("preparation_deadline_exhausted", {"step": step})
    ATTEMPTED_BUILD_STEPS.add(step)
    try:
        result = common.run(argv, timeout=remaining, diagnostic_output=True)
    except BaseException:
        stop_build_container(step)
        raise
    prove_build_container_absent(step)
    return result.stdout


def stop_build_container(step: str) -> None:
    if step not in BUILD_STEPS:
        raise common.RunError("builder_step_not_allowlisted", {"step": step})
    name = f"{CONTAINER}-build-{step}"
    present = common.run(
        ["docker", "ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"],
        timeout=15, check=False,
    ).stdout.decode("ascii").strip()
    if present:
        owner = common.run(
            [
                "docker", "inspect", name, "--format",
                f'{{{{ index .Config.Labels "{BUILD_OWNER_LABEL}" }}}}',
            ],
            timeout=15,
            check=False,
        )
        if owner.returncode != 0 or owner.stdout.decode("utf-8", "replace").strip() != BUILD_OWNER_TOKEN:
            raise common.RunError("builder_cleanup_ownership_not_proven", {"step": step})
        result = common.run(["docker", "rm", "-f", name], timeout=60, check=False)
        if result.returncode != 0:
            raise common.RunError("builder_cleanup_not_proven", {"step": step})
    remaining = common.run(
        ["docker", "ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"],
        timeout=15, check=False,
    ).stdout.decode("ascii").strip()
    if remaining:
        raise common.RunError("builder_cleanup_not_proven", {"step": step})
    ATTEMPTED_BUILD_STEPS.discard(step)


def prove_build_container_absent(step: str) -> None:
    if step not in BUILD_STEPS:
        raise common.RunError("builder_step_not_allowlisted", {"step": step})
    name = f"{CONTAINER}-build-{step}"
    remaining = common.run(
        ["docker", "ps", "-a", "--filter", f"name=^/{name}$", "--format", "{{.Names}}"],
        timeout=15,
        check=False,
    ).stdout.decode("ascii").strip()
    if remaining:
        raise common.RunError("builder_cleanup_not_proven", {"step": step})
    ATTEMPTED_BUILD_STEPS.discard(step)


def stop_all_build_containers() -> None:
    failures: list[str] = []
    for step in tuple(sorted(ATTEMPTED_BUILD_STEPS)):
        try:
            stop_build_container(step)
        except common.RunError:
            failures.append(step)
    if failures:
        raise common.RunError("builder_cleanup_not_proven", {"steps": failures})


def seed_calibration_model(deadline: float) -> dict[str, str]:
    CAL_MODEL.mkdir(parents=True, mode=0o700)
    for source in SOURCE_MODEL.iterdir():
        if source.is_symlink() or not source.is_file():
            raise common.RunError("source_model_file_set_invalid")
        os.link(source, CAL_MODEL / source.name)
    for blob in BACKUP_BLOBS:
        os.link(blob.ingress, CAL_MODEL / blob.installed_name)
    raw_extra = CAL_MODEL / "model_extra_tensors.safetensors"
    raw_extra_backup = CAL_MODEL / "model_extra_tensors.safetensors.bak-mtp"
    os.link(raw_extra, raw_extra_backup)
    if (
        raw_extra.stat().st_ino != raw_extra_backup.stat().st_ino
        or raw_extra.stat().st_size != raw_extra_backup.stat().st_size
    ):
        raise common.RunError("raw_mtp_backup_identity_invalid")
    script = (
        "import hashlib,json,pathlib,torch;"
        "m=pathlib.Path('/model');p=pathlib.Path('/app/prepare/draft_vocab_ids.json');"
        "ids=json.loads(p.read_text());"
        "assert len(ids)==40960 and ids==sorted(set(ids));"
        "raw=json.dumps(ids,separators=(',',':')).encode();"
        "assert hashlib.sha256(raw).hexdigest()=='" + GENERIC_IDS_CANONICAL_SHA256 + "';"
        "(m/'draft_vocab_ids.json').write_bytes(raw);"
        "torch.save(torch.tensor(ids,dtype=torch.int64),m/'mtp_draft_vocab_ids.pt');"
        "paths=" + repr(PREPARE_SCRIPTS) + ";"
        "print(json.dumps({x:hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest() for x in paths},sort_keys=True))"
    )
    raw = run_builder(
        "seed", ["/app/venv/bin/python", "-B", "-c", script],
        mounts=((CAL_MODEL, "/model", False),), deadline=deadline, gpu=False,
    )
    try:
        return json.loads(raw.decode("utf-8").splitlines()[-1])
    except (UnicodeDecodeError, json.JSONDecodeError, IndexError) as exc:
        raise common.RunError("recipe_script_digest_capture_failed") from exc


def verify_builder_gpu_ready(deadline: float) -> dict[str, Any]:
    script = (
        "import json,torch;"
        "from vllm.platforms import current_platform;"
        "x=torch.ones(1,device='cuda');"
        "print(json.dumps({'cuda_available':bool(torch.cuda.is_available()),"
        "'cuda_device_count':int(torch.cuda.device_count()),"
        "'accelerator_device_count':int(torch.accelerator.device_count()),"
        "'platform_class':type(current_platform).__name__,"
        "'cuda_compute':float(x.sum().item())},sort_keys=True))"
    )
    raw = run_builder(
        "gpu-preflight", ["/app/venv/bin/python", "-B", "-c", script],
        mounts=(), deadline=deadline,
    )
    try:
        value = json.loads(raw.decode("utf-8").splitlines()[-1])
    except (UnicodeDecodeError, json.JSONDecodeError, IndexError) as exc:
        raise common.RunError("builder_gpu_preflight_invalid") from exc
    expected = {
        "cuda_available": True,
        "cuda_device_count": 1,
        "accelerator_device_count": 1,
        "platform_class": "NvmlCudaPlatform",
        "cuda_compute": 1.0,
    }
    if value != expected:
        raise common.RunError("builder_gpu_preflight_invalid", {"observed": value})
    return value


def captured_rows() -> int:
    try:
        seqs = json.loads((CAL_DATA / "seqs.json").read_text(encoding="utf-8"))
        rows = sum(int(item["n"]) for item in seqs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise common.RunError("calibration_rows_invalid") from exc
    if not isinstance(seqs, list) or len(seqs) != CALIBRATION_RECORDS or rows <= 20_000:
        raise common.RunError("calibration_rows_invalid", {"sequences": len(seqs) if isinstance(seqs, list) else None, "rows": rows})
    expected_min = rows * 5_120 * 2
    try:
        if (CAL_DATA / "hidden.npy").stat().st_size < expected_min or (CAL_DATA / "tokens.npy").stat().st_size < rows * 4:
            raise common.RunError("calibration_rows_invalid")
    except OSError as exc:
        raise common.RunError("calibration_rows_invalid") from exc
    return rows


def validate_capture_stdout(raw: bytes) -> str:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise common.RunError("capture_completion_invalid") from exc
    if not any(line.strip() == "done; incomplete sequences: 0 []" for line in lines):
        raise common.RunError("capture_incomplete_sequences")
    return hashlib.sha256(raw).hexdigest()


def read_hessian_diagnostics(*, require_complete: bool) -> list[dict[str, Any]]:
    if not HESSIAN_DIAGNOSTICS.exists():
        if require_complete:
            raise common.RunError("hessian_diagnostics_missing")
        return []
    try:
        info = HESSIAN_DIAGNOSTICS.lstat()
        if HESSIAN_DIAGNOSTICS.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > 128 * 1024:
            raise common.RunError("hessian_diagnostics_invalid")
        records = [json.loads(line) for line in HESSIAN_DIAGNOSTICS.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise common.RunError("hessian_diagnostics_invalid") from exc
    if len(records) > len(MTP_MODULES) or any(not isinstance(record, dict) for record in records):
        raise common.RunError("hessian_diagnostics_invalid")
    for index, record in enumerate(records):
        if record.get("module") != MTP_MODULES[index] or not isinstance(record.get("finite"), bool):
            raise common.RunError("hessian_diagnostics_invalid")
        attempts = record.get("attempts")
        if not isinstance(attempts, list):
            raise common.RunError("hessian_diagnostics_invalid")
        if record["finite"] is False:
            if (
                record.get("status") != "FAIL"
                or record.get("error_code") != "hessian_nonfinite"
                or attempts
            ):
                raise common.RunError("hessian_diagnostics_invalid")
            continue
        if not attempts:
            raise common.RunError("hessian_diagnostics_invalid")
        factors = [attempt.get("factor") for attempt in attempts if isinstance(attempt, dict)]
        infos = [attempt.get("cholesky_info") for attempt in attempts if isinstance(attempt, dict)]
        if len(factors) != len(attempts) or len(infos) != len(attempts) or factors != sorted(set(factors)):
            raise common.RunError("hessian_diagnostics_invalid")
        if record.get("status") == "PASS":
            if infos[-1] != 0 or record.get("chosen_damping_factor") != factors[-1]:
                raise common.RunError("hessian_diagnostics_invalid")
        elif "error_code" not in record:
            raise common.RunError("hessian_diagnostics_invalid")
    if require_complete and (
        len(records) != len(MTP_MODULES) or any(record.get("status") != "PASS" for record in records)
    ):
        raise common.RunError("hessian_diagnostics_incomplete")
    return records


def make_extra_distinct() -> None:
    source = SOURCE_MODEL / "model_extra_tensors.safetensors"
    MTP_WORK_DIR.mkdir(mode=0o700)
    for path in LM4_DIR.iterdir():
        destination = MTP_WORK_DIR / path.name
        if destination.exists() or destination.is_symlink():
            raise common.RunError("fresh_distinct_copy_path_required")
        if path.name in {source.name, "config.json", "model.safetensors.index.json"}:
            shutil.copyfile(path, destination)
            os.chmod(destination, 0o644)
        else:
            os.link(path, destination)
    destination = MTP_WORK_DIR / source.name
    if destination.lstat().st_ino == source.lstat().st_ino or sha256_file(destination) != sha256_file(source):
        raise common.RunError("raw_extra_distinct_copy_invalid")


def publish_candidate_model() -> None:
    if (
        not BUILD_CANDIDATE_MODEL.is_dir()
        or BUILD_CANDIDATE_MODEL.is_symlink()
        or TARGET_MODEL.exists()
        or TARGET_MODEL.is_symlink()
    ):
        raise common.RunError("candidate_publish_precondition_invalid")
    rename_noreplace(BUILD_CANDIDATE_MODEL, TARGET_MODEL)
    if BUILD_CANDIDATE_MODEL.exists() or BUILD_CANDIDATE_MODEL.is_symlink() or not TARGET_MODEL.is_dir():
        raise common.RunError("candidate_publish_not_proven")


def prepare_candidate(
    storage: StorageGuard, deadline: float, patched_recipe_digests: dict[str, str]
) -> dict[str, Any]:
    if RESUME_CURRENT_K4 and not RESUME_TREE_VALIDATED:
        raise common.RunError("resume_tree_not_validated")
    resume_exist_ok = RESUME_TREE_VALIDATED
    if RESUME_TARGET_LEAF_REUSABLE:
        work_info = _validate_root_path(WORK_ROOT, mode=0o700)
        target_info = _validate_root_path(TARGET_LEAF, mode=0o700)
        _validate_empty_resume_target(target_info, work_info)
    WORK_ROOT.mkdir(parents=True, mode=0o700, exist_ok=resume_exist_ok)
    TARGET_LEAF.mkdir(parents=True, mode=0o700, exist_ok=RESUME_TARGET_LEAF_REUSABLE)
    INGRESS_DIR.mkdir(mode=0o700, exist_ok=resume_exist_ok)
    CAL_DATA.mkdir(mode=0o700, exist_ok=resume_exist_ok)
    RUN_DIR.mkdir(parents=True, mode=0o700, exist_ok=resume_exist_ok)
    storage.check("directories_created")
    for blob in BACKUP_BLOBS:
        download_blob(blob, deadline)
        storage.check(f"downloaded_{blob.source_name}")
    corpus_sha = write_calibration_prompts()
    script_digests = seed_calibration_model(deadline)
    storage.check("calibration_model_seeded")
    verify_builder_gpu_ready(deadline)
    storage.check("builder_gpu_ready")

    common_mounts = ((CAL_MODEL, "/model", False), (CAL_DATA, "/app/drafter/data", False))
    initial_commands = preparation_commands(85_536)
    run_builder(
        "gen-data", initial_commands["gen_data"],
        mounts=common_mounts, deadline=deadline,
        env=("MODEL=/model", "MAX_THINK=1024", "MAX_NOTHINK=1024", "MAX_SEQS=8", "GPU_UTIL=0.90"),
    )
    storage.check("generated_calibration_tokens")
    capture_raw = run_builder(
        "capture", initial_commands["capture"],
        mounts=common_mounts, deadline=deadline, env=("MODEL=/model", "MAX_SEQS=8", "GPU_UTIL=0.90"),
    )
    capture_log_sha = validate_capture_stdout(capture_raw)
    rows = captured_rows()
    commands = preparation_commands(rows)
    storage.check("captured_hidden_states")
    run_builder(
        "hessians",
        commands["hessians"],
        mounts=((CAL_MODEL, "/model", False), (CAL_DATA, "/app/drafter/data", True), (RUN_DIR, "/work/run", False)),
        deadline=deadline,
    )
    storage.check("dumped_mtp_hessians")
    if not HESSIANS.is_file() or HESSIANS.is_symlink():
        raise common.RunError("mtp_hessians_missing")

    calib_rows = int(commands["gptq_lm_head"][-1])
    run_builder(
        "gptq-lm-head",
        commands["gptq_lm_head"],
        mounts=((WORK_ROOT, "/work", False), (CAL_DATA, "/app/drafter/data", True)),
        deadline=deadline,
    )
    storage.check("gptq_int4_lm_head")
    make_extra_distinct()
    storage.check("raw_extra_made_distinct")
    run_builder(
        "quant-mtp-int8",
        commands["quant_mtp_int8"],
        mounts=((WORK_ROOT, "/work", False),), deadline=deadline,
    )
    storage.check("intermediate_int8_mtp")
    run_builder(
        "draft-vocab",
        commands["draft_vocab"],
        mounts=((WORK_ROOT, "/work", False),), deadline=deadline,
    )
    shutil.copyfile(CAL_MODEL / "draft_vocab_ids.json", MTP_WORK_DIR / "draft_vocab_ids.json")
    storage.check("generic_int4_draft_head")
    run_builder(
        "requant-mtp-int4",
        commands["requant_mtp_int4"],
        mounts=(
            (WORK_ROOT, "/work", False),
            (PATCHED_GPTQ_UTILS, "/app/drafter/gptq_utils.py", True),
            (PATCHED_REQUANT_MTP, "/app/drafter/requant_mtp_gptq.py", True),
        ),
        deadline=deadline,
        env=("GPTQ_DIAGNOSTICS_PATH=/work/runs/eval/gptq_hessian_diagnostics.jsonl",),
    )
    hessian_diagnostics = read_hessian_diagnostics(require_complete=True)
    shutil.copyfile(MTP_WORK_DIR / "draft_vocab_ids.json", BUILD_CANDIDATE_MODEL / "draft_vocab_ids.json")
    publish_candidate_model()
    storage.check("final_int4_mtp")
    verification = verify_candidate_tree(deadline)
    receipt = seal_candidate(
        corpus_sha=corpus_sha,
        script_digests=script_digests,
        rows=rows,
        calib_rows=calib_rows,
        storage=storage,
        verification=verification,
        capture_log_sha=capture_log_sha,
        patched_recipe_digests=patched_recipe_digests,
        hessian_diagnostics=hessian_diagnostics,
        deadline=deadline,
    )
    storage.check("candidate_sealed")
    return receipt


def verify_candidate_tree(deadline: float) -> dict[str, Any]:
    try:
        config = json.loads((TARGET_MODEL / "config.json").read_text(encoding="utf-8"))
        index = json.loads((TARGET_MODEL / "model.safetensors.index.json").read_text(encoding="utf-8"))
        source_index = json.loads((SOURCE_MODEL / "model.safetensors.index.json").read_text(encoding="utf-8"))
        ids = json.loads((TARGET_MODEL / "draft_vocab_ids.json").read_text(encoding="utf-8"))
        quant = config["quantization_config"]
        lm_group = quant["config_groups"]["group_1"]
        mtp_group = quant["config_groups"]["group_3"]
        weight_map = index["weight_map"]
        source_weight_map = source_index["weight_map"]
        vocab_size = int(config["text_config"]["vocab_size"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise common.RunError("candidate_metadata_invalid") from exc
    if (
        lm_group.get("weights", {}).get("num_bits") != 4
        or mtp_group.get("targets") != [r"re:^mtp\..*"]
        or mtp_group.get("weights", {}).get("num_bits") != 4
        or mtp_group.get("weights", {}).get("group_size") != 128
        or not isinstance(ids, list)
        or len(ids) != DRAFT_VOCAB_SIZE
        or ids != sorted(set(ids))
        or vocab_size != INT4_TENSOR_DIMS["lm_head"][0]
    ):
        raise common.RunError("candidate_quantization_contract_invalid")
    if sha256_file(TARGET_MODEL / "draft_vocab_ids.json", deadline) != GENERIC_IDS_CANONICAL_SHA256:
        raise common.RunError("candidate_draft_vocab_identity_invalid")
    lm_keys = {f"lm_head.{suffix}" for suffix in ("weight_packed", "weight_scale", "weight_shape")}
    lm_shard = source_weight_map.get("lm_head.weight_packed")
    if (
        not isinstance(lm_shard, str)
        or any(source_weight_map.get(key) != lm_shard for key in lm_keys)
        or any(weight_map.get(key) != lm_shard for key in lm_keys)
        or "lm_head.weight" in weight_map
    ):
        raise common.RunError("candidate_lm_head_contract_invalid")
    for source in SOURCE_MODEL.iterdir():
        destination = TARGET_MODEL / source.name
        if source.name.startswith("model-0000") and source.name.endswith(".safetensors") and source.name != lm_shard:
            if not destination.is_file() or (source.stat().st_dev, source.stat().st_ino) != (destination.stat().st_dev, destination.stat().st_ino):
                raise common.RunError("candidate_body_hardlink_invalid", {"file": source.name})
    mutated_file_sha256: dict[str, str] = {}
    for name in (lm_shard, "config.json", "model.safetensors.index.json", "model_extra_tensors.safetensors"):
        destination = TARGET_MODEL / name
        source = SOURCE_MODEL / name
        if not destination.is_file() or destination.stat().st_ino == source.stat().st_ino:
            raise common.RunError("candidate_mutated_file_not_distinct", {"file": name})
        mutated_file_sha256[name] = sha256_file(destination, deadline)
    fast_overlay_matching_files = verify_fast_overlay_absent(mutated_file_sha256)
    probe_names = sorted(
        {f"{prefix}.{suffix}" for prefix in INT4_TENSOR_DIMS for suffix in ("weight_packed", "weight_scale", "weight_shape", "weight")}
    )
    probe = (
        "import json,pathlib,torch;from safetensors import safe_open;"
        "p='/candidate/';idx=json.load(open(p+'model.safetensors.index.json'))['weight_map'];"
        "files={};wanted=set(" + repr(probe_names) + ");"
        "\nfor path in sorted(pathlib.Path(p).glob('*.safetensors')):\n"
        " with safe_open(str(path),framework='pt') as f:\n"
        "  files[path.name]={}\n"
        "  for k in sorted(wanted.intersection(f.keys())):\n"
        "   t=f.get_tensor(k);files[path.name][k]={'shape':list(t.shape),'dtype':str(t.dtype)}\n"
        "   if k.endswith('.weight_shape'): files[path.name][k]['value']=t.tolist()\n"
        "draft=torch.load(p+'mtp_draft_vocab_ids.pt',map_location='cpu',weights_only=True).tolist();"
        "print(json.dumps({'files':files,'draft_ids':draft},sort_keys=True))"
    )
    raw = run_builder(
        "verify", ["/app/venv/bin/python", "-B", "-c", probe],
        mounts=((TARGET_MODEL, "/candidate", True),), deadline=deadline, gpu=False,
    )
    try:
        inventory = json.loads(raw.decode("utf-8").splitlines()[-1])
        tensors_by_file = inventory["files"]
        torch_draft_ids = inventory["draft_ids"]
    except (UnicodeDecodeError, json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        raise common.RunError("candidate_tensor_inventory_invalid") from exc
    if not isinstance(tensors_by_file, dict) or torch_draft_ids != ids:
        raise common.RunError("candidate_tensor_inventory_invalid")
    locations: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for file_name, values in tensors_by_file.items():
        if not isinstance(file_name, str) or not isinstance(values, dict):
            raise common.RunError("candidate_tensor_inventory_invalid")
        for tensor_name, tensor_record in values.items():
            if not isinstance(tensor_record, dict):
                raise common.RunError("candidate_tensor_inventory_invalid")
            locations.setdefault(tensor_name, []).append((file_name, tensor_record))
    for prefix, (out_features, in_features) in INT4_TENSOR_DIMS.items():
        raw_key = f"{prefix}.weight"
        if raw_key in weight_map or raw_key in locations:
            raise common.RunError("candidate_raw_weight_present", {"tensor": raw_key})
        expected = {
            f"{prefix}.weight_packed": {"shape": [out_features, in_features // 8], "dtype": "torch.int32"},
            f"{prefix}.weight_scale": {"shape": [out_features, in_features // 128], "dtype": "torch.float16"},
            f"{prefix}.weight_shape": {
                "shape": [2], "dtype": "torch.int64", "value": [out_features, in_features],
            },
        }
        for tensor_name, exact in expected.items():
            mapped_file = weight_map.get(tensor_name)
            found = locations.get(tensor_name, [])
            if not isinstance(mapped_file, str) or len(found) != 1 or found[0][0] != mapped_file or found[0][1] != exact:
                raise common.RunError("candidate_int4_tensor_contract_invalid", {"tensor": tensor_name})
    return {
        "lm_head_bits": 4,
        "mtp_bits": 4,
        "group_size": 128,
        "draft_vocab_size": DRAFT_VOCAB_SIZE,
        "exact_target_body_hardlinks_proven": True,
        "dbirks_fast_overlay_absent": True,
        "dbirks_fast_overlay_matching_files": fast_overlay_matching_files,
        "tensor_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def seal_candidate(
    *, corpus_sha: str, script_digests: dict[str, str], rows: int, calib_rows: int,
    storage: StorageGuard, verification: dict[str, Any], capture_log_sha: str,
    patched_recipe_digests: dict[str, str], hessian_diagnostics: list[dict[str, Any]], deadline: float,
) -> dict[str, Any]:
    files = []
    for path in sorted(TARGET_MODEL.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise common.RunError("candidate_file_set_invalid")
        info = path.stat()
        files.append({
            "name": path.name,
            "sha256": sha256_file(path, deadline),
            "size": info.st_size,
            "allocated_bytes": info.st_blocks * 512,
        })
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "synthetic_calibration_content_retained": True,
        "user_content_recorded": False,
        "secrets_recorded": False,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "receipt_sha256": SOURCE_RECEIPT_SHA256,
        },
        "recipe": {
            "repository": SYV_REPOSITORY,
            "revision": SYV_REVISION,
            "image_manifest_digest": base.MANIFEST_DIGEST,
            "scripts": script_digests,
            "patched_scripts": patched_recipe_digests,
            "hessian_stabilization": {
                "method": "finite-check-blockwise-symmetrize-bounded-adaptive-damping",
                "damping_factors": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0],
                "rtn_fallback": False,
                "diagnostics": hessian_diagnostics,
            },
            "lm_head_bits": 4,
            "intermediate_mtp_bits": 8,
            "final_mtp_bits": 4,
            "group_size": 128,
            "draft_vocab_size": DRAFT_VOCAB_SIZE,
            "draft_tokens": DRAFT_TOKENS,
            "draft_vocab_ids_sha256": GENERIC_IDS_CANONICAL_SHA256,
            "fast_overlay_repository": None,
        },
        "calibration": {
            "source": "pinned syv bench/prompts_real.jsonl at the recipe revision",
            "base_prompt_count": len(PINNED_PROMPTS),
            "record_count": CALIBRATION_RECORDS,
            "max_output_tokens": CALIBRATION_OUTPUT_TOKENS,
            "corpus_sha256": corpus_sha,
            "captured_rows": rows,
            "capture_complete": True,
            "capture_log_sha256": capture_log_sha,
            "lm_head_calibration_rows": calib_rows,
            "mtp_hessians_sha256": sha256_file(HESSIANS, deadline),
        },
        "backups": [
            {"installed_name": blob.installed_name, "size": blob.size, "sha256": blob.sha256}
            for blob in BACKUP_BLOBS
        ],
        "storage": {
            "required_reserve_bytes": RESERVE_BYTES,
            "preparation_peak_bound_bytes": PREPARATION_PEAK_BOUND_BYTES,
            "resume_retained_bytes": RESUME_RETAINED_BYTES,
            "remaining_preparation_peak_bound_bytes": storage.remaining_preparation_peak_bound_bytes,
            "follow_on_peak_bound_bytes": FOLLOW_ON_PEAK_BOUND_BYTES,
            "matrix_peak_bound_bytes": storage.matrix_peak_bound_bytes,
            "components": storage_components(),
            "available_before_bytes": storage.available_before,
            "phase_samples": storage.samples,
        },
        "verification": verification,
        "files": files,
    }
    raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    preparation_timeout(deadline, 1)
    with TARGET_RECEIPT.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    for path in TARGET_MODEL.iterdir():
        preparation_timeout(deadline, 1)
        os.chmod(path, 0o444)
    os.chmod(TARGET_MODEL, 0o555)
    os.chmod(TARGET_RECEIPT, 0o444)
    os.chmod(TARGET_LEAF, 0o555)
    receipt["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    return receipt


def verify_sealed_candidate(deadline: float) -> str:
    try:
        raw = TARGET_RECEIPT.read_bytes()
        receipt = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise common.RunError("candidate_receipt_invalid") from exc
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("source", {}).get("revision") != SOURCE_REVISION:
        raise common.RunError("candidate_receipt_invalid")
    if receipt.get("recipe", {}).get("fast_overlay_repository") is not None:
        raise common.RunError("dbirks_fast_overlay_identity_detected")
    for record in receipt.get("files", []):
        path = TARGET_MODEL / record["name"]
        if path.is_symlink() or not path.is_file() or sha256_file(path, deadline) != record["sha256"] or path.stat().st_size != record["size"]:
            raise common.RunError("candidate_file_identity_invalid", {"file": record.get("name")})
    return hashlib.sha256(raw).hexdigest()


_ORIGINAL_CONTAINER_ARGS = base.container_args
_ORIGINAL_VERIFY_CONTAINER_CONTRACT = base.verify_container_contract
_ORIGINAL_STARTUP_EVIDENCE = base.startup_evidence


def container_args() -> list[str]:
    original_target, original_container = base.TARGET_MODEL_DIR, base.CONTAINER
    try:
        base.TARGET_MODEL_DIR, base.CONTAINER = TARGET_MODEL, CONTAINER
        argv = _ORIGINAL_CONTAINER_ARGS()
    finally:
        base.TARGET_MODEL_DIR, base.CONTAINER = original_target, original_container
    verify_entries = [index for index, value in enumerate(argv) if value == "VERIFY=1"]
    if len(verify_entries) != 1:
        raise common.RunError("candidate_container_contract_invalid")
    # The pinned upstream verifier only accepts an int8 lm_head. This exact
    # candidate intentionally uses an independently verified int4 lm_head.
    argv[verify_entries[0]] = "VERIFY=0"
    gpu_entries = [
        index for index, value in enumerate(argv[:-1])
        if value == "--gpus" and argv[index + 1] == "1"
    ]
    if len(gpu_entries) != 1:
        raise common.RunError("candidate_container_contract_invalid")
    gpu_index = gpu_entries[0]
    argv[gpu_index:gpu_index + 2] = ["--device", CDI_GPU_DEVICE]
    image_index = argv.index(IMAGE)
    argv[image_index:image_index] = [
        "--env", NVIDIA_VISIBLE_DEVICES,
        "--env", NVIDIA_DRIVER_CAPABILITIES,
        "--env", "GPU_UTIL=0.93",
        "--env", "PREFIX_CACHE=0",
        "--env", f"DRAFT_TOKENS={DRAFT_TOKENS}",
    ]
    return argv


def verify_runtime_contract(deadline: float) -> dict[str, Any]:
    original_container, original_target = base.CONTAINER, base.TARGET_MODEL_DIR
    try:
        base.CONTAINER, base.TARGET_MODEL_DIR = CONTAINER, TARGET_MODEL
        contract = _ORIGINAL_VERIFY_CONTAINER_CONTRACT(deadline, expected_verify="0")
    finally:
        base.CONTAINER, base.TARGET_MODEL_DIR = original_container, original_target
    raw = common.run(
        ["docker", "inspect", CONTAINER],
        timeout=base.bounded_timeout(deadline, 15),
    ).stdout
    try:
        values = json.loads(raw.decode("utf-8"))
        value = values[0]
        env = set(value["Config"]["Env"])
        device_requests = value["HostConfig"]["DeviceRequests"]
    except (UnicodeDecodeError, json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
        raise common.RunError("candidate_container_contract_invalid") from exc
    expected_device_request = {
        "Driver": "cdi",
        "Count": 0,
        "DeviceIDs": [CDI_GPU_DEVICE],
        "Capabilities": None,
        "Options": None,
    }
    if (
        device_requests != [expected_device_request]
        or not {
            NVIDIA_VISIBLE_DEVICES,
            NVIDIA_DRIVER_CAPABILITIES,
            "GPU_UTIL=0.93",
            "PREFIX_CACHE=0",
            f"DRAFT_TOKENS={DRAFT_TOKENS}",
        } <= env
        or "NVIDIA_VISIBLE_DEVICES=all" in env
    ):
        raise common.RunError("candidate_container_contract_invalid")
    contract.update({
        "gpu_device_request": expected_device_request,
        "nvidia_visible_devices": "void",
        "nvidia_driver_capabilities": "compute,utility",
        "gpu_memory_utilization": 0.93,
        "prefix_cache": False,
        "draft_tokens": DRAFT_TOKENS,
        "upstream_model_verify": "skipped_incompatible_int8_only_check",
        "custom_candidate_verification": True,
    })
    return contract


def startup_evidence(deadline: float, receipt_sha: str) -> dict[str, Any]:
    try:
        receipt_raw = TARGET_RECEIPT.read_bytes()
        receipt = json.loads(receipt_raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise common.RunError("candidate_receipt_invalid") from exc
    verification = receipt.get("verification")
    if (
        hashlib.sha256(receipt_raw).hexdigest() != receipt_sha
        or not isinstance(verification, dict)
        or verification.get("lm_head_bits") != 4
        or verification.get("mtp_bits") != 4
        or verification.get("draft_vocab_size") != DRAFT_VOCAB_SIZE
    ):
        raise common.RunError("candidate_receipt_invalid")
    original_container = base.CONTAINER
    try:
        base.CONTAINER = CONTAINER
        proof = _ORIGINAL_STARTUP_EVIDENCE(deadline)
    finally:
        base.CONTAINER = original_container
    lines = proof.get("matched_lines") or []
    proof["mtp_embedding_runtime_proven"] = any(
        "Detected MTP model. Sharing target model embedding weights with the draft model." in line
        for line in lines
    )
    proof["mtp_lm_head_runtime_proven"] = any(
        "Detected MTP model. Sharing target model lm_head weights with the draft model." in line
        for line in lines
    )
    proof["draft_vocab_runtime_proven"] = any(
        f"MTP drafter uses a {DRAFT_VOCAB_SIZE}-token draft head" in line
        for line in lines
    )
    proof["sealed_candidate_contract"] = {
        "receipt_sha256": receipt_sha,
        "lm_head_bits": verification["lm_head_bits"],
        "mtp_bits": verification["mtp_bits"],
        "draft_vocab_size": verification["draft_vocab_size"],
    }
    proof["mtp_runtime_proven"] = bool(
        proof.get("mtp_runtime_proven")
        and proof["mtp_embedding_runtime_proven"]
        and proof["mtp_lm_head_runtime_proven"]
        and proof["draft_vocab_runtime_proven"]
    )
    return proof


def start_candidate(deadline: float) -> None:
    common.run(container_args(), timeout=base.bounded_timeout(deadline, 60), diagnostic_output=True)


def stop_candidate(deadline: float | None = None) -> None:
    original_container = base.CONTAINER
    try:
        base.CONTAINER = CONTAINER
        base.stop_candidate(deadline)
    finally:
        base.CONTAINER = original_container


def cleanup_transient_staging() -> dict[str, Any]:
    expected = (
        TRANSIENT_STAGING_BASE / "leminkozey-int4-fast-mtp-adaptive-work-r1" / SOURCE_REVISION,
        TRANSIENT_STAGING_BASE / "leminkozey-int4-fast-mtp-adaptive-r1" / SOURCE_REVISION,
    )
    actual = (WORK_ROOT, TARGET_LEAF)
    if actual != expected:
        raise common.RunError("transient_cleanup_path_contract_invalid")
    free_before = shutil.disk_usage(TRANSIENT_STAGING_BASE).free
    removed: list[str] = []
    already_absent: list[str] = []
    for path in actual:
        if not os.path.lexists(path):
            already_absent.append(str(path))
            continue
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise common.RunError("transient_cleanup_root_type_invalid", {"path": str(path)})
        if path.resolve(strict=True) != path or info.st_uid != 0 or info.st_gid != 0 or os.path.ismount(path):
            raise common.RunError("transient_cleanup_root_identity_invalid", {"path": str(path)})
        for root, directories, files in os.walk(path, followlinks=False):
            root_path = Path(root)
            if root_path != path and os.path.ismount(root_path):
                raise common.RunError("transient_cleanup_nested_mount_refused", {"path": str(root_path)})
            for name in directories + files:
                child = root_path / name
                child_info = child.lstat()
                if stat.S_ISLNK(child_info.st_mode):
                    raise common.RunError("transient_cleanup_symlink_refused", {"path": str(child)})
                if child_info.st_uid != 0 or child_info.st_gid != 0:
                    raise common.RunError("transient_cleanup_owner_invalid", {"path": str(child)})
        shutil.rmtree(path)
        if os.path.lexists(path):
            raise common.RunError("transient_cleanup_not_proven", {"path": str(path)})
        removed.append(str(path))
    free_after = shutil.disk_usage(TRANSIENT_STAGING_BASE).free
    return {
        "status": "PASS",
        "removed": removed,
        "already_absent": already_absent,
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "freed_bytes": max(0, free_after - free_before),
    }


def cleanup_transient_staging_safely() -> dict[str, Any]:
    try:
        return cleanup_transient_staging()
    except common.RunError:
        raise
    except OSError as exc:
        raise common.RunError(
            "transient_cleanup_io_failure", {"exception_type": type(exc).__name__}
        ) from exc


def close_admission_once(operator: common.Operator) -> None:
    try:
        operator.advance("closed")
    except common.RunError as exc:
        if exc.code != "operator_outcome_unknown":
            raise
        status = operator.status()
        if not common.admission_safe(status, "closed"):
            raise common.RunError("admission_close_outcome_unknown") from exc


def qwen_transition_once(operator: common.Operator, verb: str) -> None:
    expected = "ready" if verb == "start" else "stopped"
    try:
        operator.qwen(verb)
    except common.RunError as exc:
        if exc.code != "operator_outcome_unknown":
            raise
        state = common.qwen_state(operator.qwen_status())
        if state != expected:
            raise common.RunError(f"qwen_{verb}_outcome_unknown", {"observed_state": state}) from exc


def restore(operator: common.Operator, *, admission_closed: bool, qwen_stopped: bool) -> dict[str, Any]:
    result = {
        "candidate_absent": False, "qwen_ready": False, "gateway_completions": 0,
        "admission_public": False, "gateway_healthy": False, "open_webui_healthy": False,
        "power_limit_w": None,
    }
    remaining = common.run(
        ["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER}$", "--format", "{{.Names}}"], timeout=15
    ).stdout.decode("ascii").strip()
    if remaining:
        raise common.RunError("candidate_cleanup_not_proven")
    result["candidate_absent"] = True
    state = common.qwen_state(operator.qwen_status())
    if state == "stopped":
        if not qwen_stopped:
            raise common.RunError("unexpected_qwen_prestate_drift")
        common.wait_gpu_idle()
        qwen_transition_once(operator, "start")
    if common.qwen_state(operator.qwen_status()) != "ready":
        raise common.RunError("qwen_restore_not_proven")
    common.verify_production_qwen_identity()
    result["qwen_ready"] = True
    result["gateway_completions"] = base.restore_admission(operator, close_attempted=admission_closed)
    result["admission_public"] = True
    result["gateway_healthy"] = base.service_healthy("ai-server-gateway")
    result["open_webui_healthy"] = base.service_healthy("ai-server-open-webui")
    if not result["gateway_healthy"] or not result["open_webui_healthy"]:
        raise common.RunError("user_service_health_not_proven")
    result["power_limit_w"] = common.gpu_rows()[3]
    if abs(result["power_limit_w"] - 300.0) > 0.05:
        raise common.RunError("power_limit_restore_not_proven")
    return result


def benchmark_candidate(deadline: float, receipt_sha: str) -> dict[str, Any]:
    start_candidate(deadline)
    contract = verify_runtime_contract(deadline)
    original_container = base.CONTAINER
    try:
        base.CONTAINER = CONTAINER
        try:
            load_seconds = base.wait_ready(deadline)
        except common.RunError as exc:
            if exc.code != "candidate_exit_before_ready":
                raise
            raise common.RunError(
                exc.code,
                {**exc.details, "startup_diagnostics": base.diagnostics(include_logs=True)},
            ) from exc
        proof = startup_evidence(deadline, receipt_sha)
        if proof.get("mtp_runtime_proven") is not True:
            raise common.RunError("mtp_runtime_not_proven")
        value = base.run_benchmark(deadline)
    finally:
        base.CONTAINER = original_container
    metrics = value.get("metrics", {})
    summary = value.get("spec_decode_summary", {})
    decode = metrics.get("decode_tokens_per_second_median")
    end_to_end = metrics.get("end_to_end_tokens_per_second_median")
    ttft = metrics.get("ttft_seconds_median")
    acceptance = summary.get("acceptance_ratio")
    if not all(isinstance(value, (int, float)) for value in (decode, end_to_end, ttft, acceptance)):
        raise common.RunError("benchmark_result_invalid")
    marker = "INT4_FAST_MTP_OK_" + receipt_sha[:12]
    probe = b"""import json,sys,urllib.request
marker=sys.argv[1]
body=json.dumps({'model':'qwen3.8-27b','messages':[{'role':'user','content':'Return exactly: '+marker}],'temperature':0,'max_tokens':64,'stream':False,'chat_template_kwargs':{'enable_thinking':False}},separators=(',',':')).encode()
request=urllib.request.Request('http://127.0.0.1:18020/v1/chat/completions',data=body,headers={'Content-Type':'application/json'},method='POST')
with urllib.request.urlopen(request,timeout=180) as response: value=json.load(response)
choice=value['choices'][0];content=choice['message']['content']
if choice.get('finish_reason')!='stop' or not isinstance(content,str) or content.strip()!=marker: raise SystemExit(22)
print('EXACT_MARKER_OK')
"""
    common.run(
        ["docker", "exec", "-i", CONTAINER, "/app/venv/bin/python", "-B", "-", marker],
        timeout=base.bounded_timeout(deadline, 210, "live_budget_exhausted_during_correctness_probe"),
        input_bytes=probe,
        diagnostic_output=True,
    )
    value.update({
        "mode_id": f"leminkozey-exact-gptq-int4-fast-mtp-adaptive-vocab40960-k{DRAFT_TOKENS}",
        "load_seconds": load_seconds,
        "runtime_contract": contract,
        "startup_evidence": proof,
        "target_receipt_sha256": receipt_sha,
        "correctness": {
            "exact_marker": True,
            "marker_sha256": hashlib.sha256(marker.encode("ascii")).hexdigest(),
            "response_text_recorded": False,
        },
        "comparison": {
            "baseline_report": str(BASELINE_REPORT),
            "baseline_report_sha256": BASELINE_REPORT_SHA256,
            "baseline_decode_tokens_per_second_median": BASELINE_DECODE_TPS,
            "candidate_decode_tokens_per_second_median": decode,
            "decode_change_percent": round((decode / BASELINE_DECODE_TPS - 1) * 100, 3),
            "baseline_end_to_end_tokens_per_second_median": BASELINE_END_TO_END_TPS,
            "candidate_end_to_end_tokens_per_second_median": end_to_end,
            "end_to_end_change_percent": round((end_to_end / BASELINE_END_TO_END_TPS - 1) * 100, 3),
            "baseline_ttft_seconds_median": BASELINE_TTFT_SECONDS,
            "candidate_ttft_seconds_median": ttft,
            "ttft_change_percent": round((ttft / BASELINE_TTFT_SECONDS - 1) * 100, 3),
            "baseline_acceptance_ratio": BASELINE_ACCEPTANCE,
            "candidate_acceptance_ratio": acceptance,
            "acceptance_change_percentage_points": round((acceptance - BASELINE_ACCEPTANCE) * 100, 3),
        },
    })
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("root_required")
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise SystemExit("output_path_invalid")
    report: dict[str, Any] = {
        "schema": "qwen38-syv-exact-int4-fast-mtp-adaptive-result/v1",
        "started_epoch": int(time.time()),
        "content_recorded": True,
        "synthetic_calibration_content_retained": True,
        "user_content_recorded": False,
        "secrets_recorded": False,
        "coordinates": {
            "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION,
            "recipe_repository": SYV_REPOSITORY, "recipe_revision": SYV_REVISION,
            "image": IMAGE, "variant_id": VARIANT_ID,
        },
        "matrix": {
            "warmups": 1, "measured_requests": 3,
            "max_output_tokens": 1024, "draft_tokens": DRAFT_TOKENS,
        },
        "excluded": [
            "DFlash2", "FreeToken", "Philbert",
            f"DRAFT_TOKENS={4 if DRAFT_TOKENS == 3 else 3} rerun",
            "Git", "promotion",
        ],
        "preparation": {"status": "PENDING"},
        "benchmark": {"status": "PENDING", "attempted": False},
        "transient_cleanup": {"status": "PENDING"},
        "restore": {},
        "outcome": "running",
        "error_code": None,
        "cleanup_error_code": None,
        "transient_cleanup_error_code": None,
    }
    operator = common.Operator()
    admission_closed = False
    qwen_stopped = False
    runtime_started = False
    storage: StorageGuard | None = None
    error: common.RunError | None = None
    cleanup_error: common.RunError | None = None
    transient_cleanup_error: common.RunError | None = None
    sampler: common.Sampler | None = None
    since_epoch = int(time.time())
    descriptor = os.open(common.LOCK, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise common.RunError("host_lock_busy") from exc
        try:
            preparation_deadline = time.monotonic() + PREPARATION_BUDGET_SECONDS
            storage, prestate = preflight(operator, preparation_deadline)
            report["prestate"] = prestate
            qwen_stopped = prestate["qwen_state"] == "stopped"
            close_admission_once(operator)
            admission_closed = True
            if not common.admission_safe(operator.status(), "closed"):
                raise common.RunError("admission_close_not_proven")
            if prestate["qwen_state"] == "ready":
                qwen_stopped = True
                qwen_transition_once(operator, "stop")
            if common.qwen_state(operator.qwen_status()) != "stopped":
                raise common.RunError("qwen_stop_not_proven")
            common.wait_gpu_idle()
            receipt = prepare_candidate(
                storage, preparation_deadline, prestate["patched_recipe_digests"]
            )
            receipt_sha = receipt["receipt_sha256"]
            if verify_sealed_candidate(preparation_deadline) != receipt_sha:
                raise common.RunError("candidate_receipt_identity_invalid")
            report["preparation"] = {
                "status": "PASS", "receipt_sha256": receipt_sha,
                "source": receipt["source"], "backups": receipt["backups"],
                "calibration": receipt["calibration"], "recipe": receipt["recipe"],
                "storage": receipt["storage"], "verification": receipt["verification"],
            }
            live_deadline = time.monotonic() + LIVE_BUDGET_SECONDS
            with common.Sampler() as active_sampler:
                sampler = active_sampler
                report["benchmark"]["attempted"] = True
                runtime_started = True
                value = benchmark_candidate(live_deadline - RESTORE_RESERVE_SECONDS, receipt_sha)
            value["resources"] = sampler.report()
            report["benchmark"] = {"status": "PASS", "attempted": True, "result": value}
        except common.RunError as exc:
            error = exc
            report["error_code"] = exc.code
            if report["preparation"]["status"] == "PENDING":
                report["preparation"] = {"status": "FAIL", "reason": exc.code, **exc.details}
                try:
                    diagnostics = read_hessian_diagnostics(require_complete=False)
                except common.RunError as diagnostic_error:
                    report["preparation"]["hessian_diagnostics_error"] = diagnostic_error.code
                else:
                    if diagnostics:
                        report["preparation"]["hessian_diagnostics"] = diagnostics
            elif report["benchmark"]["status"] == "PENDING":
                report["benchmark"] = {"status": "FAIL", "attempted": report["benchmark"]["attempted"], "reason": exc.code, **exc.details}
        except Exception as exc:
            error = common.RunError("unexpected_runner_failure", {"exception_type": type(exc).__name__})
            report["error_code"] = error.code
        finally:
            runtime_cleanup_failures: list[common.RunError] = []
            if runtime_started:
                try:
                    stop_candidate(live_deadline)
                except common.RunError as exc:
                    runtime_cleanup_failures.append(exc)
            try:
                stop_all_build_containers()
            except common.RunError as exc:
                runtime_cleanup_failures.append(exc)
            if runtime_cleanup_failures:
                cleanup_error = runtime_cleanup_failures[0]
                report["cleanup_error_code"] = cleanup_error.code
                report["cleanup_failures"] = [exc.code for exc in runtime_cleanup_failures]
                report["transient_cleanup"] = {
                    "status": "SKIPPED", "reason": "runtime_cleanup_not_proven"
                }
            else:
                if RETAIN_TRANSIENT:
                    report["transient_cleanup"] = {
                        "status": "RETAINED",
                        "reason": "explicit_cleanup_authority_required",
                        "paths": [str(WORK_ROOT), str(TARGET_LEAF)],
                        "available_bytes": available_bytes(),
                    }
                else:
                    try:
                        report["transient_cleanup"] = cleanup_transient_staging_safely()
                    except common.RunError as exc:
                        transient_cleanup_error = exc
                        report["transient_cleanup_error_code"] = exc.code
                        report["transient_cleanup"] = {
                            "status": "FAIL", "reason": exc.code, **exc.details
                        }
                try:
                    report["restore"] = restore(
                        operator, admission_closed=admission_closed, qwen_stopped=qwen_stopped
                    )
                    if storage is not None:
                        phase = "restored_with_candidate_retained" if RETAIN_TRANSIENT else "restored"
                        report["storage_after_restore"] = storage.check(phase)
                except common.RunError as exc:
                    cleanup_error = exc
                    report["cleanup_error_code"] = exc.code
    finally:
        os.close(descriptor)
    report["kernel_fault_free"] = common.kernel_fault_free(since_epoch)
    report["ended_epoch"] = int(time.time())
    if cleanup_error is not None:
        report["outcome"] = "restore_unknown"
    elif error is not None or transient_cleanup_error is not None or report["kernel_fault_free"] is not True:
        report["outcome"] = "failed_restored"
    elif report["preparation"].get("status") == "PASS" and report["benchmark"].get("status") == "PASS":
        report["outcome"] = "passed_restored"
    else:
        report["outcome"] = "failed_restored"
    common.write_report(args.output, report)
    print(json.dumps({"outcome": report["outcome"], "report": str(args.output)}, sort_keys=True))
    return 0 if report["outcome"] == "passed_restored" else 1


if __name__ == "__main__":
    raise SystemExit(main())
