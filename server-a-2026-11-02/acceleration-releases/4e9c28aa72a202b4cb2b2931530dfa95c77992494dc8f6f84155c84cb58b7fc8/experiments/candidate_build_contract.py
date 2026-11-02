"""Fail-closed, offline-verifiable build contracts for acceleration candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable
from urllib.parse import urlsplit


CUDA_BASE = (
    "nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:"
    "092b75ee0ab6ff2da2fa56799866b23910129c37cf3454a2a4941c422f5304e5"
)
FREETOKEN_BASE_DIGEST = "sha256:092b75ee0ab6ff2da2fa56799866b23910129c37cf3454a2a4941c422f5304e5"
LLAMA_CUDA_BUILDER_BASE = (
    "nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:"
    "4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43"
)
LLAMA_CUDA_RUNTIME_BASE = (
    "nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:"
    "828c4d878adcaa4265d80c95d8ec877149b49bb2419a4cf3bb6aa889bbb7ca2e"
)
FREETOKEN_REVISION = "9db1a39455a3fb107f3db83e381d10ceadfe5d99"
FREETOKEN_WHEEL = "freetoken-0.1.2-cp312-cp312-manylinux_2_27_x86_64.whl"
FREETOKEN_WHEEL_SHA256 = (
    "993afeb4ef1ee3a1c5302b3c46dea6545d86f4a6facc3ea386985e02f4466a2f"
)
FREETOKEN_WHEEL_SIZE = 1_238_381
FREETOKEN_WHEEL_URL = (
    "https://github.com/FlashML-org/FreeToken/releases/download/v0.1.2/"
    + FREETOKEN_WHEEL
)
FREETOKEN_SOURCE_ARCHIVE_SHA256 = (
    "c98d43493f898703c905c9dc192d8b12c9362b07794d6de7865507fee2da3091"
)
PYTHON_STANDALONE = (
    "cpython-3.12.14+20260825-x86_64-unknown-linux-gnu-"
    "install_only_stripped.tar.gz"
)
PYTHON_STANDALONE_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    "20260825/cpython-3.12.14%2B20260825-x86_64-unknown-linux-gnu-"
    "install_only_stripped.tar.gz"
)
PYTHON_STANDALONE_SIZE = 34_130_340
PYTHON_STANDALONE_SHA256 = (
    "7ce4a71285d913955a76053cc7605ea96da8ecada54dba9cf395245961816421"
)
FREETOKEN_PYLOCK_SHA256 = (
    "88b95aa6e966d857e80da356d6657b12ec07d07e237036c0cdd5357a4ef8c32b"
)
FREETOKEN_SUPPLEMENTS_SHA256 = (
    "561c34202d6b6ba3057d782107c01659dc43beaf457637e81ba3e68d8d60ab69"
)
FREETOKEN_ARTIFACT_MANIFEST_SHA256 = (
    "f69f08f96f59a208a462f8d35c1f347e556234a9eaeedaf863c29c5a466581ed"
)
FREETOKEN_REQUIREMENTS_LOCK_SHA256 = (
    "89e57a4f6efae127c1922a9c60436a0fcf656e64b3ec32da95613650b98c4981"
)
FREETOKEN_REQUIREMENTS_OFFLINE_LOCK_SHA256 = (
    "c5a8478d127dc3c8bf268956d6b09f1c552daa25a495c0f395fc9e53096be441"
)
FREETOKEN_RESOLVER_RECEIPT_SHA256 = (
    "bd0d5d065806fd0c7326d5a7f354adba7949d4671ab3a151c1fb761e1b69e45f"
)
FREETOKEN_DOCKERFILE_SHA256 = (
    "ca71914bc19882999276958eae752eb5fd6424e0aa2adf2c15ac328a9a2fd622"
)
FREETOKEN_OCI_RECEIPT_SCHEMA_SHA256 = (
    "5e0050007d4e7b4c4a13079406b38d624d89741653f02c8798cf9be31c8af44e"
)
FREETOKEN_COMPOSE_SHA256 = (
    "713cd9f02f54b81d719d355c348a2bf44f8a9fc373d06d1288b404001a9459e5"
)
LLAMA_REVISION = "b10f9ca58c89ccfc3653ac01e979dd085d582b76"
LLAMA_SOURCE_ARCHIVE_SHA256 = (
    "0ce1e39d49da65dfba450827f1e8b689b1d7f31721fba06d5bd298dc4177c773"
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
DIRECT_REQUIREMENT = re.compile(
    r"([A-Za-z0-9_.-]+)\s+@\s+(https://\S+)\s+--hash=sha256:([0-9a-f]{64})\Z"
)
PINNED_REQUIREMENT = re.compile(
    r"([a-z0-9]+(?:-[a-z0-9]+)*)==([^\s]+)\s+--hash=sha256:([0-9a-f]{64})\Z"
)
WHEEL_FILENAME = re.compile(
    r"^(?P<distribution>.+?)-(?P<version>[^-]+)-(?P<python>[^-]+)-"
    r"(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$",
    re.IGNORECASE,
)


class BuildContractError(RuntimeError):
    """A candidate build input is mutable, incomplete, or inconsistent."""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise BuildContractError(f"cannot read {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise BuildContractError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_tokens(text: str, tokens: Iterable[str], label: str) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        raise BuildContractError(f"{label} missing required tokens: {missing}")


def _reject_mutable_or_networked_build(text: str, label: str) -> None:
    lowered = text.lower()
    forbidden = {
        "mutable image tag": re.compile(r"(?:from|image:)\s+[^\s@]+:(?:latest|main)(?:\s|$)"),
        "curl pipe shell": re.compile(r"curl[^\n|]*\|\s*(?:sh|bash)"),
        "wget pipe shell": re.compile(r"wget[^\n|]*\|\s*(?:sh|bash)"),
        "package index resolution": re.compile(r"\b(?:apt-get|apt|apk|dnf|yum)\s+(?:install|update)\b"),
        "source network fetch": re.compile(r"\b(?:git\s+clone|curl\s+https?://|wget\s+https?://)"),
        "pip network resolution": re.compile(r"pip\s+install(?![\s\S]{0,180}--no-index)"),
    }
    for reason, pattern in forbidden.items():
        if pattern.search(lowered):
            raise BuildContractError(f"{label} contains {reason}")
    if re.search(r"^\s*entrypoint\b", lowered, flags=re.MULTILINE):
        raise BuildContractError(f"{label} must not define ENTRYPOINT")
    runtime_marker = " as runtime"
    marker_index = lowered.rfind(runtime_marker)
    runtime = lowered[marker_index:] if marker_index >= 0 else ""
    if re.search(r"\b(?:pip\s+install|git\s+clone|curl\s+https?://|wget\s+https?://)\b", runtime):
        raise BuildContractError(f"{label} runtime stage performs network-capable installation")


def validate_freetoken_argv(argv: Iterable[str]) -> tuple[str, ...]:
    """Allow only the two explicitly reviewed FreeToken command families."""

    command = tuple(argv)
    allowed_prefixes = (("ft", "bench", "bw"), ("ft", "serve"))
    if not any(command[: len(prefix)] == prefix for prefix in allowed_prefixes):
        raise BuildContractError("FreeToken argv is not an allowlisted bench-bw or serve command")
    if any(not item or "\x00" in item for item in command):
        raise BuildContractError("FreeToken argv contains an invalid item")
    return command


def validate_freetoken_assets(root: Path) -> dict[str, str]:
    directory = root / "freetoken"
    dockerfile = _read(directory / "Dockerfile")
    compose = _read(directory / "compose.yaml")
    requirements_in = _read(directory / "requirements.in")
    metadata = {
        "pylock.freetoken.toml": FREETOKEN_PYLOCK_SHA256,
        "artifact-supplements.json": FREETOKEN_SUPPLEMENTS_SHA256,
        "artifact-manifest.json": FREETOKEN_ARTIFACT_MANIFEST_SHA256,
        "requirements.lock": FREETOKEN_REQUIREMENTS_LOCK_SHA256,
        "requirements.offline.lock": FREETOKEN_REQUIREMENTS_OFFLINE_LOCK_SHA256,
        "resolver-receipt.json": FREETOKEN_RESOLVER_RECEIPT_SHA256,
        "oci-build-receipt.schema.json": FREETOKEN_OCI_RECEIPT_SCHEMA_SHA256,
    }
    for filename, expected_sha256 in metadata.items():
        if _sha256(directory / filename) != expected_sha256:
            raise BuildContractError(f"FreeToken {filename} digest mismatch")
    _require_tokens(
        dockerfile,
        (
            f"FROM --platform=linux/amd64 {CUDA_BASE} AS builder",
            f"FROM --platform=linux/amd64 {CUDA_BASE} AS runtime",
            FREETOKEN_REVISION,
            FREETOKEN_WHEEL_SHA256,
            FREETOKEN_SOURCE_ARCHIVE_SHA256,
            FREETOKEN_REQUIREMENTS_LOCK_SHA256,
            FREETOKEN_REQUIREMENTS_OFFLINE_LOCK_SHA256,
            FREETOKEN_ARTIFACT_MANIFEST_SHA256,
            FREETOKEN_RESOLVER_RECEIPT_SHA256,
            PYTHON_STANDALONE,
            PYTHON_STANDALONE_SHA256,
            'ai.server.healthcheck.runtime="python-stdlib-urllib"',
            "import freetoken, urllib.request",
            "ARG SEALED_INPUT_MANIFEST_SHA256=UNSET",
            "sha256sum -c SHA256SUMS",
            "/opt/python/bin/python3.12",
            "--no-index",
            "--require-hashes",
            "RUN --network=none",
            "USER 65532:65532",
            'HF_HUB_OFFLINE="1"',
            'TRANSFORMERS_OFFLINE="1"',
        ),
        "FreeToken Dockerfile",
    )
    _reject_mutable_or_networked_build(dockerfile, "FreeToken Dockerfile")
    if _sha256(directory / "Dockerfile") != FREETOKEN_DOCKERFILE_SHA256:
        raise BuildContractError("FreeToken Dockerfile digest mismatch")
    if _sha256(directory / "compose.yaml") != FREETOKEN_COMPOSE_SHA256:
        raise BuildContractError("FreeToken compose digest mismatch")
    _require_tokens(
        compose,
        (
            'test: ["CMD", "/opt/python/bin/python3.12", "-c",',
            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:1919/health', timeout=5).read(1)",
        ),
        "FreeToken compose healthcheck",
    )
    if "curl" in compose.lower():
        raise BuildContractError("FreeToken compose healthcheck must not require curl")
    _load_json(directory / "oci-build-receipt.schema.json")
    if requirements_in.strip().splitlines()[-1] != "freetoken==0.1.2":
        raise BuildContractError("requirements.in must have one exact direct pin")
    _validate_freetoken_resolution(directory)
    return {"candidate": "freetoken", "status": "sealed_resolution_verified"}


def _validate_oci_descriptor(value: object, media_types: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"media_type", "digest", "size"}:
        raise BuildContractError(f"invalid {label} OCI descriptor schema")
    if (
        value["media_type"] not in media_types
        or not isinstance(value["digest"], str)
        or not value["digest"].startswith("sha256:")
        or HEX64.fullmatch(value["digest"][7:]) is None
        or not isinstance(value["size"], int)
        or isinstance(value["size"], bool)
        or value["size"] <= 0
    ):
        raise BuildContractError(f"invalid {label} OCI descriptor identity")
    return value


def validate_freetoken_oci_receipt(
    receipt: object,
    *,
    expected_sealed_input_manifest_sha256: str,
) -> dict:
    """Validate actual OCI build output without inventing an output digest."""

    if HEX64.fullmatch(expected_sealed_input_manifest_sha256) is None:
        raise BuildContractError("expected sealed-input manifest SHA256 must be lowercase hex")
    if not isinstance(receipt, dict) or set(receipt) != {
        "format", "candidate_id", "base", "sealed_input_manifest_sha256",
        "dockerfile_sha256", "build", "output",
    }:
        raise BuildContractError("unexpected FreeToken OCI receipt schema")
    if receipt["format"] != "ai-server-freetoken-oci-build-receipt-v1" or receipt["candidate_id"] != "freetoken":
        raise BuildContractError("unexpected FreeToken OCI receipt identity")
    if receipt["base"] != {"reference": CUDA_BASE, "digest": FREETOKEN_BASE_DIGEST}:
        raise BuildContractError("FreeToken OCI receipt base image mismatch")
    if receipt["sealed_input_manifest_sha256"] != expected_sealed_input_manifest_sha256:
        raise BuildContractError("FreeToken OCI receipt sealed-input mismatch")
    if receipt["dockerfile_sha256"] != FREETOKEN_DOCKERFILE_SHA256:
        raise BuildContractError("FreeToken OCI receipt Dockerfile mismatch")
    if receipt["build"] != {
        "platform": "linux/amd64",
        "network": "none",
        "pull": False,
        "no_cache": True,
        "args": {"SEALED_INPUT_MANIFEST_SHA256": expected_sealed_input_manifest_sha256},
    }:
        raise BuildContractError("FreeToken OCI build invocation is not fail-closed")

    output = receipt["output"]
    if not isinstance(output, dict) or set(output) != {
        "reference", "digest", "index", "manifest", "config", "layers", "labels",
    }:
        raise BuildContractError("unexpected FreeToken OCI output schema")
    index = _validate_oci_descriptor(
        output["index"], {"application/vnd.oci.image.index.v1+json"}, "index",
    )
    _validate_oci_descriptor(
        output["manifest"], {"application/vnd.oci.image.manifest.v1+json"}, "manifest",
    )
    _validate_oci_descriptor(
        output["config"], {"application/vnd.oci.image.config.v1+json"}, "config",
    )
    layers = output["layers"]
    if not isinstance(layers, list) or not layers:
        raise BuildContractError("FreeToken OCI receipt must contain layers")
    layer_digests: set[str] = set()
    for layer in layers:
        descriptor = _validate_oci_descriptor(
            layer,
            {
                "application/vnd.oci.image.layer.v1.tar+gzip",
                "application/vnd.oci.image.layer.v1.tar+zstd",
            },
            "layer",
        )
        if descriptor["digest"] in layer_digests:
            raise BuildContractError("duplicate FreeToken OCI layer digest")
        layer_digests.add(descriptor["digest"])
    digest = output["digest"]
    reference = output["reference"]
    if digest != index["digest"] or not isinstance(reference, str) or not reference.endswith("@" + digest) or any(character.isspace() for character in reference):
        raise BuildContractError("FreeToken OCI output reference/index mismatch")
    expected_labels = {
        "org.opencontainers.image.source": "https://github.com/FlashML-org/FreeToken",
        "org.opencontainers.image.revision": FREETOKEN_REVISION,
        "ai.server.freetoken.version": "0.1.2",
        "ai.server.freetoken.wheel.sha256": FREETOKEN_WHEEL_SHA256,
        "ai.server.freetoken.requirements.sha256": FREETOKEN_REQUIREMENTS_LOCK_SHA256,
        "ai.server.freetoken.requirements.offline.sha256": FREETOKEN_REQUIREMENTS_OFFLINE_LOCK_SHA256,
        "ai.server.freetoken.artifacts.sha256": FREETOKEN_ARTIFACT_MANIFEST_SHA256,
        "ai.server.freetoken.resolver-receipt.sha256": FREETOKEN_RESOLVER_RECEIPT_SHA256,
        "ai.server.python-standalone.sha256": PYTHON_STANDALONE_SHA256,
        "ai.server.healthcheck.runtime": "python-stdlib-urllib",
    }
    if output["labels"] != expected_labels:
        raise BuildContractError("FreeToken OCI output labels mismatch")
    return receipt


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(_read(path))
    except json.JSONDecodeError as exc:
        raise BuildContractError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BuildContractError(f"{path} must contain a JSON object")
    return value


def _logical_lock_requirements(text: str) -> list[str]:
    logical: list[str] = []
    pending = ""
    for physical in text.splitlines():
        stripped = physical.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pending = f"{pending} {stripped}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical.append(pending)
        pending = ""
    if pending:
        logical.append(pending)
    return logical


def _is_supported_freetoken_wheel(match: re.Match[str]) -> bool:
    python_tags = match.group("python").lower().split(".")
    abi_tags = match.group("abi").lower().split(".")
    platform = match.group("platform").lower()
    supported_platform = platform == "any" or (
        "linux" in platform and "x86_64" in platform and "musllinux" not in platform
    )
    if not supported_platform:
        return False
    if "cp312" in python_tags and "cp312" in abi_tags:
        return True
    if "abi3" in abi_tags and any(
        re.fullmatch(r"cp3(?:7|8|9|10|11|12)", tag) for tag in python_tags
    ):
        return True
    return (
        match.group("python").lower() in {"py3", "py2.py3"}
        and match.group("abi").lower() == "none"
        and (platform == "any" or "x86_64" in platform)
    )


def _validate_freetoken_resolution(directory: Path) -> None:
    receipt = _load_json(directory / "resolver-receipt.json")
    expected_receipt = {
        "artifact_count": 101,
        "artifact_manifest_sha256": FREETOKEN_ARTIFACT_MANIFEST_SHA256,
        "artifact_supplements_sha256": FREETOKEN_SUPPLEMENTS_SHA256,
        "format": "ai-server-freetoken-resolver-receipt-v1",
        "requirements_lock_sha256": FREETOKEN_REQUIREMENTS_LOCK_SHA256,
        "requirements_offline_lock_sha256": FREETOKEN_REQUIREMENTS_OFFLINE_LOCK_SHA256,
        "source_pylock": "pylock.freetoken.toml",
        "source_pylock_sha256": FREETOKEN_PYLOCK_SHA256,
    }
    if receipt != expected_receipt:
        raise BuildContractError("FreeToken resolver receipt is not the sealed identity")

    manifest = _load_json(directory / "artifact-manifest.json")
    if set(manifest) != {
        "artifact_supplements_sha256", "artifacts", "format", "platform",
        "source_pylock_sha256",
    }:
        raise BuildContractError("unexpected FreeToken artifact manifest schema")
    if (
        manifest["format"] != "ai-server-freetoken-wheelhouse-v1"
        or manifest["platform"] != "cp312-linux-x86_64"
        or manifest["source_pylock_sha256"] != FREETOKEN_PYLOCK_SHA256
        or manifest["artifact_supplements_sha256"] != FREETOKEN_SUPPLEMENTS_SHA256
    ):
        raise BuildContractError("FreeToken artifact manifest identity mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != receipt["artifact_count"]:
        raise BuildContractError("FreeToken artifact count mismatch")

    by_requirement: dict[tuple[str, str, str], dict] = {}
    by_offline_requirement: dict[tuple[str, str, str], dict] = {}
    filenames: set[str] = set()
    normalized_names: set[str] = set()
    allowed_hosts = {"files.pythonhosted.org", "download-r2.pytorch.org", "pypi.nvidia.com", "github.com"}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {
            "filename", "name", "sha256", "size", "url", "version",
        }:
            raise BuildContractError("invalid FreeToken artifact entry")
        filename = artifact["filename"]
        name = artifact["name"]
        version = artifact["version"]
        url = artifact["url"]
        sha256 = artifact["sha256"]
        wheel_match = WHEEL_FILENAME.fullmatch(filename) if isinstance(filename, str) else None
        normalized_name = re.sub(r"[-_.]+", "-", name).lower() if isinstance(name, str) else None
        if (
            not isinstance(filename, str) or Path(filename).name != filename
            or filename in filenames or not filename.endswith(".whl")
            or not isinstance(name, str) or name != normalized_name
            or normalized_name in normalized_names
            or not isinstance(version, str) or not version
            or wheel_match is None
            or not _is_supported_freetoken_wheel(wheel_match)
            or re.sub(r"[-_.]+", "-", wheel_match.group("distribution")).lower() != normalized_name
            or wheel_match.group("version").replace("_", "-") != version.replace("_", "-")
            or not isinstance(artifact["size"], int) or artifact["size"] <= 0
            or not isinstance(sha256, str) or HEX64.fullmatch(sha256) is None
            or not isinstance(url, str) or urlsplit(url).scheme != "https"
            or urlsplit(url).hostname not in allowed_hosts
        ):
            raise BuildContractError("unsafe or incomplete FreeToken artifact entry")
        filenames.add(filename)
        normalized_names.add(normalized_name)
        key = (name, url, sha256)
        if key in by_requirement:
            raise BuildContractError("duplicate FreeToken artifact identity")
        by_requirement[key] = artifact
        by_offline_requirement[(name, version, sha256)] = artifact

    requirements = _logical_lock_requirements(_read(directory / "requirements.lock"))
    if len(requirements) != len(artifacts):
        raise BuildContractError("FreeToken lock/artifact count mismatch")
    seen: set[tuple[str, str, str]] = set()
    for requirement in requirements:
        match = DIRECT_REQUIREMENT.fullmatch(requirement)
        if match is None:
            raise BuildContractError(f"unsupported FreeToken lock entry: {requirement}")
        key = (match.group(1), match.group(2), match.group(3))
        if key not in by_requirement or key in seen:
            raise BuildContractError("FreeToken lock does not exactly match artifact manifest")
        seen.add(key)
    if seen != set(by_requirement):
        raise BuildContractError("FreeToken artifact manifest has unlocked entries")

    offline_requirements = _logical_lock_requirements(
        _read(directory / "requirements.offline.lock")
    )
    if len(offline_requirements) != len(artifacts):
        raise BuildContractError("FreeToken offline lock/artifact count mismatch")
    seen_offline: set[tuple[str, str, str]] = set()
    for requirement in offline_requirements:
        match = PINNED_REQUIREMENT.fullmatch(requirement)
        if match is None:
            raise BuildContractError(f"unsupported FreeToken offline lock entry: {requirement}")
        key = (match.group(1), match.group(2), match.group(3))
        if key not in by_offline_requirement or key in seen_offline:
            raise BuildContractError(
                "FreeToken offline lock does not exactly match artifact manifest"
            )
        seen_offline.add(key)
    if seen_offline != set(by_offline_requirement):
        raise BuildContractError("FreeToken artifact manifest has offline-unlocked entries")

    wheel = by_requirement.get(("freetoken", FREETOKEN_WHEEL_URL, FREETOKEN_WHEEL_SHA256))
    if wheel is None or wheel["filename"] != FREETOKEN_WHEEL or wheel["size"] != FREETOKEN_WHEEL_SIZE:
        raise BuildContractError("FreeToken cp312 wheel identity mismatch")


def validate_llama_assets(root: Path) -> dict[str, str]:
    dockerfile = _read(root / "llama_dflash2_official" / "Dockerfile")
    _require_tokens(
        dockerfile,
        (
            f"FROM --platform=linux/amd64 {LLAMA_CUDA_BUILDER_BASE} AS builder",
            f"FROM --platform=linux/amd64 {LLAMA_CUDA_RUNTIME_BASE} AS runtime",
            LLAMA_REVISION,
            LLAMA_SOURCE_ARCHIVE_SHA256,
            "ARG SEALED_INPUT_MANIFEST_SHA256=UNSET",
            "ARG TOOLCHAIN_LOCK_SHA256=UNSET",
            "ARG RUNTIME_CLOSURE_LOCK_SHA256=UNSET",
            "sha256sum -c SHA256SUMS",
            "toolchain-lock.json",
            "runtime-closure-lock.json",
            "set -- /opt/sealed-input/toolchain-debs/*.deb",
            'dpkg --unpack "$@"',
            "dpkg --configure --pending",
            "dpkg --audit",
            "seal-elf-runtime.sh",
            "runtime-sonames.txt",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DGGML_CUDA=ON",
            "-DGGML_NATIVE=OFF",
            "-DCMAKE_CUDA_ARCHITECTURES=86",
            "-DLLAMA_OPENSSL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_UI=OFF",
            "-DLLAMA_USE_PREBUILT_UI=OFF",
            "-DLLAMA_LLGUIDANCE=OFF",
            "-DGGML_CUDA_NCCL=OFF",
            "--target llama-server",
            "RUN --network=none",
            "USER 65532:65532",
        ),
        "llama.cpp Dockerfile",
    )
    _reject_mutable_or_networked_build(dockerfile, "llama.cpp Dockerfile")
    if "build-toolchain.tar" in dockerfile:
        raise BuildContractError("llama.cpp Dockerfile uses an opaque toolchain archive")
    return {"candidate": "llama-dflash2-official", "status": "blocked_pending_sealed_input"}


def _validate_debian_closure_lock(
    directory: Path,
    entries: dict[PurePosixPath, str],
    *,
    filename: str,
    role: str,
    base_image: str,
    artifact_directory: str,
) -> None:
    lock_path = PurePosixPath(filename)
    if lock_path not in entries:
        raise BuildContractError(f"llama.cpp {filename} is missing")
    lock = _load_json(directory / filename)
    expected_keys = {"artifacts", "base_image", "format", "platform", "role"}
    if role == "runtime":
        expected_keys.add("soname_allowlist")
    if set(lock) != expected_keys:
        raise BuildContractError(f"invalid llama.cpp {filename} schema")
    if (
        lock["format"] != "ai-server-debian-closure-v1"
        or lock["platform"] != "linux/amd64"
        or lock["role"] != role
        or lock["base_image"] != base_image
    ):
        raise BuildContractError(f"llama.cpp {filename} identity mismatch")
    artifacts = lock["artifacts"]
    if not isinstance(artifacts, list) or (role == "builder" and not artifacts):
        raise BuildContractError(f"llama.cpp {filename} has invalid sealed artifacts")

    if role == "runtime":
        allowlist = lock["soname_allowlist"]
        if not isinstance(allowlist, dict) or set(allowlist) != {"filename", "sha256"}:
            raise BuildContractError("invalid llama.cpp runtime SONAME allowlist")
        allowlist_path = _safe_manifest_path(str(allowlist.get("filename", "")))
        allowlist_sha256 = allowlist.get("sha256")
        if (
            allowlist_path != PurePosixPath("runtime-sonames.txt")
            or not isinstance(allowlist_sha256, str)
            or HEX64.fullmatch(allowlist_sha256) is None
            or entries.get(allowlist_path) != allowlist_sha256
        ):
            raise BuildContractError("unresolved llama.cpp runtime SONAME allowlist")
        sonames = _read(directory / "runtime-sonames.txt").splitlines()
        if (
            not sonames
            or sonames != sorted(set(sonames))
            or any(re.fullmatch(r"[A-Za-z0-9_.+-]+\.so(?:\.[0-9]+)*", item) is None for item in sonames)
        ):
            raise BuildContractError("invalid llama.cpp runtime SONAME allowlist content")

    locked_paths: set[PurePosixPath] = set()
    packages: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {
            "filename", "package", "sha256", "size", "version",
        }:
            raise BuildContractError(f"invalid artifact in llama.cpp {filename}")
        artifact_filename = artifact["filename"]
        package = artifact["package"]
        version = artifact["version"]
        sha256 = artifact["sha256"]
        size = artifact["size"]
        if (
            not isinstance(artifact_filename, str)
            or Path(artifact_filename).name != artifact_filename
            or not artifact_filename.endswith(".deb")
            or not isinstance(package, str) or not package
            or not isinstance(version, str) or not version
            or not isinstance(sha256, str) or HEX64.fullmatch(sha256) is None
            or not isinstance(size, int) or size <= 0
        ):
            raise BuildContractError(f"unresolved artifact in llama.cpp {filename}")
        path = PurePosixPath(artifact_directory, artifact_filename)
        if path in locked_paths or package in packages:
            raise BuildContractError(f"duplicate artifact in llama.cpp {filename}")
        if entries.get(path) != sha256:
            raise BuildContractError(f"llama.cpp {filename} artifact digest mismatch: {path}")
        if (directory.joinpath(*path.parts)).stat().st_size != size:
            raise BuildContractError(f"llama.cpp {filename} artifact size mismatch: {path}")
        locked_paths.add(path)
        packages.add(package)

    sealed_paths = {
        path for path in entries
        if len(path.parts) == 2 and path.parts[0] == artifact_directory
    }
    if sealed_paths != locked_paths:
        raise BuildContractError(f"llama.cpp {filename} does not exactly cover its closure")
    required = {"cmake", "ninja-build", "g++"} if role == "builder" else set()
    if not required.issubset(packages):
        raise BuildContractError(f"llama.cpp {filename} lacks required packages")


def _safe_manifest_path(raw: str) -> PurePosixPath:
    value = PurePosixPath(raw)
    if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise BuildContractError(f"unsafe sealed-input path: {raw!r}")
    return value


def _parse_sha256sums(text: str) -> dict[PurePosixPath, str]:
    entries: dict[PurePosixPath, str] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        if match is None:
            raise BuildContractError(f"invalid SHA256SUMS line {number}")
        path = _safe_manifest_path(match.group(2))
        if path in entries:
            raise BuildContractError(f"duplicate SHA256SUMS path: {path}")
        entries[path] = match.group(1)
    if not entries:
        raise BuildContractError("empty SHA256SUMS")
    return entries


def verify_sealed_input(
    directory: Path,
    expected_manifest_sha256: str,
    *,
    candidate: str,
    source_archive_sha256: str | None = None,
) -> dict[str, str | int]:
    """Verify an already-generated build input without network access."""

    directory = directory.resolve()
    if HEX64.fullmatch(expected_manifest_sha256) is None:
        raise BuildContractError("expected manifest SHA256 must be lowercase hex")
    manifest = directory / "SHA256SUMS"
    if _sha256(manifest) != expected_manifest_sha256:
        raise BuildContractError("sealed-input manifest digest mismatch")
    entries = _parse_sha256sums(_read(manifest))
    for relative, expected in entries.items():
        path = directory.joinpath(*relative.parts).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise BuildContractError(f"sealed path escaped root: {relative}") from exc
        if not path.is_file() or _sha256(path) != expected:
            raise BuildContractError(f"sealed artifact mismatch: {relative}")

    actual = {
        PurePosixPath(path.relative_to(directory).as_posix())
        for path in directory.rglob("*")
        if path.is_file() and path != manifest
    }
    if actual != set(entries):
        raise BuildContractError("SHA256SUMS must cover every sealed artifact exactly")

    if candidate == "freetoken":
        source_archive = PurePosixPath(f"FreeToken-{FREETOKEN_REVISION}.tar.gz")
        required = {
            PurePosixPath("requirements.lock"),
            PurePosixPath("requirements.offline.lock"),
            PurePosixPath("artifact-manifest.json"),
            PurePosixPath("resolver-receipt.json"),
            PurePosixPath(PYTHON_STANDALONE),
            PurePosixPath(f"wheels/{FREETOKEN_WHEEL}"),
            source_archive,
        }
        if not required.issubset(entries):
            raise BuildContractError("FreeToken sealed input is incomplete")
        if entries[source_archive] != FREETOKEN_SOURCE_ARCHIVE_SHA256:
            raise BuildContractError("FreeToken source archive digest mismatch")
        if entries[PurePosixPath("requirements.lock")] != FREETOKEN_REQUIREMENTS_LOCK_SHA256:
            raise BuildContractError("FreeToken requirements lock digest mismatch")
        if (
            entries[PurePosixPath("requirements.offline.lock")]
            != FREETOKEN_REQUIREMENTS_OFFLINE_LOCK_SHA256
        ):
            raise BuildContractError("FreeToken offline requirements lock digest mismatch")
        if entries[PurePosixPath("artifact-manifest.json")] != FREETOKEN_ARTIFACT_MANIFEST_SHA256:
            raise BuildContractError("FreeToken artifact manifest digest mismatch")
        if entries[PurePosixPath("resolver-receipt.json")] != FREETOKEN_RESOLVER_RECEIPT_SHA256:
            raise BuildContractError("FreeToken resolver receipt digest mismatch")
        if entries[PurePosixPath(f"wheels/{FREETOKEN_WHEEL}")] != FREETOKEN_WHEEL_SHA256:
            raise BuildContractError("FreeToken wheel digest mismatch")
        if entries[PurePosixPath(PYTHON_STANDALONE)] != PYTHON_STANDALONE_SHA256:
            raise BuildContractError("Python standalone archive digest mismatch")
        if (directory / PYTHON_STANDALONE).stat().st_size != PYTHON_STANDALONE_SIZE:
            raise BuildContractError("Python standalone archive size mismatch")
        if (directory / "wheels" / FREETOKEN_WHEEL).stat().st_size != FREETOKEN_WHEEL_SIZE:
            raise BuildContractError("FreeToken wheel size mismatch")
        _validate_freetoken_resolution(directory)
    elif candidate == "llama-dflash2-official":
        archive = PurePosixPath(f"llama.cpp-{LLAMA_REVISION}.tar.gz")
        required = {
            PurePosixPath("toolchain-lock.json"),
            PurePosixPath("runtime-closure-lock.json"),
            archive,
        }
        if not required.issubset(entries):
            raise BuildContractError("llama.cpp sealed input is incomplete")
        if PurePosixPath("build-toolchain.tar") in entries:
            raise BuildContractError("llama.cpp sealed input contains an opaque toolchain archive")
        if source_archive_sha256 is not None and source_archive_sha256 != LLAMA_SOURCE_ARCHIVE_SHA256:
            raise BuildContractError("llama.cpp requested source digest is not allowlisted")
        if entries[archive] != LLAMA_SOURCE_ARCHIVE_SHA256:
            raise BuildContractError("llama.cpp source archive digest mismatch")
        _validate_debian_closure_lock(
            directory,
            entries,
            filename="toolchain-lock.json",
            role="builder",
            base_image=LLAMA_CUDA_BUILDER_BASE,
            artifact_directory="toolchain-debs",
        )
        _validate_debian_closure_lock(
            directory,
            entries,
            filename="runtime-closure-lock.json",
            role="runtime",
            base_image=LLAMA_CUDA_RUNTIME_BASE,
            artifact_directory="runtime-debs",
        )
    else:
        raise BuildContractError(f"unknown build candidate: {candidate}")
    return {
        "candidate": candidate,
        "manifest_sha256": expected_manifest_sha256,
        "artifact_count": len(entries),
    }


def validate_repository_build_assets(root: Path | None = None) -> list[dict[str, str]]:
    base = root or Path(__file__).resolve().parent
    return [validate_freetoken_assets(base), validate_llama_assets(base)]


if __name__ == "__main__":
    for result in validate_repository_build_assets():
        print(f"{result['candidate']}: {result['status']}")
