"""Fail-closed contracts for the allowlisted acceleration experiments.

The checked-in manifests are source candidates, not live authorization.  Values
which can only be produced by the immutable package build remain ``null`` and
are listed in ``build_time_required``.  ``require_live=True`` rejects every
such unresolved value.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit


class ContractError(ValueError):
    """The candidate is outside the reviewed allowlist."""


ROOT = Path(__file__).resolve().parent
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

_DIRS = {
    "qwen38-vllm": "qwen38_vllm",
    "freetoken": "freetoken",
    "llama-dflash2-official": "llama_dflash2_official",
    "higgs-tts3": "higgs_tts3",
}

_RUNTIME_IDENTITIES = {
    "qwen38-vllm": {"uid": 0, "gid": 0},
    "freetoken": {"uid": 65532, "gid": 65532},
    "llama-dflash2-official": {"uid": 65532, "gid": 65532},
    "higgs-tts3": {"uid": 0, "gid": 0},
}

_SOURCES = {
    "qwen38-vllm": (
        "https://github.com/syv-ai/qwen38-27b-rtx3090",
        "69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c",
        "91d02477a91721be01350915534d1ebf882ea45f",
        "Apache-2.0",
    ),
    "freetoken": (
        "https://github.com/FlashML-org/FreeToken",
        "9db1a39455a3fb107f3db83e381d10ceadfe5d99",
        "ff3ff1de70209467631a63a8f441c239d03ff6d0",
        "Apache-2.0",
    ),
    "llama-dflash2-official": (
        "https://github.com/ggml-org/llama.cpp",
        "b10f9ca58c89ccfc3653ac01e979dd085d582b76",
        "4bd9becc238e24ef8a3740e2f4d938e38a3008b8",
        "MIT",
    ),
    "higgs-tts3": (
        "https://github.com/sgl-project/sglang-omni",
        "bdb6748882465e0b4c0b3298f95cb6e47adf5e10",
        "c11cb483ce143f036a2d546b84ce35381c1c6b7c",
        "Apache-2.0",
    ),
}

_SOURCE_ARCHIVES = {
    "qwen38-vllm": (7197512, "e92392376cf886c4fec1b326d5a4caf6b9596e26a5b816d26d5c14e81c8d60ea"),
    "freetoken": (1470833, "c98d43493f898703c905c9dc192d8b12c9362b07794d6de7865507fee2da3091"),
    "llama-dflash2-official": (37021763, "0ce1e39d49da65dfba450827f1e8b689b1d7f31721fba06d5bd298dc4177c773"),
    "higgs-tts3": (19743143, "b448536e274a904add75fcde6fee00794f099c97d80c65a29e7038b45664d421"),
}

_MODEL_IDENTITIES = {
    "qwen38-vllm": [
        ("target", "dbirks/Qwen3.8-27B-W4A16-AutoRound", "1f05c441c4e64ae0549de44fa9ea5a6d43610314"),
        ("drafter", "syvai/Qwen3.8-27B-DFlash2-W4A16", "4d30ec736ffc6b8688dc2ae2b502d9b48bdec279"),
        ("fast-variant", "syvai/qwen3.8-27b-3090-fast-variant", "124c14e7e8c7d2f5402933b9af368e772a9fcf0c"),
    ],
    "freetoken": [
        ("target", "openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
    ],
    "llama-dflash2-official": [
        ("target", "ggml-org/Qwen3.8-27B-GGUF", "0669b98607d47046c7c2b3f801011d54a08cfccf"),
        ("drafter", "incoai/Qwen3.8-27B-DFlash2-GGUF", "51962825493a48b846b40126d35c799ac4093ad0"),
    ],
    "higgs-tts3": [
        ("target", "bosonai/higgs-tts-3-4b", "7556c17e05201fccd9c8cc120bc216dcc7b5d561"),
    ],
}

_MODEL_LICENSE_URLS = {
    (candidate, role): (
        f"https://huggingface.co/{repository}/blob/{revision}/"
        + ("LICENSE" if candidate in {"freetoken", "higgs-tts3"} else "README.md")
    )
    for candidate, identities in _MODEL_IDENTITIES.items()
    for role, repository, revision in identities
}

_BLOBS = {
    ("qwen38-vllm", "target"): [
        ("chat_template.jinja", 8952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
        ("config.json", 15587, "3176e8ef1f8a6a11ca189436a8ad9245973d420a4eb5acca67be3fb0a5d2f98c"),
        ("generation_config.json", 219, "1a450b75a54bf9e6ffdc681d8b764d453061dd2d271df33fcfed46949b67c3de"),
        ("model-00001-of-00007.safetensors", 3211003472, "6b2adc4f1b93386eedd033a62c63221bf85e789c09ada8ff43939c55e9f06250"),
        ("model-00002-of-00007.safetensors", 3195027104, "a223fa487834749a14d86d558af9a4d5799b79be9bc28bdefa43eed055d10578"),
        ("model-00003-of-00007.safetensors", 3195027104, "8719567ac822768ac61872894604d3a4d13f2edbd22c9bd3c4680ebc83e24ca5"),
        ("model-00004-of-00007.safetensors", 3217442568, "8faf8d4a57c69b9b96e27c4eabf1ba0caadd3d48344d6f65df17f30da2a7e6e3"),
        ("model-00005-of-00007.safetensors", 699275200, "0994bb8d91c07ceefc4bfffd3a7b7fd110cc9bd3ce865bdcb2821abf02a02fab"),
        ("model-00006-of-00007.safetensors", 2542807272, "55a14ee79d3e5a65a8731d89426f4df477e8bdc7daa7976d41254d8afb9432f0"),
        ("model-00007-of-00007.safetensors", 2542796896, "6866cf8adcccc4cc6a00e74bc025f1a774fb52103b70f2674d0288272951a733"),
        ("model.safetensors.index.json", 198290, "43b30ffd1e2f940a08e2ea03f6039ad26008949af74158855beeec2b2cd79a86"),
        ("model_extra_tensors.safetensors", 849400392, "1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe"),
        ("processor_config.json", 1191, "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1"),
        ("quantization_config.json", 11112, "2f3122aa57dd2f35fda8e08f3e1c0ca4e16b72a6eda3f26a435455dfe7fd2d50"),
        ("tokenizer.json", 19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
        ("tokenizer_config.json", 1166, "5f7aa0d810000c4a0940b35c8f16e56c3439260fccc13a321826ebc3ccf2501f"),
    ],
    ("qwen38-vllm", "drafter"): [
        ("config.json", 2264, "61d6276fe8d76295232cb02d26cbb0d29c25565911f50441e779c88c9220c556"),
        ("model.safetensors", 1280633960, "ec26996e6a0745ab5edb857117220ce1e219ad524f71e6e149b703804947d8e7"),
    ],
    ("qwen38-vllm", "fast-variant"): [
        ("config.json", 17060, "5a3e7312dee3a6c5909beb854b05ac818be836683730e4facba81f547b77fd62"),
        ("draft_vocab_ids.json", 279256, "b64b6dfcf5441eb995ddf77d3d37b018e91b88c56ad1b4c5774ad8fbfac1c388"),
        ("model-00007-of-00007.safetensors", 655565120, "ac4511ddfbdcd8399dffee570c31708c13f974a18e46d3c337ca652a46a1a437"),
        ("model.safetensors.index.json", 200180, "0821f46e4a9888262e8f63d5a3c39e3be93e62fa3acd95febeb9a2c9bd9131a1"),
        ("model_extra_tensors.safetensors", 327162544, "3643128012845562d066540990eceb012e6dc9d1ba794c668d774a91edcd9dbb"),
        ("mtp_draft_vocab_ids.pt", 329341, "8af9028616e41932e0c949cd5364d2da0a69d15375af9bb404cab1e4d723f581"),
    ],
    ("freetoken", "target"): [
        ("chat_template.jinja", 16738, "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146"),
        ("config.json", 1806, "3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce"),
        ("generation_config.json", 177, "f9970ada892d2d1f72e3ed0a6535ccebadd11897318794ca671d8c7014c957da"),
        ("model-00000-of-00002.safetensors", 4792272488, "16d0f997dcfc4462089d536bffe51b4bcea2f872f5c430be09ef8ed392312427"),
        ("model-00001-of-00002.safetensors", 4798702184, "4fbe328ab445455d6f58dc73852b85873bd626986310abd91cd4d2ce3245eaea"),
        ("model-00002-of-00002.safetensors", 4170342232, "a18106b209e9ab35c3406db4f6f12a927364a058b21e9d1373d682e20674b303"),
        ("model.safetensors.index.json", 36355, "0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1"),
        ("special_tokens_map.json", 98, "dd5e191d20c12d2fee1da5bae14ca1db0f5f4215300af691f23cdee97120a293"),
        ("tokenizer.json", 27868174, "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3"),
        ("tokenizer_config.json", 4200, "9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070"),
    ],
    ("llama-dflash2-official", "target"): [
        ("Qwen3.8-27B-Q4_K_M.gguf", 18973870432, "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"),
    ],
    ("llama-dflash2-official", "drafter"): [
        ("Qwen3.8-27B-DFlash2-Q4_K_M.gguf", 1143006816, "1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd"),
    ],
    ("higgs-tts3", "target"): [
        (".gitattributes", 1570, "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930"),
        ("AGENTS.md", 9505, "dad259b0c375bcacf10de6a277354cc2e8b96c85d8e713f6a2dc61b1dfbca568"),
        ("LICENSE", 32184, "0382189330b87c38517530e4e5bf0c9dfeb6c7bfbe62812577ddcb73e09a60c5"),
        ("PROMPTING.md", 2812, "8e6a23d19eca73b0f0a302076607edb7447c1b2ddd48c382fcd509e3292369d6"),
        ("README.md", 46017, "8e20d72c9000a823a208ab42f9deec2f9ceef65260ce65fddca0cd343dcc23e2"),
        ("assets/model_architecture.png", 63078, "f03b182a50ddc9fb3d9ee7d5861c9d9243512471085fafd49acd29827dd03074"),
        ("chat_template.jinja", 2427, "44d5f08f3f72b837eaad09f13a54c1f9f4eb58d75240334548b7fd52a5437fa5"),
        ("config.json", 2755, "2ead4442c079ee35c2123a5b197e126e18eccfc0bdb65d31c94767e75d7864d4"),
        ("model.safetensors", 9309834930, "2f7965264c360b38180885006944aa16bd1de20f4e6cff79f6473bfcf8ae3d5a"),
        ("model.safetensors.index.json", 90103, "cf2689739828b9b77b7c09f66bc400d147541377f0b58fd6871f826cf4c16010"),
        ("tokenizer.json", 11433924, "eb883de2de5adc5113f1f02b54830a0ea7cd6ef191cde65c41aceb3737d4d1c1"),
        ("tokenizer_config.json", 1937, "b4d632e1239569fb1829bf0bfa3c674fa54f22c42e9cb2669d77c438d4b4e02c"),
    ],
}

_MODE_FIELDS = {
    "compose_file", "compose_sha256", "project_name", "service",
    "container_name", "endpoint", "argv", "readiness_path",
    "readiness_timeout_seconds", "auth_mode", "capabilities", "evaluator",
    "auth_file", "auth_file_mode",
}

_MODES = {
    "qwen38-vllm": {
        "mtp": ("compose.mtp.yaml", "11f99f2357ecfc8e3755dde735f068c15d2ae5d19b3d19191345e9f763d0a83a", "ai-exp-qwen38-vllm-mtp", "qwen38-vllm-mtp", "ai-exp-qwen38-vllm-mtp", "http://127.0.0.1:18020", ["single"], 900, "bearer", "generic-openai-v1"),
        "dflash2": ("compose.dflash2.yaml", "8ea998052ee3c36cf29b35941eb039860f83478a5a1b46a021c67f7ed0ea2a7c", "ai-exp-qwen38-vllm-dflash2", "qwen38-vllm-dflash2", "ai-exp-qwen38-vllm-dflash2", "http://127.0.0.1:18020", ["single"], 900, "bearer", "generic-openai-v1"),
    },
    "freetoken": {
        "serve": ("compose.yaml", "713cd9f02f54b81d719d355c348a2bf44f8a9fc373d06d1288b404001a9459e5", "ai-exp-freetoken-serve", "freetoken-serve", "ai-exp-freetoken-serve", "http://127.0.0.1:1919", ["ft", "serve", "--model", "/models/openai-gpt-oss-20b", "--served-model-name", "gpt-oss-20b", "--host", "0.0.0.0", "--port", "1919", "--max-running-requests", "1"], 900, "none_loopback", "generic-openai-v1"),
    },
    "llama-dflash2-official": {
        "plain": ("compose.plain.yaml", "a2ece009b0e2328ccf0aa4a358517f940a5e766ff14a5b5e83dfd63b57389131", "ai-exp-llama-dflash2-plain", "llama-dflash2-plain", "ai-exp-llama-dflash2-plain", "http://127.0.0.1:18021", ["llama-server", "--model", "/models/Qwen3.8-27B-Q4_K_M.gguf", "--host", "0.0.0.0", "--port", "18021", "--ctx-size", "4096", "--n-gpu-layers", "all", "--parallel", "1", "--api-key-file", "<redacted-secret-path>"], 600, "bearer", "llama-strict-v1"),
        "mtp": ("compose.mtp.yaml", "b6a3635b6f07969c00f913c84f34d71e6b1c64f718be0e6fa4f8aca74787901e", "ai-exp-llama-dflash2-mtp", "llama-dflash2-mtp", "ai-exp-llama-dflash2-mtp", "http://127.0.0.1:18021", ["llama-server", "--model", "/models/Qwen3.8-27B-Q4_K_M.gguf", "--host", "0.0.0.0", "--port", "18021", "--ctx-size", "4096", "--n-gpu-layers", "all", "--parallel", "1", "--spec-type", "draft-mtp", "--api-key-file", "<redacted-secret-path>"], 600, "bearer", "llama-strict-v1"),
        "dflash2-n4": ("compose.dflash2.yaml", "ea7b5520444f55d910df39534ca8d154698d05b762fb5ba8ab45d1c243696511", "ai-exp-llama-dflash2-dflash2", "llama-dflash2-n4", "ai-exp-llama-dflash2-n4", "http://127.0.0.1:18021", ["llama-server", "--model", "/models/Qwen3.8-27B-Q4_K_M.gguf", "--model-draft", "/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf", "--host", "0.0.0.0", "--port", "18021", "--ctx-size", "4096", "--n-gpu-layers", "all", "--parallel", "1", "--spec-type", "draft-dflash", "--spec-draft-n-max", "4", "--api-key-file", "<redacted-secret-path>"], 600, "bearer", "llama-strict-v1"),
        "dflash2-n7": ("compose.dflash2.yaml", "ea7b5520444f55d910df39534ca8d154698d05b762fb5ba8ab45d1c243696511", "ai-exp-llama-dflash2-dflash2", "llama-dflash2-n7", "ai-exp-llama-dflash2-n7", "http://127.0.0.1:18021", ["llama-server", "--model", "/models/Qwen3.8-27B-Q4_K_M.gguf", "--model-draft", "/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf", "--host", "0.0.0.0", "--port", "18021", "--ctx-size", "4096", "--n-gpu-layers", "all", "--parallel", "1", "--spec-type", "draft-dflash", "--spec-draft-n-max", "7", "--api-key-file", "<redacted-secret-path>"], 600, "bearer", "llama-strict-v1"),
    },
    "higgs-tts3": {
        "tts": ("compose.yaml", "6b7181a54fd2537f97e156e09cc16b7e087be98360a7fd1d1a73c5eb90e0d20e", "ai-exp-higgs-tts3", "higgs-tts3", "ai-exp-higgs-tts3", "http://127.0.0.1:18095", ["python", "-m", "sglang_omni.cli", "serve", "--config", "/config/higgs-3090.yaml", "--host", "0.0.0.0", "--port", "8000"], 900, "none_loopback", "higgs-russian-tts-smoke-v1"),
    },
}

_BUILD_FIELDS = {
    "qwen38-vllm": set(),
    "freetoken": {"image.reference", "image.digest", "image.lock_sha256"},
    "llama-dflash2-official": {"image.reference", "image.digest", "image.lock_sha256"},
    "higgs-tts3": set(),
}

_LOCAL_IMAGE_REFERENCES = {
    "freetoken": "local/ai-server-freetoken:9db1a39455a3fb107f3db83e381d10ceadfe5d99",
    "llama-dflash2-official": "local/ai-server-llama-dflash2-official:b10f9ca58c89ccfc3653ac01e979dd085d582b76",
}

_QUALITY_SUITE_SHA256 = "50451fcd5d850c9d94a08d2bd8466baca737ff1bac6ea7821e63eddb7200fb48"
_FREETOKEN_BANDWIDTH_PROFILE_SHA256 = "78105382dcc5cd2b0b2e0ed03b7224a2042f94c336bcd140b1cde874297bf688"
_MODEL_MANIFESTS = {
    ("qwen38-vllm", "target"): ("qwen38-vllm.target.json", "1b19a6fb7a4cf7043f024dda30201af763a11db7b02b6cdda5c9d06c418a14cf"),
    ("qwen38-vllm", "drafter"): ("qwen38-vllm.drafter.json", "123e5ac7f250f4552f0e5412a8beda01cfb7dfc3333d17895e895a579e88398c"),
    ("qwen38-vllm", "fast-variant"): ("qwen38-vllm.fast-variant.json", "4dbdb9dbf55976d6b4d3fda56e1ad04d460c6c0d7b5c5f1e4a5703c954f1545a"),
    ("freetoken", "target"): ("freetoken.target.json", "e5ab8081b4e957763cdf28d1c09148a255353c059cb3440c03cb35bb4679ad93"),
    ("llama-dflash2-official", "target"): ("llama-dflash2-official.target.json", "4b57e317e3f8931bf834e1207111040e3ac44bf8eab3ebdb2c8c5c72008fb26a"),
    ("llama-dflash2-official", "drafter"): ("llama-dflash2-official.drafter.json", "368dc79a912258dc051899f4a953f6aa0001ef1872e91a7f63042bee7e8266e9"),
    ("higgs-tts3", "target"): ("higgs-tts3.target.json", "a7a59f2d347c94b71791bf4ce5117a328bca66b352519b902e1e9fc7d59eb9eb"),
}
_SYV_IMAGE_LOCK = ("qwen38-vllm.oci-lock.json", "707fac2a24fe546eb7dedbdea3853fecba0e00086bd0c418054e9551f83ae22d")
_HIGGS_IMAGE_RECEIPT = ("oci-layer-receipt.json", "a808ab7f9011764aa9353566dca8b864df9bb53e13bfb2ad1fe4ec4790404d45")
_HIGGS_SMOKE_SUITE = ("higgs-tts3-russian-smoke.json", "3b9e708565e4a6deb8605ee87626a1f8f492c7845bda0d9ba26c7b7921fde84f")


def _fail(message: str) -> None:
    raise ContractError(message)


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _canonical_hf_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or any(ord(char) < 32 for char in value):
        _fail("Hugging Face blob path is not a canonical relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        _fail("Hugging Face blob path is not a canonical relative path")
    return path


def huggingface_resolve_url(repository: str, revision: str, filename: str) -> str:
    """Derive the immutable upstream URL for one validated model-tree path."""

    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None or _HEX40.fullmatch(revision) is None:
        _fail("Hugging Face repository or revision is not immutable")
    path = _canonical_hf_path(filename)
    encoded_path = "/".join(quote(part, safe="") for part in path.parts)
    return f"https://huggingface.co/{repository}/resolve/{revision}/{encoded_path}"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate immutable metadata key: {key}")
        result[key] = value
    return result


def _load_immutable_metadata(experiments_root: Path, relative: Path) -> dict[str, Any]:
    unresolved = experiments_root / "immutable_metadata" / relative
    if unresolved.is_symlink():
        _fail("immutable metadata must not be a symlink")
    path = unresolved.resolve()
    try:
        path.relative_to(experiments_root)
    except ValueError:
        _fail("immutable metadata path escapes experiments root")
    if not path.is_file():
        _fail(f"missing immutable metadata: {relative.as_posix()}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load immutable metadata: {exc}") from exc
    if not isinstance(value, dict):
        _fail("immutable metadata root must be an object")

    def reject_circular(item: Any) -> None:
        if isinstance(item, dict):
            if "manifest_sha256" in item or "lock_sha256" in item:
                _fail("immutable metadata must not contain its own candidate digest")
            for child in item.values():
                reject_circular(child)
        elif isinstance(item, list):
            for child in item:
                reject_circular(child)

    reject_circular(value)
    return value


def _validate_model_manifest(experiments_root: Path, candidate_id: str, model: dict[str, Any]) -> None:
    key = (candidate_id, model["role"])
    if key not in _MODEL_MANIFESTS:
        return
    filename, expected_sha256 = _MODEL_MANIFESTS[key]
    if model.get("manifest_sha256") != expected_sha256:
        _fail(f"model manifest digest is not allowlisted for {candidate_id}/{model['role']}")
    manifest = _load_immutable_metadata(experiments_root, Path("models") / filename)
    if _canonical_sha256(manifest) != expected_sha256:
        _fail(f"model manifest hash mismatch for {candidate_id}/{model['role']}")
    expected = {
        "schema_version": 1,
        "kind": "huggingface-model-manifest",
        "repository": model["repository"],
        "revision": model["revision"],
        "license": model["license"],
        "blobs": model["blobs"],
    }
    if "overlay" in model:
        expected["overlay"] = model["overlay"]
    if manifest != expected:
        _fail(f"model manifest content mismatch for {candidate_id}/{model['role']}")


def _validate_syv_image_lock(experiments_root: Path, image: dict[str, Any]) -> None:
    filename, expected_sha256 = _SYV_IMAGE_LOCK
    if image.get("lock_sha256") != expected_sha256:
        _fail("syv OCI metadata lock digest is not allowlisted")
    lock = _load_immutable_metadata(experiments_root, Path("images") / filename)
    if _canonical_sha256(lock) != expected_sha256:
        _fail("syv OCI metadata lock hash mismatch")
    if set(lock) != {"schema_version", "kind", "registry", "repository", "manifest", "config_metadata"}:
        _fail("unexpected syv OCI metadata lock schema")
    if (
        lock.get("schema_version") != 1
        or lock.get("kind") != "oci-image-metadata-lock"
        or lock.get("registry") != "ghcr.io"
        or lock.get("repository") != "syv-ai/qwen38-27b-rtx3090"
    ):
        _fail("unexpected syv OCI metadata lock identity")
    manifest = lock.get("manifest")
    if not isinstance(manifest, dict) or set(manifest) != {"media_type", "digest", "size", "config", "layers"}:
        _fail("unexpected syv OCI manifest metadata")
    if (
        manifest.get("media_type") != "application/vnd.oci.image.manifest.v1+json"
        or manifest.get("digest") != image.get("digest")
        or manifest.get("size") != 2773
    ):
        _fail("syv OCI manifest identity mismatch")
    config = manifest.get("config")
    expected_config = {
        "media_type": "application/vnd.oci.image.config.v1+json",
        "digest": "sha256:b0516eab4a689391cd73629593d53111777691c5f3c453d5064e77fe55981101",
        "size": 14631,
    }
    if config != expected_config:
        _fail("syv OCI config descriptor mismatch")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != 13:
        _fail("unexpected syv OCI layer descriptor set")
    for layer in layers:
        if (
            not isinstance(layer, dict)
            or set(layer) != {"media_type", "digest", "size"}
            or layer.get("media_type") != "application/vnd.oci.image.layer.v1.tar+gzip"
            or not _DIGEST.fullmatch(str(layer.get("digest", "")))
            or not isinstance(layer.get("size"), int)
            or layer["size"] <= 0
        ):
            _fail("invalid syv OCI layer descriptor")
    metadata = lock.get("config_metadata")
    if not isinstance(metadata, dict) or set(metadata) != {"architecture", "os", "created", "entrypoint", "cmd", "working_dir", "exposed_ports", "labels"}:
        _fail("unexpected syv OCI config metadata schema")
    expected_labels = {
        "org.opencontainers.image.created": "2026-08-27T08:52:29.474Z",
        "org.opencontainers.image.licenses": "Apache-2.0",
        "org.opencontainers.image.revision": "69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c",
        "org.opencontainers.image.source": "https://github.com/syv-ai/qwen38-27b-rtx3090",
        "org.opencontainers.image.title": "qwen38-27b-rtx3090",
    }
    if metadata != {
        "architecture": "amd64",
        "os": "linux",
        "created": "2026-08-27T08:55:40.479541114Z",
        "entrypoint": ["bash", "docker/entrypoint.sh"],
        "cmd": ["single"],
        "working_dir": "/app",
        "exposed_ports": ["18020/tcp"],
        "labels": expected_labels,
    }:
        _fail("syv OCI config metadata mismatch")


def _validate_higgs_image_receipt(experiments_root: Path, image: dict[str, Any]) -> None:
    filename, expected_sha256 = _HIGGS_IMAGE_RECEIPT
    path = experiments_root / _DIRS["higgs-tts3"] / filename
    if path.is_symlink() or not path.is_file():
        _fail("Higgs OCI receipt is missing")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256 or image.get("lock_sha256") != expected_sha256:
        _fail("Higgs OCI receipt hash mismatch")
    try:
        receipt = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load Higgs OCI receipt: {exc}") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "ai-server-oci-layer-receipt/v1"
        or receipt.get("repository") != "lmsysorg/sglang-omni"
        or receipt.get("manifest_digest") != image.get("digest")
        or receipt.get("config_digest") != "sha256:803f7cbaf1b859458302348680ef35ae0e458a0ff9193d2d1a622a8f8f50cffe"
        or receipt.get("manifest_bytes") != 11046
        or receipt.get("config_bytes") != 66364
        or receipt.get("layer_count") != 66
        or receipt.get("unique_layer_count") != 62
    ):
        _fail("Higgs OCI receipt identity mismatch")


def _value_at(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if isinstance(value, list):
            if not part.isdigit() or int(part) >= len(value):
                _fail(f"invalid build-time field path: {dotted}")
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            _fail(f"invalid build-time field path: {dotted}")
    return value


def _validate_build_value(candidate_id: str, path: str, value: Any) -> None:
    if path == "wheel.filename":
        if not isinstance(value, str) or Path(value).name != value or not value.endswith(".whl"):
            _fail("resolved wheel.filename must be a contained wheel basename")
    elif path == "image.reference":
        expected_local = _LOCAL_IMAGE_REFERENCES.get(candidate_id)
        if expected_local is not None:
            if value != expected_local:
                _fail("resolved local image.reference is not allowlisted")
        elif not isinstance(value, str) or "@sha256:" not in value or ":latest@" in value.lower() or "/main@" in value.lower():
            _fail("resolved image.reference must contain an immutable digest")
    elif path == "image.digest":
        if not isinstance(value, str) or not _DIGEST.fullmatch(value):
            _fail("resolved image.digest must be an OCI SHA256 digest")
    elif not isinstance(value, str) or not _SHA256.fullmatch(value):
        _fail(f"resolved {path} must be a lowercase SHA256")


def _validate_compose(candidate_id: str, base: Path, mode: dict[str, Any], image_reference: Any, require_live: bool) -> None:
    filename = mode["compose_file"]
    if Path(filename).name != filename or not filename.endswith((".yaml", ".yml")):
        _fail("compose file must be a contained basename")
    path = base / filename
    if path.is_symlink() or not path.is_file():
        _fail(f"missing compose file: {filename}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != mode["compose_sha256"]:
        _fail(f"compose hash mismatch: {filename}")
    text = raw.decode("utf-8")
    lower = text.lower()
    forbidden = ("${", ": latest", ":latest", "/var/run/docker.sock", "/srv/ai-server/model-store", "0.0.0.0:", "rm -rf", "/dev/shm/*")
    if any(token in lower for token in forbidden):
        _fail(f"unsafe or mutable compose content: {filename}")
    if "restart: \"no\"" not in text or "read_only: true" not in text:
        _fail(f"compose service is not fail-closed: {filename}")
    images = {match.group(1).strip('"\'') for match in re.finditer(r"(?m)^\s*image:\s*(\S+)\s*$", text)}
    service_count = len(re.findall(r"(?m)^\s*image:\s*\S+\s*$", text))
    expected_user = f'{_RUNTIME_IDENTITIES[candidate_id]["uid"]}:{_RUNTIME_IDENTITIES[candidate_id]["gid"]}'
    users = [match.group(1).strip('"\'') for match in re.finditer(r"(?m)^\s*user:\s*(\S+)\s*$", text)]
    if service_count == 0 or users != [expected_user] * service_count:
        _fail(f"compose runtime identity does not match every service: {filename}")
    unresolved_image = {
        "freetoken": "build-time-required.invalid/freetoken",
        "llama-dflash2-official": "build-time-required.invalid/llama-dflash2",
    }.get(candidate_id)
    expected_image = image_reference if image_reference is not None else unresolved_image
    if images != {expected_image}:
        _fail(f"compose image does not match candidate identity: {filename}")
    for line in text.splitlines():
        stripped = line.strip().strip("- ").strip('"').strip("'")
        if re.match(r"^(?:[0-9.]+):\d+:\d+$", stripped) and not stripped.startswith("127.0.0.1:"):
            _fail(f"non-loopback published port: {filename}")
    if "build-time-required.invalid/" in text and require_live:
        _fail(f"unresolved build-time image in {filename}")
    if "ipc: host" in text:
        _fail("host IPC is forbidden for every candidate")
    if candidate_id == "qwen38-vllm":
        if "ipc: private" not in text or 'shm_size: "16gb"' not in text or 'VLLM_OFFLOAD_KEEP_SHM: "1"' not in text:
            _fail("syv candidate requires reviewed bounded private IPC")
    if candidate_id == "higgs-tts3" and (
        "ipc: private" not in text
        or "shm_size: 4gb" not in text
        or "cap_drop:\n      - ALL" not in text
        or "no-new-privileges:true" not in text
        or "127.0.0.1:18095:8000" not in text
        or "TRITON_CACHE_DIR: /runtime-cache" not in text
        or "FLASHINFER_WORKSPACE_BASE: /runtime-cache" not in text
        or "SPEAKER_SAMPLES_DIR: /runtime-cache/sglang-omni/speakers" not in text
        or re.findall(r"(?m)^\s*-\s+(/runtime-cache:[^\r\n]+)$", text)
        != ["/runtime-cache:rw,nosuid,nodev,exec,size=2g"]
    ):
        _fail("Higgs candidate requires the reviewed private loopback sandbox")


def validate_candidate_contract(
    candidate_id: str,
    mode: str,
    *,
    root: Path | str | None = None,
    require_live: bool = False,
) -> dict[str, Any]:
    """Load and validate one allowlisted candidate/mode.

    No caller-supplied image, argv, environment, path, URL, or Compose arguments
    are accepted.  The returned object is safe only for the phase requested;
    callers which can mutate Server A must pass ``require_live=True``.
    """

    if candidate_id not in _DIRS:
        _fail(f"candidate is not allowlisted: {candidate_id}")
    if mode not in _MODES[candidate_id]:
        _fail(f"mode is not allowlisted for {candidate_id}: {mode}")
    experiments_root = Path(root).resolve() if root is not None else ROOT
    unresolved_base = experiments_root / _DIRS[candidate_id]
    if unresolved_base.is_symlink():
        _fail("candidate directory must not be a symlink")
    base = unresolved_base.resolve()
    try:
        base.relative_to(experiments_root)
    except ValueError:
        _fail("candidate path escapes experiments root")
    manifest_path = base / "candidate.json"
    if manifest_path.is_symlink():
        _fail("candidate manifest must not be a symlink")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load candidate manifest: {exc}") from exc

    common_keys = {"schema_version", "candidate_id", "source", "image", "models", "runtime_identity", "storage_root", "writable_cas", "ipc", "modes", "benchmark", "clean_stop", "restore", "build_time_required"}
    expected_keys = common_keys | ({"wheel", "build_inputs"} if candidate_id == "freetoken" else set())
    if set(data) != expected_keys or data.get("schema_version") != 1 or data.get("candidate_id") != candidate_id:
        _fail("unexpected manifest schema or candidate identity")
    if data.get("writable_cas") is not False:
        _fail("candidate must not have writable production CAS")
    if data.get("runtime_identity") != _RUNTIME_IDENTITIES[candidate_id]:
        _fail("candidate runtime identity is not allowlisted")
    root_value = data.get("storage_root")
    storage_pin = (
        _MODEL_IDENTITIES[candidate_id][0][2]
        if candidate_id == "higgs-tts3"
        else _SOURCES[candidate_id][1]
    )
    expected_storage_root = f"/srv/ai-server/staging/experiments/{candidate_id}/{storage_pin}"
    if root_value != expected_storage_root or ".." in PurePosixPath(str(root_value)).parts:
        _fail("storage root is outside candidate staging")

    source = data.get("source")
    if not isinstance(source, dict) or set(source) != {"repository", "revision", "tree", "archive_sha256", "archive_size", "license"}:
        _fail("unexpected source schema")
    repo, revision, tree, source_license = _SOURCES[candidate_id]
    if (source.get("repository"), source.get("revision"), source.get("tree")) != (repo, revision, tree):
        _fail("source pin is not allowlisted")
    if not _HEX40.fullmatch(revision) or not _HEX40.fullmatch(tree):
        _fail("source revision/tree is not an exact Git identity")
    if (source.get("archive_size"), source.get("archive_sha256")) != _SOURCE_ARCHIVES[candidate_id]:
        _fail("source archive size/hash is not allowlisted")
    license_data = source.get("license")
    if not isinstance(license_data, dict) or set(license_data) != {"id", "url"} or license_data.get("id") != source_license or not str(license_data.get("url", "")).startswith(repo + "/blob/" + revision + "/"):
        _fail("source license is not allowlisted")

    image = data.get("image")
    if not isinstance(image, dict) or set(image) != {"reference", "digest", "revision_label", "lock_sha256"} or image.get("revision_label") != revision:
        _fail("unexpected image contract")
    if candidate_id == "qwen38-vllm":
        expected_digest = "sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0"
        expected_ref = "ghcr.io/syv-ai/qwen38-27b-rtx3090:sha-69ba4d0@" + expected_digest
        if image.get("reference") != expected_ref or image.get("digest") != expected_digest:
            _fail("syv OCI image is not the reviewed immutable image")
        _validate_syv_image_lock(experiments_root, image)
    elif candidate_id == "higgs-tts3":
        expected_digest = "sha256:46235435997d1fa93fc81fb1c2d5b7fd8470d77395a5c348c0176094ffddf95e"
        expected_ref = "lmsysorg/sglang-omni@" + expected_digest
        if image.get("reference") != expected_ref or image.get("digest") != expected_digest:
            _fail("Higgs OCI image is not the reviewed immutable image")
        _validate_higgs_image_receipt(experiments_root, image)
    elif image.get("reference") is not None and image.get("reference") != _LOCAL_IMAGE_REFERENCES[candidate_id]:
        _fail("local candidate image reference is not allowlisted")
    elif image.get("digest") is not None and not _DIGEST.fullmatch(str(image["digest"])):
        _fail("image digest is invalid")
    if candidate_id == "qwen38-vllm" and isinstance(image.get("reference"), str) and isinstance(image.get("digest"), str) and not image["reference"].endswith("@" + image["digest"]):
        _fail("image reference and digest disagree")

    models = data.get("models")
    if not isinstance(models, list) or len(models) != len(_MODEL_IDENTITIES[candidate_id]):
        _fail("unexpected model set")
    for index, (model, expected_identity) in enumerate(zip(models, _MODEL_IDENTITIES[candidate_id], strict=True)):
        model_keys = {"role", "repository", "revision", "manifest_sha256", "license", "blobs"}
        if candidate_id == "qwen38-vllm" and index == 2:
            model_keys.add("overlay")
        if not isinstance(model, dict) or set(model) != model_keys:
            _fail(f"unexpected model schema at index {index}")
        if (model.get("role"), model.get("repository"), model.get("revision")) != expected_identity:
            _fail(f"model identity is not allowlisted at index {index}")
        model_license = model.get("license")
        if (
            not isinstance(model_license, dict)
            or set(model_license) != {"id", "url"}
            or model_license.get("id") != (
                "LicenseRef-BosonAI" if candidate_id == "higgs-tts3" else "Apache-2.0"
            )
            or model_license.get("url") != _MODEL_LICENSE_URLS[(candidate_id, model["role"])]
        ):
            _fail(f"model license is not allowlisted at index {index}")
        actual_blobs = []
        for blob in model.get("blobs", []):
            if not isinstance(blob, dict) or set(blob) != {"filename", "size", "sha256"} or not isinstance(blob.get("size"), int) or blob["size"] <= 0 or not _SHA256.fullmatch(str(blob.get("sha256", ""))):
                _fail(f"invalid model blob at index {index}")
            _canonical_hf_path(blob.get("filename"))
            huggingface_resolve_url(model["repository"], model["revision"], blob["filename"])
            actual_blobs.append((blob["filename"], blob["size"], blob["sha256"]))
        if [blob[0] for blob in actual_blobs] != sorted({blob[0] for blob in actual_blobs}):
            _fail(f"model blob paths must be unique and sorted at index {index}")
        if actual_blobs != _BLOBS[(candidate_id, model["role"])]:
            _fail(f"model blob inventory is not allowlisted at index {index}")
        _validate_model_manifest(experiments_root, candidate_id, model)

    if candidate_id == "qwen38-vllm":
        base_model, overlay_model = models[0], models[2]
        overlay = overlay_model.get("overlay")
        expected_overlay = {
            "base_manifest_sha256": base_model["manifest_sha256"],
            "replaces": [
                "config.json",
                "model-00007-of-00007.safetensors",
                "model.safetensors.index.json",
                "model_extra_tensors.safetensors",
            ],
            "adds": ["draft_vocab_ids.json", "mtp_draft_vocab_ids.pt"],
        }
        if overlay != expected_overlay:
            _fail("qwen fast variant overlay contract is not allowlisted")
        replacements = overlay["replaces"]
        additions = overlay["adds"]
        if replacements != sorted(replacements) or additions != sorted(additions) or set(replacements) & set(additions):
            _fail("qwen fast variant overlay paths must be sorted and disjoint")
        base_paths = {blob["filename"] for blob in base_model["blobs"]}
        overlay_paths = {blob["filename"] for blob in overlay_model["blobs"]}
        if not set(replacements) <= base_paths or set(additions) & base_paths or set(replacements) | set(additions) != overlay_paths:
            _fail("qwen fast variant overlay does not exactly cover replacements and additions")

    if candidate_id == "freetoken":
        wheel = data.get("wheel")
        expected_wheel = {
            "filename": "freetoken-0.1.2-cp312-cp312-manylinux_2_27_x86_64.whl",
            "url": "https://github.com/FlashML-org/FreeToken/releases/download/v0.1.2/freetoken-0.1.2-cp312-cp312-manylinux_2_27_x86_64.whl",
            "size": 1238381,
            "sha256": "993afeb4ef1ee3a1c5302b3c46dea6545d86f4a6facc3ea386985e02f4466a2f",
        }
        if wheel != expected_wheel:
            _fail("FreeToken wheel is not allowlisted")
        build_inputs = data.get("build_inputs")
        expected_build_inputs = {
            "python_standalone": {
                "filename": "cpython-3.12.14+20260825-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
                "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20260825/cpython-3.12.14%2B20260825-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
                "size": 34130340,
                "sha256": "7ce4a71285d913955a76053cc7605ea96da8ecada54dba9cf395245961816421",
            },
            "requirements_lock": {"filename": "requirements.lock", "sha256": "89e57a4f6efae127c1922a9c60436a0fcf656e64b3ec32da95613650b98c4981"},
            "requirements_offline_lock": {"filename": "requirements.offline.lock", "sha256": "c5a8478d127dc3c8bf268956d6b09f1c552daa25a495c0f395fc9e53096be441"},
            "artifact_manifest": {"filename": "artifact-manifest.json", "sha256": "f69f08f96f59a208a462f8d35c1f347e556234a9eaeedaf863c29c5a466581ed"},
            "resolver_receipt": {"filename": "resolver-receipt.json", "sha256": "bd0d5d065806fd0c7326d5a7f354adba7949d4671ab3a151c1fb761e1b69e45f"},
            "dockerfile": {"filename": "Dockerfile", "sha256": "ca71914bc19882999276958eae752eb5fd6424e0aa2adf2c15ac328a9a2fd622"},
            "oci_receipt_schema": {"filename": "oci-build-receipt.schema.json", "sha256": "5e0050007d4e7b4c4a13079406b38d624d89741653f02c8798cf9be31c8af44e"},
        }
        if build_inputs != expected_build_inputs:
            _fail("FreeToken build inputs are not the sealed allowlist")
    ipc = data.get("ipc")
    expected_ipc = {"mode": "private", "shm_size_bytes": 17179869184, "cleanup": "container_scoped"} if candidate_id == "qwen38-vllm" else {"mode": "private", "cleanup": "none", "created_names": []}
    if ipc != expected_ipc:
        _fail("IPC or cleanup contract is not allowlisted")
    benchmark = data.get("benchmark")
    expected_benchmark_keys = {"suite", "suite_sha256", "bandwidth_profile_sha256"} if candidate_id == "freetoken" else {"suite", "suite_sha256"}
    expected_suite = "higgs-russian-tts-smoke-v1" if candidate_id == "higgs-tts3" else "acceleration-quality-v1"
    if not isinstance(benchmark, dict) or set(benchmark) != expected_benchmark_keys or benchmark.get("suite") != expected_suite:
        _fail("benchmark contract is not allowlisted")
    expected_suite_sha256 = _HIGGS_SMOKE_SUITE[1] if candidate_id == "higgs-tts3" else _QUALITY_SUITE_SHA256
    if benchmark.get("suite_sha256") != expected_suite_sha256:
        _fail("benchmark suite digest is not the canonical acceleration-quality-v1 digest")
    if candidate_id == "higgs-tts3":
        suite = _load_immutable_metadata(
            experiments_root, Path("suites") / _HIGGS_SMOKE_SUITE[0]
        )
        if _canonical_sha256(suite) != _HIGGS_SMOKE_SUITE[1]:
            _fail("Higgs Russian smoke suite hash mismatch")
    if candidate_id == "freetoken":
        if benchmark.get("bandwidth_profile_sha256") != _FREETOKEN_BANDWIDTH_PROFILE_SHA256:
            _fail("FreeToken bandwidth profile digest is not allowlisted")
        profile_path = experiments_root.parent / "benchmarks" / "suites" / "acceleration-freetoken-bandwidth-v1.json"
        if profile_path.is_symlink() or not profile_path.is_file():
            _fail("FreeToken bandwidth profile is missing")
        try:
            profile_value = json.loads(
                profile_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot load FreeToken bandwidth profile: {exc}") from exc
        if _canonical_sha256(profile_value) != _FREETOKEN_BANDWIDTH_PROFILE_SHA256:
            _fail("FreeToken bandwidth profile hash mismatch")

    modes = data.get("modes")
    if not isinstance(modes, dict) or set(modes) != set(_MODES[candidate_id]):
        _fail("manifest mode set does not match allowlist")
    for mode_name, mode_data in modes.items():
        if not isinstance(mode_data, dict) or set(mode_data) != _MODE_FIELDS:
            _fail(f"unexpected mode schema: {mode_name}")
        expected = _MODES[candidate_id][mode_name]
        projection = (
            mode_data["compose_file"], mode_data["compose_sha256"], mode_data["project_name"],
            mode_data["service"], mode_data["container_name"], mode_data["endpoint"],
            mode_data["argv"], mode_data["readiness_timeout_seconds"], mode_data["auth_mode"],
            mode_data["evaluator"],
        )
        expected_capabilities = ["audio"] if candidate_id == "higgs-tts3" else ["text", "tools"]
        if projection != expected or mode_data["readiness_path"] != "/health" or mode_data["capabilities"] != expected_capabilities:
            _fail(f"mode values do not match the reviewed allowlist: {mode_name}")
        parsed_endpoint = urlsplit(mode_data["endpoint"])
        if parsed_endpoint.scheme != "http" or parsed_endpoint.hostname != "127.0.0.1" or parsed_endpoint.path not in ("", "/") or parsed_endpoint.username or parsed_endpoint.password or parsed_endpoint.query or parsed_endpoint.fragment:
            _fail("candidate endpoint must be literal loopback HTTP")
        if mode_data["auth_mode"] == "none_loopback" and candidate_id not in {"freetoken", "higgs-tts3"}:
            _fail("none_loopback is not allowlisted for this candidate")
        if mode_data["auth_mode"] not in {"bearer", "none_loopback"}:
            _fail("unsupported auth mode")
        if mode_data["auth_mode"] == "bearer":
            expected_secret_mode = 0o600 if candidate_id == "qwen38-vllm" else 0o440
            if mode_data["auth_file"] != root_value + "<redacted-secret-path>" or mode_data["auth_file_mode"] != expected_secret_mode:
                _fail("bearer auth must use the exact candidate-owned sealed mode")
        elif mode_data["auth_file"] is not None or mode_data["auth_file_mode"] is not None:
            _fail("none_loopback must not declare an auth secret")
        _validate_compose(candidate_id, base, mode_data, image.get("reference"), False)

    expected_build_fields = _BUILD_FIELDS[candidate_id]
    if set(data.get("build_time_required", [])) != expected_build_fields:
        _fail("build-time field set does not match allowlist")
    unresolved = []
    for field in sorted(expected_build_fields):
        value = _value_at(data, field)
        if value is None:
            unresolved.append(field)
        else:
            _validate_build_value(candidate_id, field, value)
    if require_live and unresolved:
        _fail("live contract has unresolved build-time fields: " + ", ".join(unresolved))
    if require_live:
        for mode_data in modes.values():
            _validate_compose(candidate_id, base, mode_data, image.get("reference"), True)

    if data.get("clean_stop") != "compose_down_then_gpu_idle" or data.get("restore") != "exact_production_qwen_then_two_authenticated_completions":
        _fail("clean-stop or restore contract is not allowlisted")
    return data
