"""Pinned syv MTP requantization with exact-layer Hessian diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import sys

import torch
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gptq_utils import HessianError, dequant, gptq_quantize  # noqa: E402


GROUP = 128
LIN = [
    "mtp.fc",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
]


def append_diagnostic(path: str, record: dict[str, object]) -> None:
    if not path:
        raise RuntimeError("gptq_diagnostics_path_required")
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        os.write(descriptor, raw.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str]) -> int:
    source = argv[1].rstrip("/") + "/"
    destination = argv[2].rstrip("/") + "/"
    hessians_path = argv[3]
    bits = int(argv[argv.index("--bits") + 1]) if "--bits" in argv else 4
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    original = (
        argv[argv.index("--orig") + 1]
        if "--orig" in argv
        else os.path.join(repository, "models", "Qwen3.8-27B-W4A16-AutoRound")
    ).rstrip("/") + "/"
    diagnostics_path = os.environ.get("GPTQ_DIAGNOSTICS_PATH", "")

    os.makedirs(destination, exist_ok=True)
    for file_name in os.listdir(source):
        if file_name.startswith("model-0000") and file_name.endswith(".safetensors") and not os.path.exists(destination + file_name):
            os.link(source + file_name, destination + file_name)
    for file_name in ["tokenizer.json"]:
        if not os.path.exists(destination + file_name):
            os.link(source + file_name, destination + file_name)
    for file_name in [
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
        "quantization_config.json",
        "tokenizer_config.json",
        "config.json",
        "model.safetensors.index.json",
        "mtp_draft_vocab_ids.pt",
    ]:
        shutil.copy(source + file_name, destination + file_name)

    with safe_open(original + "model_extra_tensors.safetensors.bak-mtp", "pt") as handle:
        bf16 = {key: handle.get_tensor(key) for key in handle.keys()}
    tensors = {}
    with safe_open(source + "model_extra_tensors.safetensors", "pt") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            if any(key.startswith(module + ".") for module in LIN):
                continue
            tensors[key] = handle.get_tensor(key)

    hessians = torch.load(hessians_path)
    for module in LIN:
        weight = bf16[module + ".weight"].cuda()
        try:
            quantized, scale, diagnostics = gptq_quantize(
                weight,
                hessians[module].cuda(),
                bits=bits,
                group=GROUP,
                blocksize=GROUP,
                module_name=module,
            )
        except HessianError as exc:
            record = dict(exc.diagnostics)
            record["error_code"] = exc.code
            append_diagnostic(diagnostics_path, record)
            print("GPTQ_HESSIAN_DIAGNOSTIC " + json.dumps(record, sort_keys=True), flush=True)
            raise
        append_diagnostic(diagnostics_path, diagnostics)
        print("GPTQ_HESSIAN_DIAGNOSTIC " + json.dumps(diagnostics, sort_keys=True), flush=True)
        relative_error = ((dequant(quantized, scale) - weight.float()).norm() / weight.float().norm()).item()
        print(f"  GPTQ {module}: {tuple(weight.shape)} int{bits} rel err {relative_error:.4f}", flush=True)
        out_features, in_features = weight.shape
        tensors[module + ".weight_packed"] = pack_to_int32(quantized.cpu(), bits, packed_dim=1).contiguous()
        tensors[module + ".weight_scale"] = scale.to(torch.float16).cpu().contiguous()
        tensors[module + ".weight_shape"] = torch.tensor([out_features, in_features], dtype=torch.int64)
        del weight, quantized, scale
        torch.cuda.empty_cache()

    if os.path.exists(destination + "model_extra_tensors.safetensors"):
        os.remove(destination + "model_extra_tensors.safetensors")
    save_file(tensors, destination + "model_extra_tensors.safetensors", metadata=metadata or {"format": "pt"})
    with open(destination + "config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    group = config["quantization_config"]["config_groups"]["group_3"]
    assert group["targets"] == [r"re:^mtp\..*"], group["targets"]
    group["weights"]["num_bits"] = bits
    with open(destination + "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print("done", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
