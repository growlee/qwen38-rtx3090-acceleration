# Server A snapshot — 2 November 2026

Everything about the text Qwen3.8 deployment on Server A (one RTX 3090) that existed
outside this repository, collected read-only before Unsloth Studio was retired from
Server A, so later tests can be replayed there. No weights, container images,
credentials, user prompts or chat history are included. Benchmark prompts are synthetic or
taken from public suites: HumanEval items keep their license (`benchmarks/*/HumanEval-LICENSE`);
the `calculator` and `voxel` tasks in `benchmarks/qwen-multiturn-context-20260910/` are adapted
from ByteDance Web-Bench (`bytedance/web-bench@7b31ca2b`, Apache-2.0, see that run's
`manifest.json` → `upstream`), and its context filler is `playwright-core@1.58.2/types/types.d.ts`
(Apache-2.0, © Microsoft, license header kept).

## What ran

Request path on Server A: Unsloth Studio provider `server-a-qwen3-8-27b`
(`llama_cpp` type, `http://172.21.250.80:8000/v1`, model `qwen3.8-27b-uncensored`) →
container `ai-server-unsloth-qwen-bridge` (image `sha256:69437c88…`, code in
[`qwen_bridge/`](qwen_bridge/), `bridge.py` sha256 `3351d8c7…`) → `ai-server-gateway:8000`
→ `ai-server-qwen` (vLLM, started on demand under the host's GPU admission).

- Runtime image: `ghcr.io/syv-ai/qwen38-27b-rtx3090@sha256:5b14f62188d90d394b6e517bc76b964b511cde81fbb572ac5a87eaf41f5ed9c0`
  (tag `sha-69ba4d0`, 14.7 GB, CUDA 13.0.1).
- Live compose: [`deploy/compose.qwen.live.yaml`](deploy/compose.qwen.live.yaml) (sha256 `4efeb1e0…`):
  `MAX_LEN=57344`, `GPU_UTIL=0.93`, `SPEC=mtp`, `CTX=fast`, `PREFIX_CACHE=0`, `VISION=1`,
  `VISION_OFFLOAD=1`, `--served-model-name qwen3.8-27b-uncensored`. The API key is read from a
  Docker secret file; the file itself is not here. `deploy/compose.qwen.llama.yaml` and
  `deploy/compose.qwen.syv.yaml` are the llama.cpp and syv candidates from the home-lab working copy.
- Studio chat settings: `maxTokens` 31296, reasoning on, effort `medium`.

## Weights (identified, not copied)

- `/srv/ai-server/artifacts/extensions/qwen-syv-k4-4d4166b891489bfc/model/` — 15 GiB, 7 safetensors
  shards. Source `leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound` at revision
  `f958acb153eb8c7bbd25638149c8c75185559226`; tensor inventory sha256 `4d4166b8…`. Per-file
  sizes and hashes, calibration and recipe: [`model-identity/COMPLETE.receipt`](model-identity/COMPLETE.receipt)
  (`RECEIPT.sha256` is the sha256 of that receipt). `config.json`, `generation_config.json` and
  `chat_template.jinja` are copied next to it.
- Vision projector, not mounted into vLLM:
  `/srv/ai-server/artifacts/extensions/qwen-vision-b33b1563/models/qwen/mmproj-Qwen3.8-27B-Uncensored-f16.gguf`,
  931,145,984 bytes, sha256 `add205b7…5288`, from `orcarouter/Qwen3.8-27B-Uncensored-GGUF` revision `b33b1563…`.
- Other images present on Server A: `local/ai-server-llama-dflash2-official` (5.77 GB),
  `ai-server/llama-cpp:server-cuda-a298422-no-vmm-sm86-20260822` (6.76 GB) and two untagged
  `ghcr.io/ggml-org/llama.cpp` builds (6.98 GB each).

## Copied material

| Folder | Source on Server A | Content |
|---|---|---|
| [`benchmarks/`](benchmarks/) | `/srv/ai-server/benchmarks` | context, quality and adaptive runs of 2026-09-08…10: runners, manifests, frozen suites, results, logs |
| [`experiments/`](experiments/) | `/srv/ai-server/artifacts/experiments` | dflash-* and freetoken013-* candidate runs (2026-09-18…24), evidence archives as `.tar.gz` |
| [`acceleration-releases/`](acceleration-releases/) | `/opt/homelab/ai-server/releases/acceleration` | the content-addressed releases of `benchmarks/` and `experiments/qwen38_vllm` used by those runs |
| [`deploy/`](deploy/), [`qwen_bridge/`](qwen_bridge/) | `/opt/homelab/ai-server/deploy`, `/opt/homelab/unsloth-integrations/releases/20260829-wai-qwen-cold-chat/qwen_bridge` | live compose and the Studio-facing bridge |

`*.partial` files, `__pycache__` and `*.pyc` were left out. Before publishing, every file and the
contents of every `.gz` archive were scanned for tokens, keys, private keys, credential
assignments, e-mail addresses and Telegram identifiers: none were found (only example addresses
from test data and library metadata). As in the rest of this archive, absolute credential paths
are not kept: 237 references to secret files in 99 text files (host `…/secrets/…` paths and the
containers' `/run/secrets/…` mounts) read `<redacted-secret-path>`. The content-addressed
`acceleration-releases/` directory names therefore no longer match the hashes of their files,
and runners that read a secret file need a path set before they can run. The `.gz` evidence
archives contain no such paths and are unchanged. Host-wide data was dropped as well: the
`experiments/cleanup-freetoken-20260924/inventory.json` host inventory (mount paths and container
names of unrelated services) is omitted, and the `docker ps` output in
`experiments/freetoken013-qwen38-nvfp4-20260918/nv-preflight-data.json` keeps only `ai-server-*`
containers. Raw `*.jsonl` results and `*.log` files stay on Server A under this repository's
`.gitignore`.

Known drift at collection time: the home-lab working copy of `compose.qwen.yaml` (`a09db0bd…`)
and of `qwen_bridge/compose.candidate.yaml` (`7227e631…`) differ from the live files kept here.
