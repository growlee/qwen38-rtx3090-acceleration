# Phase 2 Qwen GPU 300 W promotion evidence

Date: 2026-08-23

Status: LIVE_ACCEPTED

The owner-attested Cougar GEX PRO 850 W PSU with two separate PCIe cable runs
closed the physical gate. The exact prior failing workload then passed at 8 GPU
layers and the full Qwen profile passed at 99 GPU layers.

## Accepted runtime

- Model: `Qwen3.8-27B-Uncensored-Q4_K_M.gguf`
- Runtime image digest:
  `sha256:995fe9f237c622ac257905948c3cdde447ab343db22e0686322445b5ec6d3131`
- Context: `16384`
- Parallel slots: `1`
- GPU layers: `99`
- Device surface: CDI `nvidia.com/gpu=all`
- Idle timeout: `1800 s`
- Persistent RTX 3090 power limit: `300 W`

Promoted Compose SHA-256:
`6eb5b0e534286d54bb27eb19b958707edd85090dcb22864b0ceede6d4657dee6`

Power-limit unit SHA-256:
`f0f4157d4556bec1a5b72e493a55b5216377e77a294eba853255020e12ed7778`

## Power-cap comparison

Each cap used the same fixed seed and prompt, one excluded warm-up, two measured
512-token completions and at least 180 seconds of request load. All measured
outputs had the same SHA-256 and every cap remained below the 79 C stop limit.

| Cap | Median decode | Median wall | Median tokens/J | Peak temperature |
|---:|---:|---:|---:|---:|
| 250 W | 27.273 tok/s | 18.968 s | 0.11095 | 66 C |
| 280 W | 33.526 tok/s | 15.421 s | 0.12196 | 70 C |
| 300 W | 37.621 tok/s | 13.754 s | 0.12853 | 72 C |

The fixed selection rule chooses the lowest eligible cap within 97% of the
fastest result. Only 300 W qualified. Hardware/software thermal slowdown,
reset, ping loss and new Xid/AER/MCE/OOM/thermal events were absent.

## Live acceptance

- Supervisor owns profile `qwen` and reports `ready` with breaker closed.
- Qwen, gateway and Open WebUI are healthy.
- Authenticated request through the actual gateway returned HTTP 200 and exact
  marker `QWEN_GPU_300_READY` without recreating the Qwen container.
- Chrome loaded `http://127.0.0.1:18080/`, showed the authenticated Open WebUI
  session and selected `qwen3.8-27b-uncensored`.
- Boot ID stayed `80572931-43be-41e9-a2ea-1f0718f8b680`.

Rollback/evidence directory:
`/srv/ai-server/state/rollback/20260823T-qwen-gpu-300w-promotion-01/`

Its `SHA256SUMS.txt` digest is
`c91e9972369bb94f0a3cb6f0106d9ec390d57a4a83c98454bc2c0350657d713f`.
