"""Bounded numerical stabilization for the pinned syv GPTQ implementation.

Derived from ``syv-ai/qwen38-27b-rtx3090`` revision
``69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c``.  This file is bind-mounted
read-only as ``/app/drafter/gptq_utils.py`` only for the exact-target
``requant-mtp-int4`` step.

Modified for this experiment on 2026-09-01: added bounded damping fallback,
deterministic Hessian diagnostics, and numerical validation.
"""

from __future__ import annotations

import math

import torch


DAMPING_FACTORS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0)
DIAGNOSTIC_BLOCK_SIZE = 512


class HessianError(RuntimeError):
    """A deterministic Hessian validation or factorization failure."""

    def __init__(self, code: str, diagnostics: dict[str, object]) -> None:
        super().__init__(code)
        self.code = code
        self.diagnostics = diagnostics


def accumulate_hessian(H, X, n_seen):
    """Accumulate GPTQ's running ``(2/n) * X.T @ X`` Hessian in fp32."""
    X = X.float()
    n = X.shape[0]
    H *= n_seen / (n_seen + n)
    n_seen += n
    X = (2.0 / n_seen) ** 0.5 * X
    H += X.t() @ X
    return H, n_seen


def _all_finite(matrix) -> bool:
    for start in range(0, matrix.shape[0], DIAGNOSTIC_BLOCK_SIZE):
        if not bool(torch.isfinite(matrix[start : start + DIAGNOSTIC_BLOCK_SIZE]).all().item()):
            return False
    return True


def _max_asymmetry(matrix) -> float:
    maximum = 0.0
    width = matrix.shape[0]
    for start in range(0, width, DIAGNOSTIC_BLOCK_SIZE):
        stop = min(start + DIAGNOSTIC_BLOCK_SIZE, width)
        value = (matrix[start:stop] - matrix[:, start:stop].t()).abs().amax().item()
        maximum = max(maximum, float(value))
    return maximum


def _symmetrize_in_place(matrix) -> None:
    """Symmetrize with bounded temporary storage instead of another full Hessian."""
    width = matrix.shape[0]
    block = DIAGNOSTIC_BLOCK_SIZE
    for row in range(0, width, block):
        row_stop = min(row + block, width)
        diagonal = matrix[row:row_stop, row:row_stop]
        average = (diagonal + diagonal.t()) * 0.5
        diagonal.copy_(average)
        for column in range(row_stop, width, block):
            column_stop = min(column + block, width)
            upper = matrix[row:row_stop, column:column_stop]
            lower = matrix[column:column_stop, row:row_stop]
            average = (upper + lower.t()) * 0.5
            upper.copy_(average)
            lower.copy_(average.t())


def stabilize_hessian(H, *, percdamp: float, module_name: str):
    """Return a Cholesky factor and bounded, JSON-safe diagnostics.

    Non-finite input is rejected before any repair.  Finite input is
    symmetrized, then receives the smallest successful multiplier from the
    fixed damping schedule.  There is deliberately no RTN fallback.
    """
    diagnostics: dict[str, object] = {
        "module": module_name,
        "shape": [int(H.shape[0]), int(H.shape[1])],
        "finite": False,
        "percdamp": float(percdamp),
        "damping_factors": list(DAMPING_FACTORS),
        "attempts": [],
        "status": "FAIL",
    }
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise HessianError("hessian_shape_invalid", diagnostics)
    if not _all_finite(H):
        raise HessianError("hessian_nonfinite", diagnostics)

    diagnostics["finite"] = True
    diagnostics["asymmetry_max_abs"] = _max_asymmetry(H)
    _symmetrize_in_place(H)
    diagonal = torch.diagonal(H)
    diag_min = float(diagonal.min().item())
    diag_mean = float(diagonal.mean().item())
    diag_max = float(diagonal.max().item())
    diagnostics.update({"diag_min": diag_min, "diag_mean": diag_mean, "diag_max": diag_max})
    base_damping = float(percdamp) * diag_mean
    diagnostics["base_damping"] = base_damping
    if not math.isfinite(base_damping) or base_damping <= 0:
        raise HessianError("hessian_damping_invalid", diagnostics)

    previous_factor = 0.0
    attempts: list[dict[str, object]] = diagnostics["attempts"]  # type: ignore[assignment]
    for factor in DAMPING_FACTORS:
        diagonal.add_(base_damping * (factor - previous_factor))
        previous_factor = factor
        factorization, info = torch.linalg.cholesky_ex(H, check_errors=False)
        info_value = int(info.item())
        attempts.append({"factor": factor, "cholesky_info": info_value})
        if info_value == 0:
            diagnostics.update(
                {
                    "status": "PASS",
                    "chosen_damping_factor": factor,
                    "chosen_damping": base_damping * factor,
                }
            )
            return factorization, diagnostics
        del factorization

    diagnostics["last_cholesky_info"] = attempts[-1]["cholesky_info"]
    raise HessianError("hessian_not_positive_definite", diagnostics)


@torch.no_grad()
def gptq_quantize(
    W,
    H,
    bits=4,
    group=128,
    blocksize=128,
    percdamp=0.01,
    mse_clip=False,
    module_name="unknown",
):
    """Symmetric group-wise GPTQ with bounded Hessian stabilization."""
    dev = W.device
    W = W.float().clone()
    N, K = W.shape
    assert K % group == 0
    qmax = 2 ** (bits - 1) - 1
    H = H.clone()
    if not _all_finite(H):
        diagnostics = {
            "module": module_name,
            "shape": [int(H.shape[0]), int(H.shape[1])],
            "finite": False,
            "percdamp": float(percdamp),
            "damping_factors": list(DAMPING_FACTORS),
            "attempts": [],
            "status": "FAIL",
        }
        raise HessianError("hessian_nonfinite", diagnostics)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    factorization, diagnostics = stabilize_hessian(H, percdamp=percdamp, module_name=module_name)
    Hinv = torch.cholesky_inverse(factorization)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    scales = torch.zeros(N, K // group, device=dev)
    scale = None
    assert blocksize == group, "one quantization group per GPTQ block keeps the scale logic simple"
    for i1 in range(0, K, blocksize):
        i2 = i1 + blocksize
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        amax = W1.abs().amax(dim=1)
        scale = torch.clamp(amax / qmax, min=1e-8)
        if mse_clip:
            best = scale.clone()
            best_err = torch.full_like(scale, float("inf"))
            for f in torch.linspace(0.75, 1.0, 11, device=dev):
                candidate_scale = torch.clamp(amax * f / qmax, min=1e-8)
                candidate = (
                    torch.clamp(torch.round(W1 / candidate_scale[:, None]), -qmax - 1, qmax)
                    * candidate_scale[:, None]
                )
                error = ((candidate - W1) ** 2).sum(1)
                better = error < best_err
                best[better] = candidate_scale[better]
                best_err[better] = error[better]
            scale = best
        scales[:, i1 // group] = scale
        for i in range(blocksize):
            w = W1[:, i]
            divisor = Hinv1[i, i]
            q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
            Q1[:, i] = q
            error = (w - q) / divisor
            W1[:, i:] -= error.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = error
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    q_int = (
        torch.round(Q / scales.repeat_interleave(group, dim=1))
        .clamp(-qmax - 1, qmax)
        .to(torch.int8)
    )
    return q_int, scales, diagnostics


def dequant(q, scale, group=128):
    N, K = q.shape
    return (q.float().reshape(N, K // group, group) * scale[..., None]).reshape(N, K)
