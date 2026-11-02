"""Versioned smaller-side recycled range finder, with no full-gradient SVD.

All randomness comes from a parameter-local CPU generator. Rank-deficient
QR/SVD completions follow the installed solver; exact bases and RNG states
are checkpointed, so loading never reconstructs a basis from a seed.
"""

from __future__ import annotations

import torch

LEGACY_POLICY = {
    "algorithm": "recycled-smaller-side-ritz-v2",
    "precision": "whole-optimizer-highest-fp32-restore-caller-v1",
    "ritz_solver": "small-cuda-gesvd-cpu-default-v1",
    "orientation": "right-if-rows-ge-cols-v1",
    "gauge": "qr-positive-diagonal-final-maxabs-positive-v1",
    "rng": "cpu-generator-seed-plus-1000003-times-parameter-index-v1",
    "moments": "retain-coordinate-arrays-v1",
    "rank_deficiency": "solver-completion-no-rank-change-v1",
}


POLICY = {
    **LEGACY_POLICY,
    "algorithm": "recycled-smaller-side-ritz-finalqr-v3",
    "gauge": "qr-positive-diagonal-final-qr-maxabs-positive-v2",
    "final_reorthogonalization": "canonical-reduced-qr-after-ritz-before-gauge-v1",
}


def projector_policy(final_qr: bool) -> dict:
    return (POLICY if final_qr else LEGACY_POLICY).copy()


def canonical_qr(matrix: torch.Tensor) -> torch.Tensor:
    """Reduced QR, fixing each nonzero diagonal sign without host sync."""
    q, r = torch.linalg.qr(matrix, mode="reduced")
    diagonal = r.diagonal()
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return q.mul_(signs.unsqueeze(0))


def canonical_columns(matrix: torch.Tensor) -> torch.Tensor:
    """Largest-magnitude entry positive; argmax selects the first index."""
    indices = matrix.abs().argmax(dim=0, keepdim=True)
    pivots = matrix.gather(0, indices)
    return matrix.mul_(torch.where(pivots < 0, -torch.ones_like(pivots), torch.ones_like(pivots)))


def recycled_basis(
    grad: torch.Tensor,
    state: dict,
    *,
    rank: int,
    oversampling: int,
    iterations: int,
    seed: int,
    parameter_index: int,
    final_qr: bool = True,
) -> torch.Tensor:
    """Return basis in GaLore's existing left/right storage orientation.

    A is d x D with d <= D. Start from old P plus independent innovation,
    take stabilized A(A.T Q) actions, then factor only an ell x ell R.T.
    QR(mode='r') avoids materializing the D x ell Q of the final QR.
    Persistent state: the returned rank-r basis and CPU generator bytes only.
    """
    right = grad.shape[0] >= grad.shape[1]
    a = grad.T if right else grad
    width = min(rank + oversampling, a.shape[0])
    previous = state.get("basis")
    generator = torch.Generator(device="cpu")
    if "projector_rng_state" in state:
        generator.set_state(state["projector_rng_state"].cpu())
    else:
        generator.manual_seed((seed + 1000003 * parameter_index) % (2**63 - 1))
    innovation_width = width if previous is None else width - rank
    innovation = torch.randn(a.shape[0], innovation_width, generator=generator, dtype=torch.float32).to(a.device)
    if previous is None:
        q = canonical_qr(innovation)
    else:
        p = previous.T if right else previous
        # Two projections reduce loss of orthogonality of a small innovation.
        innovation = innovation - p @ (p.T @ innovation)
        innovation = innovation - p @ (p.T @ innovation)
        q = canonical_qr(torch.cat((p, innovation), dim=1))
    del innovation
    for _ in range(iterations):
        z = canonical_qr(a.T @ q)
        q = canonical_qr(a @ z)
        del z
    compressed_transpose = a.T @ q
    _, triangular = torch.linalg.qr(compressed_transpose, mode="r")
    del compressed_transpose
    left, _, _ = torch.linalg.svd(triangular.T, full_matrices=False, driver="gesvd" if triangular.is_cuda else None)
    p = q @ left[:, :rank]
    # This is a versioned numerical rule: Adam coordinates can change under
    # even a tiny QR rotation, so legacy checkpoints retain final_qr=False.
    if final_qr:
        p = canonical_qr(p)
    p = canonical_columns(p)
    state["projector_rng_state"] = generator.get_state().clone()
    state["projector_parameter_index"] = parameter_index
    state["projector_refreshes"] = int(state.get("projector_refreshes", 0)) + 1
    return p.T.contiguous() if right else p.contiguous()
