"""Compare feasible-row filtering with the previous all-row KL projection."""
import time
from unittest.mock import patch

import numpy as np
import torch

import credal_gaussian_experiment as exp


@torch.no_grad()
def reference_all_rows(log_probs, vertices, vertex_mask, A_ub, b_ub,
                       constraint_mask, bisection_steps=32,
                       feasibility_tol=1e-6, eps=1e-12):
    """Unoptimized implementation retained here solely as an oracle."""
    probs = log_probs.exp()
    lhs = torch.einsum("bmc,bc->bm", A_ub, probs)
    feasible = ((lhs <= b_ub + feasibility_tol) | ~constraint_mask).all(dim=1)
    vertex_kl = exp.categorical_loss_from_log_probs(
        vertices, log_probs[:, None, :], "kl", eps=eps
    ).masked_fill(~vertex_mask, float("inf"))
    best_vertex_idx = torch.argmin(vertex_kl, dim=1)
    batch_indices = torch.arange(len(log_probs))
    best_q = vertices[batch_indices, best_vertex_idx]
    best_kl = vertex_kl[batch_indices, best_vertex_idx]
    vmax = vertices.shape[1]
    if vmax >= 2:
        pairs = torch.triu_indices(vmax, vmax, offset=1)
        left_idx, right_idx = pairs[0], pairs[1]
        pair_mask = vertex_mask[:, left_idx] & vertex_mask[:, right_idx]
        left = vertices[:, left_idx, :]
        direction = vertices[:, right_idx, :] - left
        low = torch.zeros_like(pair_mask, dtype=vertices.dtype)
        high = torch.ones_like(low)
        for _ in range(bisection_steps):
            mid = (low + high) * .5
            q_mid = left + mid[..., None] * direction
            gradient = torch.log(q_mid.clamp_min(eps)) - log_probs[:, None, :] + 1.
            move_right = (direction * gradient).sum(dim=-1) < 0
            low = torch.where(move_right, mid, low)
            high = torch.where(move_right, high, mid)
        segment_q = left + ((low + high) * .5)[..., None] * direction
        segment_kl = exp.categorical_loss_from_log_probs(
            segment_q, log_probs[:, None, :], "kl", eps=eps
        ).masked_fill(~pair_mask, float("inf"))
        values, indices = segment_kl.min(dim=1)
        segment_best = segment_q[batch_indices, indices]
        best_q = torch.where((values < best_kl)[:, None], segment_best, best_q)
    return torch.where(feasible[:, None], probs, best_q).detach()


def inputs(method, n=10):
    labels = np.tile(np.array([[.2, .3, .5]], dtype=np.float64), (n, 1))
    bank = exp.prepare_vertex_bank(labels, labels, method, .1, torch.device("cpu"))
    predictions = labels.copy()
    predictions[1::2] = np.array([.999, .0005, .0005])
    log_probs = torch.log(torch.tensor(predictions, dtype=torch.float32))
    lhs = torch.einsum("bmc,bc->bm", bank.A_ub, log_probs.exp())
    feasible = ((lhs <= bank.b_ub + 1e-6) | ~bank.constraint_mask).all(dim=1)
    return bank, log_probs, feasible


def main():
    for method in ("pmm", "lv", "qcor", "tv"):
        bank, log_probs, feasible = inputs(method)
        assert feasible.any() and (~feasible).any()
        expected = reference_all_rows(log_probs, bank.vertices, bank.mask,
                                      bank.A_ub, bank.b_ub, bank.constraint_mask)
        seen_batch_sizes = []
        original_loss = exp.categorical_loss_from_log_probs
        def observe(targets, *args, **kwargs):
            seen_batch_sizes.append(targets.shape[0])
            return original_loss(targets, *args, **kwargs)
        with patch.object(exp, "categorical_loss_from_log_probs", observe):
            actual = exp.minimum_kl_target_in_credal_set(
                log_probs, bank.vertices, bank.mask,
                bank.A_ub, bank.b_ub, bank.constraint_mask)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        assert seen_batch_sizes and set(seen_batch_sizes) == {int((~feasible).sum())}
        torch.testing.assert_close(actual[feasible], log_probs.exp()[feasible])

        all_feasible_log_probs = torch.log(torch.tensor(
            np.tile(np.array([[.2, .3, .5]]), (len(log_probs), 1)),
            dtype=torch.float32,
        ))
        with patch.object(exp, "categorical_loss_from_log_probs",
                          side_effect=AssertionError("segment search was not bypassed")):
            direct = exp.minimum_kl_target_in_credal_set(
                all_feasible_log_probs, bank.vertices, bank.mask,
                bank.A_ub, bank.b_ub, bank.constraint_mask)
        torch.testing.assert_close(direct, all_feasible_log_probs.exp())

        _, all_bad, all_bad_mask = inputs(method, n=8)
        all_bad = torch.log(torch.tensor(
            np.tile(np.array([[.999, .0005, .0005]]), (8, 1)),
            dtype=torch.float32,
        ))
        lhs = torch.einsum("bmc,bc->bm", bank.A_ub[:8], all_bad.exp())
        assert not (((lhs <= bank.b_ub[:8] + 1e-6) |
                     ~bank.constraint_mask[:8]).all(dim=1)).any()
        expected_bad = reference_all_rows(all_bad, bank.vertices[:8], bank.mask[:8],
                                          bank.A_ub[:8], bank.b_ub[:8],
                                          bank.constraint_mask[:8])
        actual_bad = exp.minimum_kl_target_in_credal_set(
            all_bad, bank.vertices[:8], bank.mask[:8], bank.A_ub[:8],
            bank.b_ub[:8], bank.constraint_mask[:8])
        torch.testing.assert_close(actual_bad, expected_bad, atol=1e-7, rtol=1e-6)

    # A small timing sample with 90% feasible rows; informative, not a pass threshold.
    bank, log_probs, _ = inputs("tv", n=128)
    predictions = np.tile(np.array([[.2, .3, .5]]), (128, 1))
    predictions[:13] = np.array([.999, .0005, .0005])
    log_probs = torch.log(torch.tensor(predictions, dtype=torch.float32))
    start = time.perf_counter()
    for _ in range(10):
        reference_all_rows(log_probs, bank.vertices, bank.mask,
                           bank.A_ub, bank.b_ub, bank.constraint_mask)
    reference_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(10):
        exp.minimum_kl_target_in_credal_set(
            log_probs, bank.vertices, bank.mask,
            bank.A_ub, bank.b_ub, bank.constraint_mask)
    filtered_seconds = time.perf_counter() - start
    print("PASS: exact reference agreement; all-feasible bypass; mixed/all-infeasible batches")
    print(f"Timing sample (90% feasible): old={reference_seconds:.4f}s, "
          f"filtered={filtered_seconds:.4f}s, ratio={filtered_seconds/reference_seconds:.3f}")


if __name__ == "__main__":
    main()
