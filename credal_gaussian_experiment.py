from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog


# ============================================================
# 0. Global configuration
# ============================================================

N_CLASSES = 3
INPUT_DIM = 2

TRANSFORMATIONS = [
    "alpha_cut",
    "inverse_pignistic",
    "dubois_prade",
    "pmm",
    "lv",
    "qcor",
    "tv",
    "comparative",
]

# Fixed separation of the three Gaussian class means in the noise study.
NOISE_PAIRWISE_DISTANCE = 2.5

# Direct noise strengths: shared mixing proportions, not KL-matched quality.
DEFAULT_RHO_VALUES = (0.1, 0.3, 0.5)
QUALITY_LEVELS = ("high", "medium", "low")


def quality_rho_levels(rho_values: Sequence[float]) -> Dict[str, float]:
    """Map high/medium/low label quality to increasing noise strengths."""
    values = sorted(float(rho) for rho in rho_values)
    if len(values) != 3 or len(set(values)) != 3:
        raise ValueError("Provide exactly three distinct rho values for high/medium/low quality")
    if any(not math.isfinite(rho) or not 0 <= rho <= 1 for rho in values):
        raise ValueError("rho values must be finite and between 0 and 1")
    return dict(zip(QUALITY_LEVELS, values))


NOISE_TYPES = ("uniform", "corruption")  # corruption always means class corruption

# Shared delta for PMM, LV, QCOR and TV.
# This is intentionally exposed because equal delta does NOT imply equal
# credal-set imprecision across transformation families.
DEFAULT_CREDAL_DELTA = 0.5


# ============================================================
# 1. Reproducibility helpers
# ============================================================


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Three-class Gaussian-mixture data with exact posteriors
# ============================================================


# NOTE: This implementation is specific to INPUT_DIM = 2. If INPUT_DIM is
# changed, this function must also be updated to return means whose second
# dimension matches INPUT_DIM.
def equilateral_means(pairwise_distance: float) -> np.ndarray:
    """Return 3 means at the vertices of an equilateral triangle."""
    radius = pairwise_distance / math.sqrt(3.0)
    angles = np.deg2rad([90.0, 210.0, 330.0])
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def log_gaussian_density_batch(
    x: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Log N(x; mean, covariance) for a batch x of shape [N, D]."""
    d = x.shape[1]
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Covariance must be positive definite.")

    diff = x - mean
    solved = np.linalg.solve(covariance, diff.T).T
    quad = np.sum(diff * solved, axis=1)
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


def exact_gmm_posterior(
    x: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    priors: np.ndarray,
) -> np.ndarray:
    """Compute exact P(Y=c | X=x) from the known GMM."""
    log_joint = np.column_stack(
        [
            np.log(priors[c])
            + log_gaussian_density_batch(x, means[c], covariances[c])
            for c in range(N_CLASSES)
        ]
    )
    log_joint -= log_joint.max(axis=1, keepdims=True)
    joint = np.exp(log_joint)
    return joint / joint.sum(axis=1, keepdims=True)


@dataclass
class GMMDataset:
    x: np.ndarray
    y: np.ndarray
    true_posterior: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    priors: np.ndarray


def generate_gmm_dataset(
    n_samples: int,
    pairwise_distance: float,
    seed: int,
    covariance_scale: float = 1.0,
) -> GMMDataset:
    """
    Standard GMM generation:
        Y ~ Categorical(priors)
        X | Y=c ~ N(mu_c, Sigma_c)

    The exact posterior P(Y|X=x) is then computed analytically.

    Calling this function with the same seed but a different pairwise_distance
    reuses the same latent labels and Gaussian residual draws, so overlap is the
    main changed factor.
    """
    rng = np.random.default_rng(seed)
    priors = np.full(N_CLASSES, 1.0 / N_CLASSES)
    means = equilateral_means(pairwise_distance)
    covariances = np.stack(
        [np.eye(INPUT_DIM) * covariance_scale for _ in range(N_CLASSES)]
    )

    y = rng.choice(N_CLASSES, size=n_samples, p=priors)
    eps = rng.normal(size=(n_samples, INPUT_DIM))
    # covariance_scale is variance here, hence sqrt for residual scale.
    x = means[y] + math.sqrt(covariance_scale) * eps

    posterior = exact_gmm_posterior(x, means, covariances, priors)
    return GMMDataset(x=x, y=y, true_posterior=posterior,
                      means=means, covariances=covariances, priors=priors)


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


# ============================================================
# 3. Probabilistic-label corruption mechanisms
# ============================================================


def rowwise_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """KL(p || q), row by row."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=1)


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def uniform_mixing(p: np.ndarray, rho: float) -> np.ndarray:
    u = np.full_like(p, 1.0 / p.shape[1])
    return (1.0 - rho) * p + rho * u


def class_corruption_to_second_best(p: np.ndarray, rho: float) -> np.ndarray:
    """
    Move a fraction rho of the probability mass toward the second-most likely
    class under the true posterior. Unlike uniform mixing,
    this can change the top-1 class and create misleading labels.
    """
    second_best = np.argsort(p, axis=1)[:, -2]
    e = np.zeros_like(p)
    e[np.arange(len(p)), second_best] = 1.0
    return (1.0 - rho) * p + rho * e


@dataclass
class NoisyLabelResult:
    probabilities: np.ndarray
    rho: float
    achieved_mean_kl: float
    top1_agreement_with_bayes: float


def make_noisy_probabilistic_labels(
    p_true: np.ndarray,
    noise_type: str,
    rho: float,
) -> NoisyLabelResult:
    """Apply a specified mixing proportion; KL is diagnostic only."""
    if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be finite and between 0 and 1")
    if noise_type == "uniform":
        q = uniform_mixing(p_true, rho)
    elif noise_type == "corruption":
        q = class_corruption_to_second_best(p_true, rho)
    else:
        raise ValueError("noise_type must be 'uniform' or 'corruption'")
    return NoisyLabelResult(
        probabilities=q,
        rho=float(rho),
        achieved_mean_kl=float(rowwise_kl(p_true, q).mean()),
        top1_agreement_with_bayes=float(
            (np.argmax(q, axis=1) == np.argmax(p_true, axis=1)).mean()
        ),
    )


# ============================================================
# 4. Credal transformations for 3 classes
# ============================================================


@dataclass
class CredalSet3:
    vertices: np.ndarray  # shape [V, 3]
    A_ub: np.ndarray      # method-specific linear inequalities A q <= b
    b_ub: np.ndarray


def _deduplicate_rows(points: List[np.ndarray], tol: float = 1e-9) -> np.ndarray:
    unique: List[np.ndarray] = []
    for p in points:
        if not any(np.linalg.norm(p - q) <= tol for q in unique):
            unique.append(p)
    if not unique:
        return np.empty((0, N_CLASSES), dtype=float)
    return np.vstack(unique)


def vertices_from_linear_constraints_3class(
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Enumerate vertices of a 3-class credal polytope:
        q >= 0, sum(q)=1, A_ub q <= b_ub.

    Since the simplex has dimension 2, a vertex is obtained by the simplex
    equality plus two active inequalities. Nonnegativity inequalities are added
    automatically.
    """
    A_ub = np.asarray(A_ub, dtype=float).reshape(-1, N_CLASSES)
    b_ub = np.asarray(b_ub, dtype=float).reshape(-1)

    A_all = np.vstack([A_ub, -np.eye(N_CLASSES)])
    b_all = np.concatenate([b_ub, np.zeros(N_CLASSES)])

    eq = np.ones((1, N_CLASSES))
    rhs_eq = np.array([1.0])

    candidates: List[np.ndarray] = []
    m = len(b_all)
    for i in range(m):
        for j in range(i + 1, m):
            M = np.vstack([eq, A_all[i][None, :], A_all[j][None, :]])
            rhs = np.concatenate([rhs_eq, [b_all[i], b_all[j]]])
            if np.linalg.matrix_rank(M) < N_CLASSES:
                continue
            q = np.linalg.solve(M, rhs)
            if (
                abs(q.sum() - 1.0) <= 1e-7
                and np.all(q >= -tol)
                and np.all(A_ub @ q <= b_ub + 1e-7)
            ):
                q = np.clip(q, 0.0, 1.0)
                q = q / q.sum()
                candidates.append(q)

    vertices = _deduplicate_rows(candidates, tol=1e-7)
    if len(vertices) == 0:
        raise RuntimeError("No vertices found; the credal constraints may be infeasible.")
    return vertices


def _possibility_constraints_from_sorted_values(
    p: np.ndarray,
    family: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    3-class possibility transformations used in the preceding analysis.

    Let p_(1) >= p_(2) >= p_(3), with corresponding class order sigma.

    alpha-cut:
        pi = (1, p_(2)+p_(3), p_(2)+p_(3))

    inverse pignistic:
        pi = (1, 2 p_(2)+p_(3), 3 p_(3))

    Dubois-Prade:
        pi = (1, p_(2)+p_(3), p_(3))

    The induced possibility credal set is represented by the nested constraints
        q_{sigma(3)} <= pi_3,
        q_{sigma(2)} + q_{sigma(3)} <= pi_2.
    """
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    p2, p3 = ps[1], ps[2]

    if family == "alpha_cut":
        pi2 = p2 + p3
        pi3 = p2 + p3
    elif family == "inverse_pignistic":
        pi2 = 2.0 * p2 + p3
        pi3 = 3.0 * p3
    elif family == "dubois_prade":
        pi2 = p2 + p3
        pi3 = p3
    else:
        raise ValueError(f"Unknown possibility family: {family}")

    A = []
    b = []

    row_tail_3 = np.zeros(N_CLASSES)
    row_tail_3[order[2]] = 1.0
    A.append(row_tail_3)
    b.append(pi3)

    row_tail_23 = np.zeros(N_CLASSES)
    row_tail_23[order[1]] = 1.0
    row_tail_23[order[2]] = 1.0
    A.append(row_tail_23)
    b.append(pi2)

    return np.asarray(A), np.asarray(b)


def credal_constraints_3class(
    p: np.ndarray,
    method: str,
    delta: float = DEFAULT_CREDAL_DELTA,
    tie_tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return method-specific A, b for A q <= b inside the probability simplex."""
    p = np.asarray(p, dtype=float)
    if p.shape != (N_CLASSES,):
        raise ValueError("p must have shape (3,)")
    if not np.isclose(p.sum(), 1.0, atol=1e-7) or np.any(p < -1e-10):
        raise ValueError("p must be a valid probability vector")

    if method in {"alpha_cut", "inverse_pignistic", "dubois_prade"}:
        return _possibility_constraints_from_sorted_values(p, method)

    A: List[np.ndarray] = []
    b: List[float] = []

    if method == "pmm":
        # PMM upper constraints:
        # q_i <= min(1, (1 + delta) * p_i).
        # The corresponding lower bounds
        # q_i >= max(0, (1 + delta) * p_i - delta)
        # follow implicitly from sum(q) = 1 and the constraints for other classes.
        upper = (1.0 + delta) * p
        for i in range(N_CLASSES):
            row = np.zeros(N_CLASSES)
            row[i] = 1.0
            A.append(row)
            b.append(upper[i])

    elif method == "lv":
        # q_i >= (1-delta) p_i  <=>  -q_i <= -(1-delta)p_i
        lower = (1.0 - delta) * p
        for i in range(N_CLASSES):
            row = np.zeros(N_CLASSES)
            row[i] = -1.0
            A.append(row)
            b.append(-lower[i])

    elif method == "qcor":
        # Exact coordinate bounds for 3 classes.
        lower = (1.0 - delta) * p / (1.0 - delta * p)
        upper = p / (1.0 - delta * (1.0 - p))
        for i in range(N_CLASSES):
            row_u = np.zeros(N_CLASSES)
            row_u[i] = 1.0
            A.append(row_u)
            b.append(upper[i])

            row_l = np.zeros(N_CLASSES)
            row_l[i] = -1.0
            A.append(row_l)
            b.append(-lower[i])

    elif method == "tv":
        # Exact coordinate bounds for the 3-class TV neighbourhood.
        lower = np.maximum(0.0, p - delta)
        upper = np.minimum(1.0, p + delta)
        for i in range(N_CLASSES):
            row_u = np.zeros(N_CLASSES)
            row_u[i] = 1.0
            A.append(row_u)
            b.append(upper[i])

            row_l = np.zeros(N_CLASSES)
            row_l[i] = -1.0
            A.append(row_l)
            b.append(-lower[i])

    elif method == "comparative":
        # Preserve the full preorder of the original probability vector:
        # p_i >= p_j  =>  q_i >= q_j.
        # For ties, both directions are added, giving equality.
        for i in range(N_CLASSES):
            for j in range(i + 1, N_CLASSES):
                if p[i] > p[j] + tie_tol:
                    # q_j - q_i <= 0
                    row = np.zeros(N_CLASSES)
                    row[j] = 1.0
                    row[i] = -1.0
                    A.append(row)
                    b.append(0.0)
                elif p[j] > p[i] + tie_tol:
                    # q_i - q_j <= 0
                    row = np.zeros(N_CLASSES)
                    row[i] = 1.0
                    row[j] = -1.0
                    A.append(row)
                    b.append(0.0)
                else:
                    row1 = np.zeros(N_CLASSES)
                    row1[j] = 1.0
                    row1[i] = -1.0
                    A.append(row1)
                    b.append(0.0)

                    row2 = -row1
                    A.append(row2)
                    b.append(0.0)

    else:
        raise ValueError(f"Unknown credal transformation: {method}")

    return np.asarray(A, dtype=float), np.asarray(b, dtype=float)


def build_credal_set_3class(
    p: np.ndarray,
    method: str,
    delta: float = DEFAULT_CREDAL_DELTA,
) -> CredalSet3:
    A, b = credal_constraints_3class(p, method, delta=delta)
    vertices = vertices_from_linear_constraints_3class(A, b)
    return CredalSet3(vertices=vertices, A_ub=A, b_ub=b)


def point_in_credal_set(
    q: np.ndarray,
    credal: CredalSet3,
    tol: float = 1e-7,
) -> bool:
    return bool(
        np.all(q >= -tol)
        and abs(q.sum() - 1.0) <= tol
        and np.all(credal.A_ub @ q <= credal.b_ub + tol)
    )


def normalized_simplex_area(vertices: np.ndarray) -> float:
    """
    Normalized area in the (q1, q2) projection.
    The full 3-class simplex has area 1/2 in these coordinates, so divide by 1/2.
    Lower-dimensional credal sets have area 0.
    """
    if len(vertices) < 3:
        return 0.0
    pts = vertices[:, :2]
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area / 0.5)


def solve_q_star_lp(
    logits: np.ndarray,
    credal: CredalSet3,
) -> np.ndarray:
    """
    Generic LP fallback:
        min_q CE(q, softmax(logits))
      = max_q q^T logits
      = min_q -logits^T q.
    """
    result = linprog(
        c=-np.asarray(logits, dtype=float),
        A_ub=credal.A_ub,
        b_ub=credal.b_ub,
        A_eq=np.ones((1, N_CLASSES)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, 1.0)] * N_CLASSES,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}")
    return result.x


# ============================================================
# 5. Precompute credal vertices for fast mini-batch training
# ============================================================


@dataclass
class VertexBank:
    vertices: torch.Tensor  # [N, Vmax, C]
    mask: torch.Tensor      # [N, Vmax]
    A_ub: torch.Tensor      # [N, Mmax, C]
    b_ub: torch.Tensor      # [N, Mmax]
    constraint_mask: torch.Tensor  # [N, Mmax]
    mean_num_vertices: float
    mean_area: float
    true_posterior_coverage: float


def prepare_vertex_bank(
    probabilistic_labels: np.ndarray,
    true_posteriors: np.ndarray,
    method: str,
    delta: float,
    device: torch.device,
) -> VertexBank:
    credal_sets: List[CredalSet3] = []
    areas = []
    coverage = []

    for p_label, p_true in zip(probabilistic_labels, true_posteriors):
        credal = build_credal_set_3class(p_label, method, delta=delta)
        credal_sets.append(credal)
        areas.append(normalized_simplex_area(credal.vertices))
        coverage.append(point_in_credal_set(p_true, credal))

    max_v = max(len(c.vertices) for c in credal_sets)
    n = len(credal_sets)
    padded = np.zeros((n, max_v, N_CLASSES), dtype=np.float32)
    mask = np.zeros((n, max_v), dtype=bool)

    max_m = max(len(c.b_ub) for c in credal_sets)
    A_padded = np.zeros((n, max_m, N_CLASSES), dtype=np.float32)
    b_padded = np.zeros((n, max_m), dtype=np.float32)
    constraint_mask = np.zeros((n, max_m), dtype=bool)

    for i, credal in enumerate(credal_sets):
        v = credal.vertices.astype(np.float32)
        padded[i, : len(v)] = v
        mask[i, : len(v)] = True

        m = len(credal.b_ub)
        A_padded[i, :m] = credal.A_ub.astype(np.float32)
        b_padded[i, :m] = credal.b_ub.astype(np.float32)
        constraint_mask[i, :m] = True

    return VertexBank(
        vertices=torch.tensor(padded, dtype=torch.float32, device=device),
        mask=torch.tensor(mask, dtype=torch.bool, device=device),
        A_ub=torch.tensor(A_padded, dtype=torch.float32, device=device),
        b_ub=torch.tensor(b_padded, dtype=torch.float32, device=device),
        constraint_mask=torch.tensor(
            constraint_mask, dtype=torch.bool, device=device
        ),
        mean_num_vertices=float(np.mean([len(c.vertices) for c in credal_sets])),
        mean_area=float(np.mean(areas)),
        true_posterior_coverage=float(np.mean(coverage)),
    )


# ============================================================
# 6. Target MLP and training objectives
# ============================================================


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: Sequence[int] = (64, 64),
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dims: Tuple[int, ...] = (64, 64)
    loss_name: str = "cross_entropy"


def categorical_loss_from_log_probs(
    targets: torch.Tensor,
    log_probs: torch.Tensor,
    loss_name: str,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return unreduced CE or KL(target || prediction)."""
    cross_entropy = -(targets * log_probs).sum(dim=-1)
    if loss_name == "cross_entropy":
        return cross_entropy
    if loss_name == "kl":
        # Define 0 * log(0) as 0, as required by categorical KL divergence.
        target_log_probs = torch.where(
            targets > 0,
            torch.log(targets.clamp_min(eps)),
            torch.zeros_like(targets),
        )
        return cross_entropy + (targets * target_log_probs).sum(dim=-1)
    raise ValueError(f"Unknown loss_name: {loss_name}")


@torch.no_grad()
def minimum_kl_target_in_credal_set(
    log_probs: torch.Tensor,
    vertices: torch.Tensor,
    vertex_mask: torch.Tensor,
    A_ub: torch.Tensor,
    b_ub: torch.Tensor,
    constraint_mask: torch.Tensor,
    bisection_steps: int = 32,
    feasibility_tol: float = 1e-6,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute argmin_q KL(q || prediction) over 3-class credal sets.

    A three-class probability simplex is two-dimensional. If the prediction is
    feasible, it is the KL minimizer. Otherwise, the minimizer lies on the
    boundary of the convex credal polygon. Every boundary edge joins two
    vertices, so minimizing KL on all vertex pairs (including all true edges)
    and taking the best candidate recovers the full-set minimum. Extra diagonal
    pairs contain only feasible points and therefore do not change the minimum.
    """
    if log_probs.ndim != 2 or log_probs.shape[1] != N_CLASSES:
        raise ValueError("log_probs must have shape [B, 3]")
    if vertices.ndim != 3 or vertices.shape[0] != len(log_probs):
        raise ValueError("vertices must have shape [B, Vmax, 3]")

    probs = log_probs.exp()
    if A_ub.shape[1] == 0:
        feasible = torch.ones(len(probs), dtype=torch.bool, device=probs.device)
    else:
        lhs = torch.einsum("bmc,bc->bm", A_ub, probs)
        constraints_satisfied = (lhs <= b_ub + feasibility_tol) | ~constraint_mask
        feasible = constraints_satisfied.all(dim=1)

    # Vertices cover singleton credal sets and provide stable endpoint candidates.
    vertex_kl = categorical_loss_from_log_probs(
        vertices,
        log_probs[:, None, :],
        "kl",
        eps=eps,
    ).masked_fill(~vertex_mask, float("inf"))
    best_vertex_idx = torch.argmin(vertex_kl, dim=1)
    batch_indices = torch.arange(len(log_probs), device=log_probs.device)
    best_q = vertices[batch_indices, best_vertex_idx]
    best_kl = vertex_kl[batch_indices, best_vertex_idx]

    vmax = vertices.shape[1]
    if vmax >= 2:
        pair_indices = torch.triu_indices(
            vmax, vmax, offset=1, device=vertices.device
        )
        left_idx, right_idx = pair_indices[0], pair_indices[1]
        pair_mask = vertex_mask[:, left_idx] & vertex_mask[:, right_idx]

        left = vertices[:, left_idx, :]
        direction = vertices[:, right_idx, :] - left
        low = torch.zeros_like(pair_mask, dtype=vertices.dtype)
        high = torch.ones_like(low)

        # KL is convex along each line segment. Its directional derivative is
        # monotone, so bisection locates the segment minimum, including limits
        # at either endpoint.
        for _ in range(bisection_steps):
            mid = (low + high) * 0.5
            q_mid = left + mid[..., None] * direction
            kl_gradient = (
                torch.log(q_mid.clamp_min(eps))
                - log_probs[:, None, :]
                + 1.0
            )
            derivative = (direction * kl_gradient).sum(dim=-1)
            move_right = derivative < 0
            low = torch.where(move_right, mid, low)
            high = torch.where(move_right, high, mid)

        segment_q = left + ((low + high) * 0.5)[..., None] * direction
        segment_kl = categorical_loss_from_log_probs(
            segment_q,
            log_probs[:, None, :],
            "kl",
            eps=eps,
        ).masked_fill(~pair_mask, float("inf"))
        best_segment_kl, best_segment_idx = segment_kl.min(dim=1)
        best_segment_q = segment_q[batch_indices, best_segment_idx]
        use_segment = best_segment_kl < best_kl
        best_q = torch.where(use_segment[:, None], best_segment_q, best_q)

    # KL(q || p) is zero at q=p, so a feasible prediction is the global optimum.
    return torch.where(feasible[:, None], probs, best_q).detach()


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mode: str,
    config: TrainConfig,
    device: torch.device,
    seed: int,
    soft_targets: Optional[np.ndarray] = None,
    vertex_bank: Optional[VertexBank] = None,
) -> SimpleMLP:
    """
    mode:
      - 'hard'   : loss(one-hot observed label, prediction)
      - 'soft'   : loss(soft target, prediction)
      - 'credal' : minimum loss over M_i

    With cross-entropy, vertex minimization is exact over the full convex credal
    set because CE is linear in its target. With KL, q_star is the full-set KL
    projection and may be an interior point or a point inside a boundary edge.
    """
    if config.loss_name not in {"cross_entropy", "kl"}:
        raise ValueError("loss_name must be 'cross_entropy' or 'kl'")

    set_all_seeds(seed)
    model = SimpleMLP(hidden_dims=config.hidden_dims).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    soft_t = None
    if soft_targets is not None:
        soft_t = torch.tensor(soft_targets, dtype=torch.float32, device=device)

    n = len(x_train)
    rng = np.random.default_rng(seed + 12345)

    for _epoch in range(config.epochs):
        permutation = rng.permutation(n)
        model.train()

        for start in range(0, n, config.batch_size):
            ids_np = permutation[start : start + config.batch_size]
            ids = torch.tensor(ids_np, dtype=torch.long, device=device)
            xb = x_t[ids]

            logits = model(xb)
            log_probs = F.log_softmax(logits, dim=-1)

            if mode == "hard":
                if config.loss_name == "cross_entropy":
                    loss = F.cross_entropy(logits, y_t[ids])
                else:
                    qb = F.one_hot(y_t[ids], num_classes=N_CLASSES).to(
                        dtype=log_probs.dtype
                    )
                    loss = categorical_loss_from_log_probs(
                        qb, log_probs, config.loss_name
                    ).mean()

            elif mode == "soft":
                if soft_t is None:
                    raise ValueError("soft_targets are required in soft mode")
                qb = soft_t[ids]
                loss = categorical_loss_from_log_probs(
                    qb, log_probs, config.loss_name
                ).mean()

            elif mode == "credal":
                if vertex_bank is None:
                    raise ValueError("vertex_bank is required in credal mode")

                # V_b: [B, Vmax, C]
                V_b = vertex_bank.vertices[ids]
                mask_b = vertex_bank.mask[ids]
                if config.loss_name == "cross_entropy":
                    # CE is linear in q, so a minimum over the full convex set
                    # is attained at a vertex.
                    losses_all = categorical_loss_from_log_probs(
                        V_b,
                        log_probs[:, None, :],
                        "cross_entropy",
                    ).masked_fill(~mask_b, float("inf"))
                    best_idx = torch.argmin(losses_all, dim=1)
                    q_star = V_b[
                        torch.arange(len(ids), device=device),
                        best_idx,
                    ].detach()
                else:
                    # KL is convex rather than linear in q. Its minimum can be
                    # inside the credal set, so optimize over the full polygon.
                    q_star = minimum_kl_target_in_credal_set(
                        log_probs=log_probs,
                        vertices=V_b,
                        vertex_mask=mask_b,
                        A_ub=vertex_bank.A_ub[ids],
                        b_ub=vertex_bank.b_ub[ids],
                        constraint_mask=vertex_bank.constraint_mask[ids],
                    )

                # Stop-gradient target q_star; gradient flows only through logits.
                loss = categorical_loss_from_log_probs(
                    q_star, log_probs, config.loss_name
                ).mean()

            else:
                raise ValueError(f"Unknown training mode: {mode}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model


# ============================================================
# 7. Evaluation
# ============================================================


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    out = []
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    for start in range(0, len(x), batch_size):
        logits = model(x_t[start : start + batch_size])
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(out)


def evaluate_model(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    p_true_test: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    probs = predict_probabilities(model, x_test, device)
    predictions = probs.argmax(axis=1)

    accuracy = float((predictions == y_test).mean())
    posterior_kl = float(rowwise_kl(p_true_test, probs).mean())
    posterior_brier = float(np.mean(np.sum((probs - p_true_test) ** 2, axis=1)))

    return {
        "accuracy": accuracy,
        "posterior_kl": posterior_kl,
        "posterior_brier": posterior_brier,
    }


def bayes_metrics(y_test: np.ndarray, p_true_test: np.ndarray) -> Dict[str, float]:
    bayes_pred = np.argmax(p_true_test, axis=1)
    return {
        # Expected Bayes accuracy conditional on the observed X values.
        "bayes_expected_accuracy": float(np.mean(np.max(p_true_test, axis=1))),
        # Realized finite-sample accuracy against the sampled hard labels.
        "bayes_empirical_accuracy": float(np.mean(bayes_pred == y_test)),
    }


# ============================================================
# 8. Noise study: independent repeats, fixed Gaussian separation
# ============================================================


def resolve_seeds(repeats: int, seeds: Optional[Sequence[int]]) -> List[int]:
    values = list(range(repeats)) if seeds is None else list(seeds)
    if repeats < 1 or len(values) != repeats or len(set(values)) != repeats:
        raise ValueError("Provide one distinct seed per repeat, with repeats >= 1")
    if any(not isinstance(seed, (int, np.integer)) or seed < 0 for seed in values):
        raise ValueError("Seeds must be nonnegative integers")
    return values


def experiment_seeds(seed: int) -> Dict[str, int]:
    """Independent streams per repeat; shared across methods and noise conditions."""
    # Keep namespace 1 to reproduce noise-study seeds from earlier versions.
    streams = np.random.SeedSequence([seed, 1]).generate_state(3)
    return dict(seed=int(seed), train_seed=int(streams[0]),
                test_seed=int(streams[1]), model_seed=int(streams[2]))


def run_noise_study(
    output_dir: Path,
    n_train: int,
    n_test: int,
    repeats: int,
    train_config: TrainConfig,
    credal_delta: float,
    device: torch.device,
    rho_values: Sequence[float] = DEFAULT_RHO_VALUES,
    seeds: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seeds = resolve_seeds(repeats, seeds)
    quality_rhos = quality_rho_levels(rho_values)
    distance = NOISE_PAIRWISE_DISTANCE

    rows: List[Dict[str, object]] = []
    quality_rows: List[Dict[str, object]] = []
    noise_types = NOISE_TYPES
    total_conditions = len(noise_types) * len(rho_values)
    total_models = repeats * 2 + total_conditions * repeats * (
        1 + len(TRANSFORMATIONS)
    )
    model_index = 0
    condition_index = 0
    print(
        f"[noise] Starting study with {total_conditions} noise conditions and "
        f"{total_models} model trainings using loss={train_config.loss_name}.",
        flush=True,
    )

    # Baselines independent of noisy-label mechanism/rho.
    for repeat, seed in enumerate(seeds):
        seed_info = experiment_seeds(seed)
        model_seed = seed_info["model_seed"]
        condition_index = 0
        print(f"[noise] Repeat {repeat + 1}/{repeats}: {seed_info}", flush=True)
        train_data = generate_gmm_dataset(n_train, distance, seed=seed_info['train_seed'])
        test_data = generate_gmm_dataset(n_test, distance, seed=seed_info['test_seed'])
        x_train, x_test = standardize_train_test(train_data.x, test_data.x)
        bayes = bayes_metrics(test_data.y, test_data.true_posterior)


        model_index += 1
        progress_label = (
            f"baseline, repeat={repeat + 1}/{repeats}, method=oracle_posterior"
        )
        print(
            f"[noise {model_index}/{total_models}] Training {progress_label}...",
            flush=True,
        )
        model = train_mlp(
            x_train,
            train_data.y,
            mode="soft",
            soft_targets=train_data.true_posterior,
            config=train_config,
            device=device,
            seed=model_seed,
        )
        metrics = evaluate_model(
            model, x_test, test_data.y, test_data.true_posterior, device
        )
        print(
            f"[noise {model_index}/{total_models}] Finished {progress_label}; "
            f"accuracy={metrics['accuracy']:.4f}.",
            flush=True,
        )
        rows.append({
            "study": "noise",
            "pairwise_distance": distance,
            "noise_type": "baseline",
            "rho": np.nan,
            "quality": "baseline",
            "achieved_mean_label_kl": 0.0,
            "top1_agreement_with_bayes": 1.0,
            "repeat": repeat,
            **seed_info,
            "method": "oracle_posterior",
            "target_type": "baseline",
            "mean_credal_area": np.nan,
            "true_posterior_coverage": np.nan,
            **bayes,
            **metrics,
        })

        model_index += 1
        progress_label = (
            f"baseline, repeat={repeat + 1}/{repeats}, method=clean_sampled_hard"
        )
        print(
            f"[noise {model_index}/{total_models}] Training {progress_label}...",
            flush=True,
        )
        model = train_mlp(
            x_train,
            train_data.y,
            mode="hard",
            config=train_config,
            device=device,
            seed=model_seed,
        )
        metrics = evaluate_model(
            model, x_test, test_data.y, test_data.true_posterior, device
        )
        print(
            f"[noise {model_index}/{total_models}] Finished {progress_label}; "
            f"accuracy={metrics['accuracy']:.4f}.",
            flush=True,
        )
        rows.append({
            "study": "noise",
            "pairwise_distance": distance,
            "noise_type": "baseline",
            "rho": np.nan,
            "quality": "baseline",
            "achieved_mean_label_kl": np.nan,
            "top1_agreement_with_bayes": np.nan,
            "repeat": repeat,
            **seed_info,
            "method": "clean_sampled_hard",
            "target_type": "baseline",
            "mean_credal_area": np.nan,
            "true_posterior_coverage": np.nan,
            **bayes,
            **metrics,
        })

        for noise_type in noise_types:
            for quality, rho in quality_rhos.items():
                condition_index += 1
                print(
                    f"[noise] Preparing condition {condition_index}/{total_conditions}: "
                    f"noise_type={noise_type}, quality={quality}, rho={rho}.",
                    flush=True,
                )
                noisy = make_noisy_probabilistic_labels(
                    train_data.true_posterior,
                    noise_type=noise_type,
                    rho=rho,
                )

                quality_rows.append({
                    "repeat": repeat,
                    **seed_info,
                    "pairwise_distance": distance,
                    "noise_type": noise_type,
                    "rho": noisy.rho,
                    "quality": quality,
                    "achieved_mean_label_kl": noisy.achieved_mean_kl,
                    "top1_agreement_with_bayes": noisy.top1_agreement_with_bayes,
                })

                # Precompute all credal sets once per noisy-label condition.
                banks: Dict[str, VertexBank] = {}
                for method_index, method in enumerate(TRANSFORMATIONS, start=1):
                    print(
                        f"[noise] Building vertex bank {method_index}/"
                        f"{len(TRANSFORMATIONS)} for method={method}...",
                        flush=True,
                    )
                    banks[method] = prepare_vertex_bank(
                        noisy.probabilities,
                        train_data.true_posterior,
                        method,
                        credal_delta,
                        device,
                    )


                # Direct noisy-soft-label baseline.
                model_index += 1
                progress_label = (
                    f"noise_type={noise_type}, quality={quality}, rho={rho}, "
                    f"repeat={repeat + 1}/{repeats}, method=noisy_soft"
                )
                print(
                    f"[noise {model_index}/{total_models}] Training "
                    f"{progress_label}...",
                    flush=True,
                )
                model = train_mlp(
                    x_train,
                    train_data.y,
                    mode="soft",
                    soft_targets=noisy.probabilities,
                    config=train_config,
                    device=device,
                    seed=model_seed,
                )
                metrics = evaluate_model(
                    model, x_test, test_data.y, test_data.true_posterior, device
                )
                print(
                    f"[noise {model_index}/{total_models}] Finished "
                    f"{progress_label}; accuracy={metrics['accuracy']:.4f}.",
                    flush=True,
                )
                rows.append({
                    "study": "noise",
                    "pairwise_distance": distance,
                    "noise_type": noise_type,
                    "rho": noisy.rho,
                    "quality": quality,
                    "achieved_mean_label_kl": noisy.achieved_mean_kl,
                    "top1_agreement_with_bayes": noisy.top1_agreement_with_bayes,
                    "repeat": repeat,
                    **seed_info,
                    "method": "noisy_soft",
                    "target_type": "baseline",
                    "mean_credal_area": np.nan,
                    "true_posterior_coverage": np.nan,
                    **bayes,
                    **metrics,
                })

                # Credal transformations.
                for method in TRANSFORMATIONS:
                    bank = banks[method]
                    model_index += 1
                    progress_label = (
                        f"noise_type={noise_type}, quality={quality}, rho={rho}, "
                        f"repeat={repeat + 1}/{repeats}, method={method}"
                    )
                    print(
                        f"[noise {model_index}/{total_models}] Training "
                        f"{progress_label}...",
                        flush=True,
                    )
                    model = train_mlp(
                        x_train,
                        train_data.y,
                        mode="credal",
                        vertex_bank=bank,
                        config=train_config,
                        device=device,
                        seed=model_seed,
                    )
                    metrics = evaluate_model(
                        model, x_test, test_data.y, test_data.true_posterior, device
                    )
                    print(
                        f"[noise {model_index}/{total_models}] Finished "
                        f"{progress_label}; accuracy={metrics['accuracy']:.4f}.",
                        flush=True,
                    )
                    rows.append({
                        "study": "noise",
                        "pairwise_distance": distance,
                        "noise_type": noise_type,
                        "rho": noisy.rho,
                        "quality": quality,
                        "achieved_mean_label_kl": noisy.achieved_mean_kl,
                        "top1_agreement_with_bayes": noisy.top1_agreement_with_bayes,
                        "repeat": repeat,
                        **seed_info,
                        "method": method,
                        "target_type": "credal",
                        "mean_credal_area": bank.mean_area,
                        "true_posterior_coverage": bank.true_posterior_coverage,
                        **bayes,
                        **metrics,
                    })

    df = pd.DataFrame(rows)
    df["loss"] = train_config.loss_name
    quality_df = pd.DataFrame(quality_rows)
    df_path = output_dir / "noise_study.csv"
    quality_path = output_dir / "noise_quality_summary.csv"
    df.to_csv(df_path, index=False)
    quality_df.to_csv(quality_path, index=False)
    print(f"Saved noise study to {df_path}", flush=True)
    print(f"Saved noise-quality summary to {quality_path}", flush=True)
    return df, quality_df


# ============================================================
# 9. Aggregation helpers
# ============================================================


def aggregate_results(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "posterior_kl",
        "posterior_brier",
        "mean_credal_area",
        "true_posterior_coverage",
    ]
    existing = [c for c in metric_cols if c in df.columns]
    agg = df.groupby(list(group_cols), dropna=False)[existing].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


# ============================================================
# 10. Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results_credal_gaussian")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of independent data/model seeds (default: 5).")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="Explicit repeat seeds; overrides --repeats.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--credal-delta", type=float, default=DEFAULT_CREDAL_DELTA)
    parser.add_argument("--delta-mode", choices=["fixed", "select", "matched"], default="fixed",
                        help="Fixed delta, validation-optimal delta, or matched training-area study.")
    parser.add_argument("--n-validation", type=int, default=1000,
                        help="Samples held out from --n-train in select/matched modes.")
    parser.add_argument("--scan-dir", type=Path,
                        help="Optional completed select-run directory supplying area scans for matched mode.")
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "kl"],
        default="cross_entropy",
        help=(
            "Training loss. In credal mode, KL is minimized over the complete "
            "credal set and can select an interior target."
        ),
    )
    parser.add_argument(
        "--rho-values", type=float, nargs=3, default=list(DEFAULT_RHO_VALUES),
        help="Three distinct rho values in [0, 1]; sorted ascending for high/medium/low quality.",
    )
    args = parser.parse_args()
    if args.seeds is not None:
        args.repeats = len(args.seeds)
    try:
        args.seeds = resolve_seeds(args.repeats, args.seeds)
    except ValueError as error:
        parser.error(str(error))
    try:
        quality_rhos = quality_rho_levels(args.rho_values)
    except ValueError as error:
        parser.error(str(error))
    args.rho_values = list(quality_rhos.values())
    if args.scan_dir is not None and args.delta_mode != "matched":
        parser.error("--scan-dir is only valid with --delta-mode matched")
    if args.scan_dir is not None and args.scan_dir.resolve() == Path(args.output_dir).resolve():
        parser.error("Use a separate output directory; do not overwrite source scans")
    if args.delta_mode in {"select", "matched"}:
        if len(args.seeds) != 5:
            parser.error("select/matched modes require exactly five distinct seeds")
        if not 0 < args.n_validation < args.n_train:
            parser.error("Require 0 < --n-validation < --n-train")
        if args.epochs < 1 or args.n_test < 1:
            parser.error("Selection requires positive --epochs and --n-test")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
    )

    config_dump = {
        "rho_values": args.rho_values,
        "quality_rho": quality_rhos,
        "noise_types": list(NOISE_TYPES),
        "pairwise_distance": NOISE_PAIRWISE_DISTANCE,
        "transformations": TRANSFORMATIONS,
        "credal_delta": args.credal_delta,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "seeds": args.seeds,
        "seed_schedule": {"noise": [experiment_seeds(seed) for seed in args.seeds]},
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "loss": args.loss,
        "corruption_kind": "class",
    }
    if args.delta_mode == "select":
        from delta_search import UNIT_DELTA_GRID, run_delta_selection_study
        config_dump["delta_selection"] = {
            "mode": "select", "methods": ["pmm", "lv", "qcor", "tv"],
            "n_validation": args.n_validation, "n_fit": args.n_train - args.n_validation,
            "selection_metric": "mean_validation_accuracy", "tie_break": "smaller_delta",
            "unit_delta_grid": list(UNIT_DELTA_GRID),
            "pmm_initial_delta": 0.05, "pmm_initial_step": 0.05,
            "pmm_step_growth": 1.5, "pmm_stop_metric": "mean_normalized_area",
            "pmm_stop_rule": "delta >= pooled_training_Q95 AND adjacent_area_increment < 0.01",
            "final_evaluation": "saved_fitting_subset_checkpoints_without_refit",
        }
    if args.delta_mode == "matched":
        config_dump["matched_imprecision"] = {
            "methods": ["pmm", "lv", "qcor", "tv"], "n_validation": args.n_validation,
            "n_fit": args.n_train - args.n_validation, "epsilon_A": .005,
            "n_tau": 5, "tau_grid": "linspace(common_min, common_max, 5), endpoints included",
            "area_data": "noisy fitting labels only", "refinement": "linear interpolation then bisection",
            "scan_dir": str(args.scan_dir.resolve()) if args.scan_dir else None,
        }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2)

    if args.delta_mode == "matched":
        from matched_imprecision import run_matched_study
        run_matched_study(output_dir, args.n_train, args.n_test, args.seeds,
                          train_config, device, args.rho_values, args.n_validation, args.scan_dir)
        print("Done.")
        return
    if args.delta_mode == "select":
        run_delta_selection_study(
            output_dir=output_dir, n_train=args.n_train, n_test=args.n_test,
            seeds=args.seeds, train_config=train_config, credal_delta=args.credal_delta,
            device=device, rho_values=args.rho_values, n_validation=args.n_validation,
        )
        print("Done.")
        return

    noise_df, _quality_df = run_noise_study(
        output_dir=output_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        repeats=args.repeats,
        seeds=args.seeds,
        train_config=train_config,
        credal_delta=args.credal_delta,
        device=device,
        rho_values=args.rho_values,
    )
    noise_summary = aggregate_results(
        noise_df,
        group_cols=["noise_type", "quality", "rho", "loss", "method", "target_type"],
    )
    noise_summary.to_csv(output_dir / "noise_study_summary.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
