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

# Pairwise distance between the three Gaussian means.
# With covariance I, these values give roughly:
#   high overlap   -> Bayes accuracy ~ 0.65
#   medium overlap -> Bayes accuracy ~ 0.82
#   low overlap    -> Bayes accuracy ~ 0.96
OVERLAP_LEVELS = {
    "low": 4.0,
    "medium": 2.5,
    "high": 1.5,
}

# Quality is matched by mean KL(true posterior || noisy soft label).
# Lower KL = higher label quality.
QUALITY_TARGET_KL = {
    "high": 0.02,
    "medium": 0.10,
    "low": 0.30,
}

# For the noise study, keep the data overlap fixed.
NOISE_STUDY_OVERLAP = "medium"

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


def temperature_scaling_from_probabilities(
    p: np.ndarray,
    temperature: float,
    eps: float = 1e-12,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.log(np.clip(p, eps, 1.0))
    return softmax_np(logits / temperature)


def class_corruption_to_second_best(p: np.ndarray, rho: float) -> np.ndarray:
    """
    Move a fraction rho of the probability mass toward the second-most likely
    class under the true posterior. Unlike uniform mixing / temperature scaling,
    this can change the top-1 class and create misleading labels.
    """
    second_best = np.argsort(p, axis=1)[:, -2]
    e = np.zeros_like(p)
    e[np.arange(len(p)), second_best] = 1.0
    return (1.0 - rho) * p + rho * e


def logit_gaussian_corruption(
    p: np.ndarray,
    sigma: float,
    fixed_noise: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    if fixed_noise.shape != p.shape:
        raise ValueError("fixed_noise must have the same shape as p")
    logits = np.log(np.clip(p, eps, 1.0)) + sigma * fixed_noise
    return softmax_np(logits)


@dataclass
class NoisyLabelResult:
    probabilities: np.ndarray
    parameter_name: str
    parameter_value: float
    achieved_mean_kl: float
    top1_agreement_with_bayes: float


def _select_candidate_closest_to_target(
    p_true: np.ndarray,
    candidates: Sequence[Tuple[float, np.ndarray]],
    parameter_name: str,
    target_kl: float,
) -> NoisyLabelResult:
    best = None
    for value, q in candidates:
        mean_kl = float(rowwise_kl(p_true, q).mean())
        error = abs(mean_kl - target_kl)
        if best is None or error < best[0]:
            best = (error, value, q, mean_kl)

    assert best is not None
    _, value, q_best, mean_kl = best
    agreement = float(
        (np.argmax(q_best, axis=1) == np.argmax(p_true, axis=1)).mean()
    )
    return NoisyLabelResult(
        probabilities=q_best,
        parameter_name=parameter_name,
        parameter_value=float(value),
        achieved_mean_kl=float(mean_kl),
        top1_agreement_with_bayes=agreement,
    )


def make_noisy_probabilistic_labels(
    p_true: np.ndarray,
    noise_type: str,
    target_kl: float,
    seed: int,
    corruption_kind: str = "class",
    temperature_direction: str = "overconfident",
) -> NoisyLabelResult:
    """
    Match soft-label quality across noise mechanisms by targeting the same
    mean KL(true posterior || noisy label).

    noise_type:
      - 'uniform'
      - 'temperature'
      - 'corruption'

    corruption_kind:
      - 'class': shift mass toward the second-best class
      - 'logit': add Gaussian noise in logit space
    """
    if noise_type == "uniform":
        grid = np.linspace(0.0, 0.98, 250)
        candidates = [(rho, uniform_mixing(p_true, rho)) for rho in grid]
        return _select_candidate_closest_to_target(
            p_true, candidates, "rho_uniform", target_kl
        )

    if noise_type == "temperature":
        if temperature_direction == "overconfident":
            # T < 1 sharpens the distribution while preserving ranking.
            grid = np.geomspace(1.0, 0.12, 250)
        elif temperature_direction == "underconfident":
            # T > 1 flattens the distribution while preserving ranking.
            grid = np.geomspace(1.0, 12.0, 250)
        else:
            raise ValueError("Unknown temperature_direction")

        candidates = [
            (T, temperature_scaling_from_probabilities(p_true, float(T)))
            for T in grid
        ]
        return _select_candidate_closest_to_target(
            p_true, candidates, "temperature", target_kl
        )

    if noise_type == "corruption":
        if corruption_kind == "class":
            grid = np.linspace(0.0, 0.98, 250)
            candidates = [
                (rho, class_corruption_to_second_best(p_true, rho))
                for rho in grid
            ]
            return _select_candidate_closest_to_target(
                p_true, candidates, "rho_class_corruption", target_kl
            )

        if corruption_kind == "logit":
            rng = np.random.default_rng(seed)
            fixed_noise = rng.normal(size=p_true.shape)
            grid = np.linspace(0.0, 4.0, 250)
            candidates = [
                (sigma, logit_gaussian_corruption(p_true, sigma, fixed_noise))
                for sigma in grid
            ]
            return _select_candidate_closest_to_target(
                p_true, candidates, "sigma_logit_corruption", target_kl
            )

        raise ValueError("corruption_kind must be 'class' or 'logit'")

    raise ValueError(f"Unknown noise_type: {noise_type}")


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
        # q_i <= (1+delta) p_i
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

    for i, credal in enumerate(credal_sets):
        v = credal.vertices.astype(np.float32)
        padded[i, : len(v)] = v
        mask[i, : len(v)] = True

    return VertexBank(
        vertices=torch.tensor(padded, dtype=torch.float32, device=device),
        mask=torch.tensor(mask, dtype=torch.bool, device=device),
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
      - 'hard'   : CE(one-hot observed label, prediction)
      - 'soft'   : CE(soft target, prediction)
      - 'credal' : min_{q in M_i} CE(q, prediction), solved over precomputed vertices
    """
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
                loss = F.cross_entropy(logits, y_t[ids])

            elif mode == "soft":
                if soft_t is None:
                    raise ValueError("soft_targets are required in soft mode")
                qb = soft_t[ids]
                loss = -(qb * log_probs).sum(dim=-1).mean()

            elif mode == "credal":
                if vertex_bank is None:
                    raise ValueError("vertex_bank is required in credal mode")

                # V_b: [B, Vmax, C]
                V_b = vertex_bank.vertices[ids]
                mask_b = vertex_bank.mask[ids]

                # CE(v, s) for every available vertex v.
                # Invalid padded vertices receive +infinity.
                ce_all = -(V_b * log_probs[:, None, :]).sum(dim=-1)
                ce_all = ce_all.masked_fill(~mask_b, float("inf"))

                best_idx = torch.argmin(ce_all, dim=1)
                q_star = V_b[
                    torch.arange(len(ids), device=device),
                    best_idx,
                ].detach()

                # Stop-gradient target q_star; gradient flows only through logits.
                loss = -(q_star * log_probs).sum(dim=-1).mean()

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
# 8. Controlled experiment A: vary overlap, no noisy labels
# ============================================================


def run_overlap_study(
    output_dir: Path,
    n_train: int,
    n_test: int,
    repeats: int,
    train_config: TrainConfig,
    credal_delta: float,
    device: torch.device,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for overlap_name, distance in OVERLAP_LEVELS.items():
        # Same random seeds across overlap levels: only Gaussian separation changes.
        train_data = generate_gmm_dataset(n_train, distance, seed=1001)
        test_data = generate_gmm_dataset(n_test, distance, seed=2001)
        x_train, x_test = standardize_train_test(train_data.x, test_data.x)
        bayes = bayes_metrics(test_data.y, test_data.true_posterior)

        # Credal sets are generated from the exact posterior: NO soft-label noise here.
        vertex_banks: Dict[str, VertexBank] = {}
        for method in TRANSFORMATIONS:
            vertex_banks[method] = prepare_vertex_bank(
                train_data.true_posterior,
                train_data.true_posterior,
                method,
                credal_delta,
                device,
            )

        for repeat in range(repeats):
            model_seed = 5000 + repeat

            # Oracle posterior supervision.
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
            rows.append({
                "study": "overlap",
                "overlap": overlap_name,
                "pairwise_distance": distance,
                "repeat": repeat,
                "method": "oracle_posterior",
                "target_type": "baseline",
                "credal_delta": np.nan,
                "mean_credal_area": np.nan,
                "true_posterior_coverage": np.nan,
                **bayes,
                **metrics,
            })

            # Clean sampled hard labels.
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
            rows.append({
                "study": "overlap",
                "overlap": overlap_name,
                "pairwise_distance": distance,
                "repeat": repeat,
                "method": "clean_sampled_hard",
                "target_type": "baseline",
                "credal_delta": np.nan,
                "mean_credal_area": np.nan,
                "true_posterior_coverage": np.nan,
                **bayes,
                **metrics,
            })

            # Credal supervision from clean exact posterior.
            for method in TRANSFORMATIONS:
                bank = vertex_banks[method]
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
                rows.append({
                    "study": "overlap",
                    "overlap": overlap_name,
                    "pairwise_distance": distance,
                    "repeat": repeat,
                    "method": method,
                    "target_type": "credal",
                    "credal_delta": credal_delta,
                    "mean_credal_area": bank.mean_area,
                    "true_posterior_coverage": bank.true_posterior_coverage,
                    **bayes,
                    **metrics,
                })

    df = pd.DataFrame(rows)
    path = output_dir / "overlap_study.csv"
    df.to_csv(path, index=False)
    print(f"Saved overlap study to {path}")
    return df


# ============================================================
# 9. Controlled experiment B: fix overlap, vary noisy-label quality
# ============================================================


def run_noise_study(
    output_dir: Path,
    n_train: int,
    n_test: int,
    repeats: int,
    train_config: TrainConfig,
    credal_delta: float,
    device: torch.device,
    corruption_kind: str,
    temperature_direction: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overlap_name = NOISE_STUDY_OVERLAP
    distance = OVERLAP_LEVELS[overlap_name]

    train_data = generate_gmm_dataset(n_train, distance, seed=3001)
    test_data = generate_gmm_dataset(n_test, distance, seed=4001)
    x_train, x_test = standardize_train_test(train_data.x, test_data.x)
    bayes = bayes_metrics(test_data.y, test_data.true_posterior)

    rows: List[Dict[str, object]] = []
    quality_rows: List[Dict[str, object]] = []

    # Baselines independent of noisy-label mechanism/quality.
    for repeat in range(repeats):
        model_seed = 7000 + repeat

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
        rows.append({
            "study": "noise",
            "overlap": overlap_name,
            "noise_type": "baseline",
            "quality": "baseline",
            "target_mean_label_kl": 0.0,
            "achieved_mean_label_kl": 0.0,
            "top1_agreement_with_bayes": 1.0,
            "noise_parameter": np.nan,
            "noise_parameter_value": np.nan,
            "repeat": repeat,
            "method": "oracle_posterior",
            "target_type": "baseline",
            "mean_credal_area": np.nan,
            "true_posterior_coverage": np.nan,
            **bayes,
            **metrics,
        })

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
        rows.append({
            "study": "noise",
            "overlap": overlap_name,
            "noise_type": "baseline",
            "quality": "baseline",
            "target_mean_label_kl": np.nan,
            "achieved_mean_label_kl": np.nan,
            "top1_agreement_with_bayes": np.nan,
            "noise_parameter": np.nan,
            "noise_parameter_value": np.nan,
            "repeat": repeat,
            "method": "clean_sampled_hard",
            "target_type": "baseline",
            "mean_credal_area": np.nan,
            "true_posterior_coverage": np.nan,
            **bayes,
            **metrics,
        })

    noise_types = ["uniform", "temperature", "corruption"]

    for noise_index, noise_type in enumerate(noise_types):
        for quality_name, target_kl in QUALITY_TARGET_KL.items():
            noisy = make_noisy_probabilistic_labels(
                train_data.true_posterior,
                noise_type=noise_type,
                target_kl=target_kl,
                seed=9000 + noise_index,
                corruption_kind=corruption_kind,
                temperature_direction=temperature_direction,
            )

            quality_rows.append({
                "overlap": overlap_name,
                "noise_type": noise_type,
                "quality": quality_name,
                "target_mean_label_kl": target_kl,
                "achieved_mean_label_kl": noisy.achieved_mean_kl,
                "top1_agreement_with_bayes": noisy.top1_agreement_with_bayes,
                "noise_parameter": noisy.parameter_name,
                "noise_parameter_value": noisy.parameter_value,
            })

            # Precompute all credal sets once per noisy-label condition.
            banks: Dict[str, VertexBank] = {}
            for method in TRANSFORMATIONS:
                banks[method] = prepare_vertex_bank(
                    noisy.probabilities,
                    train_data.true_posterior,
                    method,
                    credal_delta,
                    device,
                )

            for repeat in range(repeats):
                model_seed = 7000 + repeat

                # Direct noisy-soft-label baseline.
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
                rows.append({
                    "study": "noise",
                    "overlap": overlap_name,
                    "noise_type": noise_type,
                    "quality": quality_name,
                    "target_mean_label_kl": target_kl,
                    "achieved_mean_label_kl": noisy.achieved_mean_kl,
                    "top1_agreement_with_bayes": noisy.top1_agreement_with_bayes,
                    "noise_parameter": noisy.parameter_name,
                    "noise_parameter_value": noisy.parameter_value,
                    "repeat": repeat,
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
                    rows.append({
                        "study": "noise",
                        "overlap": overlap_name,
                        "noise_type": noise_type,
                        "quality": quality_name,
                        "target_mean_label_kl": target_kl,
                        "achieved_mean_label_kl": noisy.achieved_mean_kl,
                        "top1_agreement_with_bayes": noisy.top1_agreement_with_bayes,
                        "noise_parameter": noisy.parameter_name,
                        "noise_parameter_value": noisy.parameter_value,
                        "repeat": repeat,
                        "method": method,
                        "target_type": "credal",
                        "mean_credal_area": bank.mean_area,
                        "true_posterior_coverage": bank.true_posterior_coverage,
                        **bayes,
                        **metrics,
                    })

    df = pd.DataFrame(rows)
    quality_df = pd.DataFrame(quality_rows)
    df_path = output_dir / "noise_study.csv"
    quality_path = output_dir / "noise_quality_summary.csv"
    df.to_csv(df_path, index=False)
    quality_df.to_csv(quality_path, index=False)
    print(f"Saved noise study to {df_path}")
    print(f"Saved noise-quality summary to {quality_path}")
    return df, quality_df


# ============================================================
# 10. Aggregation helpers
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
# 11. Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results_credal_gaussian")
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--credal-delta", type=float, default=DEFAULT_CREDAL_DELTA)
    parser.add_argument(
        "--study",
        choices=["overlap", "noise", "both"],
        default="both",
    )
    parser.add_argument(
        "--corruption-kind",
        choices=["class", "logit"],
        default="class",
        help="Third noise family: targeted class-mass shift or Gaussian logit noise.",
    )
    parser.add_argument(
        "--temperature-direction",
        choices=["overconfident", "underconfident"],
        default="overconfident",
        help="T<1 or T>1 for the temperature-noise study.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    config_dump = {
        "overlap_levels": OVERLAP_LEVELS,
        "quality_target_kl": QUALITY_TARGET_KL,
        "noise_study_overlap": NOISE_STUDY_OVERLAP,
        "transformations": TRANSFORMATIONS,
        "credal_delta": args.credal_delta,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "corruption_kind": args.corruption_kind,
        "temperature_direction": args.temperature_direction,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2)

    if args.study in {"overlap", "both"}:
        overlap_df = run_overlap_study(
            output_dir=output_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            repeats=args.repeats,
            train_config=train_config,
            credal_delta=args.credal_delta,
            device=device,
        )
        overlap_summary = aggregate_results(
            overlap_df,
            group_cols=["overlap", "method", "target_type"],
        )
        overlap_summary.to_csv(output_dir / "overlap_study_summary.csv", index=False)

    if args.study in {"noise", "both"}:
        noise_df, _quality_df = run_noise_study(
            output_dir=output_dir,
            n_train=args.n_train,
            n_test=args.n_test,
            repeats=args.repeats,
            train_config=train_config,
            credal_delta=args.credal_delta,
            device=device,
            corruption_kind=args.corruption_kind,
            temperature_direction=args.temperature_direction,
        )
        noise_summary = aggregate_results(
            noise_df,
            group_cols=["noise_type", "quality", "method", "target_type"],
        )
        noise_summary.to_csv(output_dir / "noise_study_summary.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
