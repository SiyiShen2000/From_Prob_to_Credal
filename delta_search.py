"""Validation-only delta selection, followed by held-out test evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch

import credal_gaussian_experiment as experiment


SEARCH_METHODS = ("pmm", "lv", "qcor", "tv")
UNIT_DELTA_GRID = tuple(i / 20 for i in range(1, 20))


@dataclass
class SearchSplit:
    repeat: int
    seed_info: Dict[str, int]
    split_seed: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    p_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    p_validation: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def prepare_search_splits(n_train: int, n_validation: int, seeds: Sequence[int]):
    """Generate training pools only; no test data are available during selection."""
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Delta selection requires exactly five distinct seeds")
    if not 0 < n_validation < n_train:
        raise ValueError("Require 0 < n_validation < n_train (training pool size)")
    splits = []
    for repeat, seed in enumerate(seeds):
        info = experiment.experiment_seeds(seed)
        data = experiment.generate_gmm_dataset(
            n_train, experiment.NOISE_PAIRWISE_DISTANCE, seed=info["train_seed"]
        )
        split_seed = int(np.random.SeedSequence([seed, 2]).generate_state(1)[0])
        permutation = np.random.default_rng(split_seed).permutation(n_train)
        val_ids, train_ids = permutation[:n_validation], permutation[n_validation:]
        x = data.x[train_ids]
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        splits.append(SearchSplit(
            repeat, info, split_seed, train_ids, val_ids,
            (x - mean) / std, data.y[train_ids], data.true_posterior[train_ids],
            (data.x[val_ids] - mean) / std,
            data.y[val_ids], data.true_posterior[val_ids], mean, std,
        ))
    return splits


def pmm_saturation_q95(training_labels: Sequence[np.ndarray]) -> float:
    """Pooled empirical Q95 of 1/min(p_i)-1 over fitting samples of all seeds."""
    if len(training_labels) != 5:
        raise ValueError("PMM saturation requires five training-label arrays")
    minimum = np.concatenate([p.min(axis=1) for p in training_labels])
    if not len(minimum) or np.any(~np.isfinite(minimum)) or np.any(minimum < 0):
        raise ValueError("Invalid training probabilities for PMM saturation")
    with np.errstate(divide="ignore", over="ignore"):
        saturation = np.sort(1.0 / minimum - 1.0)
    # Explicit interpolation avoids inf-inf warnings for labels with zero entries.
    position = .95 * (len(saturation) - 1)
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    if not np.isfinite(saturation[hi]):
        raise ValueError(
            "PMM saturation Q95 is infinite: too many zero/underflowed training "
            "probabilities. Use a rho with finite Q95 (class corruption rho=1 "
            "cannot have a finite saturation range)."
        )
    return float(saturation[lo] + (position - lo) * (saturation[hi] - saturation[lo]))


def pmm_delta_candidates(q95: float):
    """Increasing spacing; visit Q95 exactly, then continue until both criteria hold."""
    if not np.isfinite(q95) or q95 <= 0:
        raise ValueError("PMM Q95 must be finite and positive")
    delta, step = min(.05, q95), .05
    while True:
        yield float(delta)
        step *= 1.5
        next_delta = delta + step
        delta = min(q95, next_delta) if delta < q95 else next_delta
        if not np.isfinite(delta):
            raise RuntimeError("PMM search overflowed before meeting both stopping criteria")


def pmm_stop_reason(previous_a, current_a, delta, q95):
    plateau = previous_a is not None and current_a - previous_a < .005
    reached = delta >= q95
    if reached and plateau:
        return "q95_reached_and_adjacent_increment_below_0.005"
    return None


def _write_rows(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def search_condition(
    splits, labels, method, condition, config, device, output_dir,
    trial_rows, summary_rows,
):
    """Choose one delta using mean validation accuracy across all five seeds."""
    q95 = pmm_saturation_q95(labels) if method == "pmm" else np.nan
    candidates = pmm_delta_candidates(q95) if method == "pmm" else UNIT_DELTA_GRID
    best_score, best_delta = -np.inf, None
    best_states, best_stats = None, None
    previous_a = None
    stop_reason = "grid_exhausted"
    for candidate_index, delta in enumerate(candidates):
        print(f"[select] {condition} method={method}, delta={delta:g}", flush=True)
        scores, areas, states, stats = [], [], [], []
        for split, noisy in zip(splits, labels):
            bank = experiment.prepare_vertex_bank(
                noisy, split.p_train, method, delta, device
            )
            model = experiment.train_mlp(
                split.x_train, split.y_train, mode="credal", config=config,
                device=device, seed=split.seed_info["model_seed"], vertex_bank=bank,
            )
            metrics = experiment.evaluate_model(
                model, split.x_validation, split.y_validation, split.p_validation, device
            )
            scores.append(metrics["accuracy"])
            areas.append(bank.mean_area)
            stats.append((bank.mean_area, bank.true_posterior_coverage))
            states.append({key: value.detach().cpu().clone()
                           for key, value in model.state_dict().items()})
            trial_rows.append({
                **condition, "method": method, "loss": config.loss_name,
                "candidate_index": candidate_index, "delta": delta,
                "repeat": split.repeat, **split.seed_info, "split_seed": split.split_seed,
                "n_fit": len(split.x_train), "n_validation": len(split.x_validation),
                "validation_accuracy": metrics["accuracy"],
                "validation_posterior_kl": metrics["posterior_kl"],
                "mean_credal_area": bank.mean_area,
            })
            print(f"  seed={split.seed_info['seed']}, validation accuracy="
                  f"{scores[-1]:.4f}", flush=True)
            del model, bank
        mean_accuracy, mean_area = float(np.mean(scores)), float(np.mean(areas))
        a = mean_area
        increment = a - previous_a if previous_a is not None else np.nan
        reason = (pmm_stop_reason(previous_a, a, delta, q95)
                  if method == "pmm" else None)
        summary_rows.append({
            **condition, "method": method, "loss": config.loss_name,
            "candidate_index": candidate_index, "delta": delta, "n_seeds": 5,
            "validation_accuracy_mean": mean_accuracy,
            "validation_accuracy_std": float(np.std(scores, ddof=1)),
            "mean_credal_area": mean_area, "A_bar": a, "A_increment": increment,
            "A_metric": "normalized_area", "pmm_q95": q95,
            "stop_reason": reason or "",
        })
        # Candidates are increasing. Exact accuracy ties favor the smaller delta.
        if mean_accuracy > best_score:
            best_score, best_delta = mean_accuracy, delta
            best_states, best_stats = states, stats
        previous_a = a
        _write_rows(output_dir / "delta_search_trials.csv", trial_rows)
        _write_rows(output_dir / "delta_search_summary.csv", summary_rows)
        print(f"  mean validation accuracy={mean_accuracy:.4f}; "
              f"A_bar={a:.6f}; increment={increment:.6f}", flush=True)
        if reason:
            stop_reason = reason
            break
    checkpoint_dir = output_dir / "selected_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoints = []
    for split, state in zip(splits, best_states):
        name = f"{condition['noise_type']}_{condition['quality']}_{method}_seed_{split.seed_info['seed']}.pt"
        path = checkpoint_dir / name
        torch.save(state, path)
        checkpoints.append(path)
    selected = {
        **condition, "method": method, "loss": config.loss_name,
        "optimal_delta": best_delta, "validation_accuracy_mean": best_score,
        "n_seeds": 5, "pmm_q95": q95, "search_stop_reason": stop_reason,
        "n_candidates": candidate_index + 1,
    }
    return selected, checkpoints, best_stats


def run_delta_selection_study(
    output_dir: Path, n_train: int, n_test: int, seeds: Sequence[int],
    train_config, credal_delta: float, device, rho_values: Sequence[float],
    n_validation: int = 1000,
):
    """Search all conditions before generating/evaluating any test set.

    Winning fitting-subset checkpoints are evaluated without refitting on the
    validation data. Untuned comparators also train on the same fitting subset.
    """
    if train_config.epochs < 1 or n_test < 1:
        raise ValueError("Delta selection requires positive epochs and test size")
    splits = prepare_search_splits(n_train, n_validation, seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "validation_split_indices.npz", **{
        f"seed_{split.seed_info['seed']}_{kind}": ids
        for split in splits
        for kind, ids in (("train", split.train_indices), ("validation", split.validation_indices))
    })
    quality_rhos = experiment.quality_rho_levels(rho_values)
    condition_labels, quality_rows = {}, []
    for noise_type in experiment.NOISE_TYPES:
        for quality, rho in quality_rhos.items():
            key = (noise_type, quality)
            condition_labels[key] = []
            for split in splits:
                noisy = experiment.make_noisy_probabilistic_labels(split.p_train, noise_type, rho)
                condition_labels[key].append(noisy.probabilities)
                quality_rows.append({
                    "noise_type": noise_type, "quality": quality, "rho": rho,
                    "repeat": split.repeat, **split.seed_info,
                    "achieved_mean_label_kl": noisy.achieved_mean_kl,
                    "top1_agreement_with_bayes": noisy.top1_agreement_with_bayes,
                })
            # Fail before expensive searches when a finite PMM range is undefined.
            pmm_saturation_q95(condition_labels[key])

    trials, summaries, optimal_rows, winners = [], [], [], {}
    print("[select] Phase 1: validation-only delta search across five seeds", flush=True)
    for (noise_type, quality), labels in condition_labels.items():
        condition = dict(noise_type=noise_type, quality=quality, rho=quality_rhos[quality])
        for method in SEARCH_METHODS:
            selected, checkpoints, stats = search_condition(
                splits, labels, method, condition, train_config, device, output_dir,
                trials, summaries,
            )
            optimal_rows.append(selected)
            winners[(noise_type, quality, method)] = (selected, checkpoints, stats)
            _write_rows(output_dir / "optimal_deltas.csv", optimal_rows)

    print("[select] Phase 2: selected deltas frozen; final test evaluation", flush=True)
    rows = []
    for split in splits:
        test = experiment.generate_gmm_dataset(
            n_test, experiment.NOISE_PAIRWISE_DISTANCE, seed=split.seed_info["test_seed"]
        )
        x_test = (test.x - split.mean) / split.std
        common = {
            "study": "noise", "delta_mode": "select", "loss": train_config.loss_name,
            "pairwise_distance": experiment.NOISE_PAIRWISE_DISTANCE,
            "repeat": split.repeat, **split.seed_info, "split_seed": split.split_seed,
            "n_fit": len(split.x_train), "n_validation": len(split.x_validation),
            **experiment.bayes_metrics(test.y, test.true_posterior),
        }

        def evaluate(model, metadata):
            metrics = experiment.evaluate_model(model, x_test, test.y, test.true_posterior, device)
            rows.append({**common, **metadata, **metrics})
            print(f"[test] seed={split.seed_info['seed']} {metadata['method']} "
                  f"{metadata['noise_type']}/{metadata['quality']}: "
                  f"accuracy={metrics['accuracy']:.4f}", flush=True)
            _write_rows(output_dir / "noise_study.csv", rows)

        for method, mode in (("oracle_posterior", "soft"), ("clean_sampled_hard", "hard")):
            model = experiment.train_mlp(
                split.x_train, split.y_train, mode=mode, config=train_config, device=device,
                seed=split.seed_info["model_seed"], soft_targets=split.p_train if mode == "soft" else None,
            )
            evaluate(model, dict(method=method, target_type="baseline", noise_type="baseline",
                                 quality="baseline", rho=np.nan, credal_delta=np.nan,
                                 mean_credal_area=np.nan, true_posterior_coverage=np.nan))
            del model
        for (noise_type, quality), labels in condition_labels.items():
            noisy = labels[split.repeat]
            condition = dict(noise_type=noise_type, quality=quality, rho=quality_rhos[quality])
            model = experiment.train_mlp(
                split.x_train, split.y_train, mode="soft", soft_targets=noisy,
                config=train_config, device=device, seed=split.seed_info["model_seed"],
            )
            evaluate(model, dict(**condition, method="noisy_soft", target_type="baseline",
                                 credal_delta=np.nan, mean_credal_area=np.nan,
                                 true_posterior_coverage=np.nan))
            del model
            for method in experiment.TRANSFORMATIONS:
                if method in SEARCH_METHODS:
                    selected, checkpoints, stats = winners[(noise_type, quality, method)]
                    model = experiment.SimpleMLP(hidden_dims=train_config.hidden_dims).to(device)
                    model.load_state_dict(torch.load(checkpoints[split.repeat], map_location=device, weights_only=True))
                    delta = selected["optimal_delta"]
                    area, coverage = stats[split.repeat]
                else:
                    delta = credal_delta
                    bank = experiment.prepare_vertex_bank(noisy, split.p_train, method, delta, device)
                    model = experiment.train_mlp(
                        split.x_train, split.y_train, mode="credal", vertex_bank=bank,
                        config=train_config, device=device, seed=split.seed_info["model_seed"],
                    )
                    area, coverage = bank.mean_area, bank.true_posterior_coverage
                    del bank
                evaluate(model, dict(**condition, method=method, target_type="credal",
                                     credal_delta=delta, mean_credal_area=area,
                                     true_posterior_coverage=coverage))
                del model
    data = pd.DataFrame(rows)
    _write_rows(output_dir / "noise_quality_summary.csv", quality_rows)
    summary = experiment.aggregate_results(
        data, ["noise_type", "quality", "rho", "loss", "method", "target_type", "credal_delta", "delta_mode"]
    )
    summary.to_csv(output_dir / "noise_study_summary.csv", index=False)
    return data, pd.DataFrame(optimal_rows)
