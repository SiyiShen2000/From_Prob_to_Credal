"""Validation-only delta selection, followed by held-out test evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch

import credal_gaussian_experiment as experiment


from credal_area_scan import (SEARCH_METHODS, UNIT_DELTA_GRID, pmm_saturation_q95,
                              pmm_delta_candidates, pmm_stop_reason)


@dataclass
class SearchSplit:
    repeat: int
    seed_info: Dict[str, int]
    validation_seed: int
    x_train: np.ndarray
    y_train: np.ndarray
    p_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def prepare_search_splits(n_train: int, n_validation: int, seeds: Sequence[int]):
    """Full training set plus independent clean-hard validation data; no test access."""
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Delta selection requires exactly five distinct seeds")
    if n_train < 1 or n_validation < 1:
        raise ValueError("Require positive independent training and validation sizes")
    splits = []
    for repeat, seed in enumerate(seeds):
        info = experiment.experiment_seeds(seed)
        data = experiment.generate_gmm_dataset(
            n_train, experiment.NOISE_PAIRWISE_DISTANCE, seed=info["train_seed"]
        )
        validation_seed = int(np.random.SeedSequence([seed, 2]).generate_state(1)[0])
        validation = experiment.generate_gmm_dataset(
            n_validation, experiment.NOISE_PAIRWISE_DISTANCE, seed=validation_seed
        )
        x = data.x
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        splits.append(SearchSplit(
            repeat, info, validation_seed,
            (x - mean) / std, data.y, data.true_posterior,
            (validation.x - mean) / std, validation.y, mean, std,
        ))
    return splits


def _write_rows(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def search_condition(
    splits, labels, method, condition, config, device, output_dir,
    trial_rows, summary_rows,
):
    """Choose one delta by maximum mean clean-hard validation accuracy over five seeds."""
    q95 = pmm_saturation_q95(labels) if method == "pmm" else np.nan
    candidates = pmm_delta_candidates(q95) if method == "pmm" else UNIT_DELTA_GRID
    best_score, best_delta = -np.inf, None
    best_states, best_stats = None, None
    previous_a = None
    stop_reason = "grid_exhausted"
    for candidate_index, delta in enumerate(candidates):
        print(f"[select] {condition} method={method}, delta={delta:g}", flush=True)
        scores, areas, trajectory_rates, trajectory_counts = [], [], [], []
        trajectory_totals = []
        states, stats = [], []
        for split, noisy in zip(splits, labels):
            bank = experiment.prepare_vertex_bank(
                noisy, split.p_train, method, delta, device
            )
            qstar_stats = {}
            model = experiment.train_mlp(
                split.x_train, split.y_train, mode="credal", config=config,
                device=device, seed=split.seed_info["model_seed"], vertex_bank=bank,
                qstar_statistics=qstar_stats,
            )
            validation_predictions = experiment.predict_probabilities(
                model, split.x_validation, device
            )
            validation_accuracy = float(np.mean(
                validation_predictions.argmax(axis=1) == split.y_validation
            ))
            scores.append(validation_accuracy)
            areas.append(bank.mean_area)
            trajectory_rates.append(qstar_stats["qstar_trajectory_boundary_rate"])
            trajectory_counts.append(qstar_stats["qstar_trajectory_boundary_count"])
            trajectory_totals.append(qstar_stats["qstar_trajectory_evaluated_count"])
            stats.append((bank.mean_area, bank.true_posterior_coverage, dict(qstar_stats)))
            states.append({key: value.detach().cpu().clone()
                           for key, value in model.state_dict().items()})
            trial_rows.append({
                **condition, "method": method, "loss": config.loss_name,
                "candidate_index": candidate_index, "delta": delta,
                "repeat": split.repeat, **split.seed_info, "validation_seed": split.validation_seed,
                "n_fit": len(split.x_train), "n_validation": len(split.x_validation),
                "validation_accuracy": validation_accuracy,
                "mean_credal_area": bank.mean_area,
                **qstar_stats,
            })
            print(f"  seed={split.seed_info['seed']}, clean-hard validation accuracy="
                  f"{scores[-1]:.4f}", flush=True)
            del model, bank
        mean_validation_accuracy, mean_area = float(np.mean(scores)), float(np.mean(areas))
        a = mean_area
        increment = a - previous_a if previous_a is not None else np.nan
        reason = (pmm_stop_reason(previous_a, a, delta, q95)
                  if method == "pmm" else None)
        summary_rows.append({
            **condition, "method": method, "loss": config.loss_name,
            "candidate_index": candidate_index, "delta": delta, "n_seeds": 5,
            "validation_accuracy_mean": mean_validation_accuracy,
            "validation_accuracy_std": float(np.std(scores, ddof=1)),
            "qstar_trajectory_boundary_count_mean": float(np.mean(trajectory_counts)),
            "qstar_trajectory_boundary_count_std": float(np.std(trajectory_counts, ddof=1)),
            "qstar_trajectory_evaluated_count_mean": float(np.mean(trajectory_totals)),
            "qstar_trajectory_evaluated_count_std": float(np.std(trajectory_totals, ddof=1)),
            "qstar_trajectory_boundary_rate_mean": float(np.mean(trajectory_rates)),
            "qstar_trajectory_boundary_rate_std": float(np.std(trajectory_rates, ddof=1)),
            "mean_credal_area": mean_area, "A_bar": a, "A_increment": increment,
            "A_metric": "normalized_area", "pmm_q95": q95,
            "stop_reason": reason or "",
        })
        # Candidates are increasing. Exact accuracy ties favor the smaller delta.
        if mean_validation_accuracy > best_score:
            best_score, best_delta = mean_validation_accuracy, delta
            best_states, best_stats = states, stats
        previous_a = a
        _write_rows(output_dir / "delta_search_trials.csv", trial_rows)
        _write_rows(output_dir / "delta_search_summary.csv", summary_rows)
        print(f"  mean clean-hard validation accuracy={mean_validation_accuracy:.6f}; "
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
        "qstar_trajectory_boundary_count_mean": float(np.mean([
            item[2]["qstar_trajectory_boundary_count"] for item in best_stats
        ])),
        "qstar_trajectory_boundary_rate_mean": float(np.mean([
            item[2]["qstar_trajectory_boundary_rate"] for item in best_stats
        ])),
        "qstar_trajectory_evaluated_count_mean": float(np.mean([
            item[2]["qstar_trajectory_evaluated_count"] for item in best_stats
        ])),
    }
    return selected, checkpoints, best_stats


def run_delta_selection_study(
    output_dir: Path, n_train: int, n_test: int, seeds: Sequence[int],
    train_config, credal_delta: float, device, rho_values: Sequence[float],
    n_validation: int = 100,
):
    """Search all conditions before generating/evaluating any test set.

    All n_train samples are used for fitting. An additional independent clean-hard
    validation set selects delta; its labels never enter training. Evaluate saved
    winning checkpoints on independent test data. Comparators use the same train set.
    """
    if train_config.epochs < 1 or n_test < 1:
        raise ValueError("Delta selection requires positive epochs and test size")
    splits = prepare_search_splits(n_train, n_validation, seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "validation_data.npz", **{
        f"seed_{s.seed_info['seed']}_{kind}": values
        for s in splits for kind, values in (
            ("x_standardized", s.x_validation), ("y_clean", s.y_validation),
            ("validation_seed", np.asarray(s.validation_seed)))
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
                splits, labels,
                method, condition, train_config, device, output_dir,
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
            "repeat": split.repeat, **split.seed_info, "validation_seed": split.validation_seed,
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
                                 mean_credal_area=np.nan, true_posterior_coverage=np.nan,
                                 **experiment.unavailable_qstar_statistics()))
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
                                 true_posterior_coverage=np.nan,
                                 **experiment.unavailable_qstar_statistics()))
            del model
            for method in experiment.TRANSFORMATIONS:
                bank = None
                if method in SEARCH_METHODS:
                    selected, checkpoints, stats = winners[(noise_type, quality, method)]
                    model = experiment.SimpleMLP(hidden_dims=train_config.hidden_dims).to(device)
                    model.load_state_dict(torch.load(checkpoints[split.repeat], map_location=device, weights_only=True))
                    delta = selected["optimal_delta"]
                    area, coverage, qstar_stats = stats[split.repeat]
                else:
                    delta = credal_delta
                    bank = experiment.prepare_vertex_bank(noisy, split.p_train, method, delta, device)
                    qstar_stats = {}
                    model = experiment.train_mlp(
                        split.x_train, split.y_train, mode="credal", vertex_bank=bank,
                        config=train_config, device=device, seed=split.seed_info["model_seed"],
                        qstar_statistics=qstar_stats,
                    )
                    area, coverage = bank.mean_area, bank.true_posterior_coverage
                evaluate(model, dict(**condition, method=method, target_type="credal",
                                     credal_delta=delta, mean_credal_area=area,
                                     true_posterior_coverage=coverage,
                                     **qstar_stats))
                del model
                if bank is not None:
                    del bank
    data = pd.DataFrame(rows)
    _write_rows(output_dir / "noise_quality_summary.csv", quality_rows)
    summary = experiment.aggregate_results(
        data, ["noise_type", "quality", "rho", "loss", "method", "target_type", "credal_delta", "delta_mode"]
    )
    summary.to_csv(output_dir / "noise_study_summary.csv", index=False)
    return data, pd.DataFrame(optimal_rows)
