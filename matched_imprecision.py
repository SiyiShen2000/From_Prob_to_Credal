"""Training-area matching, independent of validation/test performance."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import credal_gaussian_experiment as exp
import credal_area_scan as search

EPSILON_A = .005


@dataclass
class TrainingData:
    repeat: int
    seed_info: dict
    x_train: np.ndarray
    y_train: np.ndarray
    p_train: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def prepare_training_data(n_train, seeds):
    if n_train < 1 or len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Require positive n_train and five distinct seeds")
    result = []
    for repeat, seed in enumerate(seeds):
        info = exp.experiment_seeds(seed)
        data = exp.generate_gmm_dataset(n_train, exp.NOISE_PAIRWISE_DISTANCE, seed=info["train_seed"])
        mean, std = data.x.mean(axis=0, keepdims=True), data.x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1., std)
        result.append(TrainingData(repeat, info, (data.x - mean) / std,
                                   data.y, data.true_posterior, mean, std))
    return result


def mean_area(labels, method, delta):
    return float(np.mean([
        exp.normalized_simplex_area(exp.build_credal_set_3class(p, method, delta).vertices)
        for p in labels
    ]))


def refine_delta(curve, tau, area_function, epsilon=EPSILON_A):
    """Interpolate within a scanned bracket, then bisect using recomputed areas."""
    curve = curve.sort_values("delta")
    ds, areas = curve.delta.to_numpy(), curve.mean_credal_area.to_numpy()
    if len(ds) < 2 or not np.isfinite(ds).all() or not np.isfinite(areas).all():
        raise ValueError("Need at least two finite scan points")
    if np.any(np.diff(ds) <= 0) or np.any(np.diff(areas) < -1e-8):
        raise ValueError("Scan must have unique increasing deltas and monotone areas")
    if not areas[0] <= tau <= areas[-1]:
        raise ValueError("Target area outside scanned range; extrapolation is forbidden")
    hi = min(int(np.searchsorted(areas, tau, side="left")), len(ds) - 1)
    lo = max(0, hi - 1)
    left, right = float(ds[lo]), float(ds[hi])
    delta = left if areas[hi] == areas[lo] else float(
        left + (tau - areas[lo]) / (areas[hi] - areas[lo]) * (right - left))
    initial = delta
    for iteration in range(61):
        actual = area_function(delta)
        if not np.isfinite(actual):
            raise ValueError("Nonfinite recomputed area")
        if abs(actual - tau) < epsilon:
            return delta, actual, initial, iteration + 1
        if actual < tau:
            left = delta
        else:
            right = delta
        next_delta = (left + right) / 2
        if next_delta == delta:
            break
        delta = next_delta
    raise RuntimeError(f"Area refinement failed: tau={tau}, area={actual}; check scan provenance")


def common_targets(scans, seeds, n_tau=9):
    """Intersect all four methods x five seeds, separately for each condition."""
    if n_tau < 2:
        raise ValueError("Require at least two tau targets")
    ranges, targets = [], []
    for (noise, quality, rho), condition in scans.groupby(["noise_type", "quality", "rho"]):
        intervals = []
        for method in search.SEARCH_METHODS:
            for seed in seeds:
                curve = condition[(condition.method == method) & (condition.seed == seed)].sort_values("delta")
                if len(curve) < 2:
                    raise ValueError(f"Missing scan: {noise}/{quality}/{method}/seed={seed}")
                ds, a = curve.delta.to_numpy(), curve.mean_credal_area.to_numpy()
                if (not np.isfinite(ds).all() or not np.isfinite(a).all()
                        or np.any(np.diff(ds) <= 0) or np.any(np.diff(a) < -1e-8)
                        or np.any(ds <= 0) or (method != "pmm" and np.any(ds >= 1))
                        or np.any(a < -1e-8) or np.any(a > 1 + 1e-8)):
                    raise ValueError("Invalid/nonmonotone scan or illegal delta")
                low, high = float(a.min()), float(a.max())
                intervals.append((low, high))
                ranges.append(dict(noise_type=noise, quality=quality, rho=rho,
                                   method=method, seed=seed, area_min=low, area_max=high))
        low, high = max(a for a, _ in intervals), min(b for _, b in intervals)
        if high - low <= 1e-12:
            raise ValueError(f"No nondegenerate common area interval for {noise}/{quality}: [{low}, {high}]")
        for index, tau in enumerate(np.linspace(low, high, n_tau)):
            targets.append(dict(noise_type=noise, quality=quality, rho=rho,
                                tau_index=index, tau=float(tau), common_area_min=low,
                                common_area_max=high))
    return pd.DataFrame(ranges), pd.DataFrame(targets)


def scan_areas(splits, labels_by_condition):
    """Same grids and PMM stopping rule as selection, without training scan models."""
    rows = []
    for (noise, quality, rho), labels in labels_by_condition.items():
        for method in search.SEARCH_METHODS:
            q95 = search.pmm_saturation_q95(labels) if method == "pmm" else np.nan
            candidates = search.pmm_delta_candidates(q95) if method == "pmm" else search.UNIT_DELTA_GRID
            previous = None
            for delta in candidates:
                areas = []
                for split, p in zip(splits, labels):
                    area = mean_area(p, method, delta)
                    areas.append(area)
                    rows.append(dict(noise_type=noise, quality=quality, rho=rho,
                                     method=method, delta=delta, mean_credal_area=area,
                                     **split.seed_info))
                current = float(np.mean(areas))
                print(f"[area scan] {noise}/{quality}/{method} delta={delta:g} area={current:.6f}", flush=True)
                if method == "pmm" and search.pmm_stop_reason(previous, current, delta, q95):
                    break
                previous = current
    return pd.DataFrame(rows)


def run_matched_study(output_dir, n_train, n_test, seeds, train_config, device,
                      rho_values, n_tau=9):
    output_dir = Path(output_dir)
    if train_config.epochs < 1 or n_test < 1:
        raise ValueError("Require positive training epochs and test size")
    if n_tau < 2:
        raise ValueError("Require at least two tau targets")
    splits = prepare_training_data(n_train, seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        (noise, quality, rho): [exp.make_noisy_probabilistic_labels(s.p_train, noise, rho).probabilities
                                for s in splits]
        for noise in exp.NOISE_TYPES for quality, rho in exp.quality_rho_levels(rho_values).items()
    }
    scans = scan_areas(splits, labels)
    observed = set(scans[["noise_type", "quality", "rho"]].itertuples(index=False, name=None))
    if observed != set(labels) or set(scans.seed) != set(seeds) or set(scans.method) != set(search.SEARCH_METHODS):
        raise ValueError("Scan conditions, methods or seeds do not match this experiment")
    scans.to_csv(output_dir / "matched_area_scan.csv", index=False)
    ranges, targets = common_targets(scans, seeds, n_tau)
    ranges.to_csv(output_dir / "matched_area_ranges.csv", index=False)
    targets.to_csv(output_dir / "matched_tau_targets.csv", index=False)
    matches = []
    # Freeze ALL matched deltas before any final training or test generation.
    for target in targets.to_dict("records"):
        key = (target["noise_type"], target["quality"], target["rho"])
        condition = scans[(scans.noise_type == key[0]) & (scans.quality == key[1]) & (scans.rho == key[2])]
        for method in search.SEARCH_METHODS:
            for split, p in zip(splits, labels[key]):
                curve = condition[(condition.method == method) & (condition.seed == split.seed_info["seed"])]
                delta, area, initial, evaluations = refine_delta(
                    curve, target["tau"], lambda d: mean_area(p, method, d))
                matches.append(dict(**target, method=method, **split.seed_info,
                                    repeat=split.repeat,
                                    delta=delta, interpolated_delta=initial, mean_credal_area=area,
                                    area_error=abs(area - target["tau"]), area_evaluations=evaluations))
        pd.DataFrame(matches).to_csv(output_dir / "matched_deltas.csv", index=False)
        print(f"[matched] {key} tau={target['tau']:.6f}: all 20 deltas verified", flush=True)
    rows, reference_rows = [], []
    for split in splits:
        test = exp.generate_gmm_dataset(n_test, exp.NOISE_PAIRWISE_DISTANCE, seed=split.seed_info["test_seed"])
        x_test = (test.x - split.mean) / split.std
        for match in matches:
            if match["seed"] != split.seed_info["seed"]:
                continue
            key = (match["noise_type"], match["quality"], match["rho"])
            bank = exp.prepare_vertex_bank(labels[key][split.repeat], split.p_train,
                                           match["method"], match["delta"], device)
            if abs(bank.mean_area - match["tau"]) >= EPSILON_A:
                raise RuntimeError("Final training bank failed area tolerance")
            qstar_stats = {}
            model = exp.train_mlp(split.x_train, split.y_train, mode="credal", config=train_config,
                                 device=device, seed=split.seed_info["model_seed"], vertex_bank=bank,
                                 qstar_statistics=qstar_stats)
            metrics = exp.evaluate_model(model, x_test, test.y, test.true_posterior, device)
            rows.append(dict(**match, study="matched_imprecision", loss=train_config.loss_name,
                             n_fit=len(split.x_train), n_validation=0, n_test=n_test,
                             true_posterior_coverage=bank.true_posterior_coverage,
                             **qstar_stats, **metrics))
            pd.DataFrame(rows).to_csv(output_dir / "matched_study.csv", index=False)
            print(f"[matched test {len(rows)}/{len(matches)}] {key}/{match['method']} "
                  f"seed={match['seed']} tau={match['tau']:.4f} accuracy={metrics['accuracy']:.4f}", flush=True)
            del model, bank
        # References are trained once per applicable condition/seed, never per tau.
        tasks = [("baseline", "baseline", np.nan, "clean_sampled_hard", "hard", None),
                 ("baseline", "baseline", np.nan, "oracle_posterior", "soft", split.p_train)]
        for (noise, quality, rho), noisy_arrays in labels.items():
            noisy = noisy_arrays[split.repeat]
            tasks.append((noise, quality, rho, "noisy_soft", "soft", noisy))
            tasks.extend((noise, quality, rho, method, "credal", noisy)
                         for method in exp.TRANSFORMATIONS if method not in search.SEARCH_METHODS)
        for noise, quality, rho, method, mode, targets_for_training in tasks:
            bank = (exp.prepare_vertex_bank(targets_for_training, split.p_train, method,
                                            exp.DEFAULT_CREDAL_DELTA, device)
                    if mode == "credal" else None)
            stats = {} if bank is not None else exp.unavailable_qstar_statistics()
            model = exp.train_mlp(split.x_train, split.y_train, mode=mode, config=train_config,
                                 device=device, seed=split.seed_info["model_seed"], vertex_bank=bank,
                                 soft_targets=targets_for_training if mode == "soft" else None,
                                 qstar_statistics=stats if bank is not None else None)
            metrics = exp.evaluate_model(model, x_test, test.y, test.true_posterior, device)
            reference_rows.append(dict(noise_type=noise, quality=quality, rho=rho, method=method,
                                       loss=train_config.loss_name, repeat=split.repeat, **split.seed_info,
                                       n_fit=n_train, n_validation=0, n_test=n_test,
                                       mean_credal_area=bank.mean_area if bank is not None else np.nan,
                                       true_posterior_coverage=bank.true_posterior_coverage if bank is not None else np.nan,
                                       **stats, **metrics))
            pd.DataFrame(reference_rows).to_csv(output_dir / "matched_references.csv", index=False)
            print(f"[matched reference] seed={split.seed_info['seed']} {noise}/{quality}/{method} "
                  f"accuracy={metrics['accuracy']:.4f}", flush=True)
            del model, bank
    data = pd.DataFrame(rows)
    groups = ["noise_type", "quality", "rho", "loss", "method", "tau_index", "tau"]
    metrics = ["accuracy", "posterior_kl", "mean_credal_area", "delta", "area_error", "true_posterior_coverage",
               "qstar_trajectory_boundary_count", "qstar_trajectory_evaluated_count",
               "qstar_trajectory_boundary_rate"]
    summary = data.groupby(groups)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "matched_study_summary.csv", index=False)
    references = pd.DataFrame(reference_rows)
    reference_summary = references.groupby(["noise_type", "quality", "rho", "loss", "method"],
                                           dropna=False)[[m for m in metrics if m not in ("delta", "area_error")]].agg(["mean", "std"])
    reference_summary.columns = [f"{metric}_{stat}" for metric, stat in reference_summary.columns]
    reference_summary.reset_index().to_csv(output_dir / "matched_references_summary.csv", index=False)
    return data
