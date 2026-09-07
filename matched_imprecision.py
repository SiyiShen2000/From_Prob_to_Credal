"""Training-area matching, independent of validation/test performance."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import credal_gaussian_experiment as exp
import delta_search as search

EPSILON_A = .005


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


def common_targets(scans, seeds):
    """Intersect all four methods x five seeds, separately for each condition."""
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
        for index, tau in enumerate(np.linspace(low, high, 5)):
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


def load_scans(directory, splits, n_train, n_validation, rho_values):
    """Reuse selection scans only when configuration and exact split indices match."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    seeds = [s.seed_info["seed"] for s in splits]
    if (config.get("seeds") != seeds or config.get("n_train") != n_train
            or config.get("delta_selection", {}).get("n_validation") != n_validation
            or config.get("pairwise_distance") != exp.NOISE_PAIRWISE_DISTANCE
            or config.get("corruption_kind") != "class"
            or config.get("rho_values") != list(exp.quality_rho_levels(rho_values).values())):
        raise ValueError("Scan configuration does not match current seeds/data/noise settings")
    with np.load(directory / "validation_split_indices.npz") as indices:
        for split in splits:
            for kind, ids in (("train", split.train_indices), ("validation", split.validation_indices)):
                if not np.array_equal(indices[f"seed_{split.seed_info['seed']}_{kind}"], ids):
                    raise ValueError("Scan training/validation split mismatch")
    scans = pd.read_csv(directory / "delta_search_trials.csv")
    required = {"noise_type", "quality", "rho", "method", "seed", "delta", "mean_credal_area"}
    if not required.issubset(scans.columns):
        raise ValueError(f"Scan is missing columns: {required - set(scans.columns)}")
    return scans


def run_matched_study(output_dir, n_train, n_test, seeds, train_config, device,
                      rho_values, n_validation=1000, scan_dir=None):
    output_dir = Path(output_dir)
    if train_config.epochs < 1 or n_test < 1:
        raise ValueError("Require positive training epochs and test size")
    splits = search.prepare_search_splits(n_train, n_validation, seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "validation_split_indices.npz", **{
        f"seed_{s.seed_info['seed']}_{kind}": ids for s in splits
        for kind, ids in (("train", s.train_indices), ("validation", s.validation_indices))})
    labels = {
        (noise, quality, rho): [exp.make_noisy_probabilistic_labels(s.p_train, noise, rho).probabilities
                                for s in splits]
        for noise in exp.NOISE_TYPES for quality, rho in exp.quality_rho_levels(rho_values).items()
    }
    scans = (load_scans(scan_dir, splits, n_train, n_validation, rho_values) if scan_dir
             else scan_areas(splits, labels))
    observed = set(scans[["noise_type", "quality", "rho"]].itertuples(index=False, name=None))
    if observed != set(labels) or set(scans.seed) != set(seeds) or set(scans.method) != set(search.SEARCH_METHODS):
        raise ValueError("Scan conditions, methods or seeds do not match this experiment")
    scans.to_csv(output_dir / "matched_area_scan.csv", index=False)
    ranges, targets = common_targets(scans, seeds)
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
                                    repeat=split.repeat, split_seed=split.split_seed,
                                    delta=delta, interpolated_delta=initial, mean_credal_area=area,
                                    area_error=abs(area - target["tau"]), area_evaluations=evaluations))
        pd.DataFrame(matches).to_csv(output_dir / "matched_deltas.csv", index=False)
        print(f"[matched] {key} tau={target['tau']:.6f}: all 20 deltas verified", flush=True)
    rows = []
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
            model = exp.train_mlp(split.x_train, split.y_train, mode="credal", config=train_config,
                                 device=device, seed=split.seed_info["model_seed"], vertex_bank=bank)
            metrics = exp.evaluate_model(model, x_test, test.y, test.true_posterior, device)
            rows.append(dict(**match, study="matched_imprecision", loss=train_config.loss_name,
                             n_fit=len(split.x_train), n_validation=n_validation, n_test=n_test,
                             **metrics))
            pd.DataFrame(rows).to_csv(output_dir / "matched_study.csv", index=False)
            print(f"[matched test {len(rows)}/{len(matches)}] {key}/{match['method']} "
                  f"seed={match['seed']} tau={match['tau']:.4f} accuracy={metrics['accuracy']:.4f}", flush=True)
            del model, bank
    data = pd.DataFrame(rows)
    groups = ["noise_type", "quality", "rho", "loss", "method", "tau_index", "tau"]
    metrics = ["accuracy", "posterior_kl", "mean_credal_area", "delta", "area_error"]
    summary = data.groupby(groups)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "matched_study_summary.csv", index=False)
    return data
