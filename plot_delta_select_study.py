"""Noise-style accuracy/KL plots annotated with validation-selected deltas."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plot_noise_study as base

METHODS = ("pmm", "lv", "qcor", "tv")
INPUT = Path(__file__).resolve().parent / "results_delta_kl" / "noise_study_summary.csv"


def plot_study(data, output):
    required = {"noise_type", "quality", "rho", "method", "accuracy_mean",
                "accuracy_std", "posterior_kl_mean", "credal_delta", "delta_mode"}
    if not required.issubset(data.columns):
        raise ValueError(f"Expected select noise_study_summary.csv; missing {required - set(data.columns)}")
    if set(data.delta_mode.dropna()) != {"select"}:
        raise ValueError("Input must contain only select-mode results")
    if "loss" in data and data.loss.nunique() != 1:
        raise ValueError("Plot one training loss at a time")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = data[data.noise_type.eq("baseline")]
    conditions = data[~data.noise_type.eq("baseline")]
    count = 0
    for (noise, quality, rho), condition in conditions.groupby(["noise_type", "quality", "rho"]):
        if noise not in base.NOISE_TYPES:
            raise ValueError(f"Unknown noise type: {noise}")
        labels = dict(base.METHOD_LABELS)
        for method in METHODS:
            rows = condition[condition.method.eq(method)]
            if len(rows) != 1:
                raise ValueError(f"Expected one selected delta for {noise}/{quality}/{method}")
            delta = float(rows.iloc[0].credal_delta)
            if not np.isfinite(delta) or delta <= 0 or (method != "pmm" and delta >= 1):
                raise ValueError(f"Invalid selected delta: {method}={delta}")
            labels[method] = f"{base.METHOD_LABELS[method]} (δ*={delta:.6g})"
        base.make_plot(data, baseline, noise, rho, output, method_labels=labels,
                       study_title="Optimal delta study", file_prefix="delta_select")
        count += 1
    if count == 0:
        raise ValueError("No experimental conditions found")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input.parent / "delta_select_plots"
    count = plot_study(pd.read_csv(args.input), output)
    print(f"Created {count} figures in {output}")


if __name__ == "__main__":
    main()
