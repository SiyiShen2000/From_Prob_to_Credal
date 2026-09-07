"""Four-method test-accuracy scatter plots at five matched training-area targets."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ("pmm", "lv", "qcor", "tv")
COLORS = {"pmm": "#0072B2", "lv": "#D55E00", "qcor": "#009E73", "tv": "#CC79A7"}
INPUT = Path(__file__).resolve().parent / "results_matched_kl" / "matched_study_summary.csv"


def make_plot(condition, output):
    if set(condition.method) != set(METHODS):
        raise ValueError("Expected exactly PMM, LV, QCOR and TV")
    if condition.duplicated(["method", "tau_index"]).any():
        raise ValueError("Expected one summary row per method/tau")
    if condition.groupby("tau_index").tau.nunique().ne(1).any():
        raise ValueError("All methods must share the same tau targets")
    targets = condition[["tau_index", "tau"]].drop_duplicates().sort_values("tau_index")
    taus = targets.tau.to_numpy(dtype=float)
    if len(taus) != 5 or not np.isfinite(taus).all() or np.any(np.diff(taus) <= 0):
        raise ValueError("Expected five distinct increasing tau values")
    for method in METHODS:
        rows = condition[condition.method.eq(method)]
        if len(rows) != 5 or set(rows.tau_index) != set(targets.tau_index):
            raise ValueError(f"Missing tau values for {method}")
    values = condition[["mean_credal_area_mean", "accuracy_mean"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1 + 1e-8):
        raise ValueError("Mean area and accuracy must be finite values in [0, 1]")
    if np.any(np.abs(condition.mean_credal_area_mean - condition.tau) >= .005):
        raise ValueError("Mean areas fail matched-area tolerance 0.005")
    first = condition.iloc[0]
    fig, axis = plt.subplots(figsize=(9, 6))
    for method in METHODS:
        rows = condition[condition.method.eq(method)].sort_values("tau_index")
        axis.scatter(rows.mean_credal_area_mean, rows.accuracy_mean, color=COLORS[method],
                     label=method.upper(), s=65, alpha=.85, edgecolors="white", linewidths=.7)
    axis.set_xticks(taus, [f"{tau:.6g}" for tau in taus])
    axis.set_xlabel("Mean normalized credal-set area")
    axis.set_ylabel("Mean test accuracy")
    noise = "Uniform" if first.noise_type == "uniform" else "Class corruption"
    axis.set_title(f"Matched-imprecision study: {noise}\n{first.quality} quality (rho={first.rho:g})")
    axis.grid(alpha=.25)
    axis.set_axisbelow(True)
    axis.legend(title="Method")
    fig.text(.5, .025, "Points: means across seeds; x = achieved training area. Ticks: target τ values.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    path = Path(output) / f"delta_matched_{first.noise_type}_{first.quality}_rho_{first.rho:g}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_study(data, output):
    required = {"noise_type", "quality", "rho", "method", "tau_index", "tau",
                "mean_credal_area_mean", "accuracy_mean"}
    if not required.issubset(data.columns):
        raise ValueError(f"Expected matched_study_summary.csv; missing {required - set(data.columns)}")
    if "loss" in data and data.loss.nunique() != 1:
        raise ValueError("Plot one training loss at a time")
    if data[list(required)].isna().any().any():
        raise ValueError("Required matched summary fields must not be missing")
    if not set(data.noise_type).issubset({"uniform", "corruption"}):
        raise ValueError("Unknown noise type")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for _, condition in data.groupby(["noise_type", "quality", "rho"]):
        make_plot(condition, output)
        count += 1
    if count == 0:
        raise ValueError("No experimental conditions found")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input.parent / "delta_matched_plots"
    count = plot_study(pd.read_csv(args.input), output)
    print(f"Created {count} figures in {output}")


if __name__ == "__main__":
    main()
