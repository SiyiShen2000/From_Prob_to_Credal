"""Compare accuracy and posterior KL for each noise type and rho."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save PNGs without requiring a GUI/Tk installation.
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results_noise_kl" / "noise_study_summary.csv"
OUTPUT = ROOT / "results_noise_kl" / "noise_plots"

NOISE_TYPES = ["uniform", "corruption"]
QUALITIES = ["high", "medium", "low"]
METHODS = [
    "alpha_cut",
    "inverse_pignistic",
    "dubois_prade",
    "pmm",
    "lv",
    "qcor",
    "tv",
    "comparative",
]
METHOD_LABELS = {
    "alpha_cut": "Alpha-cut",
    "inverse_pignistic": "Inverse pignistic",
    "dubois_prade": "Dubois-Prade",
    "pmm": "PMM",
    "lv": "LV",
    "qcor": "QCOR",
    "tv": "TV",
    "comparative": "Comparative",
}
COLORS = {
    "alpha_cut": "#4C78A8",
    "inverse_pignistic": "#F58518",
    "dubois_prade": "#54A24B",
    "pmm": "#E45756",
    "lv": "#72B7B2",
    "qcor": "#B279A2",
    "tv": "#FF9DA6",
    "comparative": "#9D755D",
}


def accuracy_std(row):
    """An undefined sample std (one repeat) must not be displayed as zero."""
    value = row["accuracy_std"]
    if pd.isna(value):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("accuracy_std must be finite and nonnegative, or NaN")
    return value


def add_baselines(axis, baseline, noisy_soft, column, title):
    rows = {
        "clean sampled-hard": baseline.loc[
            baseline["method"].eq("clean_sampled_hard")
        ].iloc[0],
        "oracle posterior": baseline.loc[
            baseline["method"].eq("oracle_posterior")
        ].iloc[0],
        "noisy soft": noisy_soft,
    }
    styles = {
        "clean sampled-hard": ("#777777", ":"),
        "oracle posterior": ("#222222", "--"),
        "noisy soft": ("#D62728", "-."),
    }
    for label, row in rows.items():
        value = row[column]
        color, linestyle = styles[label]
        if column == "accuracy_mean":
            std = accuracy_std(row)
            if std is not None:
                axis.axhspan(value - std, value + std, color=color, alpha=0.10, zorder=0)
        axis.axhline(
            value,
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
            label=f"Baseline: {label}",
        )
    axis.set_title(title)


def make_plot(data, baseline, noise_type, rho, output=OUTPUT, *,
              method_labels=None, study_title="Noise study", file_prefix="noise"):
    method_labels = METHOD_LABELS if method_labels is None else method_labels
    condition = data[
        data["noise_type"].eq(noise_type) & data["rho"].eq(rho)
    ]
    if condition["method"].duplicated().any():
        raise ValueError("Expected one summary row per method and noise/rho condition.")
    for method in ("oracle_posterior", "clean_sampled_hard"):
        if baseline["method"].eq(method).sum() != 1:
            raise ValueError(f"Expected exactly one global baseline row for {method}.")
    subset = condition.set_index("method").loc[METHODS].reset_index()
    noisy_soft_rows = condition[condition["method"].eq("noisy_soft")]
    if len(noisy_soft_rows) != 1:
        raise ValueError(
            "Expected exactly one noisy_soft row for "
            f"noise_type={noise_type}, rho={rho}; "
            f"found {len(noisy_soft_rows)}."
        )
    noisy_soft = noisy_soft_rows.iloc[0]
    quality = noisy_soft["quality"]

    figure, axes = plt.subplots(1, 2, figsize=(13, 6))
    metrics = [
        ("accuracy_mean", "Accuracy", "Accuracy (mean ± 1 SD across seeds)"),
        ("posterior_kl_mean", "Posterior KL", "Posterior KL (lower is better)"),
    ]

    for axis, (column, ylabel, title) in zip(axes, metrics):
        for method in METHODS:
            row = subset[subset["method"].eq(method)].iloc[0]
            std = accuracy_std(row) if column == "accuracy_mean" else None
            axis.bar(
                method_labels[method],
                row[column],
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.7,
                yerr=std,
                capsize=4,
                error_kw={"ecolor": "#222222", "elinewidth": 1.2, "capthick": 1.2},
            )
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", labelrotation=55, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        add_baselines(axis, baseline, noisy_soft, column, title)

    std_rows = pd.concat([subset, noisy_soft.to_frame().T, baseline])
    note = "Accuracy: error bars and baseline bands show ±1 sample SD (not confidence intervals)."
    if std_rows["accuracy_std"].isna().any():
        note += "\nUndefined SD is omitted (e.g. only one repeat)."
    figure.text(0.5, 0.025, note, ha="center", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=2,
            frameon=False,
        )
    figure.suptitle(
        f"{study_title}: {'Class corruption' if noise_type == 'corruption' else 'Uniform'} noise, {quality} label quality (rho={rho:g})",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    figure.subplots_adjust(top=0.72, bottom=0.30, wspace=0.24)
    figure.savefig(
        output / f"{file_prefix}_{noise_type}_{quality}_rho_{rho}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    if "rho" not in data.columns:
        raise ValueError("This CSV uses the old KL-calibrated format. Run the rho-based noise study first.")
    required = {"noise_type", "method", "accuracy_mean", "accuracy_std", "posterior_kl_mean"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Expected a noise_study_summary.csv; missing columns: {sorted(missing)}")
    # Also label existing direct-rho results, without rerunning training.
    for noise_type in NOISE_TYPES:
        selected = data["noise_type"].eq(noise_type)
        rhos = sorted(data.loc[selected, "rho"].dropna().unique())
        if len(rhos) != 3:
            raise ValueError(f"{noise_type}: expected three distinct rho values for high/medium/low quality")
        mapping = dict(zip(rhos, QUALITIES))
        data.loc[selected, "quality"] = data.loc[selected, "rho"].map(mapping)
    if "loss" in data.columns and data["loss"].nunique() > 1:
        raise ValueError("Plot one training loss at a time; input contains multiple losses.")
    baseline = data[data["noise_type"].eq("baseline")]
    output = args.output_dir or args.input.parent / "noise_plots"
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for noise_type in NOISE_TYPES:
        rhos = sorted(data.loc[data["noise_type"].eq(noise_type), "rho"].dropna().unique())
        for rho in rhos:
            make_plot(data, baseline, noise_type, rho, output)
            count += 1
    print(f"Created {count} figures in {output}")


if __name__ == "__main__":
    main()
