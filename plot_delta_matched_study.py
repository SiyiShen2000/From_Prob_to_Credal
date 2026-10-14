"""Accuracy and posterior KL versus target training credal-set area."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_noise_study as base

METHODS = ("pmm", "lv", "qcor", "tv")
COLORS = {"pmm": "#0072B2", "lv": "#D55E00", "qcor": "#009E73", "tv": "#CC79A7"}
REFERENCES = ("clean_sampled_hard", "oracle_posterior", "noisy_soft")
INPUT = Path(__file__).resolve().parent / "results_matched_full_kl" / "matched_study_summary.csv"


def validate_condition(condition):
    if set(condition.method) != set(METHODS) or condition.duplicated(["method", "tau_index"]).any():
        raise ValueError("Expected one summary row per matched method/tau")
    if condition.groupby("tau_index").tau.nunique().ne(1).any():
        raise ValueError("All methods must share tau targets")
    targets = condition[["tau_index", "tau"]].drop_duplicates().sort_values("tau_index")
    taus = targets.tau.to_numpy(dtype=float)
    if len(taus) < 2 or not np.isfinite(taus).all() or np.any(np.diff(taus) <= 0):
        raise ValueError("Expected at least two increasing tau targets")
    for method in METHODS:
        rows = condition[condition.method.eq(method)]
        if len(rows) != len(taus) or set(rows.tau_index) != set(targets.tau_index):
            raise ValueError(f"Missing tau values for {method}")
    if np.any(np.abs(condition.mean_credal_area_mean - condition.tau) >= .005):
        raise ValueError("Mean areas fail matched-area tolerance 0.005")
    return taus


def condition_references(references, first):
    references = references[references.method.isin(REFERENCES)]
    subset = references[references.noise_type.eq("baseline") | (
        references.noise_type.eq(first.noise_type) & references.quality.eq(first.quality) &
        references.rho.eq(first.rho))]
    if set(subset.method) != set(REFERENCES) or subset.method.duplicated().any():
        raise ValueError(f"Missing/duplicate references for {first.noise_type}/{first.quality}")
    return subset.set_index("method").loc[list(REFERENCES)]


def make_plot(condition, output, references):
    taus = validate_condition(condition)
    first = condition.iloc[0]
    reference_rows = condition_references(references, first)
    metrics = [("accuracy_mean", "Mean test accuracy"),
               ("posterior_kl_mean", "Mean posterior KL (lower is better)")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    styles = {"clean_sampled_hard": ("#777777", ":", "Baseline: clean sampled-hard"),
              "oracle_posterior": ("#222222", "--", "Baseline: oracle posterior"),
              "noisy_soft": ("#D62728", "-.", "Baseline: noisy soft")}
    for axis, (metric, ylabel) in zip(axes, metrics):
        for method in METHODS:
            rows = condition[condition.method.eq(method)].sort_values("tau_index")
            axis.plot(rows.tau, rows[metric], "o-", color=COLORS[method],
                      label=method.upper(), markersize=4, linewidth=1.5)
            if metric == "accuracy_mean":
                axis.fill_between(rows.tau.to_numpy(), (rows[metric] - rows.accuracy_std).to_numpy(),
                                  (rows[metric] + rows.accuracy_std).to_numpy(),
                                  color=COLORS[method], alpha=.12)
        for method, row in reference_rows.iterrows():
            color, style, label = styles[method]
            axis.axhline(row[metric], color=color, linestyle=style, linewidth=1.1, label=label)
            if metric == "accuracy_mean" and base.accuracy_std(row) is not None:
                axis.axhspan(row[metric] - row.accuracy_std, row[metric] + row.accuracy_std,
                             color=color, alpha=.06, zorder=0)
        axis.set_xticks(taus, [f"{tau:.4g}" for tau in taus], rotation=40)
        axis.set_xlabel("Target mean normalized training credal-set area (τ)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.25)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(.80, .52), fontsize=9)
    noise = "Uniform" if first.noise_type == "uniform" else "Class corruption"
    fig.suptitle(f"Matched-imprecision study: {noise}, {first.quality} quality (rho={first.rho:g}, loss={first.loss})",
                 fontsize=14, fontweight="bold")
    # fig.text(.5, .025, "Curves: all training-area targets; no selection by test performance.\n"
    #          "Accuracy bands: ±1 sample SD across seeds (not confidence intervals); undefined SD omitted.\n"
    #          "Horizontal references are trained once per seed/condition and are not area-matched.",
    #          ha="center", fontsize=9)
    fig.subplots_adjust(top=.87, bottom=.28, right=.78, wspace=.32)
    path = Path(output) / f"delta_matched_curves_{first.noise_type}_{first.quality}_rho_{first.rho:g}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_study(data, output, references):
    required = {"noise_type", "quality", "rho", "method", "loss", "tau_index", "tau",
                "mean_credal_area_mean", "accuracy_mean", "accuracy_std", "posterior_kl_mean"}
    if not required.issubset(data.columns):
        raise ValueError(f"Matched summary missing {required - set(data.columns)}")
    reference_required = required - {"tau_index", "tau", "mean_credal_area_mean"}
    if not reference_required.issubset(references.columns):
        raise ValueError(f"Reference summary missing {reference_required - set(references.columns)}")
    if data.empty or data.loss.nunique() != 1 or set(references.loss) != set(data.loss):
        raise ValueError("Use nonempty matched/reference summaries from the same training loss")
    if data[list(required - {"accuracy_std"})].isna().any().any():
        raise ValueError("Required matched fields must not be missing")
    if not set(data.noise_type).issubset({"uniform", "corruption"}):
        raise ValueError("Unknown noise type")
    for frame in (data, references):
        for _, row in frame.iterrows():
            base.accuracy_std(row)
            if not np.isfinite(row.accuracy_mean) or not 0 <= row.accuracy_mean <= 1:
                raise ValueError("Invalid accuracy")
            if not np.isfinite(row.posterior_kl_mean) or row.posterior_kl_mean < 0:
                raise ValueError("Invalid posterior KL")
    areas = data.mean_credal_area_mean.to_numpy()
    if not np.isfinite(areas).all() or np.any((areas < 0) | (areas > 1)):
        raise ValueError("Invalid mean normalized area")
    conditions = list(data.groupby(["noise_type", "quality", "rho"]))
    for _, condition in conditions:
        validate_condition(condition)
        condition_references(references, condition.iloc[0])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for _, condition in conditions:
        make_plot(condition, output, references)
    return len(conditions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input.parent / "delta_matched_plots"
    reference_path = args.input.parent / "matched_references_summary.csv"
    if not reference_path.exists():
        parser.error("Missing matched_references_summary.csv; run the independent matched study first")
    data = pd.read_csv(args.input)
    count = plot_study(data, output, pd.read_csv(reference_path))
    print(f"Created {count} curve figures in {output}")


if __name__ == "__main__":
    main()
