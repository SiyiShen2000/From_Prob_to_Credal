"""Publication figures from existing summaries; never train models or edit source results."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METHODS = ("pmm", "lv", "qcor", "tv")
ALL_METHODS = ("alpha_cut", "inverse_pignistic", "dubois_prade", "pmm", "lv", "qcor", "tv", "comparative", "noisy_soft")
LABELS = dict(zip(ALL_METHODS, ("Alpha-cut", "Inverse pignistic", "Dubois-Prade", "PMM", "LV", "QCOR", "TV", "Comparative", "Noisy soft")))
COLORS = dict(zip(METHODS, ("#0072B2", "#D55E00", "#009E73", "#CC79A7")))
QUALITIES = ("high", "medium", "low")
LOSSES = ("ce", "kl")
LOSS_NAMES = {"ce": "cross_entropy", "kl": "kl"}
STYLES = {"ce": "--", "kl": "-"}


def one(frame, **filters):
    for key, value in filters.items():
        frame = frame[frame[key].eq(value)]
    if len(frame) != 1:
        raise ValueError(f"Expected exactly one row for {filters}, found {len(frame)}")
    return frame.iloc[0]


def save(fig, output, name):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{extension}", dpi=400, facecolor="white")
    plt.close(fig)


def fixed_figure(fixed, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5))
    norms = {"accuracy_mean": Normalize(0, 100), "posterior_kl_mean": LogNorm(1e-3, 20)}
    # Scale only the displayed methods; the oracle baseline is reported in the caption.
    shown = pd.concat(fixed.values()).query("method in @ALL_METHODS")
    if shown.posterior_kl_mean.min() < 1e-3 or shown.posterior_kl_mean.max() > 20:
        norms["posterior_kl_mean"] = LogNorm(shown.posterior_kl_mean.min(), shown.posterior_kl_mean.max())
    for row_index, noise in enumerate(("uniform", "corruption")):
        for col_index, metric in enumerate(("accuracy_mean", "posterior_kl_mean")):
            ax = axes[row_index, col_index]
            matrix = np.array([[one(fixed[loss], noise_type=noise, quality=q, method=m)[metric]
                                for q in QUALITIES for loss in LOSSES] for m in ALL_METHODS])
            if metric == "accuracy_mean":
                matrix *= 100
            cmap = plt.get_cmap("viridis" if metric == "accuracy_mean" else "viridis_r")
            im = ax.imshow(matrix, cmap=cmap, norm=norms[metric], aspect="auto")
            for (i, j), value in np.ndenumerate(matrix):
                color = cmap(norms[metric](value))
                luminance = .2126*color[0]+.7152*color[1]+.0722*color[2]
                ax.text(j, i, f"{value:.1f}" if metric == "accuracy_mean" else f"{value:.2g}",
                        ha="center", va="center", fontsize=6, color="black" if luminance > .5 else "white")
            ax.set_xticks(range(6), [f"{'Med.' if q == 'medium' else q.title()}\n{loss.upper()}" for q in QUALITIES for loss in LOSSES])
            ax.set_yticks(range(len(ALL_METHODS)), [LABELS[m] for m in ALL_METHODS] if col_index == 0 else [])
            for divider in (1.5, 3.5):
                ax.axvline(divider, color="white", linewidth=1)
            ax.axhline(7.5, color="white", linewidth=1)
            name = "Uniform" if noise == "uniform" else "Class corruption"
            title = "Accuracy (%) ↑" if metric == "accuracy_mean" else "Posterior KL ↓"
            ax.set_title(f"({chr(97 + row_index*2 + col_index)}) {name}: {title}", loc="left", fontsize=8)
            cb = fig.colorbar(im, ax=ax, fraction=.047, pad=.025)
            cb.ax.tick_params(labelsize=6)
            if metric == "posterior_kl_mean":
                cb.ax.yaxis.set_minor_formatter(NullFormatter())
                cb.set_label("Log color scale", fontsize=7)
    fig.subplots_adjust(left=.18, right=.90, top=.96, bottom=.08, wspace=.24, hspace=.27)
    save(fig, output, "figure1_fixed_delta")


def matched_figure(matched, refs, noise, output, number, kl_limits):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2), sharey="row", sharex="col")
    for column, quality in enumerate(QUALITIES):
        for loss in LOSSES:
            condition = matched[loss][matched[loss].noise_type.eq(noise) & matched[loss].quality.eq(quality)]
            for method in METHODS:
                rows = condition[condition.method.eq(method)].sort_values("tau_index")
                if len(rows) < 2 or rows.tau.duplicated().any():
                    raise ValueError("Incomplete matched curve")
                x = rows.tau.to_numpy()
                y, sd = rows.accuracy_mean.to_numpy()*100, rows.accuracy_std.to_numpy()*100
                axes[0, column].fill_between(x, y-sd, y+sd, color=COLORS[method], alpha=.07, linewidth=0)
                for ax, values in ((axes[0, column], y), (axes[1, column], rows.posterior_kl_mean)):
                    ax.plot(x, values, color=COLORS[method], linestyle=STYLES[loss], linewidth=1.15,
                            marker="o" if loss == "kl" else "s", markersize=2.2,
                            markerfacecolor=COLORS[method] if loss == "kl" else "white", markeredgewidth=.6)
        # Shared reference lines are legitimate only when the CE/KL values agree.
        for method, linestyle, color in (("noisy_soft", "-.", "#555555"),
                                         ("oracle_posterior", ":", "#111111")):
            filters = dict(method=method, noise_type="baseline") if method == "oracle_posterior" else dict(method=method, noise_type=noise, quality=quality)
            a, b = one(refs["ce"], **filters), one(refs["kl"], **filters)
            for metric in ("accuracy_mean", "posterior_kl_mean"):
                if not np.isclose(a[metric], b[metric], rtol=1e-5, atol=1e-8):
                    raise ValueError("CE/KL baselines differ; cannot draw a shared reference")
            axes[0, column].axhline(a.accuracy_mean*100, color=color, linestyle=linestyle, linewidth=.9)
            axes[1, column].axhline(a.posterior_kl_mean, color=color, linestyle=linestyle, linewidth=.9)
        rho = condition.rho.unique()
        if len(rho) != 1:
            raise ValueError("Ambiguous quality/rho mapping")
        axes[0, column].set_title(f"{quality.title()} quality ($\\rho={rho[0]:g}$)", fontsize=8)
        for row in range(2):
            ax = axes[row, column]
            ax.text(.03, .95, f"({chr(97+row*3+column)})", transform=ax.transAxes, va="top", fontsize=8)
            ax.grid(alpha=.2, linewidth=.5)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelsize=7)
        axes[0, column].set_ylim(0, 100)
        axes[1, column].set_yscale("log")
        axes[1, column].set_ylim(*kl_limits)
        axes[1, column].yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        axes[1, column].yaxis.set_minor_formatter(NullFormatter())
        axes[1, column].set_xlabel("Target mean normalized area, $\\tau$", fontsize=7)
    axes[0, 0].set_ylabel("Test accuracy (%)", fontsize=8)
    axes[1, 0].set_ylabel("Posterior KL (log scale)", fontsize=8)
    handles = [Line2D([], [], color=COLORS[m], label=m.upper(), linewidth=1.5) for m in METHODS]
    handles += [Line2D([], [], color="#333333", linestyle=STYLES[l], marker="s" if l == "ce" else "o",
                       markerfacecolor="white" if l == "ce" else "#333333", markersize=3, label=l.upper()) for l in LOSSES]
    handles += [Line2D([], [], color="#555555", linestyle="-.", label="Noisy soft"),
                Line2D([], [], color="#111111", linestyle=":", label="Oracle posterior")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .99), ncol=4,
               frameon=False, fontsize=7, columnspacing=1.7, handlelength=2.5)
    fig.subplots_adjust(left=.10, right=.99, top=.83, bottom=.10, hspace=.14, wspace=.12)
    save(fig, output, f"figure{number}_matched_{noise}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_figures")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output_dir.resolve()
    sources = {}
    def read(relative):
        path = root / relative
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)
    fixed = {l: read(f"results_noise_{l}/noise_study_summary.csv") for l in LOSSES}
    matched = {l: read(f"results_matched_full_{l}/matched_study_summary.csv") for l in LOSSES}
    refs = {l: read(f"results_matched_full_{l}/matched_references_summary.csv") for l in LOSSES}
    configs = []
    for folder in ("results_noise_ce", "results_noise_kl", "results_matched_full_ce", "results_matched_full_kl"):
        if output.is_relative_to(root / folder):
            raise ValueError("Paper figures must be outside the original result directories")
        path = root / folder / "config.json"
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        configs.append(json.loads(path.read_text(encoding="utf-8")))
    for key in ("n_train", "n_test", "epochs", "batch_size", "seeds", "seed_schedule", "pairwise_distance", "learning_rate", "weight_decay", "rho_values"):
        if any(c[key] != configs[0][key] for c in configs):
            raise ValueError(f"Incompatible experiment configurations: {key}")
    if configs[0]["credal_delta"] != configs[1]["credal_delta"]:
        raise ValueError("Fixed-delta settings differ")
    for l in LOSSES:
        for data in (fixed[l], matched[l], refs[l]):
            if set(data.loss) != {LOSS_NAMES[l]}:
                raise ValueError("Unexpected loss labels")
            if (not np.isfinite(data.posterior_kl_mean).all() or (data.posterior_kl_mean <= 0).any()
                    or not data.accuracy_mean.between(0, 1).all()):
                raise ValueError("Invalid metrics (log-scale KL must be positive)")
    keys = ["noise_type", "quality", "method", "tau_index"]
    paired = matched["ce"].merge(matched["kl"], on=keys, suffixes=("_ce", "_kl"), validate="one_to_one")
    if len(paired) != len(matched["ce"]) or len(paired) != len(matched["kl"]) or not np.allclose(paired.tau_ce, paired.tau_kl):
        raise ValueError("Matched CE/KL targets differ")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": .6,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fixed_figure(fixed, output)
    all_kl = pd.concat([*matched.values(), *refs.values()]).posterior_kl_mean
    limits = (10**np.floor(np.log10(all_kl.min())), 10**np.ceil(np.log10(all_kl.max())))
    matched_figure(matched, refs, "corruption", output, 2, limits)
    matched_figure(matched, refs, "uniform", output, 3, limits)
    pd.concat(fixed.values()).to_csv(output / "figure1_source_data.csv", index=False)
    pd.concat(matched.values()).to_csv(output / "figures2_3_source_data.csv", index=False)
    pd.concat(refs.values()).to_csv(output / "reference_source_data.csv", index=False)
    (output / "source_manifest.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    for source, digest in sources.items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise RuntimeError("A source file changed during plotting")
    print(f"Created three figures in PDF, SVG and PNG plus source tables in {output}")


if __name__ == "__main__":
    main()
