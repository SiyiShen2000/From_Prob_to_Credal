"""Synthetic summaries verify labels, scatter coordinates and condition-specific ticks."""
from pathlib import Path
import tempfile
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

import plot_noise_study as base
import plot_delta_select_study as select
import plot_delta_matched_study as matched


def main():
    selected, aligned = [], []
    metrics = dict(accuracy_mean=.8, accuracy_std=.02, posterior_kl_mean=.1,
                   delta_mode="select", loss="kl")
    for method in ("oracle_posterior", "clean_sampled_hard"):
        selected.append(dict(**metrics, noise_type="baseline", quality="baseline",
                             rho=np.nan, method=method, credal_delta=np.nan))
    for noise in base.NOISE_TYPES:
        for index, (quality, rho) in enumerate(zip(base.QUALITIES, (.1, .3, .5))):
            for method in base.METHODS + ["noisy_soft"]:
                delta = 2.125 if method == "pmm" else .35
                selected.append(dict(**metrics, noise_type=noise, quality=quality, rho=rho,
                                     method=method, credal_delta=delta))
            for method in matched.METHODS:
                for i, tau in enumerate(np.linspace(.1 + index*.01, .8, 5)):
                    aligned.append(dict(noise_type=noise, quality=quality, rho=rho, loss="kl",
                                        method=method, tau_index=i, tau=tau,
                                        mean_credal_area_mean=tau + .001, accuracy_mean=.7+i*.01))
    with tempfile.TemporaryDirectory(prefix="delta_plots_") as directory:
        out = Path(directory)
        assert select.plot_study(pd.DataFrame(selected), out) == 6
        assert matched.plot_study(pd.DataFrame(aligned), out) == 6
        assert len(list(out.glob("*.png"))) == 12
        captured = []
        def capture(fig, *args, **kwargs):
            captured.append(fig)
        with patch.object(Figure, "savefig", capture):
            select.plot_study(pd.DataFrame(selected), out)
            for fig in captured:
                for ax in fig.axes:
                    labels = [tick.get_text() for tick in ax.get_xticklabels()]
                    assert "PMM (δ*=2.125)" in labels
                    assert all(f"{m} (δ*=0.35)" in labels for m in ("LV", "QCOR", "TV"))
                    assert len(ax.patches) >= 8
            captured.clear()
            condition = pd.DataFrame(aligned).query("noise_type == 'uniform' and quality == 'high'")
            matched.make_plot(condition, out)
            ax = captured[0].axes[0]
            np.testing.assert_allclose(ax.get_xticks(), np.linspace(.1, .8, 5))
            assert len(ax.collections) == 4
            for points, method in zip(ax.collections, matched.METHODS):
                rows = condition[condition.method == method]
                np.testing.assert_allclose(points.get_offsets(), rows[["mean_credal_area_mean", "accuracy_mean"]])
            assert len({tuple(c.get_facecolors()[0]) for c in ax.collections}) == 4
        try:
            matched.plot_study(pd.DataFrame(aligned).iloc[1:], out)
            raise AssertionError("Missing tau accepted")
        except ValueError:
            pass
    plt.close("all")
    print("PASS: 12 PNGs, selected delta labels, four colors, actual-area points and five tau ticks")


if __name__ == "__main__":
    main()
