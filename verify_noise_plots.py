"""Verify current multiseed CSV compatibility and plotted accuracy SD values."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from matplotlib.container import BarContainer

import credal_gaussian_experiment as exp
import plot_noise_study as plot


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="credal_plot_") as directory:
        out = Path(directory)
        with contextlib.redirect_stdout(io.StringIO()):
            data, _ = exp.run_noise_study(
                out, 8, 12, 2, exp.TrainConfig(epochs=1, batch_size=8),
                .5, torch.device("cpu"), seeds=[11, 29],
            )
        keys = ["noise_type", "quality", "rho", "loss", "method", "target_type"]
        summary = exp.aggregate_results(data, keys)
        path = out / "noise_study_summary.csv"
        summary.to_csv(path, index=False)
        with patch.object(sys, "argv", ["plot", "--input", str(path)]):
            plot.main()
        assert len(list((out / "noise_plots").glob("*.png"))) == 6
        baseline = summary[summary.noise_type.eq("baseline")]
        rows = summary[summary.noise_type.eq("uniform") & summary.rho.eq(.1)].set_index("method")
        # Check actual plotted endpoints, not merely the existence of error bars.
        for name, value in rows.iterrows():
            raw = data[data.noise_type.eq("uniform") & data.rho.eq(.1) & data.method.eq(name)]
            np.testing.assert_allclose(value.accuracy_std, raw.accuracy.std(ddof=1))
        with patch.object(plot.plt, "close"):
            plot.make_plot(summary, baseline, "uniform", .1, out)
            figure = plot.plt.gcf()
        try:
            accuracy_axis, kl_axis = figure.axes
            bars = [c for c in accuracy_axis.containers if isinstance(c, BarContainer)]
            assert len(bars) == len(plot.METHODS)
            for method, bar in zip(plot.METHODS, bars):
                row = rows.loc[method]
                segment = bar.errorbar.lines[2][0].get_segments()[0]
                np.testing.assert_allclose(segment[:, 1], [
                    row.accuracy_mean-row.accuracy_std,
                    row.accuracy_mean+row.accuracy_std,
                ])
            assert len(accuracy_axis.lines) >= 3
            # Axhspan adds one rectangle per baseline, after the 8 bars.
            bands = accuracy_axis.patches[len(plot.METHODS):]
            assert len(bands) == 3
            baseline_rows = [
                baseline[baseline.method.eq("clean_sampled_hard")].iloc[0],
                baseline[baseline.method.eq("oracle_posterior")].iloc[0],
                rows.loc["noisy_soft"],
            ]
            for band, row in zip(bands, baseline_rows):
                np.testing.assert_allclose(band.get_y(), row.accuracy_mean-row.accuracy_std)
                np.testing.assert_allclose(band.get_height(), 2*row.accuracy_std)
            assert all(c.errorbar is None for c in kl_axis.containers if isinstance(c, BarContainer))
        finally:
            plot.plt.close(figure)
        # Single-repeat data should be drawable without inventing zero SD.
        single = exp.aggregate_results(data[data.repeat.eq(0)], keys)
        assert single.accuracy_std.isna().all()
        with patch.object(plot.plt, "close"):
            plot.make_plot(single, single[single.noise_type.eq("baseline")], "uniform", .1, out)
            figure = plot.plt.gcf()
        try:
            assert all(c.errorbar is None for c in figure.axes[0].containers if isinstance(c, BarContainer))
            assert any("Undefined SD" in text.get_text() for text in figure.texts)
        finally:
            plot.plt.close(figure)
    print("PASS: six plots from current multiseed output, exact SD error bars/bands, single-repeat NaNs")


if __name__ == "__main__":
    main()
