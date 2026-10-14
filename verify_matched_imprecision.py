"""Small real-KL end-to-end check of matched areas and seed-wise aggregation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import credal_gaussian_experiment as exp
import credal_area_scan as search
import plot_delta_matched_study as plotter
import matched_imprecision as matched


def main():
    torch.set_num_threads(1)
    curve = pd.DataFrame({"delta": [.05, .95], "mean_credal_area": [.05**2, .95**2]})
    delta, area, initial, count = matched.refine_delta(curve, .25, lambda d: d*d)
    assert abs(area - .25) < .005 and count > 1 and delta != initial
    for tau in (.05**2, .95**2):
        assert abs(matched.refine_delta(curve, tau, lambda d: d*d)[1] - tau) < .005
    plateau = pd.DataFrame({"delta": [.05, .1, .2], "mean_credal_area": [.1, .1, .2]})
    assert matched.refine_delta(plateau, .1, lambda d: .1)[1] == .1
    try:
        matched.refine_delta(curve, .99, lambda d: d*d)
        raise AssertionError("Extrapolation accepted")
    except ValueError:
        pass
    rows = [dict(noise_type="uniform", quality="high", rho=.1, seed=s,
                 method=m, delta=d, mean_credal_area=a)
            for s in range(5) for m in search.SEARCH_METHODS
            for d, a in ((.05, .1 if s != 4 else .2), (.95, .8 if m != "tv" else .7))]
    ranges, targets = matched.common_targets(pd.DataFrame(rows), list(range(5)))
    assert len(ranges) == 20
    np.testing.assert_allclose(targets.tau, np.linspace(.2, .7, 9))
    bad = pd.DataFrame(rows)
    bad.loc[(bad.seed == 4) & (bad.method == "pmm"), "mean_credal_area"] = [.8, .9]
    try:
        matched.common_targets(bad, list(range(5)))
        raise AssertionError("Empty intersection accepted")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory(prefix="credal_matched_") as directory:
        out = Path(directory)
        real_generate, real_train = exp.generate_gmm_dataset, exp.train_mlp
        calls = []
        def generate(*args, **kwargs):
            if kwargs["seed"] in {exp.experiment_seeds(s)["test_seed"] for s in range(5)}:
                assert len(pd.read_csv(out / "matched_deltas.csv")) == 240
            return real_generate(*args, **kwargs)
        def train(*args, **kwargs):
            assert len(args[0]) == 9
            assert len(pd.read_csv(out / "matched_deltas.csv")) == 240
            calls.append(kwargs["seed"])
            return real_train(*args, **kwargs)
        config = exp.TrainConfig(epochs=1, batch_size=8, hidden_dims=(8, 8), loss_name="kl")
        with patch.object(exp, "generate_gmm_dataset", generate), patch.object(exp, "train_mlp", train), \
             patch.object(search, "UNIT_DELTA_GRID", (.05, .5, .95)), contextlib.redirect_stdout(io.StringIO()):
            data = matched.run_matched_study(out, 9, 7, list(range(5)), config, torch.device("cpu"),
                                             (.1, .3, .5), n_tau=2)
        assert len(data) == 240 and len(calls) == 400
        assert data.n_fit.eq(9).all() and data.n_validation.eq(0).all()
        assert not (out / 'validation_split_indices.npz').exists()
        assert (data.area_error < .005).all()
        assert data.qstar_trajectory_evaluated_count.eq(9).all()
        assert data.qstar_trajectory_boundary_rate.between(0, 1).all()
        assert data.groupby(["noise_type", "quality", "tau_index"]).tau.nunique().eq(1).all()
        summary = pd.read_csv(out / "matched_study_summary.csv")
        assert len(summary) == 48 and summary.accuracy_std.notna().all()
        assert summary.qstar_trajectory_boundary_rate_mean.between(0, 1).all()
        expected = data.groupby(["noise_type", "quality", "rho", "loss", "method", "tau_index", "tau"]).accuracy.std()
        np.testing.assert_allclose(summary.accuracy_std, expected.to_numpy())
        references = pd.read_csv(out / "matched_references.csv")
        assert len(references) == 160
        assert references.n_fit.eq(9).all() and references.n_validation.eq(0).all()
        assert not references.duplicated(["noise_type", "quality", "rho", "method", "seed"]).any()
        credal_references = references[references.method.isin(exp.TRANSFORMATIONS)]
        assert credal_references.qstar_trajectory_evaluated_count.eq(9).all()
        reference_summary = pd.read_csv(out / "matched_references_summary.csv")
        assert len(reference_summary) == 32
        expected_refs = references.groupby(["noise_type", "quality", "rho", "loss", "method"],
                                          dropna=False).accuracy.std()
        np.testing.assert_allclose(reference_summary.accuracy_std, expected_refs.to_numpy())
        assert plotter.plot_study(summary, out / "plots", reference_summary) == 6
        assert len(list((out / "plots").glob("*.png"))) == 6
        assert not list(out.rglob("*best_tau*"))
    print("PASS: 400 real KL fits, all training samples, model-free scans/test isolation, references, summaries and six curves")


if __name__ == "__main__":
    main()
