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
import delta_search as search
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
    np.testing.assert_allclose(targets.tau, np.linspace(.2, .7, 5))
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
                assert len(pd.read_csv(out / "matched_deltas.csv")) == 600
            return real_generate(*args, **kwargs)
        def train(*args, **kwargs):
            assert len(args[0]) == 6
            assert len(pd.read_csv(out / "matched_deltas.csv")) == 600
            calls.append(kwargs["seed"])
            return real_train(*args, **kwargs)
        config = exp.TrainConfig(epochs=1, batch_size=8, hidden_dims=(8, 8), loss_name="kl")
        with patch.object(exp, "generate_gmm_dataset", generate), patch.object(exp, "train_mlp", train), \
             patch.object(search, "UNIT_DELTA_GRID", (.05, .5, .95)), contextlib.redirect_stdout(io.StringIO()):
            data = matched.run_matched_study(out, 9, 7, list(range(5)), config, torch.device("cpu"),
                                             (.1, .3, .5), n_validation=3)
        assert len(data) == len(calls) == 600
        assert (data.area_error < .005).all()
        assert data.groupby(["noise_type", "quality", "tau_index"]).tau.nunique().eq(1).all()
        summary = pd.read_csv(out / "matched_study_summary.csv")
        assert len(summary) == 120 and summary.accuracy_std.notna().all()
        expected = data.groupby(["noise_type", "quality", "rho", "loss", "method", "tau_index", "tau"]).accuracy.std()
        np.testing.assert_allclose(summary.accuracy_std, expected.to_numpy())
        # Exercise compatible select-scan loading and mismatched-config rejection.
        scans = pd.read_csv(out / "matched_area_scan.csv")
        scans.to_csv(out / "delta_search_trials.csv", index=False)
        source_config = {"seeds": list(range(5)), "n_train": 9,
                         "delta_selection": {"n_validation": 3}, "pairwise_distance": 2.5,
                         "corruption_kind": "class", "rho_values": [.1, .3, .5]}
        # A temporary fixture, not a project-file edit.
        with (out / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(source_config, handle)
        splits = search.prepare_search_splits(9, 3, list(range(5)))
        assert len(matched.load_scans(out, splits, 9, 3, (.1, .3, .5))) == len(scans)
        try:
            matched.load_scans(out, splits, 10, 3, (.1, .3, .5))
            raise AssertionError("Mismatched training size accepted")
        except ValueError:
            pass
    print("PASS: common intervals, five taus, interpolation/refinement, 600 real KL fits, test isolation, summaries, scan reuse")


if __name__ == "__main__":
    main()
