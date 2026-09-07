"""Small-data KL integration and selection-isolation regression checks."""
import contextlib
import io
import itertools
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import credal_gaussian_experiment as exp
import delta_search as search
import plot_noise_study as plots


def main():
    torch.set_num_threads(1)
    device = torch.device("cpu")
    seeds = list(range(5))
    splits = search.prepare_search_splits(12, 4, seeds)
    for split in splits:
        assert not set(split.train_indices) & set(split.validation_indices)
        assert len(set(split.train_indices) | set(split.validation_indices)) == 12
        np.testing.assert_allclose(split.x_train.mean(0), 0, atol=1e-12)
        np.testing.assert_allclose(split.x_train.std(0), 1, atol=1e-12)
        pool = exp.generate_gmm_dataset(12, 2.5, seed=split.seed_info["train_seed"])
        np.testing.assert_allclose(split.mean, pool.x[split.train_indices].mean(0, keepdims=True))
        np.testing.assert_allclose(split.x_validation,
                                   (pool.x[split.validation_indices] - split.mean) / split.std)
    assert search.UNIT_DELTA_GRID == tuple(i * .05 for i in range(1, 20)) or np.allclose(search.UNIT_DELTA_GRID, np.arange(.05, 1, .05))
    labels = [np.array([[.2, .3, .5], [.1, .2, .7]])] * 5
    expected = np.quantile(np.concatenate([1 / p.min(1) - 1 for p in labels]), .95)
    assert search.pmm_saturation_q95(labels) == expected
    try:
        search.pmm_saturation_q95([np.array([[0., .5, .5]])] * 5)
        raise AssertionError("Infinite Q95 accepted")
    except ValueError:
        pass
    candidates = list(itertools.islice(search.pmm_delta_candidates(2.), 12))
    assert 2. in candidates and candidates[-1] > 2.
    assert np.all(np.diff(candidates) > 0)
    assert search.pmm_stop_reason(.5, .505, 1, 2) is None
    assert search.pmm_stop_reason(.5, .52, 2, 2) is None
    assert search.pmm_stop_reason(.5, .505, 2, 2)

    config = exp.TrainConfig(epochs=1, batch_size=8, hidden_dims=(8, 8), loss_name="kl")
    with tempfile.TemporaryDirectory(prefix="credal_delta_") as directory:
        out = Path(directory)
        # Deterministic score test: choose across seeds, not the best single seed.
        scores = iter([.99, .1, .1, .1, .1] + [.4] * 5 + [.4] * 5)
        def evaluate(*args, **kwargs):
            return {"accuracy": next(scores), "posterior_kl": 0.}
        with patch.object(search, "UNIT_DELTA_GRID", (.05, .1, .15)), \
             patch.object(exp, "evaluate_model", evaluate), contextlib.redirect_stdout(io.StringIO()):
            selected, checkpoints, _ = search.search_condition(
                splits, [s.p_train for s in splits], "lv",
                dict(noise_type="uniform", quality="high", rho=.1),
                config, device, out, [], [])
        assert selected["optimal_delta"] == .1  # Tie selects smaller delta.
        assert all(p.exists() for p in checkpoints)

        real_generate = exp.generate_gmm_dataset
        generated = []
        def generate(*args, **kwargs):
            if kwargs["seed"] in {exp.experiment_seeds(s)["test_seed"] for s in seeds}:
                assert len(pd.read_csv(out / "optimal_deltas.csv")) == 24
            generated.append(kwargs["seed"])
            return real_generate(*args, **kwargs)
        # Short unit grid, but real PMM adaptive search, real banks and real KL training.
        with patch.object(search, "UNIT_DELTA_GRID", (.05, .5)), \
             patch.object(exp, "generate_gmm_dataset", generate), \
             contextlib.redirect_stdout(io.StringIO()):
            data, optimal = search.run_delta_selection_study(
                out, 12, 9, seeds, config, .5, device, (.1, .3, .5), n_validation=4)
        assert len(generated) == 10 and len(data) == 280 and len(optimal) == 24
        trials = pd.read_csv(out / "delta_search_trials.csv")
        summary = pd.read_csv(out / "delta_search_summary.csv")
        assert set(trials.method) == set(search.SEARCH_METHODS)
        assert trials.groupby(["noise_type", "quality", "method", "delta"]).size().eq(5).all()
        assert data.n_fit.eq(8).all() and data.n_validation.eq(4).all()
        for row in optimal.itertuples():
            history = summary[(summary.noise_type == row.noise_type) &
                              (summary.quality == row.quality) & (summary.method == row.method)]
            winner = history.sort_values(["validation_accuracy_mean", "delta"], ascending=[False, True]).iloc[0]
            assert np.isclose(row.optimal_delta, winner.delta)
            tested = data[(data.noise_type == row.noise_type) &
                          (data.quality == row.quality) & (data.method == row.method)]
            assert len(tested) == 5 and np.allclose(tested.credal_delta, row.optimal_delta)
            if row.method == "pmm":
                last = history.iloc[-1]
                assert last.delta >= last.pmm_q95 and last.A_increment < .01
                np.testing.assert_allclose(history.A_bar, history.mean_credal_area)
        with patch.object(sys, "argv", ["plot_noise_study.py", "--input", str(out / "noise_study_summary.csv")]), \
             contextlib.redirect_stdout(io.StringIO()):
            plots.main()
        assert len(list((out / "noise_plots").glob("*.png"))) == 6
    print("PASS: five-seed splits, Q95, AND stopping, shared selection/ties, test isolation, KL integration, six plots")


if __name__ == "__main__":
    main()
