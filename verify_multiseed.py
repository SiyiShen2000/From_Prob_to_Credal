"""Small-data checks for independent repeats and paired comparisons."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import credal_gaussian_experiment as exp


def main():
    torch.set_num_threads(1)
    seeds = [11, 29]
    config = exp.TrainConfig(epochs=1, batch_size=8)
    real_generate, real_train = exp.generate_gmm_dataset, exp.train_mlp
    generated, trained = [], []

    def generate(*args, **kwargs):
        data = real_generate(*args, **kwargs)
        generated.append((kwargs["seed"], data))
        return data

    def train(*args, **kwargs):
        digest = hashlib.sha256(args[0].tobytes()).hexdigest()
        trained.append((kwargs["seed"], digest))
        return real_train(*args, **kwargs)

    with tempfile.TemporaryDirectory(prefix="credal_seeds_") as directory:
        out = Path(directory)
        with patch.object(exp, "generate_gmm_dataset", generate), patch.object(exp, "train_mlp", train), contextlib.redirect_stdout(io.StringIO()):
            noise, quality = exp.run_noise_study(out, 8, 12, 2, config, .5, torch.device("cpu"), seeds=seeds)
        assert len(noise) == 112 and len(quality) == 12
        assert noise.pairwise_distance.eq(2.5).all()
        assert "overlap" not in noise.columns
        for df in (noise,):
            assert set(df.seed) == set(seeds)
            for _, group in df.groupby("repeat"):
                assert group.train_seed.nunique() == group.test_seed.nunique() == group.model_seed.nunique() == 1
            assert df.train_seed.nunique() == df.test_seed.nunique() == df.model_seed.nunique() == 2
        assert quality.groupby(["noise_type", "quality"]).size().eq(2).all()
        # Two datasets per repeat; all noise conditions share them.
        assert len(generated) == 4
        assert not np.array_equal(generated[0][1].x, generated[2][1].x)
        assert not np.array_equal(generated[1][1].x, generated[3][1].x)
        for first, size in [(0,56),(56,56)]:
            assert len(set(trained[first:first+size])) == 1
        summary = exp.aggregate_results(noise, ["noise_type", "quality", "rho", "loss", "method", "target_type"])
        assert len(summary) == 56
        assert summary.accuracy_std.notna().all()
        assert summary.posterior_kl_std.notna().all()
        # Exact repeated CPU run for a fixed seed schedule.
        with contextlib.redirect_stdout(io.StringIO()):
            again, again_quality = exp.run_noise_study(out, 8, 12, 2, config, .5, torch.device("cpu"), seeds=seeds)
        pd.testing.assert_frame_equal(noise, again)
        pd.testing.assert_frame_equal(quality, again_quality)
    print("PASS: noise-only study, independent datasets, paired methods, seed metadata, summaries and reproducibility")


if __name__ == "__main__":
    main()
