# Delta selection experiment

The default `--delta-mode fixed` experiment is unchanged. To run KL with delta selection:

```powershell
py credal_gaussian_experiment.py --delta-mode select --loss kl --output-dir results_delta_kl
py plot_noise_study.py --input results_delta_kl/noise_study_summary.csv
```

Exactly five distinct seeds are required (default: 0, 1, 2, 3, 4). Each original
5000-sample training pool is split reproducibly into 4000 fitting samples and
1000 validation samples. Standardization uses only fitting samples. The test sets
contain 5000 samples per seed. `--n-train`, `--n-validation`, and `--n-test` can
override these sizes. Fixed mode does not reserve validation data.

For each noise type, quality level, and searched method, select one shared delta
by the highest mean validation **accuracy against sampled clean class labels**
across five seeds. Exact ties select the smaller delta. This yields 24 selections,
not one delta shared across different methods or noise conditions. The optimal
delta is best among evaluated candidates, not a guarantee of a global optimum.

- LV, QCOR, TV: 0.05, 0.10, ..., 0.95 (19 candidates).
- PMM: compute `1/min(p)-1` for each noisy fitting label in all five seeds, pool
  the values, and compute their empirical Q95 (linear interpolation).
  Start at 0.05, with spacing initially 0.05 and multiplied by 1.5 before each
  next step: 0.05, 0.125, 0.2375, ... . Include Q95 exactly when crossing it.
  Continue beyond Q95 if necessary: stop only when **both** the current delta
  is at least Q95 **and** the adjacent mean normalized-area increase is < 0.005.
  Area is averaged over fitting samples, then over the five equally sized seeds;
  the full probability simplex has normalized area 1. Area controls stopping,
  whereas validation accuracy selects the winner. Infinite Q95 (e.g. class
  corruption with rho=1) raises an error instead of silently clipping labels.

All selection completes before test data are generated or evaluated. The five
saved winning checkpoints are tested directly, without refitting on validation
samples. Other methods and baselines train on the same 4000 fitting samples in
this branch; they do not search delta. The original fixed experiment still trains
on all 5000 samples, so its training budget differs from this branch.

Outputs:

- `delta_search_trials.csv`: each seed/candidate's validation accuracy, posterior KL and area.
- `delta_search_summary.csv`: five-seed mean accuracy, accuracy SD, `A_bar`,
  `A_increment`, Q95 and stopping reason for each candidate.
- `optimal_deltas.csv`: selected delta and mean validation accuracy for each condition/method.
- `validation_split_indices.npz`, `selected_checkpoints/`: reproducible splits and winning weights.
- `noise_study.csv`, `noise_study_summary.csv`: final per-seed tests and mean/SD,
  compatible with the existing accuracy/KL plots; tested delta is `credal_delta`.
- `config.json`, `noise_quality_summary.csv`: configuration and fitting-label diagnostics.

Use separate output directories for fixed/select and CE/KL runs. Searching is
substantially more expensive: the three fixed grids alone require 1710 training
runs (2 noise types x 3 qualities x 3 methods x 19 candidates x 5 seeds).
PMM adds adaptive trials, and untuned methods/baselines add 160 runs. Winning
searched models are loaded, not retrained, for final testing.

Small-data verification: `py verify_delta_search.py` (real KL training with a shortened
LV/QCOR/TV grid; checks the real PMM stopping rule and plot compatibility).
