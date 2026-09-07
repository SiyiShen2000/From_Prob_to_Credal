# Matched-imprecision study

Run from the project directory, reusing a compatible select experiment:

```powershell
py credal_gaussian_experiment.py --delta-mode matched --loss kl --scan-dir results_delta_kl --output-dir results_matched_kl
```

Or omit `--scan-dir` to compute the same delta/area scans without training a model
at every scan point:

```powershell
py credal_gaussian_experiment.py --delta-mode matched --loss kl --output-dir results_matched_kl
```

The fixed/select modes remain separate and unchanged. Matching includes only PMM,
LV, QCOR and TV. Each condition is one noise type and quality/rho combination.
The same five seeds, 4000 fitting samples, 1000 held-out validation samples and
5000 test samples per seed are used by default. Matching and final training use
only the fitting subset. Validation performance is not used in this study.

For each condition:

1. For every method/seed, take the minimum and maximum scanned training-label
   mean normalized credal area. The full simplex has area 1.
2. Intersect all 20 intervals: lower endpoint is the largest minimum, upper
   endpoint the smallest maximum. Empty or singleton intersections raise an error.
3. Select five equally spaced tau values, including both common endpoints.
4. Independently for every method/seed/tau, interpolate delta between bracketing
   scan points. Recompute the actual training area. If the absolute error is not
   strictly below 0.005, bisect the local bracket and recompute until it is.
   No extrapolation or tolerance-violating fallback is allowed.
5. Freeze all matched deltas, then train and test a fresh model at every matched
   delta. Within a seed, initialization, fitting data, batches and test data are
   paired across methods/tau. Validation data are not added back into training.

Unlike optimal-delta selection, different seeds can use different deltas for the
same method/tau. Aggregation therefore groups by condition/method/tau, **not delta**.
Tau grids are condition-specific; do not assume the same tau_index denotes the
same numerical area across noise conditions.

Default final training count: 2 noise types x 3 qualities x 5 taus x 4 methods x
5 seeds = **600**. No optimal-validation model selection is performed here.

Fresh scans use LV/QCOR/TV's existing grid and PMM's existing adaptive generator
and stopping function in `delta_search.py`. Reused scans use the ranges actually
recorded in `delta_search_trials.csv`. Source config and exact split indices must
match; use the same `--seeds`, `--n-train`, `--n-validation`, `--rho-values` as the
source run. Loss may differ because area does not depend on training loss.
Use a separate output directory to preserve the select results.

Outputs:

- `matched_area_scan.csv`: all method/seed/delta training areas.
- `matched_area_ranges.csv`: each method/seed interval (120 rows by default).
- `matched_tau_targets.csv`: common intervals and five taus (30 rows).
- `matched_deltas.csv`: matched deltas, initial interpolation, actual area, absolute
  error and refinement evaluations (600 rows).
- `matched_study.csv`: all per-seed final results including accuracy and posterior KL.
- `matched_study_summary.csv`: five-seed mean and sample SD of accuracy, posterior KL,
  area, delta and area error (120 rows). SD uses ddof=1.
- `config.json`, `validation_split_indices.npz`: configuration and split indices.

Plot matched results with the dedicated scatter plotter (one figure per condition):

```powershell
py plot_delta_matched_study.py --input results_matched_kl/matched_study_summary.csv
```

Points use achieved mean training area and mean test accuracy. The five x-axis
ticks are the condition's target taus; colors distinguish PMM/LV/QCOR/TV.
The existing `plot_noise_study.py` does not support the additional tau axis.

Verification: `py verify_matched_imprecision.py` exercises 600 tiny real-KL fits,
local refinement, all-seed intersections, test isolation, summary SD and scan reuse.
