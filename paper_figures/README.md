# Paper figures

These figures use the existing experiment CSVs. No models were retrained and no
original plots or result files were changed. Regenerate from the project directory:

```powershell
py plot_paper_figures.py
```

Each figure is exported as PDF (embedded TrueType fonts), SVG (editable text), and
400-dpi PNG. The canvas width is 7.2 inches, intended for a double-column figure.
Use PDF for manuscript inclusion and PNG for quick inspection. SVG appearance
depends on the viewer having DejaVu Sans installed. Final sizing should follow
the target journal's template; avoid shrinking these multi-panel figures to a
single column.

## Suggested English captions

**Figure 1. Fixed-parameter comparison under noisy probabilistic supervision.**
Mean test accuracy (left, percent) and posterior KL divergence (right) under
uniform mixing (top) and class corruption (bottom). Columns compare CE and KL
training at high, medium and low label quality, corresponding to rho = 0.1, 0.3
and 0.5. The four neighborhood methods use a fixed delta = 0.5; other credal
transformations do not depend on this parameter. Each entry averages five seeds,
with 5000 training and 5000 test examples per seed and 120 training epochs.
Accuracy uses a common linear color scale and posterior KL a common logarithmic
color scale across noise types. Cell annotations show the original metric values,
not their logarithms; brighter colors indicate better values for both metrics.
The final row shows the noisy-soft baseline; the oracle-posterior baseline is
omitted from the heatmaps and their color scales. Values are rounded for
display; full means and sample standard deviations are supplied in the source
table. The clean sampled-hard baseline achieves 82.408% accuracy and posterior
KL 0.004330. As a theoretical reference, an ideal oracle that outputs the true
posterior distribution has zero posterior KL divergence and, under argmax
classification, attains Bayes-optimal expected accuracy.

**Figure 2. Matched-imprecision responses under class corruption.**
Test accuracy (top) and posterior KL divergence (bottom) as functions of the
target mean normalized training credal-set area, tau. Columns correspond to
rho = 0.1, 0.3 and 0.5. Color identifies the credal method; dashed lines with open
squares represent CE and solid lines with filled circles represent KL. For every
method, seed and target, delta is matched using training labels only, with absolute
area error below 0.005. All nine precomputed targets are shown; no point is selected
by test performance. Lines connect observed target values and do not imply results
were evaluated between them. Curves average five seeds; faint accuracy bands show
plus/minus one sample standard deviation, not confidence intervals (display is
bounded to the valid 0--100% accuracy range). Gray horizontal lines show noisy-soft
and oracle-posterior references, whose CE/KL results agree numerically. References
are not area-matched. Posterior KL uses a logarithmic vertical axis. All models use
5000 training and 5000 test examples per seed and 120 epochs. Axis limits are shared
with Figure 3; attainable tau ranges may differ between quality conditions.

**Figure 3. Matched-imprecision responses under uniform mixing.**
The layout, method colors, loss encodings, uncertainty bands, references and axis
scales follow Figure 2. Columns correspond to rho = 0.1, 0.3 and 0.5. The complete
training-area target grid is shown for each condition, without test-based selection.

## Source tables and reproducibility

- `figure1_source_data.csv`: original fixed-delta summary rows, including means,
  sample SD, mean areas and the clean/oracle baselines.
- `figures2_3_source_data.csv`: all matched method/condition/tau rows, including
  achieved areas and mean matched deltas.
- `reference_source_data.csv`: all reference summaries, including clean-hard and
  the nonmatched methods omitted from the main curve figures to reduce clutter.
- `source_manifest.json`: absolute source CSV/config paths and SHA-256 hashes.

No coverage or q-star boundary diagnostics are displayed. Fixed-delta points are
not overlaid on the matched curves. Panel labels (a)--(f) are intended for use in
the manuscript text. Figures deliberately omit large global titles and bottom
notes: the experimental settings and explanations belong in the captions above.
