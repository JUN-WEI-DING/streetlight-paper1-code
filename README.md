# Dispatch-aware PV-battery streetlight research code

Research code for *A dispatch-aware framework for resource-efficient distributed
electrification: A case study of photovoltaic-battery streetlights*.

This is the independent public source repository for the study. It contains the
calculation methods, study configuration, synthetic examples, and independent
unit tests. Observed time series, geographic datasets, licensed LCA exports,
manuscripts, reviewer correspondence, and paper result files are not tracked in Git.
The synthetic example demonstrates execution; it does not reproduce the
paper's numbers.
The [v1.0.0 release](https://github.com/JUN-WEI-DING/streetlight-paper1-code/releases/tag/v1.0.0) provides code and a companion data bundle containing
processed inputs, capacity-search and sensitivity results, and numerical figure
tables; public raw source files are obtained from their providers.

Two entry points are documented in [the bundle guide](docs/data-bundle.md):

- **Quick replay:** reuse the 99,000-row capacity panel and scenario intermediates
  to inspect selection, recalculate downstream values, and draw numerical figures.
- **Analysis-suite rerun:** start from processed inputs and rerun capacity search
  and the documented sensitivity analyses, including fuel allocation.

Both routes download NASA weather and official county boundaries separately.
Neither route reconstructs the original satellite archive or licensed LCA process
model. Code and data are publicly downloadable without login.
Processed municipal PAR is retained with JAXA/P-Tree attribution under the
project’s adopted interpretation of the JAXA research-data policy; see the
third-party notices for processing and source terms.

The [release details](docs/release-draft.md) identify the fixed code revision,
artifacts and data attribution.

## Run the synthetic example

Use Python 3.12 and `uv`, from this checkout:

```bash
uv sync --locked --extra pv-benchmark
uv run --locked python examples/synthetic/run_demo.py
uv run --locked --extra pv-benchmark pytest -q
```

The deterministic example builds two days of artificial PV, lighting load, and
AEF at 10-minute intervals, runs the original dispatch and emissions/cost engine,
and writes a time series and summary under `outputs/synthetic/`. Its repeated
20-year scaling is illustrative, not a future prediction. No download is needed
for the example after dependencies are installed.

## Code layout

- `src/streetlight/`: PAR processing, regional AEF, storage dispatch, costs,
  capacity search, hardware LCA aggregates, and Pareto plotting.
- `scripts/analysis/`: study workflows and supplementary sensitivity calculations.
- `config/`: original scientific settings and region-to-city allocation.
- `examples/synthetic/`: artificial input generator and runnable demonstration.
- `tests/`: checks that need no study dataset.

[Reproduction guide](docs/reproduction.md) explains inputs, entry points,
scientific boundaries, and which calculations require externally supplied files.
Run from the checkout using the editable environment above; this is not a
standalone wheel with bundled scientific inputs.

## Provenance and scope

Exported from private research commit
`7b1e607cdc66aada9ac4a176985c80405ffd1508` with a fresh Git history.
The 25 package source files are retained byte-for-byte. Analysis scripts retain
their algorithms; the three raw-input command-line tools require explicit input
paths instead of defaulting to the author's machine. The value builder adds a
base-only phase to remove the PV/uncertainty dependency cycle; calibration
robustness converts datetime differences explicitly to hours across timestamp
resolutions; new standalone
entries stage and verify a data bundle and regenerate numerical figures.
The manuscript-value builder is retained because it implements regression, selection, discount-rate,
and battery-lifetime calculations. Its older table/token formatting is not a
manuscript export interface. Manuscript and original editorial figure-layout
tools are excluded. Numerical figure reproductions use separate, simpler layouts.

Original code and documentation use the MIT license. Third-party data and
software retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
