# Dispatch-aware PV-battery streetlight research code

Research code for *A dispatch-aware framework for resource-efficient distributed
electrification: A case study of photovoltaic-battery streetlights*.

This is a code-only repository prepared for author review. It contains the
calculation methods, study configuration, synthetic examples, and independent
unit tests. Observed time series, geographic datasets, licensed LCA exports,
manuscripts, reviewer correspondence, and paper result files are not included.
The example demonstrates execution; it does not reproduce the paper's numbers.

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
paths instead of defaulting to the author's machine. The manuscript-value
builder is retained because it implements regression, selection, discount-rate,
and battery-lifetime calculations. Its older table/token formatting is not a
manuscript export interface. Manuscript and figure-layout tools are excluded.

Original code and documentation use the MIT license. Third-party data and
software retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
