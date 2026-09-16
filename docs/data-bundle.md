# Processed-input and intermediate-result bundle

This is a **local author-review package**, not a published dataset. The archive
is not covered by the code's MIT license. Public redistribution of the specific
PAR product and downstream derivatives is still unresolved; no public download
URL or access-on-request promise is made.

## What is included

The selection is executable in `scripts/release/build_local_data_bundle.py`.
The initial package contains 175 source files, 405,273,481 uncompressed bytes
(386.5 MiB), approximately 74.1 MiB gzip-compressed. `bundle.json` records source
commit, original relative paths, sizes and SHA-256 hashes. The accompanying
`.sha256` verifies the archive. No raw satellite stores, raw download Parquet,
licensed process database exports, manuscript, reviews, or credentials are
included. Historic processing manifests retain original staging-path strings as
provenance; these are not runtime paths.

| Group | Purpose |
|---|---|
| Municipal PAR and imputation flags | PV forcing, calibration, quality audit |
| Seven regional AEF files and storage pools | Dispatch emissions and storage audit |
| Regional generation, national generation/capacity/categories, flow, selected quality records | Fuel/storage/static-grid/marginal analyses and audit |
| County boundaries, 2024 municipal population | Solar-zenith locations and population weighting |
| 14 NASA POWER meteorology responses | Offline PVWatts comparison; full 2024 hourly UTC weather |
| Baseline 4,500-design × 22-city panel and compact scenario products | Capacity reselection and downstream analysis without rerunning every sweep |
| 13 numerical figure tables and cogeneration sensitivity table | Frozen upstream evidence for supplemental summaries |
| Three final JSON snapshots | Expected answers, kept exclusively under `reference/` |

Weekly/long-format PAR duplicates, alternate scenarios' large candidate panels,
and duplicate baseline folders are omitted. The power records make this bundle
larger than a dispatch-only pack: the manuscript's SI quality audit consumes
national generation/imputation records, while the marginal lens needs capacity.

## Quick reproduction

Start with a **fresh checkout**; existing `reference/` or `outputs/final_runs/`
causes staging to stop instead of replacing work. Place the archive anywhere
outside those paths. Commands run from the repository root:

```bash
uv sync --locked --extra pv-benchmark
# In the archive's directory:
sha256sum -c paper1-data-review.tar.gz.sha256
# Back in the checkout:
uv run --locked python scripts/analysis/reproduce_from_bundle.py \
  --archive /absolute/path/to/paper1-data-review.tar.gz
```

The script validates all file hashes, stages inputs/intermediates, and keeps
expected final JSON answers under `reference/`. It then:

1. Recomputes the Pareto frontier and knee from the full candidate panel.
2. Regenerates Figure 2–4 numerical source tables, including hardware-stage LCA.
3. Builds upstream manuscript values without consuming PV/uncertainty outputs.
4. Recalculates selected-design dispatch and two PVWatts alternatives from PAR,
   AEF, county geometry and cached weather; no network fetching is requested.
5. Recalculates 12 conditional economic uncertainty cases, with up to 50,000
   draws each and a separate seed check.
6. Rebuilds the complete value object, including regression/composition,
   lifecycle/population selection, battery life/discounting, and AEF quality/
   storage audits. Scenario summaries remain frozen inputs in this mode.
7. Compares numerical fields against the three expected JSONs with
   `rtol=1e-8`, `atol=1e-8`; missing keys, list lengths and Boolean/null changes
   fail. Provenance/code hashes, paths, versions, formatted prose and general
   text are not numerical comparison targets. Input-byte hashes are separately
   verified. This is numerical agreement, not a manuscript-text comparison.
8. Checks the regenerated Figure 2–4 CSVs against their references (4,500,
   12 and 22 rows respectively), then writes six numerical figures as PNG/PDF.

Outputs: `outputs/reproduction/verification.json`,
`outputs/reproduction/figures/`, and regenerated JSONs under
`outputs/final_runs/paper1_canonical_results/`.

The dependency cycle is explicitly broken with the value builder's
`--base-only` option. Final reference answers are never copied to working
output paths. Frozen scenario summaries and figure tables ARE upstream inputs;
this mode must not be described as a complete rerun from observations.

## Figure mapping to the current EIAR manuscript

| EIAR figure | Source / output | Reproduction boundary |
|---|---|---|
| 1 study area | County geometry + region allocation | Simplified accounting-region map; decorative terrain, power-line and plant overlays omitted |
| 2 Pareto | Regenerated `f2_pareto_frontier_source.csv` | Full candidate panel → frontier → knee; simplified layout |
| 3 lifecycle | Regenerated `f3_lca_balance_components.csv` | Hardware engine + selected operational emissions; simplified layout |
| 4 municipal comparison | Regenerated `f4_input_output_maps_source.csv` | Calibration, AEF and selected candidate rows; simplified map layout |
| 5 ranking shifts | Bundled `f5_rank_shift_bump_data.csv` | Plot frozen ranking table; alternative sweeps are not rerun in quick mode |
| 6 static grid states | Bundled `f6_future_grid_retention_points.csv` | Plot 21 regional/state points without pooled regression |

Legacy `f7`/`f8` and `figS*` filenames identify supplemental numerical sources,
not Figures 7 and 8 in the six-figure EIAR main manuscript. Their CSVs are
included; the manuscript's exact editorial layouts are not claimed reproduced.

## Recompute the computationally expensive layers

Use another fresh checkout, stage without quick replay, then run the existing
analysis entries. Explicitly pin the root because parent directories can contain
unrelated Git repositories:

```bash
uv run --locked python scripts/analysis/reproduce_from_bundle.py \
  --archive /absolute/path/to/paper1-data-review.tar.gz --stage-only
export STREETLIGHT_ROOT="$PWD"
export STREETLIGHT_CONFIG=config/paper_baseline.yaml
uv run --locked python scripts/analysis/run_paper_baseline.py
```

This recomputes calibration, full dispatch/capacity search and baseline outputs
from processed inputs. It does not redo the raw satellite or raw Taipower
preprocessing. `run_paper1_suite.py` runs the older full suite, including full
alternative sweeps; see `docs/reproduction.md` for individual entries. It is
computationally expensive and was **not run for this handoff**. Check resources
and run long jobs with persistent logging before choosing that route.

Do not label a suite run alone as complete EIAR reconstruction: it predates PV
and conditional analyses; the complete supplemental table/figure transformation
sequence is not yet extracted from the private editorial workflow. Quick replay
has explicitly identified frozen inputs for these stages.

The optional wind-classification path is absent in the source snapshot. The
calculator uses its existing prefix fallback. The storage audit in quick replay
checks the resulting baseline against the frozen AEF. Preserve this behavior
for comparison; a new onshore/offshore mapping would be a changed scientific
input. Full future-grid rerun agreement remains unverified.

## Data origins and publication status

Checked 2026-09-16; provider links establish terms/context, not an assertion that
every historical local file has a fully resolved redistribution chain.

- **PAR:** JAXA P-Tree Himawari-derived research product. The
  [P-Tree terms](https://www.eorc.jaxa.jp/ptree/terms.html) distinguish JAXA
  geophysical products from JMA Himawari Standard Data; the latter has an
  explicit redistribution restriction. [JAXA site policy](https://global.jaxa.jp/policy.html)
  also applies. Confirm the specific PAR product/version and derived-data
  redistribution before publication. Do not blanket-label PAR or its derived
  results MIT/CC0, or infer that every PAR product is categorically prohibited.
- **Power:** Taiwan Power Company, [generation](https://data.gov.tw/en/datasets/37331)
  and [regional flows](https://data.gov.tw/en/datasets/37326), whose provider
  pages list Open Government Data License 1.0. Retain provider attribution and
  processing descriptions; these data have been transformed by the research.
- **Boundaries:** National Land Surveying and Mapping Center / Ministry of
  the Interior. The historical project links the
  [county-boundary service](https://data.gov.tw/dataset/32158), which lists
  OGDL 1.0. The precise downloadable shapefile release still needs provenance
  confirmation; the service listing alone does not establish its version.
- **Population:** Ministry of the Interior, 2024 year-end registered population;
  exact source resource, cells and hash are embedded in the supplied JSON.
  Registered population is a deployment-weight proxy, not streetlight inventory.
- **Weather:** NASA POWER T2M/WS10M responses. Preserve request URLs and response
  metadata; see [meteorological sources](https://power.larc.nasa.gov/docs/methodology/data/sources/).
  Exact cached responses support repeatability; refreshed API data can differ.
- **LCA:** original code uses calibrated aggregate parameters. No SimaPro or
  ecoinvent process database is supplied; numerical replay does not reconstruct
  the licensed background process model.

## Author-side packaging

```bash
python scripts/release/build_local_data_bundle.py \
  --source /absolute/path/to/research-repo \
  --output dist/paper1-data-review.tar.gz --dry-run
python scripts/release/build_local_data_bundle.py \
  --source /absolute/path/to/research-repo \
  --output dist/paper1-data-review.tar.gz
```

Only this packaging command needs the private research repo. Reader commands
consume the archive and standalone code. The archive is ignored by Git; do not
commit or publish it until the outstanding source-specific rights questions
are resolved.
