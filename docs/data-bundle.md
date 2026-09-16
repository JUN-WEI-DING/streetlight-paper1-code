# Processed-input and intermediate-result bundle

This is a **local author-review package**, not a published dataset. The archive
is not covered by the code's MIT license. Public redistribution of the specific
PAR product and downstream derivatives is still unresolved; no public download
URL or access-on-request promise is made.

## What is included

The selection is executable in `scripts/release/build_local_data_bundle.py`.
NASA POWER API responses are omitted and downloaded separately. The packaging
command reports the current file count and archive size. `bundle.json` records source
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
| NASA POWER request coordinates and value hashes (no responses) | Download 14 full-year 2024 UTC responses and check their values |
| Baseline 4,500-design × 22-city panel and compact scenario products | Capacity reselection and downstream analysis without rerunning every sweep |
| 13 numerical figure tables and allocation-sensitivity results | Quick-mode intermediates; regenerated in full mode |
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
  --archive /absolute/path/to/paper1-data-review.tar.gz --stage-only
uv run --locked python scripts/analysis/fetch_bundle_weather.py
uv run --locked python scripts/analysis/reproduce_from_bundle.py
```

The script validates all file hashes, stages inputs/intermediates, and keeps
expected final JSON answers under `reference/`. It then:

1. Recomputes the Pareto frontier and knee from the full candidate panel.
2. Regenerates Figure 2–4 numerical source tables, including hardware-stage LCA.
3. Builds upstream manuscript values without consuming PV/uncertainty outputs.
4. Recalculates selected-design dispatch and two PVWatts alternatives from PAR,
   AEF, county geometry and separately downloaded, validated weather. The
   download command requires network access; calculation itself does not.
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

## Rerun the analysis suite from processed inputs

In a fresh checkout, use the same archive with `--full`:

```bash
uv run --locked python scripts/analysis/reproduce_from_bundle.py \
  --archive /absolute/path/to/paper1-data-review.tar.gz --full --stage-only
uv run --locked python scripts/analysis/fetch_bundle_weather.py
uv run --locked python scripts/analysis/reproduce_from_bundle.py --full
```

This mode leaves bundled analysis results and figure tables exclusively under
`reference/`. It stages processed inputs, geometry and weather. It reruns the full
capacity search, the 16-case fuel-allocation sensitivity, alternative
methods, future static grid states, sensitivities, economic analyses, PV
comparison and conditional uncertainty; it rebuilds all 13 numerical figure
source tables. Numerical JSON outputs, all bundled result CSVs and all figure
source CSVs are compared with the preserved references at `rtol=atol=1e-8`.
The same `outputs/reproduction/verification.json` records differences and scope.
This is a processed-input reconstruction, not raw satellite/power processing or
licensed SimaPro background-model reconstruction. Figure layouts are simplified.

Run long jobs in tmux and retain console output. The suite writes timing/progress
after each successful stage. If a stage fails, inspect the cause before resuming:

```bash
export STREETLIGHT_ROOT="$PWD"
export STREETLIGHT_CONFIG=config/paper_baseline.yaml
uv run --locked python scripts/analysis/run_paper1_suite.py --start-at STEP_NAME
```

Resuming requires the earlier generated stages to remain in place; it does not
make a partial run independently complete. After the suite, the downstream order
is `rebuild_core_figure_data.py`, `rebuild_supplementary_figure_data.py`,
`build_paper1_manuscript_values.py --base-only`, `paper1_pv_benchmark.py`,
`paper1_conditional_uncertainty.py`, then `build_paper1_manuscript_values.py`.
Finally run `reproduce_from_bundle.py --full --verify-only` to compare the
regenerated outputs without repeating the calculations.

### Verification result

For the current 165-file bundle, all retained files passed checksum checks in a
fresh temporary source checkout. All 14 NASA responses were downloaded without
using author caches and matched the recorded meteorological-value hashes. Quick
replay matched 8,833 numerical fields and three core figure CSVs at 1e-8 and
produced six numerical figures. Ten focused bundle/weather/replay tests passed.
This packaging change did not rerun the full capacity/scenario suite. The exact
MOI population ODS was also downloaded and matched its recorded SHA-256.


Before the public-weather packaging change, isolated validation recomputed 99,000 baseline design/city rows in the
previous full-suite run. The new baseline-plus-16 allocation stage was then
independently executed in that isolated checkout, followed by rebuilding the
final value object. Unchanged stages retain the prior verified outputs; the
whole suite was not rerun for this addition. All 8,859 numerical fields across
the three final JSONs and allocation summary, and all 79 figure/result CSVs
(including 13 numerical figure tables), agree at `rtol=atol=1e-8`. All 179
files in that earlier bundle passed their size and SHA-256 checks. The suite manifest records
the separate allocation run; reported cumulative timing includes both runs.

The calibration-robustness entry converts timedeltas explicitly to hours, with
tests for nanosecond and microsecond timestamp storage. Its
`mean_annual_lighting_hours` values are 4408.6212–4545.3030 hours; raw datetime
integers must not be divided by a presumed nanosecond conversion constant.
This correction is synchronized to the canonical research code and regenerated
reference table. The bundle records the exact research commit in `bundle.json`.
The allocation calculation below updates the final SI sensitivity row and its
reviewer-response explanation; the main manuscript remains unchanged.

### Reproducible allocation sensitivity

`scripts/analysis/run_allocation_sensitivity.py` replaces the historical,
rounded five-region QA summary with an explicitly specified calculation on the
current seven-region inputs. It evaluates a baseline plus 16 one-at-a-time cases:
two fuels, four mainland regions, and relative share changes of -20%/+20%.
For target region r, s'_r = (1 +/- 0.2)s_r; other mainland shares become
s'_j = s_j(1 - s'_r)/(1 - s_r). Only each fuel's `-_other_allocated`
columns change. The national fuel total is conserved at every timestamp;
named generation, island inputs, transfer inputs, and storage settings remain fixed.

Each regional baseline AEF series is checked against its frozen reference before
scenario evaluation. The AEF summary uses the equal mean across seven regions
at common valid timestamps. Original selected capacities, zero-SOC dispatch,
costs and operational comparator are retained; operational abatement is the
equal-municipality mean, and MAC is fixed mean incremental cost divided by actual
scenario mean abatement. The last row of EIAR Table S18 reports independently
selected minima/maxima across cases, not a lower/higher-input pair. Other S18
screening rows retain their first-order MAC convention.

The calculation writes `scenario_summary.csv`, `regional_summary.csv`,
`city_results.csv`, `allocation_shares.csv`, and `summary.json` under
`outputs/final_runs/paper1_canonical_results/allocation_sensitivity/`.
The regional diagnostic reports changes in power-balance residuals with transfer
inputs fixed; absolute residuals cannot be reconstructed without regional load
observations. This is a deterministic accounting stress test, not a balanced
physical-grid forecast, confidence interval, or probabilistic AEF error model.
The historical `outputs/qa/cogen_biomass_sensitivity.csv` is no longer consumed
or bundled. Quick replay uses the new scenario results as intermediates; full
mode recalculates them and compares every output CSV and numerical summary field.

The optional wind-classification path is absent in the source snapshot. Preserve
the calculator's existing prefix fallback when comparing with reference results;
adding a new mapping changes scientific inputs.

## Data origins and publication status

Checked 2026-09-16; provider links establish terms/context, not an assertion that
every historical local file has a fully resolved redistribution chain.

- **PAR:** JAXA P-Tree Himawari-derived research product. The
  [P-Tree terms](https://www.eorc.jaxa.jp/ptree/terms.html) distinguish JAXA
  geophysical products from JMA Himawari Standard Data; the latter has an
  explicit redistribution restriction. [JAXA site policy](https://global.jaxa.jp/policy.html)
  also applies. The [registration page](https://www.eorc.jaxa.jp/ptree/registration_top.html)
  additionally states redistribution restrictions and asks users to contact the
  secretariat before publicly releasing research results. These instructions
  require clarification for our specific processed product; no permission is
  inferred here. Confirm the specific PAR product/version and derived-data
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
  Responses are not bundled. `fetch_bundle_weather.py` downloads the exact
  requested cells/period and checks units, UTC coverage, and a hash of the
  meteorological values. API metadata may change without changing those values.
  Changed values stop replay before calculations; no silent substitution occurs.
- **LCA:** original code uses calibrated aggregate parameters. No SimaPro or
  ecoinvent process database is supplied; numerical replay does not reconstruct
  the licensed background process model.

## Public-source inputs and remaining decisions

Public raw observations are obtained from their providers, not redistributed by
this package. Research-generated intermediate tables remain in the review bundle.

| Input | Reader route | Exact-study limitation |
|---|---|---|
| NASA POWER hourly weather | Stage bundle, then run `fetch_bundle_weather.py` | Current responses must match recorded parameter hashes; metadata-only differences are allowed |
| Taipower generation and flows | Provider links below; preprocessing entry points in `docs/reproduction.md` | Full 2024 acquisition has not been verified from the current provider service. The bundle retains cleaned/aligned generation and flow intermediates, not raw download archives |
| Municipal population | Exact MOI ODS resource, sheet/cells and checksum in `data/geo/municipal_population_2024.json` | The extracted, reordered JSON is a research input intermediate and remains included; the source ODS is not bundled |
| County boundaries | Official dataset 32158 | Exact local release remains unidentified. Existing geometry remains in the private review bundle pending the author's decision; replacing it with today's geometry could change representative points and calculations |
| PAR | Registered P-Tree access; externally prepared weekly Zarr stores | Product/version, redistribution conditions and raw-to-weekly-store acquisition route remain unresolved. Processed municipal PAR remains local-review only |

There is no verified end-to-end public-download reconstruction of all 2024 inputs.
The reproducible route starts from research intermediates plus downloaded weather.
The boundary and PAR decisions must be resolved before claiming a public,
complete reproduction package. No source archive or manuscript availability
statement should imply that the private review bundle is already downloadable.

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
