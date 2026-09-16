# Processed-input and intermediate-result bundle

This is a **local author-review package**, not a published dataset. The source
repository is private. The code's MIT license does not cover third-party data.
Raw public-source files are obtained from their providers; the bundle retains
research-generated inputs, intermediates and reference results.
Processed municipal PAR is retained with JAXA/P-Tree attribution under the
project’s adopted JAXA research-data policy interpretation; see the data-origin
section below.

## What is included

The current archive contains 160 data/metadata files, plus `bundle.json` and
`THIRD_PARTY_NOTICES.md`, and is
approximately 69.4 MiB compressed. Paths are preserved from the research workflow.
The selection is executable in `scripts/release/build_local_data_bundle.py`.

| Role | Contents | Files | Use |
|---|---|---:|---|
| Processed input | Municipal population weights | 1 | Population-weighted selection |
| Processed input | Seven regional AEF series and storage pools | 8 | Dispatch emissions and storage audit |
| Input documentation | Preparation manifests and quality/provenance records | 3 | Interpret the prepared inputs |
| Processed input | Municipal PAR and imputation flags | 2 | PV forcing and calibration |
| Processed input and audit | Aligned regional/national power, capacity, flows and quality records | 40 | AEF, fuel/storage, marginal and input-quality analyses |
| Analysis intermediates and reference results | Capacity panel, allocation, calibration, sensitivities and summary JSONs | 93 | Inspect selection or compare recalculated results |
| Figure intermediates and reference results | Numerical figure source tables | 13 | Draw figures or compare regenerated tables |
| **Total** | | **160** | |

The capacity panel contains 4,500 PV/battery combinations × 22 municipalities
(99,000 rows). It lets readers inspect candidate designs and reproduce Pareto
selection without first repeating every dispatch simulation. The 93 analysis files
include the three final JSON snapshots used as expected answers; these are staged
only under `reference/`, never substituted for newly computed final outputs.
Quick mode reuses scenario intermediates; full mode keeps analysis and figure
references separate and recalculates them from processed inputs.

`bundle.json` records the source commit, relative paths, sizes and SHA-256 hashes;
the accompanying `.sha256` checks archive integrity. It also records requests for
14 NASA weather cells and the five boundary components, without bundling those
source files. Historic processing manifests retain author staging paths as
provenance; these are not runtime paths.

Raw satellite stores, raw Taipower downloads, the source population ODS, NASA API
responses, county geometry, licensed process databases, manuscripts, reviewer
correspondence and credentials are excluded. Weekly/long-format PAR duplicates,
large alternative-scenario capacity panels and duplicate baseline folders are
also excluded. Processed power tables remain necessary for the documented analyses
even though their original source is public.

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
uv run --locked python scripts/analysis/fetch_bundle_boundaries.py
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
uv run --locked python scripts/analysis/fetch_bundle_boundaries.py
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

The delivery-description update retains all 160 payload files byte-for-byte.
The rebuilt archive passes its checksum; quick and full staging both pass in
fresh temporary directories, keeping final reference answers separate from working
outputs. Quick staging retains all 99,000 capacity rows. Eleven focused
bundle/replay/external-input tests pass. No numerical model changed, so the
capacity sweep, numerical replay and provider downloads were not repeated for
this documentation update. Earlier numerical validation is described below.

The current 160-file bundle additionally omits all five county-boundary files.
In a fresh temporary checkout, all 160 retained files passed size/hash checks;
the new boundary command downloaded the official ZIP and verified all five
components byte-for-byte. Eleven focused input/bundle/replay tests passed.
This boundary packaging change does not change coordinates or scientific inputs;
the numerical replay below was not repeated solely for identical geometry.


For the preceding 165-file bundle, all retained files passed checksum checks in a
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

- **PAR:** original Himawari-derived PAR was supplied by JAXA's P-Tree System.
  The authors prepared municipal 10-minute time series and accompanying
  imputation flags. The project adopts the [JAXA research-data
  policy](https://earth.jaxa.jp/en/data/policy/index.html) as the sharing basis for
  these derivatives, with source attribution and processing disclosure. This
  is the authors' policy interpretation, not individual JAXA approval. Retain
  the credit and policy link in `THIRD_PARTY_NOTICES.md`; do not apply the
  code's MIT license to these data. Raw gridded products are not bundled.
- **Power:** Taiwan Power Company, [generation](https://data.gov.tw/en/datasets/37331)
  and [regional flows](https://data.gov.tw/en/datasets/37326), whose provider
  pages list Open Government Data License 1.0. Retain provider attribution and
  processing descriptions; these data have been transformed by the research.
- **Boundaries:** National Land Surveying and Mapping Center / Ministry of
  the Interior. [Dataset 7442](https://data.gov.tw/dataset/7442) supplies the
  1140318 (2025-03-18) TWD97 longitude/latitude release under OGDL 1.0.
  All five downloaded COUNTY_MOI_1140318 components match the study files
  byte-for-byte. `fetch_bundle_boundaries.py` verifies their hashes before
  writing them to `data/geo/county_boundaries.*`; it does not change the CRS
  or coordinates. If direct access is unavailable, download the official ZIP
  manually and pass `--archive /path/to/file.zip`. Attribution: National Land
  Surveying and Mapping Center, Ministry of the Interior, Taiwan, 2025,
  county boundaries release 1140318, [OGDL 1.0](https://data.gov.tw/license).
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

## Public-source inputs and reproduction routes

| Source | Obtain from | Preparation / use |
|---|---|---|
| PAR | Individual [JAXA P-Tree registration](https://www.eorc.jaxa.jp/ptree/registration_top.html) | Raw preprocessing expects weekly PAR Zarr stores; the two bundle routes instead use municipal PAR tables |
| Taipower generation and regional power/flows | [Generation](https://data.gov.tw/en/datasets/37331), [regional power](https://data.gov.tw/en/datasets/37326) | Prepare study-period archives for the raw pipeline; the bundle retains cleaned/aligned power tables |
| Population | MOI resource URL and sheet/cells recorded in the bundled `data/geo/municipal_population_2024.json` | The extracted municipal weights are included; the original ODS is not |
| NASA POWER weather | `fetch_bundle_weather.py` after staging | Downloads the 14 requested 2024 hourly UTC cells |
| County boundaries | [NLSC dataset 7442](https://data.gov.tw/dataset/7442), release 1140318; `fetch_bundle_boundaries.py` | Downloads the five geometry components; manual ZIP input is also supported |

The default quick route reuses research intermediates. The advanced `--full`
route reruns analyses from processed inputs. Both require the separately fetched
weather and geometry. The raw-input interfaces and schemas are documented in
[the reproduction guide](reproduction.md); acquisition and conversion of raw PAR
to weekly stores are not automated by either bundle route. A complete fresh
reconstruction of all 2024 raw inputs has not been verified.

Raw PAR is available through provider registration; the author's original archive
is temporarily unavailable on this machine. This does not prevent preparing code
and retained intermediates. Before public release, finalize the package access
route and fixed version; do not describe the private repository or local bundle
as already accessible to readers. Bundle and download checks verify the inputs
used for numerical comparisons.

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
