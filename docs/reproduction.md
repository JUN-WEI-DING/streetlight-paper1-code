# Reproduction guide

## Scope and quickstart

Use the README's locked Python 3.12 environment from this checkout. Python 3.11
is also permitted by the dependency specification; validation here uses 3.12.
The artificial example and unit tests run without study data. They validate
software behavior, not agreement with observed Taiwan results. The separate [local review bundle](data-bundle.md) supplies processed inputs
and intermediates for an independently tested replay. It has no public download
URL yet; no access-on-request arrangement is promised.

The example generates 288 UTC timestamps at 10-minute intervals, sinusoidal PV,
a 6.4 kW night load, and artificial AEF. Its fixed-hour Asia/Taipei schedule does
not implement the study's geographic solar-zenith schedule. Its battery is
20 kWh / 8 kW with 90% round-trip efficiency and zero initial SOC, not the selected
paper design. The 48 artificial hours are repeated to 20 × 365 days only to
exercise the accounting interface.

`dispatch.csv` contains power in kW, SOC in kWh, and AEF in kg CO2e/kWh.
`summary.json` reports 64-light installation totals: energy in kWh, emissions in
tonnes CO2e, and lifecycle cost in NTD. Divide by 64 for per-light values.

## External inputs

| Input | Representation and units | Consumer |
|---|---|---|
| Municipal PAR | CSV with first-column UTC datetime and one numeric PAR column per city (µmol photons m⁻² s⁻¹), or Parquet with DatetimeIndex. Recalibrate if changing units; do not substitute W/m² directly. | calibration, baseline |
| Regional AEF | Seven CSVs named `north`, `central`, `south`, `east`, `island_penghu`, `island_kinmen`, `island_lienchiang` with `.csv`; first column UTC datetime, `FLOW_UNIT_FINAL_AEF` in kg CO2e/kWh (`AEF` fallback supported). | dispatch |
| Allocation | Included `config/region_city_map.json`; city names match PAR columns. | baseline/aggregation |
| County geometry | External `data/geo/county_boundaries.shp` plus companion files, official TWD97 longitude/latitude release 1140318 and `COUNTYNAME`; download with `fetch_bundle_boundaries.py`. | representative points for solar-zenith lighting |
| Generation and demand | Provider-format Parquet, as handled by `streetlight.processing.power_adapter`; artificial schema examples are in `test_paper1_final_data_pipeline.py`. Taipower local time converts to UTC. | raw-data pipeline |
| Satellite | Weekly PAR Zarr stores checked by `check_par_weekly_root.py`, with time/latitude/longitude coordinates. | PAR aggregation |
| Regional power/flow | Pipeline-produced interval-energy tables with signed flows and storage columns. | AEF and fuel/storage sensitivities |
| Weather | Separately downloaded NASA POWER hourly UTC T2M/WS10M; use `fetch_bundle_weather.py` after bundle staging to verify study values. | PV benchmark |
| Wind classification | External `data/power/wind_unit_classification.csv` configured by `data.wind_unit_classification_path`; unit names mapped to onshore/offshore categories. The supplied snapshot has no classification file and uses the existing prefix fallback; do not invent a mapping for reference comparison. | AEF and future-grid calculations |
| Population | External study-year municipality mapping. | population-weighted selection |
| Intermediate results | Baseline/Pareto/selected-allocation CSVs, structured-values JSON, PV comparison JSON, figure-data tables. | supplementary summaries |

Naive timestamps are treated as UTC by the lighting implementation. Use
consistent UTC indices across PAR and AEF; lighting converts to Asia/Taipei.
Input readers alone do not establish data quality or completeness.

After preparing the PAR/AEF inputs and county geometry, a baseline can be run:

```bash
STREETLIGHT_CONFIG=config/paper_baseline.yaml \
PAPER_PAR_PATH=/absolute/path/to/par_wide.csv \
PAPER_AEF_DIR=/absolute/path/to/aef \
uv run --locked python scripts/analysis/run_paper_baseline.py
```

Study output defaults to `outputs/final_runs/paper1_canonical_results/`.
`PAPER_OUTPUT_DIR` relocates the baseline; scripts using `paper1_config.py` also
read YAML `paper.workflow` paths. Adjust those paths consistently when relocating
supplementary outputs. `STREETLIGHT_CONFIG` selects the study configuration.

Inspect raw-data interfaces with:

```bash
uv run --locked python scripts/analysis/check_paper1_canonical_inputs_preflight.py --help
uv run --locked python scripts/analysis/run_paper1_final_data_pipeline.py --help
```

Both require explicit `--demand-path`, `--generation-path`, and
`--par-weekly-root`. The standalone weekly checker requires `--weekly-root`.
Run preflight before costly reconstruction. Inspect promotion with
`--promote-dry-run` before choosing `--promote` to replace local generated inputs.
The full pipeline and capacity sweep are outside the quickstart.

## Scientific calculation map

Script names below are under `scripts/analysis/`. Topics are used instead of
changing manuscript table numbers.

| Main-text / SI topic | Code entry | Upstream inputs |
|---|---|---|
| Dispatch, capacity design, operating emissions/cost | `run_paper_baseline.py`; `src/streetlight/simulation/` | PAR, AEF, allocation, geometry |
| Regional attribution and storage pool | `run_paper1_final_data_pipeline.py`; `src/streetlight/aef/` | power, flow, storage |
| Hardware lifecycle comparison | `src/streetlight/lca/hardware.py` | selected factors and embedded aggregate coefficients |
| Calibration checks | `calibrate_par_to_kw_factor.py`, `run_calibration_robustness.py` | PAR, geometry, baseline |
| Method simplifications | `run_method_simplifications.py` | baseline, PAR, AEF |
| Future static grid states | `run_future_grid_scenarios.py` | baseline, regional power |
| Regional allocation sensitivity | `run_allocation_sensitivity.py` | regional power, PAR/AEF, geometry, selected baseline design |
| One-way/fuel/storage sensitivities | `run_paper_sensitivity.py`, `run_subjective_sensitivity.py` | baseline, power for fuel/storage cases |
| Degradation and payback | `run_degradation_sensitivity.py`, `run_carbon_payback_time.py` | selected designs and baseline |
| Economic and marginal alternatives | `run_marginal_cost_economic_lens.py`, `run_merit_order_marginal_carbon.py` | baseline and generation inputs |
| PV model comparison | `paper1_pv_benchmark.py` | baseline, PAR/AEF, geometry, weather |
| Input quality / storage audit | `paper1_aef_quality.py`, `paper1_aef_storage_audit.py` | canonical input snapshots |
| Regression and composition | `_build_regression_tokens` in `build_paper1_manuscript_values.py` | municipal AEF/PAR/abatement table |
| Lifecycle-aware / population selection | `_build_selection_sensitivity_values` in the same module | candidate panel, population |
| Battery life / discount rate | `_build_battery_lifetime_values`, `_build_discount_rate_values` in the same module | selected allocation, energy totals |
| Conditional cost uncertainty | `paper1_conditional_uncertainty.py` | values JSON and PV comparison JSON |
| Earlier complete analysis sequence | `run_paper1_suite.py` | all inputs needed by its component analyses |

The suite predates some supplementary additions and does not call every newer
PV/quality/conditional analysis. The value builder consumes derived figure tables supplied in the separate
local review bundle. The standalone replay regenerates the core Figure 2–4
tables and uses frozen supplemental scenario tables as upstream inputs. Its
base-only phase breaks the PV/uncertainty dependency cycle. See the bundle guide
for exact commands and scope: quick replay is not a full rerun of every scenario.
Historical rendering tokens remain for compatibility, not as an editorial workflow.

## Scientific settings that must remain explicit

- Runtime calibration determines the PAR-to-kW coefficient. The study's effective
  value was 0.0077; YAML 0.0081 is a fallback. Inspect `paper_contract.json`.
- The installation has 64 lights × 0.1 kW. Selected study factors 1.85/8.45 imply
  about 0.52 kWp PV and 1.32 kWh battery per light; these differ from the demo.
- The sweep compares `max(load − PV, 0)` before storage against imports after
  storage. Legacy keys named `grid_only` do not by themselves establish a
  full-load/no-PV comparison. Separate full-load diagnostics use another boundary.
- The sweep starts at zero SOC; a separate diagnostic has `init_soc=0.5`.
  Exporting code must not silently harmonize those historical defaults.
- The study's common 50,892 ten-minute samples use scaling
  `20 * 8760 / (50892 / 6)`. There is no annual SOC reset or elapsed-gap loss
  simulation in that scaling.
- The 2024 reference is conditional on sampled conditions. Static 2030/2050 AEF
  stress states retain hardware/dispatch; they are not a 20-year forecast.
  Attributional AEF does not establish causal marginal emissions.
- Hardware Python code uses calibrated aggregates, not a SimaPro process rebuild.
- Conditional uncertainty samples four cost inputs separately within each PV
  model × battery-life condition. It does not assign a joint probability to
  future physical conditions.

## Validation boundary

Tests cover synthetic AEF/storage/flow, timestamp adapters, quality checks, PV
interfaces, regression, selection geometry, replacement accounting, dispatch,
discounting, and cost sampling. Private result-snapshot and manuscript tests are
excluded. Export verification compares the source and export using identical
artificial inputs and checks byte identity of package files. The synthetic validation does not establish agreement with study results.
The separate processed-input validation reruns the documented capacity/scenario
suite and downstream analyses; see `docs/data-bundle.md` for scope and results.

Validated on Python 3.12.3 with the checked-in lockfile: 111 tests passed. All
package and analysis modules imported, six CLI help commands completed, and a
small synthetic Zarr dataset round-tripped successfully. Original/export
outputs matched for dispatch, an eight-row capacity sweep, hardware LCA, and
regression on identical artificial inputs.

The isolated processed-input validation compared 8,859 numerical values and
79 figure/result CSVs with the current reference bundle at relative/absolute
tolerances of 1e-8, with no differences. This combines the previously verified
full-suite run with a separate rerun of the new allocation stage and final value
builder; unchanged stages were not repeated. All 179 files in the earlier bundle passed
SHA-256 verification. The earlier 165-file bundle omitted the 14 public NASA responses: all
165 retained files passed verification in a fresh temporary checkout. All 14
weather cells were downloaded afresh and matched the recorded scientific values;
API metadata differed. Quick replay then matched 8,833 numerical values and three
core figure tables and generated six figures. The full suite was not repeated
for this packaging-only change. Reference answers are kept separate from working outputs.
The 19 focused allocation, bundle, and suite-resume tests also pass. See
`docs/data-bundle.md` for commands, validation scope, and figure mapping.

The current 160-file bundle also omits the five official boundary components.
The separate boundary download was verified in a fresh checkout: all components
match the study byte-for-byte. Eleven focused input/bundle/replay tests pass.
No scientific input or calculation changed in this further packaging step.
