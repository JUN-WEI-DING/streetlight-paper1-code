# Reproduction guide

## Scope and quickstart

Use the README's locked Python 3.12 environment from this checkout. Python 3.11
is also permitted by the dependency specification; validation here uses 3.12.
The artificial example and unit tests run without study data. They validate
software behavior, not agreement with observed Taiwan results. The separate [public data bundle](data-bundle.md) supplies processed inputs
and intermediates for an independently tested replay. The bundle is publicly available in
[release v1.0.0](https://github.com/JUN-WEI-DING/streetlight-paper1-code/releases/tag/v1.0.0).

The example generates 288 UTC timestamps at 10-minute intervals, sinusoidal PV,
a 6.4 kW night load, and artificial AEF. Its fixed-hour Asia/Taipei schedule does
not implement the study's geographic solar-zenith schedule. Its battery is
20 kWh / 8 kW with 90% round-trip efficiency and zero initial SOC, not the selected
paper design. The 48 artificial hours are repeated to 20 × 365 days only to
exercise the accounting interface.

`dispatch.csv` contains power in kW, SOC in kWh, and AEF in kg CO2e/kWh.
`summary.json` reports 64-light installation totals: energy in kWh, emissions in
tonnes CO2e, and lifecycle cost in NTD. Divide by 64 for per-light values.

## Choose a reproduction route

Use [the bundle guide](data-bundle.md) for both executable routes. **Quick replay**
starts with the capacity panel and scenario intermediates; **full analysis**
recalculates capacity and sensitivity results from the processed inputs. Both
obtain weather and county geometry separately. The bundle contains municipal PAR,
regional AEF, cleaned power records and population weights, so readers do not need
to collect all raw observations before using these routes.

Raw-data preprocessing is a separate advanced workflow described below. It expects
prepared study-period Taipower archives and weekly PAR Zarr stores; it is not an
automated provider-download-to-paper workflow. Keep provider acquisition, our
processed data, and the expected numerical answers distinct.

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
| Population | Processed 2024 municipal weights included in the review bundle; source ODS obtained from MOI. | population-weighted selection |
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
public data bundle. The standalone replay regenerates the core Figure 2–4
tables and uses frozen supplemental scenario tables as upstream inputs. Its
base-only phase breaks the PV/uncertainty dependency cycle. See the bundle guide
for exact commands and scope: quick replay is not a full rerun of every scenario.
Historical rendering tokens remain for compatibility, not as an editorial workflow.

## Scientific settings that must remain explicit

- Runtime calibration determines the PAR-to-kW coefficient. The study's effective
  value was 0.0077 kW per PAR unit at solar factor 1, calibrated before the
  PAR–AEF intersection; YAML 0.0081 is a fallback. Inspect `paper_contract.json`.
- The installation has 64 lights × 0.1 kW. Selected study factors 1.85/8.45 imply
  0.5203125 kWp PV and 1.3203125 kWh battery per light; these differ from the demo.
  PV capacity is `18 * solar_factor` kW; battery capacity is
  `10 * battery_factor` kWh and power is `min(5 * battery_factor, 9.6)` kW
  per installation (9.6 kW, or 0.15 kW per light, at the selected design).
- Reproduce the exact factor sequences: solar `0.25 + 0.2*i`, `i=0,...,49`;
  battery `0.25 + 0.2*j`, `j=0,...,89`. In `run_paper_baseline.py`, rounded
  interval counts produce endpoints 10.05 and 18.05 despite YAML bounds 10
  and 18. These give 4,500 designs and 99,000 municipality–design rows.
- Lighting draws 6.4 kW at solar zenith ≥90.833° and zero otherwise. Storage
  uses the full `[0, capacity]` SOC range and charge/discharge efficiencies
  `sqrt(0.90)`. There is no further depth-of-discharge factor, self-discharge,
  grid charging, surplus credit, spin-up, terminal-SOC constraint or terminal
  credit. Charging and discharging are mutually exclusive.
- The sweep compares `max(load − PV, 0)` before storage against imports after
  storage. Legacy keys named `grid_only` do not by themselves establish a
  full-load/no-PV comparison. `compute_city_emission` stores this deficit under
  the grid-only-import label; `streetlight_sim` uses the separate full-load
  comparator. Use the stored Pareto rows for the reported selected allocation.
  In the PV-profile check, removing below-horizon PV changes the comparator
  by at most 0.00017493 tCO2e and 0.818305 NTD per installation
  (2.733 gCO2e and USD 0.000398 per light); it is not exactly full load.
- The sweep starts at zero SOC; a separate diagnostic has `init_soc=0.5`.
  That diagnostic uses the 10 kWh reference battery, not the selected 84.5 kWh
  installation. Exporting code must not harmonize those historical defaults.
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

## Staged audit and original-run traceability

Staged dispatch reads `outputs/final_runs/paper1_canonical_inputs/par/par_wide.csv`
and the seven regional CSVs under the same root's `aef/`. Quality/storage audits
also need the matching `power/` tables, `aef/storage_pools.csv`, preparation
reports, flags and manifests. Retain the input hashes with results. The original
run recorded uncommitted source changes: its base Git revision alone does not
identify the executed source. Source hashes and agreement with frozen outputs
identify the staged replay. The public locked environment supports the documented
routes; it is not a recovered lockfile for that original raw-input run.

The staged AEF audit records 105,565 flagged generation cells over 589 timestamps
(568 restored timestamps and 21 partially missing rows). The low-generation
screen rejects 10,388 source rows across 59 timestamps; three timestamps are
interpolated and none survives final alignment. Flow preparation removes 63
all-zero regional timestamps, averages one duplicate and excludes one off-grid
output timestamp; rounding permits a two-minute distance from a ten-minute grid
point. Legacy storage-pool magnitudes convert with `1/6` for MWh and `1000/6`
for kg CO2e; these conversions preserve AEF ratios. The staged replay matches
all seven frozen AEF series within 1e-10 kg CO2e/kWh. These are snapshot audit
checks, not verification of a fresh reconstruction from the raw archives.

## PV and conditional-sampling implementation

`paper1_pv_benchmark.py` uses municipal polygon representative points at altitude
0 m. The 22 municipalities map to 14 nearest NASA POWER grid centers
(0.5° latitude × 0.625° longitude), each with 8,784 complete hourly UTC T2M/WS10M
records for 2024. Each hourly mean is held over six ten-minute intervals.
Both the horizontal and south-facing 20° cases use 1 kWp DC / 1 kW AC,
`gamma_pdc=-0.0047` per °C, and nominal inverter efficiency 0.96 (inverter
`pdc0=1000/0.96` W). Model settings are Erbs decomposition, Perez transposition,
albedo 0.2, physical incidence-angle losses on direct irradiance, and no spectral
correction. SAPM `open_rack_glass_glass` temperature parameters are `a=-3.47`,
`b=-0.0594`, `deltaT=3` °C; this is not the complete PVWatts V5 Fuentes chain.

Multiplicative loss allowances are soiling 2%, shading 3%, snow 0%, mismatch 2%,
wiring 2%, connections 0.5%, light-induced degradation 1.5%, nameplate 1%,
age 0% and availability 3%: 14.0757% combined before loading-dependent inverter
loss and clipping. Irradiance and output are zeroed at geometric zenith ≥90°;
this removes less than 0.001% of annual PAR energy in each municipality.
First-year production is repeated over the horizon without chronological aging.

`paper1_conditional_uncertainty.py` recovers alternative profiles' annual
avoided-import changes from stored incremental-cost differences using
32.108 NTD/USD, the 3.7556 NTD/kWh tariff and the 20-year present-worth factor at
5%, keeping non-electricity costs fixed. Repricing the 12 deterministic cases
agrees within 1e-8 USD per light. Each row contains four independent PCG64
uniform draws (seed `20260914`), transformed by inverse triangular CDFs.
The same rows serve all cases; P5/P50/P95 use linear quantiles. Nested
2,000/10,000-row prefixes are compared with the primary 50,000 rows; an independent
50,000-row check uses seed `20260915`. The maximum cost/MAC quantile differences
are respectively 4.146 USD per light / 1.165 USD/tCO2e at 2,000 rows,
1.142 / 0.337 at 10,000, and 0.705 / 0.218 for the independent seed.
Parameters, source hashes and full checks are in `pv_model_comparison.json`
and `conditional_uncertainty.json`.

## Result lookup and licensed foreground boundary

Final reference JSONs are staged under
`reference/outputs/final_runs/paper1_canonical_results/`; recalculated JSONs use
the corresponding `outputs/` path. In `paper1_manuscript_values.json`, use:

| JSON field | Contents / checkpoint |
|---|---|
| `values.timestamp_scaling` | 50,892 common timestamps; 8,482 modeled hours; scale 20.6555057769 |
| `values.frontier.knee`, `values.battery_lifetime` | Selected factors 1.85/8.45; mean per-light operational abatement 4.5765305224 tCO2e, net lifecycle abatement 3.5584596948 tCO2e and incremental cost 1,020.5284958 USD |
| `values.regression`, `values.regression.sensitivity` | Full-precision coefficients, correlations, partial R² and variance decompositions for ten comparison sets, including regional omissions |
| `values.selection_sensitivity.population_weights.records` | 22 municipal counts at 31 December 2024 (total 23,400,220) and normalized weights |
| `values.pv_model_comparison.cases.horizontal.cities`, `values.pv_model_comparison.cases.south_tilt20.cities` | Municipal annual yields: `simple_yield_kwh_per_kwp` for calibrated output, `benchmark_yield_kwh_per_kwp` for the alternative |

Spearman calculations round inputs to 12 decimal places and use average tied
ranks; other regression calculations retain full-precision inputs.

The calibrated `src/streetlight/lca/hardware.py` calculation runs without SimaPro.
A process-level rebuild instead requires SimaPro 10.2.0.3, ecoinvent v3.10,
IPCC 2021 GWP100 V1.03, and the author's foreground mappings/calibration.
The private research workspace retains a foreground import CSV, generator and
manifest under `artifacts/lca/simapro_dual_system/`, and process/exchange mapping
tables under `artifacts/lca/simapro_index/`. These foreground materials and the
private `LCA_REBUILD_WITH_LICENSED_DATABASES.md` are absent from public v1.0.0;
they are not paths readers can execute in this checkout. Calibrated coefficients
do not replace them, and no foreground-delivery route or fresh licensed
process-level rerun is established by the public reproduction routes.

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
