"""Rebuild remaining numerical figure tables from analysis outputs.

Numerical transformations extracted from the canonical research renderer.
No manuscript rendering or private-repository imports are required.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from paper1_config import load_paper1_config

PAPER1_CONFIG = load_paper1_config()
PAPER_BASELINE = RESULTS = PAPER1_CONFIG.canonical_results_dir
INPUTS = PAPER1_CONFIG.canonical_inputs_dir
DATA_OUT = PAPER1_CONFIG.figure_data_dir
N_STUDY_LIGHTS = PAPER1_CONFIG.study_light_count
N_LIGHTS_PER_CITY = PAPER1_CONFIG.lights_per_city
KNEE_PV_FACTOR = PAPER1_CONFIG.selected_solar_panel_factor
KNEE_BATTERY_FACTOR = PAPER1_CONFIG.selected_battery_factor
REPORTING_REGIONS = tuple(PAPER1_CONFIG.load_side_regions)
USD_PER_NTD = 1 / PAPER1_CONFIG.ntd_per_usd

def _require_existing_paths(paths, context):
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f'{context}: {path}')

def make_r3_simplification() -> None:
    """SI Figure S2 — cost-boundary sign flip as signed cost distributions."""
    print("R3 cost_boundary_sign_flip (SI Figure S2)")
    base_dir = PAPER_BASELINE / "method_simplifications"
    n_lights = N_STUDY_LIGHTS

    base_path = base_dir / "baseline_results.csv"
    elec_path = base_dir / "electricity_only_cost_results.csv"
    _require_existing_paths([base_path, elec_path], "SI Figure S2 cost-boundary sign flip")

    def per_design(path):
        df = pd.read_csv(path)
        g = df.groupby(["solar_panel_factor", "battery_factor"]).agg(
            abate_t=("abatement_t", "sum"),
            delta_cost_ntd=("delta_cost", "sum"),
        ).reset_index()
        g["cost_per_streetlight_usd"] = g["delta_cost_ntd"] * USD_PER_NTD / n_lights
        g["abatement_t_per_streetlight"] = g["abate_t"] / n_lights
        return g

    base = per_design(base_path)
    elec = per_design(elec_path)
    assert len(base) == len(elec), \
        f"design count mismatch: base={len(base)}, elec={len(elec)}"

    paired = base.merge(
        elec, on=["solar_panel_factor", "battery_factor"],
        suffixes=("_base", "_elec"))
    paired.to_csv(DATA_OUT / "figS2_cost_boundary_source.csv", index=False)


    base_cost = base["cost_per_streetlight_usd"]
    elec_cost = elec["cost_per_streetlight_usd"]
    bin_edges = np.arange(-800.0, 5700.0 + 100.0, 100.0)

    def _share(values: pd.Series) -> np.ndarray:
        counts, _ = np.histogram(values.to_numpy(dtype=float), bins=bin_edges)
        return counts / len(values) * 100.0

    elec_share = _share(elec_cost)
    base_share = _share(base_cost)
    pd.DataFrame({
        "bin_left_usd": bin_edges[:-1],
        "bin_right_usd": bin_edges[1:],
        "electricity_only_share_pct": elec_share,
        "life_cycle_share_pct": base_share,
    }).to_csv(DATA_OUT / "figS2_cost_boundary_bins.csv", index=False)

    def _summary(label: str, values: pd.Series, shares: np.ndarray) -> dict[str, float | int | str]:
        return {
            "boundary": label,
            "n_designs": int(len(values)),
            "negative_count": int((values < 0).sum()),
            "positive_count": int((values > 0).sum()),
            "min_usd_per_streetlight": float(values.min()),
            "max_usd_per_streetlight": float(values.max()),
            "median_usd_per_streetlight": float(values.median()),
            "max_bin_share_pct": float(shares.max()),
        }

    pd.DataFrame([
        _summary("electricity_only", elec_cost, elec_share),
        _summary("life_cycle", base_cost, base_share),
    ]).to_csv(DATA_OUT / "figS2_cost_boundary_summary.csv", index=False)



def _load_bump_data() -> pd.DataFrame:
    """Build the 22-city × 4-rank table at the selected allocation.

    All method-simplification inputs must come from the canonical Paper 1
    result bundle.
    """
    primary = RESULTS / "method_simplifications"
    files = {
        "baseline":  "baseline_results.csv",
        "static":    "static_average_aef_results.csv",
        "uniform":   "uniform_solar_yield_results.csv",
        "combined":  "combined_typical_simplifications_results.csv",
    }

    def _read(name: str) -> pd.DataFrame:
        p = primary / files[name]
        _require_existing_paths([p], f"Figure 5 rank-shift source {name}")
        df = pd.read_csv(p)
        sub = df[(df["solar_panel_factor"] == KNEE_PV_FACTOR)
                 & (df["battery_factor"] == KNEE_BATTERY_FACTOR)].copy()
        sub = sub.sort_values("abatement_t", ascending=False).reset_index(drop=True)
        sub[f"r_{name}"] = sub.index + 1
        return sub[["city", "region", f"r_{name}"]]

    b = _read("baseline")
    s = _read("static")
    u = _read("uniform")
    c = _read("combined")
    df = (b.merge(s[["city", "r_static"]], on="city")
            .merge(u[["city", "r_uniform"]], on="city")
            .merge(c[["city", "r_combined"]], on="city"))

    # Validation per spec §2
    assert len(df) == 22, f"expected 22 cities, got {len(df)}"
    for col in ("r_baseline", "r_static", "r_uniform", "r_combined"):
        assert sorted(df[col].tolist()) == list(range(1, 23)), \
            f"{col} is not a permutation of 1..22"
    assert set(df["region"]).issubset(REPORTING_REGIONS), \
        f"unexpected regions: {set(df['region']) - set(REPORTING_REGIONS)}"
    return df


def make_f6_future_grid_retention() -> None:
    """Figure 6 — attributional operational reduction by grid state.

    Single panel: X = post-flow regional grid AEF (kg CO2e/kWh assigned
    to residual streetlight imports after inter-regional power flow), Y =
    selected-allocation attributional operational emission reduction
    (t CO2e per streetlight, nominal 20 yr). Four
    Taipower main-island load-side regions plus separate county-level
    outlying-island load-serving zones x three grid states.

    The figure compares independently applied static grid states with the
    repeated-2024 reference under a common nominal 20-year horizon and fixed
    physical dispatch. The 21 region-state points share regional inputs and
    dispatch assumptions, and AEF enters the reduction calculation itself.

    Region is encoded by colour (F2 region tints), and grid state is
    encoded by marker shape. No pooled regression or correlation is shown.
    Fuel-mix assumptions and retention values are documented in the
    manuscript and SI.
    """
    print("F6 future_grid_retention")
    n_per_city = N_LIGHTS_PER_CITY

    base_results = PAPER_BASELINE / "pareto" / "results.csv"
    fg_root = PAPER_BASELINE / "future_grid_scenarios"
    _require_existing_paths([base_results, fg_root], "Figure 6 future-grid retention")

    INPUTS_AEF = INPUTS / "aef"
    scen_sources = [
        ("2024 baseline",      2024, base_results, INPUTS_AEF),
        ('2030 MOEA "532"',    2030,
         fg_root / "official_2030_grid_target" / "results.csv",
         fg_root / "official_2030_grid_target" / "aef"),
        ("2050 National Development Council net-zero pathway",  2050,
         fg_root / "pathway_2050_midpoint_proxy" / "results.csv",
         fg_root / "pathway_2050_midpoint_proxy" / "aef"),
    ]
    REGIONS = list(REPORTING_REGIONS)

    def _region_knee_per_pole(results_path: Path) -> dict:
        df = pd.read_csv(results_path)
        k = df[(df["solar_panel_factor"] == KNEE_PV_FACTOR)
               & (df["battery_factor"] == KNEE_BATTERY_FACTOR)]
        g = k.groupby("region").agg(abate=("abatement_t", "sum"),
                                    ncity=("city", "nunique")).reset_index()
        return {r["region"]: r["abate"] / (r["ncity"] * n_per_city)
                for _, r in g.iterrows()}

    def _region_post_flow_aef(aef_dir: Path) -> dict:
        out = {}
        for r in REGIONS:
            path = aef_dir / f"{r}.csv"
            _require_existing_paths([path], f"Figure 6 AEF source {aef_dir.name}:{r}")
            d = pd.read_csv(path, usecols=["FLOW_UNIT_FINAL_AEF"])
            out[r] = float(d["FLOW_UNIT_FINAL_AEF"].mean())
        return out

    rows = []
    for label, year, results_path, aef_dir in scen_sources:
        _require_existing_paths([results_path, aef_dir], f"Figure 6 source {label}")
        knee_by_region = _region_knee_per_pole(results_path)
        aef_by_region = _region_post_flow_aef(aef_dir)
        for r in REGIONS:
            rows.append({"region": r, "year": year, "scen_label": label,
                         "aef": aef_by_region[r],
                         "abate": knee_by_region.get(r, np.nan)})
    pts = pd.DataFrame(rows)
    pts.to_csv(DATA_OUT / "f6_future_grid_retention_points.csv", index=False)

    retention_path = fg_root / "fixed_representative_retention.csv"
    if not retention_path.exists():
        raise FileNotFoundError(retention_path)
    retention = pd.read_csv(retention_path)
    retention_source = retention[
        [
            "scenario",
            "knee_abatement_per_pole_2024_t",
            "uniform_aef_knee_abatement_per_pole_t",
            "generation_by_fuel_knee_abatement_per_pole_t",
            "reselected_knee_abatement_per_pole_t",
            "uniform_retention_pct",
            "generation_by_fuel_retention_pct",
            "reselected_knee_retention_pct",
            "diurnal_asymmetry_gap_pp",
        ]
    ].copy()
    retention_source = retention_source.rename(
        columns={
            "knee_abatement_per_pole_2024_t": "repeated_2024_reference_t_per_streetlight",
            "uniform_aef_knee_abatement_per_pole_t": "uniform_aef_prediction_t_per_streetlight",
            "generation_by_fuel_knee_abatement_per_pole_t": "fixed_allocation_t_per_streetlight",
            "reselected_knee_abatement_per_pole_t": "reselected_knee_t_per_streetlight",
        }
    )
    retention_source.to_csv(DATA_OUT / "f6_future_grid_retention_headline_source.csv", index=False)



def make_f8_monte_carlo_uncertainty() -> None:
    """Figure 8 — Monte Carlo classification robustness for the selected allocation."""
    print("F8 monte_carlo_uncertainty")
    samples_path = PAPER_BASELINE / "probabilistic_uncertainty" / "knee_samples.csv"
    _require_existing_paths([samples_path], "Figure 8 Monte Carlo uncertainty")

    samples = pd.read_csv(samples_path)
    n_lights = N_STUDY_LIGHTS
    delta_usd = samples["delta_cost"].astype(float) / (PAPER1_CONFIG.ntd_per_usd * n_lights)
    macc_usd = samples["macc"].astype(float) / PAPER1_CONFIG.ntd_per_usd
    p_cost_saving = float((delta_usd < 0).mean())

    def _q(values: pd.Series) -> dict[str, float]:
        qs = values.quantile([0.05, 0.50, 0.95])
        return {"p05": float(qs.loc[0.05]), "p50": float(qs.loc[0.50]), "p95": float(qs.loc[0.95])}

    delta_q = _q(delta_usd)
    macc_q = _q(macc_usd)
    summary = pd.DataFrame([
        {"metric": "delta_cost_usd_per_streetlight", **delta_q, "probability_below_zero": p_cost_saving},
        {"metric": "macc_usd_per_tco2e", **macc_q, "probability_below_zero": np.nan},
    ])
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DATA_OUT / "f8_monte_carlo_uncertainty_summary.csv", index=False)
    pd.DataFrame({
        "delta_cost_usd_per_streetlight": delta_usd,
        "macc_usd_per_tco2e": macc_usd,
    }).to_csv(DATA_OUT / "f8_monte_carlo_uncertainty_samples.csv", index=False)



def make_sensitivity():
    # ----- Source 1: one-way sensitivity (paper_baseline/sensitivity) ----------
    ows = pd.read_csv(PAPER_BASELINE / "sensitivity" / "one_way_sensitivity.csv")
    base_macc = float(ows[ows.parameter == "baseline"]["macc"].iloc[0])

    def _macc_pair(name, lo_label, hi_label):
        lo = float(ows[(ows.parameter == name)
                       & (ows.scenario_label == lo_label)]["macc"].iloc[0])
        hi = float(ows[(ows.parameter == name)
                       & (ows.scenario_label == hi_label)]["macc"].iloc[0])
        return lo, hi

    def _pct_macc(name, lo_label, hi_label):
        lo, hi = _macc_pair(name, lo_label, hi_label)
        return ((lo - base_macc) / base_macc * 100.0,
                (hi - base_macc) / base_macc * 100.0)

    # All cost-side rows now sweep at ±20% directly (matched to fuel-EF
    # convention; no rescaling needed — battery_capex_multipliers and
    # pv_capex_multipliers in config/paper_baseline.yaml are both [0.8,1.0,1.2]).
    pv_lo, pv_hi = _pct_macc("pv_capex", "x0.80", "x1.20")
    bcap_lo, bcap_hi = _pct_macc("battery_capex", "x0.80", "x1.20")
    elec_lo, elec_hi = _pct_macc("electricity_price", "x0.80", "x1.20")
    par_lo, par_hi = _pct_macc("par_to_kw_factor", "x0.80", "x1.20")
    disc_lo, disc_hi = _pct_macc("discount_rate", "0.03", "0.08")

    # ----- Battery replacement-year slope from joint MC regression -----------
    mc = pd.read_csv(PAPER_BASELINE / "probabilistic_uncertainty"
                     / "knee_samples.csv")
    X = mc[["electricity_price", "discount_rate", "pv_capex_ntd_per_kw",
            "battery_multiplier", "battery_replacement_year"]].copy()
    X_std = (X - X.mean()) / X.std()
    coefficients = np.linalg.lstsq(
        np.column_stack([np.ones(len(X_std)), X_std]), mc["macc"], rcond=None)[0]
    coefs = dict(zip(X.columns, coefficients[1:]))
    mu_macc = mc["macc"].mean()
    sigma_brep = mc["battery_replacement_year"].std()
    brep_pct = (2.0 / sigma_brep) * coefs["battery_replacement_year"] \
        / mu_macc * 100
    # Low/high cases are year 10 / year 14. Earlier replacement raises MAC;
    # later replacement lowers it.
    brep_lo, brep_hi = -brep_pct, +brep_pct

    # ----- Source 3: degradation_sensitivity scenario summary ------------------
    deg = pd.read_csv(PAPER_BASELINE / "degradation_sensitivity"
                      / "scenario_summary.csv")
    deg_macc = {r["scenario"]: r["knee_delta_cost"] / r["knee_abatement_t"]
                for _, r in deg.iterrows()}
    deg_lo = (deg_macc["low_degradation"] - deg_macc["base_degradation"]) \
        / deg_macc["base_degradation"] * 100
    deg_hi = (deg_macc["high_degradation"] - deg_macc["base_degradation"]) \
        / deg_macc["base_degradation"] * 100

    # ----- Source 4: subjective AEF inputs -----------------------------------
    subj = pd.read_csv(PAPER_BASELINE / "subjective_sensitivity" / "summary.csv")

    # The source analysis perturbs each fuel EF by +/-20%. MAC response
    # here is first-order propagation of its panel-mean AEF response.
    def _fuel_pct(fuel_param: str) -> tuple[float, float]:
        """Return (lo, hi) Δ knee MAC% for a single fuel via -ΔAEF/AEF propagation."""
        sel_lo = subj[(subj["parameter"] == fuel_param)
                      & (subj["value"].astype(str).str.startswith("lo="))]
        sel_hi = subj[(subj["parameter"] == fuel_param)
                      & (subj["value"].astype(str).str.startswith("hi="))]
        lo_aef = float(sel_lo["delta_vs_baseline_pct"].iloc[0]) if not sel_lo.empty else 0.0
        hi_aef = float(sel_hi["delta_vs_baseline_pct"].iloc[0]) if not sel_hi.empty else 0.0
        # ΔMAC ≈ -ΔAEF/AEF (cost invariant; abatement ∝ AEF at fixed knee)
        return (-lo_aef, -hi_aef)

    # Per-fuel sensitivity rows — uniform ±20% perturbation on each fuel's
    # lifecycle EF median (matching the ±20% convention used elsewhere on
    # the tornado). Bar magnitudes are therefore directly comparable across
    # rows. Wider literature ranges (IPCC AR5 lo / hi tertiles, ecoinvent
    # uncertainty bands) are documented in SI Table S10 as a separate
    # reality-check view.
    fuel_specs = [
        ("fuel_ef_coal", "Coal EF"),
        ("fuel_ef_lng", "LNG EF"),
        ("fuel_ef_diesel", "Diesel EF"),
        ("fuel_ef_oil", "Oil EF"),
        ("fuel_ef_solar", "Solar PV EF"),
        ("fuel_ef_co_gen", "Co-Gen EF"),
        ("fuel_ef_wind", "Wind onshore EF"),
        ("fuel_ef_offshore_wind", "Wind offshore EF"),
        ("fuel_ef_hydro", "Hydro EF"),
        ("fuel_ef_geothermal", "Geothermal EF"),
        ("fuel_ef_biomass", "Biomass EF"),
        ("fuel_ef_nuclear", "Nuclear EF"),
    ]

    fuel_rows = []
    for param_key, label in fuel_specs:
        lo, hi = _fuel_pct(param_key)
        fuel_rows.append(
            dict(
                label=label,
                lo=lo,
                hi=hi,
                source_parameter=param_key,
            )
        )

    fuel_by_label = {row["label"]: row for row in fuel_rows}
    minor_fuel_labels = (
        "Solar PV EF",
        "Co-Gen EF",
        "Wind onshore EF",
        "Wind offshore EF",
        "Hydro EF",
        "Geothermal EF",
        "Biomass EF",
        "Nuclear EF",
    )
    minor_fuel_rows = [fuel_by_label[label] for label in minor_fuel_labels]
    other_fuel = max(
        minor_fuel_rows,
        key=lambda r: max(abs(r["lo"]), abs(r["hi"])),
    )
    main_fuel_rows = [
        fuel_by_label["Coal EF"],
        fuel_by_label["LNG EF"],
        fuel_by_label["Diesel EF"],
        fuel_by_label["Oil EF"],
        dict(
            label="Other fuel EFs\n(max individual)",
            lo=other_fuel["lo"],
            hi=other_fuel["hi"],
            source_parameter=other_fuel.get("source_parameter"),
            source_note=f"maximum individual residual fuel row: {other_fuel['label']}",
        ),
    ]

    phs_low_row = subj[
        (subj["parameter"] == "phs_round_trip_efficiency")
        & (subj["value"].astype(str) == "0.75")
    ]
    phs_high_row = subj[
        (subj["parameter"] == "phs_round_trip_efficiency")
        & (subj["value"].astype(str) == "0.82")
    ]
    phs_low_aef = (
        float(phs_low_row["delta_vs_baseline_pct"].iloc[0])
        if not phs_low_row.empty else 0.0
    )
    phs_high_aef = (
        float(phs_high_row["delta_vs_baseline_pct"].iloc[0])
        if not phs_high_row.empty else 0.0
    )
    phs_lo, phs_hi = -phs_low_aef, -phs_high_aef

    def _row_record(
        *,
        panel: str,
        label: str,
        lo: float,
        hi: float,
        low_case: str,
        high_case: str,
        calculation_basis: str,
        source: str,
        baseline_macc_ntd: float | None = None,
        low_macc_ntd: float | None = None,
        high_macc_ntd: float | None = None,
    ) -> dict:
        baseline = base_macc if baseline_macc_ntd is None else baseline_macc_ntd
        low_macc = baseline * (1.0 + lo / 100.0) if low_macc_ntd is None else low_macc_ntd
        high_macc = baseline * (1.0 + hi / 100.0) if high_macc_ntd is None else high_macc_ntd
        return {
            "figure_panel": panel,
            "parameter": label.replace("\n", " "),
            "low_case": low_case,
            "high_case": high_case,
            "baseline_macc_usd_per_t": baseline * USD_PER_NTD,
            "low_macc_usd_per_t": low_macc * USD_PER_NTD,
            "high_macc_usd_per_t": high_macc * USD_PER_NTD,
            "low_delta_pct": lo,
            "high_delta_pct": hi,
            "calculation_basis": calculation_basis,
            "source": source,
        }

    f8_records = [
        _row_record(
            panel="a_uniform_perturbation",
            label="Battery CAPEX",
            lo=bcap_lo,
            hi=bcap_hi,
            low_case="0.8 x battery CAPEX",
            high_case="1.2 x battery CAPEX",
            calculation_basis="analytic cost recomputation at fixed dispatch",
            source="sensitivity/one_way_sensitivity.csv",
            low_macc_ntd=_macc_pair("battery_capex", "x0.80", "x1.20")[0],
            high_macc_ntd=_macc_pair("battery_capex", "x0.80", "x1.20")[1],
        ),
        _row_record(
            panel="a_uniform_perturbation",
            label="PV CAPEX",
            lo=pv_lo,
            hi=pv_hi,
            low_case="0.8 x PV CAPEX",
            high_case="1.2 x PV CAPEX",
            calculation_basis="analytic cost recomputation at fixed dispatch",
            source="sensitivity/one_way_sensitivity.csv",
            low_macc_ntd=_macc_pair("pv_capex", "x0.80", "x1.20")[0],
            high_macc_ntd=_macc_pair("pv_capex", "x0.80", "x1.20")[1],
        ),
        _row_record(
            panel="a_uniform_perturbation",
            label="Electricity price",
            lo=elec_lo,
            hi=elec_hi,
            low_case="0.8 x electricity price",
            high_case="1.2 x electricity price",
            calculation_basis="analytic cost recomputation at fixed dispatch",
            source="sensitivity/one_way_sensitivity.csv",
            low_macc_ntd=_macc_pair("electricity_price", "x0.80", "x1.20")[0],
            high_macc_ntd=_macc_pair("electricity_price", "x0.80", "x1.20")[1],
        ),
        _row_record(
            panel="a_uniform_perturbation",
            label="PAR estimate",
            lo=par_lo,
            hi=par_hi,
            low_case="0.8 x PAR-to-PV coefficient",
            high_case="1.2 x PAR-to-PV coefficient",
            calculation_basis="dispatch rerun at fixed selected allocation",
            source="sensitivity/one_way_sensitivity.csv",
            low_macc_ntd=_macc_pair("par_to_kw_factor", "x0.80", "x1.20")[0],
            high_macc_ntd=_macc_pair("par_to_kw_factor", "x0.80", "x1.20")[1],
        ),
    ]
    for row in main_fuel_rows:
        f8_records.append(
            _row_record(
                panel="a_uniform_perturbation",
                label=row["label"],
                lo=row["lo"],
                hi=row["hi"],
                low_case="-20% fuel EF",
                high_case="+20% fuel EF",
                calculation_basis="full AEF-pipeline rerun with MAC propagated from relative AEF response",
                source=row.get("source_note") or f"subjective_sensitivity/summary.csv:{row.get('source_parameter')}",
            )
        )
    deg_base = deg_macc["base_degradation"]
    f8_records.extend([
        _row_record(
            panel="b_scenario_perturbation",
            label="PV-battery degradation",
            lo=deg_lo,
            hi=deg_hi,
            low_case="low degradation pathway",
            high_case="high degradation pathway",
            calculation_basis="frontier rerun relative to the base degradation pathway",
            source="degradation_sensitivity/scenario_summary.csv",
            baseline_macc_ntd=deg_base,
            low_macc_ntd=deg_macc["low_degradation"],
            high_macc_ntd=deg_macc["high_degradation"],
        ),
        _row_record(
            panel="b_scenario_perturbation",
            label="Discount rate",
            lo=disc_lo,
            hi=disc_hi,
            low_case="3% discount rate",
            high_case="8% discount rate",
            calculation_basis="analytic present-value recomputation at fixed dispatch",
            source="sensitivity/one_way_sensitivity.csv",
            low_macc_ntd=_macc_pair("discount_rate", "0.03", "0.08")[0],
            high_macc_ntd=_macc_pair("discount_rate", "0.03", "0.08")[1],
        ),
        _row_record(
            panel="b_scenario_perturbation",
            label="Battery replacement timing",
            lo=brep_lo,
            hi=brep_hi,
            low_case="year-10 replacement",
            high_case="year-14 replacement",
            calculation_basis="standardised MC regression coefficient converted to year-10/year-14 response",
            source="probabilistic_uncertainty/knee_samples.csv",
        ),
        _row_record(
            panel="b_scenario_perturbation",
            label="PHS round-trip efficiency",
            lo=phs_lo,
            hi=phs_hi,
            low_case="0.75 round-trip efficiency",
            high_case="0.82 round-trip efficiency",
            calculation_basis="full AEF-pipeline rerun with MAC propagated from relative AEF response",
            source="subjective_sensitivity/summary.csv:phs_round_trip_efficiency",
        ),
    ])
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(f8_records).to_csv(DATA_OUT / "f7_sensitivity_tornado_source.csv", index=False)



def make_aef_week():
    source = pd.DataFrame()
    for region in REPORTING_REGIONS:
        frame = pd.read_csv(INPUTS / 'aef' / f'{region}.csv', index_col=0, parse_dates=True)
        source[f'{region}_final_aef_kg_per_kwh'] = frame.loc[
            '2024-03-18':'2024-03-24 15:10', 'FLOW_UNIT_FINAL_AEF'].dropna()
    source.index.name = 'timestamp'
    source.reset_index().to_csv(DATA_OUT / 'figS1_aef_pool_week_source.csv', index=False)

def main():
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    make_r3_simplification()
    _load_bump_data().to_csv(DATA_OUT / 'f5_rank_shift_bump_data.csv', index=False)
    make_f6_future_grid_retention()
    make_f8_monte_carlo_uncertainty()
    make_sensitivity()
    make_aef_week()

if __name__ == '__main__':
    main()
