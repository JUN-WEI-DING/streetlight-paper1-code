#!/usr/bin/env python3
"""Build the structured value contract used to render Paper 1 prose and SI.

The publication figures already write CSV source tables under
``outputs/paper_assets/paper1/figure_data``.  This script collects those
figure sources plus canonical analysis outputs into one JSON file so manuscript
text and supplementary tables can be regenerated instead of hand-edited.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from paper1_config import build_config_render_blocks, build_config_render_tokens, load_paper1_config
from paper1_provenance import normalize_repo_paths
from paper1_aef_quality import build_aef_quality
from paper1_aef_storage_audit import build_aef_storage_audit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = load_paper1_config()
DEFAULT_RESULTS = DEFAULT_CONFIG.canonical_results_dir
DEFAULT_FIGURE_DATA = DEFAULT_CONFIG.figure_data_dir
DEFAULT_OUTPUT = DEFAULT_CONFIG.manuscript_values_path
OUTLYING_ISLAND_REGIONS = {
    "island_penghu": "Penghu",
    "island_kinmen": "Kinmen",
    "island_lienchiang": "Lienchiang",
}
OUTLYING_ISLAND_PROXY_LABELS = {
    "island_penghu": "Penghu Jianshan oil-generation proxy",
    "island_kinmen": "Kinmen Tashan diesel-generation proxy",
    "island_lienchiang": "Matsu Zhushan diesel-generation proxy",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path.relative_to(ROOT))
    return pd.read_csv(path)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


def _is_split_outlying_island_mainline(regions: pd.Series) -> bool:
    return set(OUTLYING_ISLAND_REGIONS).issubset(set(regions.astype(str)))


def _island_other_generation_summary(canonical_inputs_dir: Path) -> dict[str, float]:
    path = canonical_inputs_dir / "power" / "island_other_unit_generation.csv"
    if not path.exists():
        return {"generation_sum_mwh": 0.0, "zero_timestep_count": 0.0}
    df = pd.read_csv(path)
    numeric = df.drop(columns=["timestamp"], errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(0.0)
    output_mw = numeric.sum(axis=1)
    return {
        "generation_sum_mwh": float(output_mw.sum() / 6.0),
        "zero_timestep_count": float((output_mw <= 0.0).sum()),
    }


def _build_mainline_outlying_island_block(
    *,
    f4: pd.DataFrame,
    f5: pd.DataFrame,
    city_label_by_local_name: dict[str, str],
    emission_factors: dict[str, float],
    canonical_inputs_dir: Path,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    rows = f5[f5["region"].astype(str).isin(OUTLYING_ISLAND_REGIONS)].copy()
    if len(rows) != 3:
        raise ValueError("Expected three split outlying-island rows in the reference ranking table")

    aef_by_city = {str(row["city"]): float(row["aef"]) for _, row in f4.iterrows()}
    rows = rows.sort_values("r_baseline")
    table_rows: list[str] = []
    records: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        city = str(row["city"])
        region = str(row["region"])
        city_label = city_label_by_local_name.get(city, city)
        proxy_label = OUTLYING_ISLAND_PROXY_LABELS[region]
        rank = int(row["r_baseline"])
        table_rows.append(
            _md_table_row(
                [
                    city_label,
                    OUTLYING_ISLAND_REGIONS[region],
                    proxy_label,
                    str(rank),
                    str(rank),
                    "+0.00 t (+0.00%)",
                ]
            )
        )
        records.append(
            {
                "city": city,
                "city_label": city_label,
                "region_baseline": region,
                "region_split": region,
                "baseline_abatement_rank": rank,
                "split_abatement_rank": rank,
                "delta_abatement_t": 0.0,
                "delta_abatement_t_pct": 0.0,
                "aef": aef_by_city.get(city),
            }
        )

    island_aefs = [record["aef"] for record in records if record["aef"] is not None]
    island_other = _island_other_generation_summary(canonical_inputs_dir)
    tokens = {
        "island_split.portfolio_delta_t_2": "+0.00",
        "island_split.portfolio_delta_pct_3": "0.000",
        "island_split.island_delta_t_2": "+0.00",
        "island_split.island_delta_pct_2": "0.00",
        "island_split.island_delta_per_streetlight_t_2": "+0.00",
        "island_split.largest_city_change_name": "none",
        "island_split.largest_city_change_t_2": "+0.00",
        "island_split.largest_city_change_pct_2": "+0.00",
        "island_split.split_order": _english_list([record["city_label"] for record in records]),
        "island_split.baseline_aef_min_3": _fmt(min(island_aefs), 3),
        "island_split.baseline_aef_mean_3": _fmt(float(np.mean(island_aefs)), 3),
        "island_split.baseline_aef_max_3": _fmt(max(island_aefs), 3),
        "island_split.oil_proxy_aef_3": _fmt(emission_factors["Oil"], 3),
        "island_split.diesel_proxy_aef_3": _fmt(emission_factors["Diesel"], 3),
        "island_split.other_generation_mwh_1": _fmt(float(island_other["generation_sum_mwh"]), 1),
        "island_split.other_zero_timestamps_int": _fmt_int(float(island_other["zero_timestep_count"])),
    }
    value = {
        "summary": {
            "reference_workflow": "split_outlying_island_load_serving_zones",
            "proxy_aef": {
                region: {"proxy_label": label}
                for region, label in OUTLYING_ISLAND_PROXY_LABELS.items()
            },
            "unit_generation": [
                {"split_region": "unassigned_island_other", **island_other}
            ],
        },
        "city_rows": records,
        "island_city_rows": records,
    }
    return "\n".join(table_rows), tokens, value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path.relative_to(ROOT))
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _row_by(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    rows = df[df[column].astype(str) == value]
    if rows.empty:
        raise KeyError(f"No row in {column!r} matching {value!r}")
    return rows.iloc[0]


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _fmt_comma(value: float, digits: int) -> str:
    return f"{value:,.{digits}f}"


def _fmt_int(value: float) -> str:
    return f"{int(round(value)):,}"


def _fmt_signed(value: float, digits: int = 1) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.{digits}f}"


def _fmt_pm(value: float, digits: int = 1) -> str:
    return f"±{abs(value):.{digits}f}"


def _fmt_sci_latex(value: float, digits: int = 2) -> str:
    if value == 0:
        return f"{0:.{digits}f} \\times 10^{{0}}"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10 ** exponent)
    return f"{mantissa:.{digits}f} \\times 10^{{{exponent}}}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def _md_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _english_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _fmt_signed_pair(low: float, high: float, digits: int = 2) -> str:
    return f"{_fmt_signed(low, digits)} / {_fmt_signed(high, digits)}"


def _fmt_response_pair(low: float, high: float, digits: int = 2, threshold: float = 0.005) -> str:
    if max(abs(low), abs(high)) < threshold:
        return f"<{10 ** -digits:.{digits}f}"
    return _fmt_signed_pair(low, high, digits)


def _pct_delta(value: float, baseline: float) -> float:
    return (float(value) - float(baseline)) / float(baseline) * 100.0


def _fuel_sensitivity_pair(subjective: pd.DataFrame, parameter: str) -> tuple[float, float]:
    rows = subjective[subjective["parameter"] == parameter].copy()
    if rows.empty:
        raise KeyError(parameter)
    values = rows["value"].astype(str)
    low = rows[values.str.startswith("lo=")]
    high = rows[values.str.startswith("hi=")]
    if low.empty or high.empty:
        raise KeyError(f"{parameter} does not have lo=/hi= rows")
    return (
        float(low.iloc[0]["delta_vs_baseline_pct"]),
        float(high.iloc[0]["delta_vs_baseline_pct"]),
    )


def _storage_sensitivity_pair(
    subjective: pd.DataFrame, low_value: float, high_value: float
) -> tuple[float, float]:
    rows = subjective[subjective["parameter"] == "phs_round_trip_efficiency"].copy()
    low = rows[np.isclose(rows["value"].astype(float), low_value)]
    high = rows[np.isclose(rows["value"].astype(float), high_value)]
    if low.empty or high.empty:
        raise KeyError("phs_round_trip_efficiency endpoints")
    return (
        float(low.iloc[0]["delta_vs_baseline_pct"]),
        float(high.iloc[0]["delta_vs_baseline_pct"]),
    )


def _subjective_factor_perturb_pct(subjective: pd.DataFrame, parameter: str, baseline: float) -> float:
    rows = subjective[subjective["parameter"] == parameter]
    if rows.empty:
        raise KeyError(parameter)
    factors = []
    for raw in rows["value"].astype(str):
        if "=" not in raw:
            continue
        factors.append(float(raw.split("=", 1)[1]))
    if not factors:
        raise KeyError(f"{parameter} does not have lo=/hi= factor values")
    return max(abs(value / baseline - 1.0) for value in factors) * 100.0


def _inventory_component_specs(selected_pv_kw: float, selected_battery_kwh: float) -> list[dict[str, str]]:
    pv_module_w = 150.0
    battery_cell_kwh = 1.4352
    pv_modules = selected_pv_kw * 1000.0 / pv_module_w
    battery_cell_units = selected_battery_kwh / battery_cell_kwh
    return [
        {
            "component": "LED street luminaire",
            "baseline": "1 unit",
            "retrofit": "1 unit",
            "mass_or_basis": _fmt(6.85, 2),
            "source": "LED streetlight LCA literature [@tahkamo2015lca; @casamayor2018led] and Taiwan solar-lighting product context [@hengs_solar_lighting]",
        },
        {
            "component": "Galvanised steel pole",
            "baseline": "1 unit",
            "retrofit": "1 unit",
            "mass_or_basis": _fmt(56.0, 2),
            "source": "Model mass anchor cross-checked against Taiwan municipal pole geometry and product-material specifications [@taipei_streetlight_pole_drawing; @chungkung_lighting_pole]",
        },
        {
            "component": "Ready-mix concrete foundation",
            "baseline": "0.64 m³",
            "retrofit": "0.64 m³",
            "mass_or_basis": _fmt_comma(1454.72, 2),
            "source": "Model concrete quantity indexed to Taiwan municipal C08/C09 foundation and base drawings [@taipei_streetlight_foundation_drawing; @taipei_streetlight_base_drawing]",
        },
        {
            "component": "Fabricated rebar",
            "baseline": "1 unit",
            "retrofit": "1 unit",
            "mass_or_basis": _fmt(25.60, 2),
            "source": "Model rebar quantity indexed to the Taiwan municipal C08 foundation drawing [@taipei_streetlight_foundation_drawing]",
        },
        {
            "component": "Low-voltage cable and polyvinyl chloride (PVC) conduit",
            "baseline": "18 m each",
            "retrofit": "18 m each",
            "mass_or_basis": _fmt(2.38554 + 2.34, 2),
            "source": "Model 18 m route-length proxy with Taiwan public-project cable, conduit, and trench-cost context [@kaohsiung_gushan_wanshou_road_proposal]",
        },
        {
            "component": "150 watt-peak (Wp) PV module",
            "baseline": "0",
            "retrofit": f"{_fmt(pv_modules, 2)} modules",
            "mass_or_basis": _fmt(pv_modules * 7.59, 2),
            "source": "150 Wp module rating and Taiwan solar-streetlight product context [@tht_solar_led_streetlight; @xtg_solar_ms4500; @hengs_solar_lighting]",
        },
        {
            "component": "LFP battery cell-unit",
            "baseline": "0",
            "retrofit": f"{_fmt(battery_cell_units, 3)} unit",
            "mass_or_basis": _fmt(8.36, 2),
            "source": "LFP inventory literature [@yudhistira2022liion; @peiseler2024lfpcarbon] and Taiwan product chemistry context [@xtg_solar_ms4500]",
        },
        {
            "component": "Battery management system and pack assembly",
            "baseline": "0",
            "retrofit": f"{_fmt(battery_cell_units, 3)} unit",
            "mass_or_basis": _fmt(0.10, 2),
            "source": "Controller, battery, and assembly context from Taiwan solar-streetlight product specifications [@tht_solar_led_streetlight; @xtg_solar_ms4500; @hengs_solar_lighting]",
        },
        {
            "component": "Battery enclosure",
            "baseline": "0",
            "retrofit": "1 unit",
            "mass_or_basis": "included in assembly exchanges",
            "source": "Battery-box/enclosure and solar-streetlight hardware context from Taiwan product specifications [@tht_solar_led_streetlight; @xtg_solar_ms4500; @hengs_solar_lighting]",
        },
        {
            "component": "Maximum power point tracking charge controller",
            "baseline": "0",
            "retrofit": "1 unit",
            "mass_or_basis": "included in assembly exchanges",
            "source": "Controller type/specification context from Taiwan solar-streetlight product specifications [@tht_solar_led_streetlight; @hengs_solar_lighting]",
        },
        {
            "component": "Grid-only switch box, switchgear, and grounding",
            "baseline": "1 allocated unit",
            "retrofit": "0",
            "mass_or_basis": _fmt(0.925 + 0.158 + 0.0843, 2),
            "source": "Taiwan municipal base/pole drawings and public-project grid-interface cost context [@taipei_streetlight_base_drawing; @taipei_streetlight_pole_drawing; @kaohsiung_gushan_wanshou_road_proposal]",
        },
    ]


def _capacity_from_factor(
    solar_factor: float,
    battery_factor: float,
    contract: dict[str, Any],
    config,
) -> tuple[float, float]:
    n_lights = float(contract["standardized_installation"]["n_lights"])
    pv_capacity_kw_per_factor = float(contract["economic_costs"]["pv_capacity_kw_per_factor"])
    storage_capacity_kwh = config.storage_capacity_kwh_per_battery_factor
    return (
        solar_factor * pv_capacity_kw_per_factor / n_lights,
        battery_factor * storage_capacity_kwh / n_lights,
    )


def _build_regression_tokens(f4: pd.DataFrame) -> tuple[dict[str, Any], dict[str, str]]:
    """Describe model outputs; region sensitivities are not inferential replicates.

    AEF is an input to abatement accounting, so correlations and the algebraic
    R2 partition below are not causal effects. No independent-city p-values or
    confidence intervals are implied by these descriptive summaries.
    """
    columns = ["aef", "mean_par", "abatement_t"]

    def summarize(frame: pd.DataFrame) -> dict[str, Any]:
        work = frame[columns].astype(float)
        standardized = (work - work.mean()) / work.std(ddof=0)
        y = standardized["abatement_t"].to_numpy()

        def fit(predictors: list[str]) -> tuple[list[float], float]:
            x = np.column_stack([np.ones(len(y)), standardized[predictors].to_numpy()])
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            r2 = 1.0 - float(np.sum((y - x @ beta) ** 2)) / float(np.sum((y - y.mean()) ** 2))
            return [float(v) for v in beta[1:]], r2

        aef_beta, aef_r2 = fit(["aef"])
        par_beta, par_r2 = fit(["mean_par"])
        both_beta, both_r2 = fit(["aef", "mean_par"])
        # Preserve scientific ties obscured by floating-point representations
        # (e.g. 0.88 versus 0.8800000000000003). Only ranking uses this rounding;
        # Pearson correlations and regressions retain the original precision.
        ranks = work.round(12).rank(method="average")
        incremental_aef = both_r2 - par_r2
        incremental_par = both_r2 - aef_r2
        result = {
            "n": len(work),
            "aef_only": {"aef_coefficient": aef_beta[0], "r2": aef_r2},
            "par_only": {"par_coefficient": par_beta[0], "r2": par_r2},
            "aef_par": {
                "aef_coefficient": both_beta[0],
                "par_coefficient": both_beta[1],
                "r2": both_r2,
            },
            "pearson": {
                "aef": float(work["abatement_t"].corr(work["aef"])),
                "par": float(work["abatement_t"].corr(work["mean_par"])),
            },
            "spearman": {
                "aef": float(ranks["abatement_t"].corr(ranks["aef"])),
                "par": float(ranks["abatement_t"].corr(ranks["mean_par"])),
            },
            "incremental_r2": {"aef_given_par": incremental_aef, "par_given_aef": incremental_par},
            "partial_r2": {
                "aef_given_par": incremental_aef / (1.0 - par_r2) if 1.0 - par_r2 > 1e-12 else None,
                "par_given_aef": incremental_par / (1.0 - aef_r2) if 1.0 - aef_r2 > 1e-12 else None,
            },
            "r2_partition": {
                "unique_aef": incremental_aef,
                "unique_par": incremental_par,
                "shared": aef_r2 + par_r2 - both_r2,
                "unexplained": 1.0 - both_r2,
            },
        }
        if "region" in frame:
            result["n_regions"] = int(frame["region"].nunique())
        return result

    values = summarize(f4)
    values["method"] = {
        "interpretation": "Descriptive associations of modeled attributional emission reductions; AEF is an accounting input, not an independently assigned exposure.",
        "standardization": "Within each reported sample, center and divide outcome and predictors by their population standard deviations; OLS includes an intercept.",
        "rank_ties": "Round each numeric variable to 12 decimal places before average-tie ranking; retain original precision for Pearson and OLS.",
        "incremental_r2": "R2(full) - R2(other predictor only).",
        "partial_r2": "Incremental R2 / (1 - R2(other predictor only)); undefined when the reduced fit is already perfect.",
        "r2_partition": "Unique AEF + unique PAR + shared = full-model R2; shared = R2(AEF only) + R2(PAR only) - R2(full). Shared may be negative because of suppression; components are not causal contribution percentages.",
        "sensitivity": "Equal-weight region means average all three variables within region; main-island analysis excludes the three named outlying-island regions; leave-one-region-out retains equal municipality weights in the remaining sample. These are composition sensitivities, not independent validation samples or confidence intervals.",
    }
    if "region" in f4:
        region_means = f4.groupby("region", sort=True)[columns].mean().reset_index()
        values["sensitivity"] = {
            "region_means": summarize(region_means),
            "main_island": summarize(f4[~f4["region"].isin(OUTLYING_ISLAND_REGIONS)]),
            "leave_one_region_out": {
                str(region): summarize(f4[f4["region"] != region])
                for region in sorted(f4["region"].unique())
            },
        }
    tokens = {
        "regression.aef_only_coef_2": _fmt(values["aef_only"]["aef_coefficient"], 2),
        "regression.aef_only_r2_3": _fmt(values["aef_only"]["r2"], 3),
        "regression.par_only_coef_2": _fmt(values["par_only"]["par_coefficient"], 2),
        "regression.par_only_r2_3": _fmt(values["par_only"]["r2"], 3),
        "regression.full_aef_coef_2": _fmt(values["aef_par"]["aef_coefficient"], 2),
        "regression.full_par_coef_2": _fmt(values["aef_par"]["par_coefficient"], 2),
        "regression.full_r2_3": _fmt(values["aef_par"]["r2"], 3),
        "regression.spearman_aef_2": _fmt(values["spearman"]["aef"], 2),
        "regression.spearman_par_2": _fmt(values["spearman"]["par"], 2),
    }
    return values, tokens


def _build_lca_values(f3: pd.DataFrame, selected_cost_usd: float) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    hardware = f3[f3["stage"] != "B6 (operational)"].copy()
    operational = f3[f3["stage"] == "B6 (operational)"].copy()
    stage_order = ["A1-A3", "A4", "A5", "B", "C"]
    labels = {
        "A1-A3": "A1-A3 manufacturing",
        "A4": "A4 transport",
        "A5": "A5 installation",
        "B": "B4 year-12 replacement",
        "C": "C end of life",
    }

    def kg(system: str, stage: str) -> float:
        rows = hardware[(hardware["system"] == system) & (hardware["stage"] == stage)]
        if rows.empty:
            return 0.0
        return float(rows["kg_co2e_per_streetlight"].iloc[0])

    def op_t(system: str) -> float:
        rows = operational[operational["system"] == system]
        if rows.empty:
            raise KeyError(system)
        return float(rows["t_co2e_per_streetlight"].iloc[0])

    base_hw_t = float(hardware[hardware["system"] == "grid_only_led"]["t_co2e_per_streetlight"].sum())
    retro_hw_t = float(hardware[hardware["system"] == "pv_battery_led"]["t_co2e_per_streetlight"].sum())
    base_op_t = op_t("grid_only_led")
    retro_op_t = op_t("pv_battery_led")
    base_total_t = base_hw_t + base_op_t
    retro_total_t = retro_hw_t + retro_op_t
    operational_abatement_t = base_op_t - retro_op_t
    hardware_addition_t = retro_hw_t - base_hw_t
    net_lc_t = base_total_t - retro_total_t
    lc_reduction_pct = net_lc_t / base_total_t * 100.0
    macc_lc = selected_cost_usd / net_lc_t

    s3b_rows = [
        _md_table_row(
            [
                labels[stage],
                _fmt_comma(kg("grid_only_led", stage), 2),
                _fmt_comma(kg("pv_battery_led", stage), 2),
            ]
        )
        for stage in stage_order
    ]
    s3b_rows.append(
        _md_table_row(
            [
                "**Total hardware-cycle**",
                f"**{_fmt_comma(base_hw_t * 1000.0, 2)}**",
                f"**{_fmt_comma(retro_hw_t * 1000.0, 2)}**",
            ]
        )
    )

    s3c_rows = [
        _md_table_row(
            [
                "Grid-only LED baseline",
                _fmt(base_hw_t, 3),
                _fmt(base_op_t, 3),
                _fmt(base_total_t, 3),
                _fmt(base_op_t / base_total_t * 100.0, 1),
                _fmt(kg("grid_only_led", "A1-A3") / 1000.0 / base_total_t * 100.0, 1),
            ]
        ),
        _md_table_row(
            [
                "PV-battery LED retrofit",
                _fmt(retro_hw_t, 3),
                _fmt(retro_op_t, 3),
                _fmt(retro_total_t, 3),
                _fmt(retro_op_t / retro_total_t * 100.0, 1),
                _fmt(kg("pv_battery_led", "A1-A3") / 1000.0 / retro_total_t * 100.0, 1),
            ]
        ),
    ]

    values = {
        "hardware_t": {"baseline": base_hw_t, "retrofit": retro_hw_t},
        "operational_t": {"baseline": base_op_t, "retrofit": retro_op_t},
        "total_t": {"baseline": base_total_t, "retrofit": retro_total_t},
        "operational_abatement_t": operational_abatement_t,
        "hardware_addition_t": hardware_addition_t,
        "net_lifecycle_abatement_t": net_lc_t,
        "lifecycle_reduction_pct": lc_reduction_pct,
        "macc_lifecycle_usd_per_t": macc_lc,
    }
    tokens = {
        "lca.net_abatement_t_2": _fmt(net_lc_t, 2),
        "lca.hardware_addition_t_2": _fmt(hardware_addition_t, 2),
        "lca.hardware_addition_t_3": _fmt(hardware_addition_t, 3),
        "lca.reduction_pct_1": _pct(lc_reduction_pct, 1),
        "lca.macc_lc_usd_int": _fmt_int(macc_lc),
        "lca.baseline_total_t_3": _fmt(base_total_t, 3),
        "lca.baseline_operational_t_3": _fmt(base_op_t, 3),
        "lca.retrofit_total_t_3": _fmt(retro_total_t, 3),
        "lca.retrofit_operational_t_3": _fmt(retro_op_t, 3),
        "lca.baseline_hardware_t_3": _fmt(base_hw_t, 3),
        "lca.retrofit_hardware_t_3": _fmt(retro_hw_t, 3),
        "lca.baseline_operational_share_pct_1": _fmt(base_op_t / base_total_t * 100.0, 1),
        "lca.retrofit_operational_share_pct_1": _fmt(retro_op_t / retro_total_t * 100.0, 1),
        "lca.baseline_a1a3_share_pct_1": _fmt(
            kg("grid_only_led", "A1-A3") / 1000.0 / base_total_t * 100.0,
            1,
        ),
        "lca.retrofit_a1a3_share_pct_1": _fmt(
            kg("pv_battery_led", "A1-A3") / 1000.0 / retro_total_t * 100.0,
            1,
        ),
    }
    blocks = {
        "table.s3b_rows": "\n".join(s3b_rows),
        "table.s3c_rows": "\n".join(s3c_rows),
    }
    return values, tokens, blocks


def _battery_replacement_years(lifetime_years: float, horizon_years: float) -> list[float]:
    """Schedule recurrent replacement, excluding purchases at service end."""
    if lifetime_years <= 0 or horizon_years <= 0:
        raise ValueError("Battery lifetime and service horizon must be positive")
    # Match storage.recompute_costs_from_results' endpoint convention.
    count = int(math.floor((horizon_years - 1e-9) / lifetime_years))
    return [lifetime_years * index for index in range(1, count + 1)]


def _build_battery_lifetime_values(
    selected: dict[str, Any], lca: dict[str, Any], config: Any
) -> dict[str, Any]:
    """Isolate recurrent battery replacement at fixed design and dispatch.

    Replace the reference discounted battery purchase stream by the scenario
    stream, and change only the battery-dependent B/C hardware burdens. The
    selected-design cost and operational reduction retain their original full
    precision. No capacity fade, efficiency fade, or frontier search is applied.
    """
    from streetlight.lca.hardware import compute_lca_at_knee

    costs = config.economic_costs
    horizon = config.analysis_years
    discount = config.discount_rate
    baseline_lifetime = float(costs["battery_replacement_year"])
    baseline_years = _battery_replacement_years(baseline_lifetime, horizon)
    capacity = selected["battery_kwh_per_streetlight"]
    power = min(
        capacity * config.storage_power_kw_per_kwh,
        config.battery_power_cap_kw_per_city / config.lights_per_city,
    )
    unit_cost_ntd = (
        capacity * float(costs["battery_replacement_energy_capex_ntd_per_kwh"])
        + power * float(costs["battery_replacement_power_capex_ntd_per_kw"])
    )
    unit_cost_usd = unit_cost_ntd / config.ntd_per_usd

    def replacement_pv(years: list[float]) -> float:
        return sum(unit_cost_usd / (1.0 + discount) ** year for year in years)

    def hardware(count: int) -> dict[str, dict[str, float]]:
        return compute_lca_at_knee(
            selected["solar_panel_factor"], selected["battery_factor"],
            battery_replacements=count,
        )

    baseline_hw = hardware(len(baseline_years))
    reference_increment_t = (
        sum(baseline_hw["SOLAR"].values()) - sum(baseline_hw["TRAD"].values())
    ) / 1000.0
    if not math.isclose(reference_increment_t, lca["hardware_addition_t"], abs_tol=1e-10):
        raise ValueError("Selected-design hardware does not match the reference LCA contract")
    baseline_pv = replacement_pv(baseline_years)
    scenarios = []
    for lifetime in (8, 10, 12, 15):
        years = _battery_replacement_years(lifetime, horizon)
        replacement_cost = replacement_pv(years)
        delta_cost = selected["delta_cost_usd_per_streetlight"] + (replacement_cost - baseline_pv)
        stages = hardware(len(years))
        b_delta_t = (stages["SOLAR"]["B"] - baseline_hw["SOLAR"]["B"]) / 1000.0
        c_delta_t = (stages["SOLAR"]["C"] - baseline_hw["SOLAR"]["C"]) / 1000.0
        hardware_delta_t = b_delta_t + c_delta_t
        incremental_hardware_t = lca["hardware_addition_t"] + hardware_delta_t
        net_t = lca["net_lifecycle_abatement_t"] - hardware_delta_t
        operational_t = selected["operational_abatement_t_per_streetlight"]
        scenarios.append({
            "battery_lifetime_years": lifetime,
            "replacement_years": years,
            "replacement_count": len(years),
            "replacement_present_value_usd_per_streetlight": replacement_cost,
            "delta_cost_usd_per_streetlight": delta_cost,
            "operational_abatement_t_per_streetlight": operational_t,
            "hardware_addition_t_per_streetlight": incremental_hardware_t,
            "replacement_stage_change_t_per_streetlight": b_delta_t,
            "end_of_life_stage_change_t_per_streetlight": c_delta_t,
            "net_lifecycle_abatement_t_per_streetlight": net_t,
            "macc_operational_usd_per_t": delta_cost / operational_t,
            "macc_lifecycle_usd_per_t": delta_cost / net_t,
        })
    return {
        "method": "Fixed selected design and dispatch; recurrent replacement cost and hardware sensitivity",
        "sources": [
            "config/paper_baseline.yaml",
            "src/streetlight/simulation/storage.py:recompute_costs_from_results",
            "src/streetlight/lca/hardware.py:compute_lca_at_knee",
            "docs/paper1/reproducibility/LCA_REBUILD_WITH_LICENSED_DATABASES.md",
            "values.frontier.knee",
            "values.lca",
        ],
        "analysis_years": horizon,
        "reference_lifetime_years": baseline_lifetime,
        "discount_rate": discount,
        "ntd_per_usd": config.ntd_per_usd,
        "battery_kwh_per_streetlight": capacity,
        "battery_kw_per_streetlight": power,
        "replacement_unit_cost_ntd_per_streetlight": unit_cost_ntd,
        "replacement_unit_cost_usd_per_streetlight": unit_cost_usd,
        "assumptions": {
            "operational_dispatch": "Unchanged reference dispatch and nominal 20-year operational emission reduction",
            "battery_performance": "Constant usable capacity and efficiency; no year-by-year degradation simulation",
            "replacement_timing": "Every lifetime interval strictly before service end; no year-20 purchase",
            "replacement_hardware": "Battery cells, BMS and freight per event; one MPPT replacement over the horizon",
            "end_of_life_hardware": "Initial and all replaced battery masses use the existing calibrated stage-C mass proxy",
            "carbon_discounting": "None; fixed GWP coefficients regardless of replacement year",
            "terminal_financial_treatment": "Reference end-of-life charges and salvage fractions retained; no remaining-life credit",
        },
        "battery_end_of_life_charge_ntd_per_installation": float(costs["battery_eol"]),
        "battery_salvage_fraction": float(config.financial_params["battery_salvage_fraction"]),
        "scenarios": scenarios,
    }


def _build_discount_rate_values(
    selected: dict[str, Any], lca: dict[str, Any], config: Any,
    city_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Reprice the fixed selected allocation from its municipal energy totals.

    The existing engine annualizes operating costs and the grid fixed-cost
    aggregate, while retaining year-zero CAPEX, scheduled replacements, and
    terminal charges. No discounting is applied to physical emission totals.
    """
    from streetlight.simulation.storage import recompute_costs_from_results

    n_lights = config.lights_per_city
    if len(city_metrics) != config.study_city_count or city_metrics["city"].nunique() != config.study_city_count:
        raise ValueError("Discount sensitivity requires one row per study municipality")
    for column, expected in (
        ("solar_panel_factor", selected["solar_panel_factor"]),
        ("battery_factor", selected["battery_factor"]),
        ("functional_unit_lights", n_lights),
    ):
        if not np.allclose(city_metrics[column].astype(float), expected, rtol=0, atol=1e-10):
            raise ValueError(f"Discount sensitivity source differs from selected design: {column}")
    frame = city_metrics.copy()
    capacity = selected["battery_kwh_per_streetlight"]
    power = min(capacity * config.storage_power_kw_per_kwh,
                config.battery_power_cap_kw_per_city / n_lights)
    frame["battery_capacity_kwh"] = capacity * n_lights
    frame["battery_power_kw"] = power * n_lights
    costs = config.economic_costs
    horizon = config.analysis_years
    fx = config.ntd_per_usd
    operational_t = selected["operational_abatement_t_per_streetlight"]
    net_t = lca["net_lifecycle_abatement_t"]
    pv_kw = selected["pv_kw_per_streetlight"]
    battery_initial = (capacity * float(costs["battery_capex_ntd_per_kwh"])
                       + power * float(costs["battery_power_capex_ntd_per_kw"]))
    replacement = (capacity * float(costs["battery_replacement_energy_capex_ntd_per_kwh"])
                   + power * float(costs["battery_replacement_power_capex_ntd_per_kw"]))
    energy = {
        "grid_only_kwh_20y_per_streetlight": float(frame["grid_energy_kwh_20y"].mean()) / n_lights,
        "retrofit_grid_kwh_20y_per_streetlight": float(frame["pv_storage_energy_kwh_20y"].mean()) / n_lights,
    }
    # Retain cash-flow primitives so the timing and zero-rate result are auditable.
    cash_flows = {
        "pv_capex_year_0": pv_kw * float(costs["pv_capex_ntd_per_kw"]),
        "battery_energy_capex_year_0": capacity * float(costs["battery_capex_ntd_per_kwh"]),
        "battery_power_capex_year_0": power * float(costs["battery_power_capex_ntd_per_kw"]),
        "other_capex_year_0": float(costs["other_capex"]) / n_lights,
        "pv_om_annual": pv_kw * float(costs["pv_om_ntd_per_kw_year"]),
        "battery_om_annual": battery_initial * float(costs["battery_om_fraction_of_capex_per_year"]),
        "other_om_annual": float(costs["other_om"]) / n_lights / horizon,
        "grid_fixed_cost_annual": float(costs["grid_fixed_cost"]) / n_lights / horizon,
        "grid_only_electricity_annual": energy["grid_only_kwh_20y_per_streetlight"] / horizon * config.electricity_price_ntd_per_kwh,
        "retrofit_electricity_annual": energy["retrofit_grid_kwh_20y_per_streetlight"] / horizon * config.electricity_price_ntd_per_kwh,
        "battery_replacement_per_event": replacement,
        "pv_end_of_life_year_20": pv_kw * float(costs["pv_eol_ntd_per_kw"]),
        "battery_end_of_life_year_20": float(costs["battery_eol"]) / n_lights,
        "other_end_of_life_year_20": float(costs["other_eol"]) / n_lights,
        "salvage_credit_year_20": (
            pv_kw * float(costs["pv_capex_ntd_per_kw"]) * config.financial_params["pv_salvage_fraction"]
            + battery_initial * config.financial_params["battery_salvage_fraction"]
            + float(costs["other_capex"]) / n_lights * config.financial_params["other_salvage_fraction"]
        ),
    }
    scenarios = []
    for rate in (0.0, 0.03, 0.05, 0.07, 0.10):
        repriced = recompute_costs_from_results(
            frame, years=horizon, elec_price=config.electricity_price_ntd_per_kwh,
            economic_costs=costs,
            financial_params={**config.financial_params, "discount_rate": rate},
        )
        mean_cost = repriced["delta_cost"].mean() / n_lights / fx
        if rate == config.discount_rate:
            if not math.isclose(mean_cost, selected["delta_cost_usd_per_streetlight"], rel_tol=0, abs_tol=1e-8):
                raise ValueError("Discount sensitivity does not reproduce the reference cost")
            # Preserve the exact serialized reference after checking agreement.
            mean_cost = selected["delta_cost_usd_per_streetlight"]
        components = {column: float(repriced[column].mean()) / n_lights / fx for column in (
            "grid_only_energy_cost", "grid_only_fixed_cost", "grid_only_cost",
            "pv_storage_energy_cost", "pv_storage_capex_cost", "pv_storage_om_cost",
            "pv_storage_eol_cost", "pv_storage_cost",
        )}
        scenarios.append({
            "discount_rate": rate,
            "delta_cost_usd_per_streetlight": mean_cost,
            "operational_abatement_t_per_streetlight": operational_t,
            "net_lifecycle_abatement_t_per_streetlight": net_t,
            "macc_operational_usd_per_t": mean_cost / operational_t,
            "macc_lifecycle_usd_per_t": mean_cost / net_t,
            "present_value_components_usd_per_streetlight": components,
        })
    return {
        "method": "Fixed selected design and municipal dispatch; cost-stream present-value recomputation",
        "sources": [
            "pareto/selected_allocation_city_metrics.csv (relative to canonical results directory)",
            "config/paper_baseline.yaml",
            "src/streetlight/simulation/storage.py:recompute_costs_from_results",
            "values.frontier.knee", "values.lca",
        ],
        "analysis_years": horizon,
        "reference_discount_rate": config.discount_rate,
        "ntd_per_usd": fx,
        "municipality_count": len(frame),
        "lights_per_municipality": n_lights,
        "energy": energy,
        "cash_flows_ntd_per_streetlight": cash_flows,
        "battery_replacement_years": _battery_replacement_years(float(costs["battery_replacement_year"]), horizon),
        "assumptions": {
            "operations": "Selected allocation, municipal energy totals, and operational and hardware emissions unchanged",
            "annual_streams": "Uniform end-of-year electricity and O&M flows in years 1–20; no escalation",
            "grid_fixed_cost": "Existing aggregate divided by 20 and treated as a uniform annual baseline expense",
            "terminal_costs": "Reference end-of-life charges and salvage fractions retained at year 20",
            "carbon_discounting": "None; both MAC denominators retain undiscounted physical emission reductions",
            "aggregation": "Mean incremental cost divided by mean emission reduction; no mean of municipal MACs",
        },
        "scenarios": scenarios,
    }


def _aggregate_selection_panel(
    panel: pd.DataFrame, weights: pd.Series, *, n_lights: float, ntd_per_usd: float,
) -> pd.DataFrame:
    """Weight complete municipal installations, then express a per-light mean."""
    keys = ["solar_panel_factor", "battery_factor"]
    numeric = keys + ["abatement_t", "delta_cost", "functional_unit_lights"]
    if panel.empty or panel["city"].isna().any() or not np.isfinite(panel[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Selection panel must contain finite numeric values and named cities")
    if not (math.isfinite(n_lights) and n_lights > 0 and math.isfinite(ntd_per_usd) and ntd_per_usd > 0):
        raise ValueError("Functional unit and exchange rate must be positive and finite")
    if not np.allclose(panel["functional_unit_lights"], n_lights, rtol=0, atol=1e-10):
        raise ValueError("Selection panel functional unit differs from the configured installation")
    cities = set(panel["city"])
    if weights.index.has_duplicates or set(weights.index) != cities:
        raise ValueError("Selection weights must match each panel municipality exactly once")
    weight_values = weights.to_numpy(dtype=float)
    total = float(weight_values.sum())
    if not np.isfinite(weight_values).all() or (weight_values < 0).any() or not math.isfinite(total) or total <= 0:
        raise ValueError("Selection weights must be finite, nonnegative, and have positive total")
    if panel.duplicated(keys + ["city"]).any() or not panel.groupby(keys).size().eq(len(cities)).all():
        raise ValueError("Selection requires a complete city-by-design panel without duplicates")
    frame = panel[keys].copy()
    row_weights = panel["city"].map(weights.astype(float) / total)
    frame["operational_abatement_t_per_streetlight"] = panel["abatement_t"] * row_weights / n_lights
    frame["delta_cost_usd_per_streetlight"] = panel["delta_cost"] * row_weights / n_lights / ntd_per_usd
    return frame.groupby(keys, as_index=False, sort=True).sum()


def _selection_frontier(points: pd.DataFrame, objective: str) -> tuple[pd.DataFrame, int]:
    """Apply the existing dominance and independently normalized chord rules."""
    from streetlight.simulation.storage import find_knee_point, pareto_frontier_min_cost_max_abatement

    frame = points.rename(columns={objective: "abatement_t", "delta_cost_usd_per_streetlight": "delta_cost"})
    frontier = pareto_frontier_min_cost_max_abatement(frame)
    index, _, _ = find_knee_point(frontier)
    if index is None:
        raise ValueError("Selection frontier has fewer than three points; no geometric knee is defined")
    return frontier, index


def _build_selection_sensitivity_values(
    selected: dict[str, Any], lca: dict[str, Any], config: Any,
    panel: pd.DataFrame, population_source: dict[str, Any],
) -> dict[str, Any]:
    """Reselect all allocations in a 2x2 weighting/objective sensitivity.

    Municipal dispatch and costs are read from the existing complete result
    panel. Hardware is the existing capacity-scaled calibrated proxy, evaluated
    for every design, with the same replacement count as the reference.
    """
    from streetlight.lca import hardware

    keys = ["solar_panel_factor", "battery_factor"]
    cities = {city for group in config.region_city_map.values() for city in group}
    if set(panel["city"]) != cities or len(cities) != config.study_city_count:
        raise ValueError("Selection panel does not match the configured study municipalities")
    expected_designs = {(round(pv, 8), round(batt, 8)) for pv in config.pareto_solar_factors for batt in config.pareto_battery_factors}
    observed_designs = {tuple(round(float(value), 8) for value in row) for row in panel[keys].drop_duplicates().itertuples(index=False, name=None)}
    if observed_designs != expected_designs:
        raise ValueError("Selection panel does not contain the full configured allocation grid")
    if not np.isfinite(panel["analysis_years"]).all() or not np.allclose(panel["analysis_years"], config.analysis_years, rtol=0, atol=1e-10):
        raise ValueError("Selection panel horizon differs from the configured analysis")
    # The existing hardware proxy is calibrated for these physical factor units.
    if (config.lights_per_city != 64 or config.pv_capacity_kw_per_solar_factor != 18
            or config.storage_capacity_kwh_per_battery_factor != 10):
        raise ValueError("Selection factor units differ from the hardware calibration (64 lights, 18 kW, 10 kWh)")
    population = pd.DataFrame(population_source["records"]).set_index("city")["population"]
    equal_weights = pd.Series(1.0, index=sorted(cities))
    years = _battery_replacement_years(float(config.economic_costs["battery_replacement_year"]), config.analysis_years)
    reference_factors = (selected["solar_panel_factor"], selected["battery_factor"])
    objectives = {
        "operational": "operational_abatement_t_per_streetlight",
        "net_lifecycle": "net_lifecycle_abatement_t_per_streetlight",
    }
    scenarios = []
    for weighting, weights in (("equal_municipality", equal_weights), ("population", population)):
        points = _aggregate_selection_panel(panel, weights, n_lights=config.lights_per_city, ntd_per_usd=config.ntd_per_usd)
        hardware_t = []
        for pv, batt in points[keys].itertuples(index=False, name=None):
            stages = hardware.compute_lca_at_knee(pv, batt, battery_replacements=len(years))
            hardware_t.append((sum(stages["SOLAR"].values()) - sum(stages["TRAD"].values())) / 1000.0)
        points["hardware_addition_t_per_streetlight"] = hardware_t
        points["net_lifecycle_abatement_t_per_streetlight"] = points[objectives["operational"]] - points["hardware_addition_t_per_streetlight"]
        points["pv_kw_per_streetlight"] = points[keys[0]] * config.pv_capacity_kw_per_solar_factor / config.lights_per_city
        points["battery_kwh_per_streetlight"] = points[keys[1]] * config.storage_capacity_kwh_per_battery_factor / config.lights_per_city
        points = points.set_index(keys, drop=False)

        def design_record(factors: tuple[float, float]) -> dict[str, Any]:
            row = {name: float(value) for name, value in points.loc[factors].items()}
            cost = row["delta_cost_usd_per_streetlight"]
            for objective, column in objectives.items():
                row[f"macc_{objective}_usd_per_t"] = cost / row[column] if row[column] > 0 else None
            return row

        reference = design_record(reference_factors)
        if weighting == "equal_municipality":
            for column, expected in (
                ("delta_cost_usd_per_streetlight", selected["delta_cost_usd_per_streetlight"]),
                (objectives["operational"], selected[objectives["operational"]]),
                ("hardware_addition_t_per_streetlight", lca["hardware_addition_t"]),
                (objectives["net_lifecycle"], lca["net_lifecycle_abatement_t"]),
            ):
                if not math.isclose(reference[column], expected, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"Selection reference differs from the existing value contract: {column}")
        for objective, column in objectives.items():
            frontier, index = _selection_frontier(points.reset_index(drop=True), column)
            knee = frontier.iloc[index]
            knee_factors = tuple(float(knee[key]) for key in keys)
            if weighting == "equal_municipality" and objective == "operational" and knee_factors != reference_factors:
                raise ValueError("Equal-municipality operational selection does not reproduce the reference knee")
            reference_on_frontier = bool(((frontier[keys] - reference_factors).abs() < 1e-10).all(axis=1).any())
            endpoint = frontier.sort_values(["abatement_t", "delta_cost"], ascending=[False, True]).iloc[0]
            scenarios.append({
                "weighting": weighting,
                "objective": objective,
                "candidate_count": len(points),
                "frontier_count": len(frontier),
                "normalization": {
                    "abatement_min_t_per_streetlight": float(frontier["abatement_t"].min()),
                    "abatement_max_t_per_streetlight": float(frontier["abatement_t"].max()),
                    "cost_min_usd_per_streetlight": float(frontier["delta_cost"].min()),
                    "cost_max_usd_per_streetlight": float(frontier["delta_cost"].max()),
                },
                "reference_on_frontier": reference_on_frontier,
                "reference_design": reference,
                "knee": design_record(knee_factors),
                "maximum_objective_endpoint": design_record(tuple(float(endpoint[key]) for key in keys)),
            })
    population_records = [{"city": city, "population": int(population.loc[city]),
                           "normalized_weight": float(population.loc[city] / population.sum())}
                          for city in sorted(cities)]
    return {
        "method": "Independent Pareto frontier and min-max normalized maximum-distance-to-endpoint-chord knee for each weighting and abatement objective",
        "sources": [
            "pareto/results.csv (under source_results_dir)",
            "data/geo/municipal_population_2024.json",
            "src/streetlight/simulation/storage.py:pareto_frontier_min_cost_max_abatement,find_knee_point",
            "src/streetlight/lca/hardware.py:compute_lca_at_knee",
        ],
        "population_weights": {**population_source, "records": population_records},
        "city_count": len(cities),
        "candidate_count": len(expected_designs),
        "panel_row_count": len(panel),
        "functional_unit_lights_per_municipality": config.lights_per_city,
        "analysis_years": config.analysis_years,
        "discount_rate": config.discount_rate,
        "ntd_per_usd": config.ntd_per_usd,
        "battery_replacement_years": years,
        "battery_replacement_count": len(years),
        "assumptions": {
            "weighting": "Normalize municipal population to sum one, average each standardized installation outcome, then divide by lights per installation; equal-municipality weights are 1/22",
            "population_scope": "Registered population is an allocation sensitivity proxy, not observed streetlight counts or a national retrofit total",
            "dispatch_and_cost": "Reuse all municipal design outcomes; unchanged dispatch, horizon, economic inputs, replacement timing and discount rate; no new dispatch simulation",
            "netting": "Net life-cycle abatement = operational abatement minus SOLAR-TRAD hardware GWP100; physical emissions are undiscounted",
            "hardware": "Existing SimaPro IPCC 2021 GWP100 V1.03 anchor-calibrated capacity proxy; anchor factors 2.0 PV/8.5 battery; coefficients scaled to each design, shared hardware retained, one MPPT replacement and calibrated stage-C mass treatment",
            "hardware_validation": "Sensitivity within the existing calibration; no independent new SimaPro process-level validation of reselected designs",
            "selection": "All allocations are reconsidered; each of four frontiers has its own min-max normalization and endpoints; the geometric knee is a compromise, not a maximum-net or minimum-MAC optimum",
            "macc": "Average incremental cost divided by the named abatement total; not the slope between adjacent frontier designs",
        },
        "scenarios": scenarios,
    }


def build_values(
    results_dir: Path,
    figure_data_dir: Path,
    config_path: Path | None = None,
    *,
    include_extensions: bool = True,
) -> dict[str, Any]:
    config = load_paper1_config(config_path)
    contract = normalize_repo_paths(_read_json(results_dir / "paper_contract.json"), repo_root=ROOT)
    f2 = _read_csv(figure_data_dir / "f2_pareto_frontier_source.csv")
    f3 = _read_csv(figure_data_dir / "f3_lca_balance_components.csv")
    f4 = _read_csv(figure_data_dir / "f4_input_output_maps_source.csv")
    f5 = _read_csv(figure_data_dir / "f5_rank_shift_bump_data.csv")
    f7 = _read_csv(figure_data_dir / "f7_sensitivity_tornado_source.csv")
    f8 = _read_csv(figure_data_dir / "f8_monte_carlo_uncertainty_summary.csv")
    future = _read_csv(results_dir / "future_grid_scenarios" / "fixed_representative_retention.csv")
    future_metadata = _read_json(results_dir / "future_grid_scenarios" / "scenario_metadata.json")
    payback = _read_csv(results_dir / "carbon_payback" / "representative_designs.csv")
    degradation = _read_csv(results_dir / "degradation_sensitivity" / "scenario_summary.csv")
    calibration = _read_json(results_dir / "calibration" / "summary.json")
    per_city_calibration = _read_csv(results_dir / "calibration" / "per_city_calibration.csv")
    data_quality = _read_csv(config.canonical_inputs_dir / "manifest" / "data_quality_summary.csv")
    pareto_summary = _read_csv(results_dir / "method_simplifications" / "scenario_summary.csv")
    pareto_sample = pd.read_csv(results_dir / "pareto" / "results.csv", nrows=1).iloc[0]
    cost_boundary = _read_csv(figure_data_dir / "figS2_cost_boundary_summary.csv")
    subjective = _read_csv(results_dir / "subjective_sensitivity" / "summary.csv")
    uncertainty_metadata = _read_json(results_dir / "probabilistic_uncertainty" / "metadata.json")
    uncertainty_samples = _read_csv(figure_data_dir / "f8_monte_carlo_uncertainty_samples.csv")
    cogen_biomass = _read_csv(ROOT / "outputs" / "qa" / "cogen_biomass_sensitivity.csv")
    global_config = _read_yaml(ROOT / "config" / "config.yaml")

    observed_city_count = int(f4["city"].nunique())
    if observed_city_count != config.study_city_count:
        raise ValueError(
            f"Config city_count={config.study_city_count} but figure data has "
            f"{observed_city_count} cities"
        )
    tokens: dict[str, str] = {
        **build_config_render_tokens(config),
        # Compatibility aliases for older templates and external checks.
        "study.city_count": str(config.study_city_count),
        "study.exchange_rate_ntd_per_usd_3": _fmt(config.ntd_per_usd, 3),
    }
    blocks: dict[str, str] = {}
    blocks.update(build_config_render_blocks(config))
    values: dict[str, Any] = {"config": config.public_dict(), "contract": contract}

    # Configured fuel factors and source rows used by the AEF reconstruction.
    emission_factors = {str(k): float(v) for k, v in dict(global_config["emission_factors"]).items()}
    s1_specs = [
        (
            "Coal (utility and independent power producer)",
            "Coal",
            "IPCC AR5 Annex III, pulverized-coal median [@ipcc_ar5_wg3_annexiii]",
        ),
        (
            "Natural gas combined cycle (LNG and independent-power-producer LNG)",
            "LNG",
            "IPCC AR5 Annex III, gas combined-cycle median [@ipcc_ar5_wg3_annexiii]",
        ),
        ("Nuclear", "Nuclear", "IPCC AR5 Annex III [@ipcc_ar5_wg3_annexiii]"),
        ("Hydropower", "Hydro", "IPCC AR5 Annex III [@ipcc_ar5_wg3_annexiii]"),
        ("Wind onshore", "Wind", "IPCC AR5 Annex III [@ipcc_ar5_wg3_annexiii]"),
        ("Wind offshore", "Offshore Wind", "IPCC AR5 Annex III [@ipcc_ar5_wg3_annexiii]"),
        ("Solar PV (utility)", "Solar", "IPCC AR5 Annex III [@ipcc_ar5_wg3_annexiii]"),
        ("Geothermal", "Geothermal", "IPCC AR5 Annex III [@ipcc_ar5_wg3_annexiii]"),
        ("Biomass (dedicated)", "Biomass", "IPCC AR5 Annex III [@ipcc_ar5_wg3_annexiii]"),
        ("Oil-fired generation", "Oil", "ecoinvent v3.10 via SimaPro 10.2.0.3 [@wernet2016ecoinvent]"),
        ("Diesel-fired generation", "Diesel", "ecoinvent v3.10 via SimaPro 10.2.0.3 [@wernet2016ecoinvent]"),
        (
            "Gas-fired cogeneration",
            "Co-Gen",
            "ecoinvent v3.10; energy allocation for combined heat and power (CHP) [@wernet2016ecoinvent]",
        ),
    ]
    fuel_factor_rows = []
    fuel_factor_values: dict[str, Any] = {}
    for label, key, source in s1_specs:
        factor = emission_factors[key]
        fuel_factor_rows.append(_md_table_row([label, _fmt(factor, 3), source]))
        fuel_factor_values[key] = {"label": label, "emission_factor_kgco2e_per_kwh": factor, "source": source}
    blocks["table.s1_rows"] = "\n".join(fuel_factor_rows)
    values["fuel_emission_factors"] = fuel_factor_values

    # Data-quality and timestamp-scaling values used by Methods/SI prose.
    def quality_row(suffix: str) -> pd.Series:
        rows = data_quality[data_quality["path"].astype(str).str.endswith(suffix)]
        if rows.empty:
            raise KeyError(suffix)
        return rows.iloc[0]

    par_quality = quality_row("/par/par_wide.csv")
    generation_quality = quality_row("/power/category_generation.csv")
    flow_quality = quality_row("/power/flow.csv")
    dt_hours = float(pareto_sample["dt_hours"])
    modeled_hours = float(pareto_sample["modeled_hours"])
    valid_timestamps = int(round(modeled_hours / dt_hours))
    expected_timestamps = int(par_quality["expected_timestamps"])
    coverage_pct = valid_timestamps / expected_timestamps * 100.0
    baseline_summary = _row_by(pareto_summary, "scenario", "baseline")
    reference_year = int(calibration["reference_year"])
    tokens.update(
        {
            "data.reference_year": str(reference_year),
            "data.timestep_minutes_int": _fmt_int(dt_hours * 60.0),
            "data.timestep_hours_fraction": "1/6",
            "data.expected_timestamps_int": _fmt_int(expected_timestamps),
            "data.short_gap_limit_records_int": "6",
            "data.par_observed_timestamps_int": _fmt_int(float(par_quality["observed_timestamps"])),
            "data.generation_observed_timestamps_int": _fmt_int(
                float(generation_quality["observed_timestamps"])
            ),
            "data.flow_missing_timestamps_int": _fmt_int(float(flow_quality["missing_timestamps"])),
            "data.valid_timestamps_int": _fmt_int(valid_timestamps),
            "data.modeled_hours_1": _fmt_comma(modeled_hours, 1),
            "data.coverage_pct_2": _fmt(coverage_pct, 2),
            "data.scaling_factor_4": _fmt(float(pareto_sample["analysis_scaling_factor"]), 4),
            "frontier.point_count_int": _fmt_int(float(baseline_summary["frontier_points"])),
        }
    )
    values["timestamp_scaling"] = {
        "expected_timestamps": expected_timestamps,
        "par_observed_timestamps": int(par_quality["observed_timestamps"]),
        "generation_observed_timestamps": int(generation_quality["observed_timestamps"]),
        "flow_missing_timestamps": int(flow_quality["missing_timestamps"]),
        "valid_timestamps": valid_timestamps,
        "modeled_hours": modeled_hours,
        "coverage_pct": coverage_pct,
        "analysis_scaling_factor": float(pareto_sample["analysis_scaling_factor"]),
        "frontier_points": int(baseline_summary["frontier_points"]),
    }

    # Frontier and selected-allocation values.
    rep = {str(row["representative_label"]): row for _, row in f2.dropna(subset=["representative_label"]).iterrows()}
    frontier_rows = []
    frontier_values: dict[str, Any] = {}
    label_map = {
        "min_cost": "Minimum-cost frontier point",
        "knee": "**Selected Pareto-knee design**",
        "max_abatement": "Maximum-abatement frontier point",
    }
    for key in ["min_cost", "knee", "max_abatement"]:
        row = rep[key]
        pv_kw, battery_kwh = _capacity_from_factor(
            float(row["solar_panel_factor"]),
            float(row["battery_factor"]),
            contract,
            config,
        )
        abatement = float(row["abatement_t_per_streetlight"])
        cost = float(row["delta_cost_usd_per_streetlight"])
        macc = cost / abatement
        frontier_values[key] = {
            "solar_panel_factor": float(row["solar_panel_factor"]),
            "battery_factor": float(row["battery_factor"]),
            "pv_kw_per_streetlight": pv_kw,
            "battery_kwh_per_streetlight": battery_kwh,
            "operational_abatement_t_per_streetlight": abatement,
            "delta_cost_usd_per_streetlight": cost,
            "macc_usd_per_tco2e": macc,
        }
        cells = [
            label_map[key],
            _fmt(pv_kw, 2),
            _fmt(battery_kwh, 2),
            _fmt(abatement, 2),
            _fmt_int(cost),
            _fmt_int(macc),
        ]
        if key == "knee":
            cells = [cells[0], *[f"**{cell}**" for cell in cells[1:]]]
        frontier_rows.append(_md_table_row(cells))
    blocks["table.s2_rows"] = "\n".join(frontier_rows)
    selected = frontier_values["knee"]
    max_abatement = frontier_values["max_abatement"]
    min_cost = frontier_values["min_cost"]
    tokens.update(
        {
            "selected.pv_kw_2": _fmt(selected["pv_kw_per_streetlight"], 2),
            "selected.battery_kwh_2": _fmt(selected["battery_kwh_per_streetlight"], 2),
            "selected.operational_abatement_t_2": _fmt(
                selected["operational_abatement_t_per_streetlight"], 2
            ),
            "selected.delta_cost_usd_int": _fmt_int(selected["delta_cost_usd_per_streetlight"]),
            "selected.macc_usd_int": _fmt_int(selected["macc_usd_per_tco2e"]),
            "frontier.selected_capture_pct_1": _pct(
                selected["operational_abatement_t_per_streetlight"]
                / max_abatement["operational_abatement_t_per_streetlight"]
                * 100.0,
                1,
            ),
            "frontier.selected_cost_pct_1": _pct(
                selected["delta_cost_usd_per_streetlight"]
                / max_abatement["delta_cost_usd_per_streetlight"]
                * 100.0,
                1,
            ),
            "frontier.min_cost_abatement_t_2": _fmt(
                min_cost["operational_abatement_t_per_streetlight"], 2
            ),
            "frontier.min_cost_pct_of_max_1": _pct(
                min_cost["operational_abatement_t_per_streetlight"]
                / max_abatement["operational_abatement_t_per_streetlight"]
                * 100.0,
                1,
            ),
            "frontier.max_cost_usd_int": _fmt_int(max_abatement["delta_cost_usd_per_streetlight"]),
            "frontier.max_extra_abatement_pct_int": _fmt_int(
                (
                    max_abatement["operational_abatement_t_per_streetlight"]
                    / selected["operational_abatement_t_per_streetlight"]
                    - 1.0
                )
                * 100.0
            ),
            "frontier.max_cost_factor_1": _fmt(
                max_abatement["delta_cost_usd_per_streetlight"]
                / selected["delta_cost_usd_per_streetlight"],
                1,
            ),
        }
    )
    values["frontier"] = frontier_values

    foreground_specs = _inventory_component_specs(
        selected["pv_kw_per_streetlight"],
        selected["battery_kwh_per_streetlight"],
    )
    blocks["table.s3a_rows"] = "\n".join(
        _md_table_row(
            [
                row["component"],
                row["baseline"],
                row["retrofit"],
                row["mass_or_basis"],
                row["source"],
            ]
        )
        for row in foreground_specs
    )
    battery_entry = next(row for row in foreground_specs if row["component"] == "LFP battery cell-unit")
    tokens["foreground.lfp_cell_entry_mass_kg_2"] = battery_entry["mass_or_basis"]
    values["foreground_inventory"] = foreground_specs

    frontier_all = f2[f2["is_frontier"].astype(bool)].copy()
    frontier_all["macc_usd_per_tco2e"] = (
        frontier_all["delta_cost_usd_per_streetlight"].astype(float)
        / frontier_all["abatement_t_per_streetlight"].astype(float)
    )
    min_macc_row = frontier_all.loc[frontier_all["macc_usd_per_tco2e"].idxmin()]
    budget_ceiling_usd = 1000.0
    budget_candidates = frontier_all[
        frontier_all["delta_cost_usd_per_streetlight"].astype(float) <= budget_ceiling_usd
    ].sort_values(["abatement_t_per_streetlight", "delta_cost_usd_per_streetlight"], ascending=[False, True])
    if budget_candidates.empty:
        raise ValueError("No budget-constrained frontier point found")
    budget_row = budget_candidates.iloc[0]

    def decision_row(label: str, row: pd.Series, *, bold: bool = False) -> tuple[str, dict[str, float]]:
        pv_kw, battery_kwh = _capacity_from_factor(
            float(row["solar_panel_factor"]),
            float(row["battery_factor"]),
            contract,
            config,
        )
        abatement = float(row["abatement_t_per_streetlight"])
        cost = float(row["delta_cost_usd_per_streetlight"])
        macc = cost / abatement
        cells = [
            label,
            _fmt(pv_kw, 2),
            _fmt(battery_kwh, 2),
            _fmt(abatement, 2),
            _fmt_int(cost),
            _fmt_int(macc),
        ]
        if bold:
            cells = [cells[0], *[f"**{cell}**" for cell in cells[1:]]]
        return _md_table_row(cells), {
            "pv_kw_per_streetlight": pv_kw,
            "battery_kwh_per_streetlight": battery_kwh,
            "operational_abatement_t_per_streetlight": abatement,
            "delta_cost_usd_per_streetlight": cost,
            "macc_usd_per_tco2e": macc,
        }

    s5_rows: list[str] = []
    decision_values: dict[str, Any] = {}
    for key, label, row, bold in [
        ("minimum_macc", "Minimum-MAC rule", min_macc_row, False),
        ("maximum_distance", "**Maximum-distance rule (selected design)**", rep["knee"], True),
        ("kneedle", "Kneedle implementation-equivalent check", rep["knee"], False),
        (
            "budget_constrained",
            f"Budget-constrained rule (<={_fmt_int(budget_ceiling_usd)} USD per streetlight)",
            budget_row,
            False,
        ),
    ]:
        row_text, row_values = decision_row(label, row, bold=bold)
        s5_rows.append(row_text)
        decision_values[key] = row_values
    blocks["table.s5_rows"] = "\n".join(s5_rows)
    tokens.update(
        {
            "decision.budget_ceiling_usd_int": _fmt_int(budget_ceiling_usd),
            "decision.minimum_macc_abatement_reduction_pct_int": _fmt_int(
                (
                    1.0
                    - decision_values["minimum_macc"]["operational_abatement_t_per_streetlight"]
                    / selected["operational_abatement_t_per_streetlight"]
                )
                * 100.0
            ),
        }
    )
    values["alternative_decision_rules"] = decision_values

    lca_values, lca_tokens, lca_blocks = _build_lca_values(
        f3, selected["delta_cost_usd_per_streetlight"]
    )
    values["lca"] = lca_values
    values["battery_lifetime"] = _build_battery_lifetime_values(selected, lca_values, config)
    values["discount_rate"] = _build_discount_rate_values(
        selected, lca_values, config,
        _read_csv(results_dir / "pareto" / "selected_allocation_city_metrics.csv"),
    )
    values["selection_sensitivity"] = _build_selection_sensitivity_values(
        selected, lca_values, config,
        pd.read_csv(results_dir / "pareto" / "results.csv", usecols=[
            "city", "solar_panel_factor", "battery_factor", "abatement_t",
            "delta_cost", "functional_unit_lights", "analysis_years",
        ]),
        _read_json(ROOT / "data" / "geo" / "municipal_population_2024.json"),
    )
    tokens.update(lca_tokens)
    blocks.update(lca_blocks)

    # Carbon payback table.
    pb_labels = {
        "min_cost": "Minimum-cost frontier point",
        "knee": "Selected Pareto-knee",
        "max_abatement": "Maximum-abatement frontier point",
    }
    s3d_rows = []
    for label in ["min_cost", "knee", "max_abatement"]:
        row = _row_by(payback, "design_label", label)
        s3d_rows.append(
            _md_table_row(
                [
                    pb_labels[label],
                    _fmt(float(row["additional_initial_carbon_tco2e_per_streetlight"]), 3),
                    _fmt(float(row["additional_non_operational_carbon_tco2e_per_streetlight"]), 3),
                    _fmt(float(row["annual_abatement_current_tco2e_per_year"]), 4),
                    _fmt(float(row["initial_carbon_payback_years_current"]), 2),
                    _fmt(float(row["non_operational_carbon_payback_years_current"]), 2),
                ]
            )
        )
    knee_pb = _row_by(payback, "design_label", "knee")
    blocks["table.s3d_rows"] = "\n".join(s3d_rows)
    tokens.update(
        {
            "payback.hardware_years_2": _fmt(
                float(knee_pb["non_operational_carbon_payback_years_current"]), 2
            ),
            "payback.hardware_years_1": _fmt(
                float(knee_pb["non_operational_carbon_payback_years_current"]), 1
            ),
            "payback.annual_abatement_t_3": _fmt(
                float(knee_pb["annual_abatement_current_tco2e_per_year"]), 3
            ),
        }
    )

    # City metrics and ranking.
    f4 = f4.copy()
    f4["city_label"] = f4["city_en"].replace(config.city_name_overrides)
    f4["macc_usd_per_tco2e"] = (
        f4["delta_cost_usd_per_streetlight"].astype(float) / f4["abatement_t"].astype(float)
    )
    f4_sorted = f4.sort_values("abatement_t", ascending=False)
    top3 = f4_sorted.head(3)
    bottom3 = f4_sorted.tail(3).sort_values("abatement_t")
    city_gap_pct = (
        (float(f4["abatement_t"].max()) - float(f4["abatement_t"].min()))
        / float(f4["abatement_t"].min())
        * 100.0
    )
    mean_row = {
        "mean_par": float(f4["mean_par"].mean()),
        "aef": float(f4["aef"].mean()),
        "abatement_t": float(f4["abatement_t"].mean()),
        "delta_cost_usd_per_streetlight": float(f4["delta_cost_usd_per_streetlight"].mean()),
    }
    mean_row["macc_usd_per_tco2e"] = (
        mean_row["delta_cost_usd_per_streetlight"] / mean_row["abatement_t"]
    )
    s13_rows = []
    for _, row in f4_sorted.iterrows():
        s13_rows.append(
            _md_table_row(
                [
                    str(row["city_label"]),
                    str(row["region"]),
                    _fmt(float(row["mean_par"]), 1),
                    _fmt(float(row["aef"]), 3),
                    _fmt(float(row["abatement_t"]), 3),
                    _fmt_int(float(row["delta_cost_usd_per_streetlight"])),
                    _fmt_int(float(row["macc_usd_per_tco2e"])),
                ]
            )
        )
    s13_rows.append(
        _md_table_row(
            [
                "**22-municipality mean**",
                "",
                f"**{_fmt(mean_row['mean_par'], 1)}**",
                f"**{_fmt(mean_row['aef'], 3)}**",
                f"**{_fmt(mean_row['abatement_t'], 3)}**",
                f"**{_fmt_int(mean_row['delta_cost_usd_per_streetlight'])}**",
                f"**{_fmt_int(mean_row['macc_usd_per_tco2e'])}**",
            ]
        )
    )
    blocks["table.s13_rows"] = "\n".join(s13_rows)
    tokens.update(
        {
            "city.gap_pct_int": _fmt_int(city_gap_pct),
            "city.min_abatement_t_2": _fmt(float(f4["abatement_t"].min()), 2),
            "city.max_abatement_t_2": _fmt(float(f4["abatement_t"].max()), 2),
            "city.top1_name": str(top3.iloc[0]["city_label"]),
            "city.top1_abatement_t_2": _fmt(float(top3.iloc[0]["abatement_t"]), 2),
            "city.top2_name": str(top3.iloc[1]["city_label"]),
            "city.top2_abatement_t_2": _fmt(float(top3.iloc[1]["abatement_t"]), 2),
            "city.top3_name": str(top3.iloc[2]["city_label"]),
            "city.top3_abatement_t_2": _fmt(float(top3.iloc[2]["abatement_t"]), 2),
            "city.bottom1_name": str(bottom3.iloc[0]["city_label"]),
            "city.bottom1_abatement_t_2": _fmt(float(bottom3.iloc[0]["abatement_t"]), 2),
            "city.bottom2_name": str(bottom3.iloc[1]["city_label"]),
            "city.bottom2_abatement_t_2": _fmt(float(bottom3.iloc[1]["abatement_t"]), 2),
            "city.bottom3_name": str(bottom3.iloc[2]["city_label"]),
            "city.bottom3_abatement_t_2": _fmt(float(bottom3.iloc[2]["abatement_t"]), 2),
            "city.aef_range_min_3": _fmt(float(f4["aef"].min()), 3),
            "city.aef_range_max_3": _fmt(float(f4["aef"].max()), 3),
        }
    )

    reg_values, reg_tokens = _build_regression_tokens(f4)
    values["regression"] = reg_values
    tokens.update(reg_tokens)
    blocks["table.s14_rows"] = "\n".join(
        [
            _md_table_row(
                [
                    "1",
                    "AEF only",
                    tokens["regression.aef_only_coef_2"],
                    "--",
                    tokens["regression.aef_only_r2_3"],
                ]
            ),
            _md_table_row(
                [
                    "2",
                    "PAR only",
                    "--",
                    tokens["regression.par_only_coef_2"],
                    tokens["regression.par_only_r2_3"],
                ]
            ),
            _md_table_row(
                [
                    "3",
                    "AEF + PAR",
                    tokens["regression.full_aef_coef_2"],
                    tokens["regression.full_par_coef_2"],
                    tokens["regression.full_r2_3"],
                ]
            ),
        ]
    )

    # Rank shifts from Figure 5 source table.
    rank_shift_static = max(abs(f5["r_static"] - f5["r_baseline"]))
    rank_shift_uniform = max(abs(f5["r_uniform"] - f5["r_baseline"]))
    rank_shift_combined = max(abs(f5["r_combined"] - f5["r_baseline"]))
    city_label_by_local_name = dict(zip(f4["city"], f4["city_label"]))
    if not _is_split_outlying_island_mainline(f5["region"]):
        raise ValueError(
            "Canonical Paper 1 figure data are not using split outlying-island "
            "load-serving zones. Rerun the Paper 1 canonical results and assets "
            "before rebuilding manuscript values."
        )
    s7a_block, island_tokens, island_values = _build_mainline_outlying_island_block(
        f4=f4,
        f5=f5,
        city_label_by_local_name=city_label_by_local_name,
        emission_factors=emission_factors,
        canonical_inputs_dir=config.canonical_inputs_dir,
    )
    blocks["table.s7a_island_split_rows"] = s7a_block
    tokens.update(island_tokens)
    values["island_split_proxy"] = island_values
    s6_rows = []
    f5_ranked = f5.sort_values("r_baseline").copy()
    for _, row in f5_ranked.iterrows():
        shifts = [
            abs(int(row["r_static"]) - int(row["r_baseline"])),
            abs(int(row["r_uniform"]) - int(row["r_baseline"])),
            abs(int(row["r_combined"]) - int(row["r_baseline"])),
        ]
        s6_rows.append(
            _md_table_row(
                [
                    city_label_by_local_name.get(str(row["city"]), str(row["city"])),
                    str(row["region"]),
                    str(int(row["r_baseline"])),
                    str(int(row["r_static"])),
                    str(int(row["r_uniform"])),
                    str(int(row["r_combined"])),
                    str(max(shifts)),
                ]
            )
        )
    blocks["table.s6_rows"] = "\n".join(s6_rows)
    top5_reference = [
        city_label_by_local_name.get(str(row["city"]), str(row["city"]))
        for _, row in f5_ranked.head(5).iterrows()
    ]
    top5_combined = [
        city_label_by_local_name.get(str(row["city"]), str(row["city"]))
        for _, row in f5.sort_values("r_combined").head(5).iterrows()
    ]
    central_top_count = int(
        ((f5["r_baseline"].astype(int) <= 8) & (f5["region"].astype(str) == "central")).sum()
    )
    static_south_gains = [
        int(row["r_baseline"]) - int(row["r_static"])
        for _, row in f5[f5["region"].astype(str) == "south"].iterrows()
        if int(row["r_baseline"]) - int(row["r_static"]) >= 10
    ]
    static_drop_by_city = {
        city_label_by_local_name.get(str(row["city"]), str(row["city"])): int(row["r_static"])
        - int(row["r_baseline"])
        for _, row in f5.iterrows()
    }
    static_summary = _row_by(pareto_summary, "scenario", "static_average_aef")
    life_cycle_boundary = _row_by(cost_boundary, "boundary", "life_cycle")
    electricity_boundary = _row_by(cost_boundary, "boundary", "electricity_only")
    tokens.update(
        {
            "rank.static_max_shift_int": str(int(rank_shift_static)),
            "rank.uniform_max_shift_int": str(int(rank_shift_uniform)),
            "rank.combined_max_shift_int": str(int(rank_shift_combined)),
            "rank.reference_top5": ", ".join(top5_reference[:-1]) + f", and {top5_reference[-1]}",
            "rank.combined_top5": ", ".join(top5_combined[:-1]) + f", and {top5_combined[-1]}",
            "rank.central_count_in_top8_int": str(central_top_count),
            "rank.top_group_count_int": "8",
            "rank.central_aef_3": _fmt(float(f4[f4["region"] == "central"]["aef"].mean()), 3),
            "rank.static_south_gain_min_int": str(min(static_south_gains)),
            "rank.static_south_gain_max_int": str(max(static_south_gains)),
            "rank.lienchiang_static_drop_int": str(static_drop_by_city["Lienchiang"]),
            "rank.kinmen_static_drop_int": str(static_drop_by_city["Kinmen"]),
            "rank.penghu_static_drop_int": str(static_drop_by_city["Penghu"]),
            "static_aef.value_3": _fmt(float(static_summary["static_aef_value"]), 3),
            "static_aef.value_5": _fmt(float(static_summary["static_aef_value"]), 5),
            "cost_boundary.life_cycle_median_usd_int": _fmt_int(
                float(life_cycle_boundary["median_usd_per_streetlight"])
            ),
            "cost_boundary.electricity_only_median_usd_int": _fmt_int(
                float(electricity_boundary["median_usd_per_streetlight"])
            ),
        }
    )
    values["rank_shifts"] = {
        "static": int(rank_shift_static),
        "uniform": int(rank_shift_uniform),
        "combined": int(rank_shift_combined),
        "city_rows": f5_ranked.to_dict(orient="records"),
        "reference_top5": top5_reference,
        "combined_top5": top5_combined,
    }
    values["cost_boundary"] = {
        "life_cycle_median_usd_per_streetlight": float(life_cycle_boundary["median_usd_per_streetlight"]),
        "electricity_only_median_usd_per_streetlight": float(
            electricity_boundary["median_usd_per_streetlight"]
        ),
    }

    # Future-grid retained abatement.
    future_rows = []
    reselected_rows = []
    for scenario in ["official_2030_grid_target", "pathway_2050_midpoint_proxy"]:
        row = _row_by(future, "scenario", scenario)
        label = config.future_grid_scenario_labels.get(scenario, scenario)
        future_rows.append(
            _md_table_row(
                [
                    label,
                    _fmt(float(row["panel_mean_canonical_scenario"]), 4),
                    _fmt(float(row["scale_factor"]), 4),
                    (
                        f"{_fmt(float(row['uniform_aef_knee_abatement_per_pole_t']), 2)} "
                        f"({_fmt(float(row['uniform_retention_pct']), 1)}%)"
                    ),
                    (
                        f"{_fmt(float(row['generation_by_fuel_knee_abatement_per_pole_t']), 2)} "
                        f"({_fmt(float(row['generation_by_fuel_retention_pct']), 1)}%)"
                    ),
                    _fmt(float(row["diurnal_asymmetry_gap_pp"]), 2),
                ]
            )
        )
        reselected_pv_kw, reselected_battery_kwh = _capacity_from_factor(
            float(row["reselected_knee_solar_panel_factor"]),
            float(row["reselected_knee_battery_factor"]),
            contract,
            config,
        )
        reselected_rows.append(
            _md_table_row(
                [
                    label,
                    _fmt(float(row["reselected_knee_abatement_per_pole_t"]), 2),
                    _fmt(float(row["reselected_knee_retention_pct"]), 1) + "%",
                    _fmt(reselected_pv_kw, 2),
                    _fmt(reselected_battery_kwh, 2),
                ]
            )
        )
    blocks["table.s9a_rows"] = "\n".join(future_rows)
    blocks["table.s9b_rows"] = "\n".join(reselected_rows)
    baseline_shares = future_metadata["baseline_fuel_shares_pct"]
    s2030 = future_metadata["scenarios"]["official_2030_grid_target"]
    s2050 = future_metadata["scenarios"]["pathway_2050_midpoint_proxy"]

    def share(mapping: dict[str, float], *keys: str) -> float:
        return sum(float(mapping.get(key, 0.0)) for key in keys)

    future_share_specs = [
        ("Coal", ("Coal",), ("Coal",), ("Coal",), ""),
        (
            "Natural gas / model LNG category",
            ("LNG",),
            ("LNG",),
            ("LNG",),
            " (gas+CCUS modeled as LNG+CCUS)",
        ),
        ("Solar", ("Solar",), ("Solar",), ("Solar",), ""),
        ("Offshore wind", ("Offshore Wind",), ("Offshore Wind",), ("Offshore Wind",), ""),
        ("Onshore wind", ("Wind",), ("Wind",), ("Wind",), ""),
        ("Hydro", ("Hydro",), ("Hydro",), ("Hydro",), ""),
        (
            "Biomass + geothermal",
            ("Biomass", "Geothermal"),
            ("Biomass", "Geothermal"),
            ("Biomass", "Geothermal"),
            "",
        ),
        ("Nuclear", ("Nuclear",), ("Nuclear",), ("Nuclear",), ""),
        ("Oil + diesel", ("Oil", "Diesel"), ("Oil", "Diesel"), ("Oil", "Diesel"), ""),
        ("Cogeneration", ("Co-Gen",), ("Co-Gen",), ("Co-Gen",), ""),
        (
            "Hydrogen",
            ("Hydrogen",),
            ("Hydrogen",),
            ("Hydrogen",),
            " (0 kg CO2e/kWh target-state assumption)",
        ),
    ]
    s7_rows = []
    future_share_values: dict[str, Any] = {}
    for label, baseline_keys, keys_2030, keys_2050, suffix_2050 in future_share_specs:
        baseline_value = share(baseline_shares, *baseline_keys)
        value_2030 = share(s2030["target_shares_pct"], *keys_2030)
        value_2050 = share(s2050["target_shares_pct"], *keys_2050)
        s7_rows.append(
            _md_table_row(
                [
                    label,
                    _fmt(baseline_value, 2),
                    _fmt(value_2030, 2),
                    f"{_fmt(value_2050, 2)}{suffix_2050}",
                ]
            )
        )
        future_share_values[label] = {
            "baseline_2024_pct": baseline_value,
            "official_2030_grid_target_pct": value_2030,
            "pathway_2050_midpoint_proxy_pct": value_2050,
        }
    blocks["table.s7_rows"] = "\n".join(s7_rows)
    blocks["table.s8_rows"] = "\n".join(
        [
            _md_table_row(
                [
                    f"Observed {reference_year} fuel profiles scaled to target shares",
                    "coal, LNG, hydro, solar, onshore wind, biomass/geothermal",
                    "hydro, solar, onshore wind, biomass/geothermal; LNG retained with CCUS-modified emission factor",
                ]
            ),
            _md_table_row(
                [
                    "Fuels set to zero outside the target mix",
                    "nuclear, oil, diesel, cogeneration",
                    "coal, nuclear, oil, diesel, cogeneration",
                ]
            ),
            _md_table_row(
                [
                    f"New target fuels without {reference_year} unit profiles",
                    (
                        f"offshore wind added at target share using the {reference_year} Wind "
                        "timing and regional-distribution fallback"
                    ),
                    (
                        f"offshore wind added with the {reference_year} Wind fallback; hydrogen "
                        "added at target share with total-demand timing, current-LNG regional "
                        "distribution, and 0 kg CO2e/kWh target-state assumption"
                    ),
                ]
            ),
            _md_table_row(
                [
                    "Island load-serving balance",
                    (
                        "diesel/oil phase-out shortfall allocated across remaining target-mix fuels "
                        f"by target-share weights; timestep shape follows the {reference_year} island "
                        "total-generation profile"
                    ),
                    "same rule",
                ]
            ),
            _md_table_row(
                [
                    "Annual generation normalization",
                    f"generation normalized to {reference_year} annual total after balancing",
                    "same",
                ]
            ),
        ]
    )
    f2030 = _row_by(future, "scenario", "official_2030_grid_target")
    f2050 = _row_by(future, "scenario", "pathway_2050_midpoint_proxy")
    tokens.update(
        {
            "future.2030_fixed_abatement_t_2": _fmt(
                float(f2030["generation_by_fuel_knee_abatement_per_pole_t"]), 2
            ),
            "future.2030_retention_pct_1": _fmt(
                float(f2030["generation_by_fuel_retention_pct"]), 1
            ),
            "future.2030_reselected_abatement_t_2": _fmt(
                float(f2030["reselected_knee_abatement_per_pole_t"]), 2
            ),
            "future.2030_reselected_retention_pct_1": _fmt(
                float(f2030["reselected_knee_retention_pct"]), 1
            ),
            "future.2050_fixed_abatement_t_2": _fmt(
                float(f2050["generation_by_fuel_knee_abatement_per_pole_t"]), 2
            ),
            "future.2050_retention_pct_1": _fmt(
                float(f2050["generation_by_fuel_retention_pct"]), 1
            ),
            "future.2050_reselected_abatement_t_2": _fmt(
                float(f2050["reselected_knee_abatement_per_pole_t"]), 2
            ),
            "future.2050_reselected_retention_pct_1": _fmt(
                float(f2050["reselected_knee_retention_pct"]), 1
            ),
            "future.2030_lng_share_pct_int": _fmt_int(
                share(s2030["target_shares_pct"], "LNG")
            ),
            "future.2030_renewables_share_pct_int": _fmt_int(
                share(
                    s2030["target_shares_pct"],
                    "Solar",
                    "Offshore Wind",
                    "Wind",
                    "Hydro",
                    "Biomass",
                    "Geothermal",
                )
            ),
            "future.2030_coal_share_pct_int": _fmt_int(
                share(s2030["target_shares_pct"], "Coal")
            ),
            "future.2050_renewable_midpoint_pct_int": _fmt_int(
                share(
                    s2050["target_shares_pct"],
                    "Solar",
                    "Offshore Wind",
                    "Wind",
                    "Hydro",
                    "Biomass",
                    "Geothermal",
                )
            ),
            "future.2050_hydrogen_midpoint_pct_1": _fmt(
                share(s2050["target_shares_pct"], "Hydrogen"), 1
            ),
            "future.2050_lng_ccus_share_pct_1": _fmt(
                share(s2050["target_shares_pct"], "LNG"), 1
            ),
            "future.2050_lng_ccus_ef_3": _fmt(float(s2050["ef_overrides_kgco2e_per_kwh"]["LNG"]), 3),
            "future.2050_renewables_range_pct": "60-70",
            "future.2050_hydrogen_range_pct": "9-12",
            "future.2050_gas_ccus_range_pct": "20-27",
            "future.2050_pumped_storage_pct_int": "1",
            "future.2050_lng_ccus_ef_min_3": _fmt(0.092, 3),
            "future.2050_lng_ccus_ef_max_3": _fmt(0.220, 3),
            "future.target_share_total_pct_int": "100",
            "future.current_static_aef_5": _fmt(
                float(future_metadata["current_regional_panel_mean_kgco2e_per_kwh"]), 5
            ),
        }
    )
    values["future_grid_shares"] = future_share_values
    values["future_policy_ranges"] = {
        "renewables_pct": "60-70",
        "hydrogen_pct": "9-12",
        "gas_ccus_pct": "20-27",
        "pumped_storage_pct": 1,
        "unece_lng_ccus_ef_range_kgco2e_per_kwh": [0.092, 0.220],
    }

    # Sensitivity and Monte Carlo.
    f7_clean = f7.copy()
    f7_clean["parameter_label"] = f7_clean["parameter"].replace(config.sensitivity_labels)
    s10a_rows = []
    for _, row in f7_clean.iterrows():
        original = str(row["parameter"])
        tested = config.sensitivity_tested_values.get(original, "")
        s10a_rows.append(
            _md_table_row(
                [
                    str(row["parameter_label"]),
                    tested,
                    _fmt(float(row["baseline_macc_usd_per_t"]), 1),
                    f"{_fmt(float(row['low_macc_usd_per_t']), 1)} / {_fmt(float(row['high_macc_usd_per_t']), 1)}",
                    f"{_fmt_signed(float(row['low_delta_pct']), 2)} / {_fmt_signed(float(row['high_delta_pct']), 2)}",
                ]
            )
        )
    blocks["table.s10a_rows"] = "\n".join(s10a_rows)

    s10b_specs = [
        ("Coal emission factor ±20%", "fuel_ef_coal"),
        ("LNG emission factor ±20%", "fuel_ef_lng"),
        ("Diesel emission factor ±20%", "fuel_ef_diesel"),
        ("Oil emission factor ±20%", "fuel_ef_oil"),
        ("Gas-fired cogeneration emission factor ±20%", "fuel_ef_co_gen"),
        ("Solar PV emission factor ±20%", "fuel_ef_solar"),
        ("Hydropower emission factor ±20%", "fuel_ef_hydro"),
        ("Wind offshore emission factor ±20%", "fuel_ef_offshore_wind"),
        ("Nuclear emission factor ±20%", "fuel_ef_nuclear"),
        ("Biomass emission factor ±20%", "fuel_ef_biomass"),
        ("Wind onshore emission factor ±20%", "fuel_ef_wind"),
        ("Geothermal emission factor ±20%", "fuel_ef_geothermal"),
    ]
    s10b_rows = []
    s10b_values: dict[str, Any] = {}
    for label, parameter in s10b_specs:
        aef_low, aef_high = _fuel_sensitivity_pair(subjective, parameter)
        s10b_rows.append(
            _md_table_row(
                [
                    label,
                    _fmt_response_pair(aef_low, aef_high, 2),
                    _fmt_response_pair(-aef_low, -aef_high, 2),
                ]
            )
        )
        s10b_values[label] = {
            "aef_response_low_input_pct": aef_low,
            "aef_response_high_input_pct": aef_high,
            "macc_response_low_input_pct": -aef_low,
            "macc_response_high_input_pct": -aef_high,
        }
    phs_low, phs_high = _storage_sensitivity_pair(subjective, 0.75, 0.82)
    s10b_rows.append(
        _md_table_row(
            [
                "Pumped-hydro storage efficiency 0.75 / 0.82",
                _fmt_response_pair(phs_low, phs_high, 2),
                _fmt_response_pair(-phs_low, -phs_high, 2),
            ]
        )
    )
    s10b_values["Pumped-hydro storage efficiency 0.75 / 0.82"] = {
        "aef_response_low_input_pct": phs_low,
        "aef_response_high_input_pct": phs_high,
        "macc_response_low_input_pct": -phs_low,
        "macc_response_high_input_pct": -phs_high,
    }
    cogen_means = cogen_biomass.groupby("scenario")["final_aef_mean"].mean()
    cogen_baseline = float(cogen_means["baseline"])
    cogen_deltas = [
        _pct_delta(value, cogen_baseline)
        for scenario, value in cogen_means.items()
        if str(scenario) != "baseline"
    ]
    cogen_low = min(cogen_deltas)
    cogen_high = max(cogen_deltas)
    s10b_rows.append(
        _md_table_row(
            [
                "Cogeneration/biomass regional-allocation sensitivity range",
                _fmt_response_pair(cogen_low, cogen_high, 2),
                _fmt_response_pair(-cogen_low, -cogen_high, 2),
            ]
        )
    )
    s10b_values["Cogeneration/biomass regional-allocation sensitivity range"] = {
        "aef_response_low_pct": cogen_low,
        "aef_response_high_pct": cogen_high,
        "macc_response_low_pct": -cogen_low,
        "macc_response_high_pct": -cogen_high,
    }
    blocks["table.s10b_rows"] = "\n".join(s10b_rows)
    uniform_fuel_perturbation_pct = _subjective_factor_perturb_pct(
        subjective, "fuel_ef_coal", emission_factors["Coal"]
    )

    def reciprocal_macc_response(aef_response_pct: float) -> float:
        return (1.0 / (1.0 + aef_response_pct / 100.0) - 1.0) * 100.0

    ecoinvent_literature_band_pct = 15
    ecoinvent_range_definition = (
        f"study-defined ±{ecoinvent_literature_band_pct}% around ecoinvent base"
    )
    literature_specs = [
        ("Coal", "selected IPCC Annex III-derived range", "Coal", 0.740, 1.050, -3.98, 11.43),
        ("LNG", "selected IPCC Annex III-derived range", "LNG", 0.410, 0.650, -4.09, 8.17),
        ("Diesel", ecoinvent_range_definition, "Diesel", None, None, -2.77, 2.77),
        ("Oil", ecoinvent_range_definition, "Oil", None, None, -1.81, 1.81),
        ("Solar PV", "selected IPCC Annex III-derived range", "Solar", 0.018, 0.180, -0.30, 1.33),
        ("Gas-fired cogeneration", ecoinvent_range_definition, "Co-Gen", None, None, -0.42, 0.42),
        ("Wind onshore", "selected IPCC Annex III-derived range", "Wind", 0.007, 0.056, -0.03, 0.37),
        ("Hydropower", "selected IPCC Annex III-derived range", "Hydro", 0.001, 0.043, -0.19, 0.15),
        ("Nuclear", "selected IPCC Annex III-derived range", "Nuclear", 0.000, 0.025, -0.08, 0.09),
    ]
    s11_rows = []
    literature_values: dict[str, Any] = {}
    for label, range_definition, fuel_key, lower, upper, aef_low, aef_high in literature_specs:
        central = emission_factors[fuel_key]
        lower_factor = central * 0.85 if lower is None else float(lower)
        upper_factor = central * 1.15 if upper is None else float(upper)
        macc_low = reciprocal_macc_response(aef_low)
        macc_high = reciprocal_macc_response(aef_high)
        s11_rows.append(
            _md_table_row(
                [
                    label,
                    range_definition,
                    f"{_fmt(lower_factor, 3)} / {_fmt(central, 3)} / {_fmt(upper_factor, 3)}",
                    _fmt_signed_pair(aef_low, aef_high, 2),
                    _fmt_signed_pair(macc_low, macc_high, 2),
                ]
            )
        )
        literature_values[label] = {
            "range_definition": range_definition,
            "lower_factor": lower_factor,
            "central_factor": central,
            "upper_factor": upper_factor,
            "aef_response_low_pct": aef_low,
            "aef_response_high_pct": aef_high,
            "macc_response_low_pct": macc_low,
            "macc_response_high_pct": macc_high,
        }
    blocks["table.s11_rows"] = "\n".join(s11_rows)
    values["grid_carbon_sensitivity"] = {
        "uniform_perturbation_pct": uniform_fuel_perturbation_pct,
        "uniform_perturbation": s10b_values,
        "literature_bounded": literature_values,
    }

    def sens_value(name: str) -> pd.Series:
        rows = f7[f7["parameter"] == name]
        if rows.empty:
            raise KeyError(name)
        return rows.iloc[0]

    battery = sens_value("Battery CAPEX")
    pv = sens_value("PV CAPEX")
    elec = sens_value("Electricity price")
    par = sens_value("PAR estimate")
    coal = sens_value("Coal EF")
    lng = sens_value("LNG EF")
    other = sens_value("Other fuel EFs (max individual)")
    degr = sens_value("PV-battery degradation")
    discount = sens_value("Discount rate")
    replacement = sens_value("Battery replacement timing")
    minor_fuel_parameters = [
        "fuel_ef_solar",
        "fuel_ef_hydro",
        "fuel_ef_wind",
        "fuel_ef_offshore_wind",
        "fuel_ef_nuclear",
    ]
    minor_fuel_threshold = max(
        max(abs(value) for value in _fuel_sensitivity_pair(subjective, parameter))
        for parameter in minor_fuel_parameters
    )
    tokens.update(
        {
            "sensitivity.battery_capex_pm_pct_1": _fmt_pm(float(battery["high_delta_pct"]), 1),
            "sensitivity.pv_capex_pm_pct_1": _fmt_pm(float(pv["high_delta_pct"]), 1),
            "sensitivity.electricity_price_pm_pct_1": _fmt_pm(float(elec["low_delta_pct"]), 1),
            "sensitivity.par_low_pct_1": _fmt_signed(float(par["low_delta_pct"]), 1),
            "sensitivity.par_high_pct_1": _fmt_signed(float(par["high_delta_pct"]), 1),
            "sensitivity.coal_pm_pct_1": _fmt_pm(float(coal["low_delta_pct"]), 1),
            "sensitivity.lng_pm_pct_1": _fmt_pm(float(lng["low_delta_pct"]), 1),
            "sensitivity.other_fuel_pm_pct_1": _fmt_pm(float(other["low_delta_pct"]), 1),
            "sensitivity.degradation_low_pct_1": _fmt_signed(float(degr["low_delta_pct"]), 1),
            "sensitivity.degradation_low_abs_pct_1": _fmt(abs(float(degr["low_delta_pct"])), 1),
            "sensitivity.degradation_high_pct_1": _fmt_signed(float(degr["high_delta_pct"]), 1),
            "sensitivity.degradation_high_abs_pct_1": _fmt(abs(float(degr["high_delta_pct"])), 1),
            "sensitivity.discount_low_pct_1": _fmt_signed(float(discount["low_delta_pct"]), 1),
            "sensitivity.discount_high_pct_1": _fmt_signed(float(discount["high_delta_pct"]), 1),
            "sensitivity.replacement_low_pct_1": _fmt_signed(float(replacement["low_delta_pct"]), 1),
            "sensitivity.replacement_high_pct_1": _fmt_signed(float(replacement["high_delta_pct"]), 1),
            "sensitivity.uniform_perturbation_pct_int": _fmt_int(uniform_fuel_perturbation_pct),
            "sensitivity.ecoinvent_literature_band_pct_int": _fmt_int(ecoinvent_literature_band_pct),
            "sensitivity.low_share_fuel_threshold_pct_1": _fmt_pm(minor_fuel_threshold, 1),
            "sensitivity.phs_macc_threshold_pct_1": _fmt(0.1, 1),
        }
    )

    deg_labels = {
        "low_degradation": "Milder degradation pathway",
        "base_degradation": "Reference degradation pathway",
        "high_degradation": "More severe degradation pathway",
    }
    s12_rows = []
    for scenario in ["low_degradation", "base_degradation", "high_degradation"]:
        row = _row_by(degradation, "scenario", scenario)
        s12_rows.append(
            _md_table_row(
                [
                    deg_labels[scenario],
                    _fmt(float(row["battery_replacement_year"]), 0),
                    _fmt(float(row["pv_degradation_rate_per_year"]) * 100.0, 1),
                    _fmt(float(row["battery_usable_capacity_multiplier"]) * 100.0, 0),
                    _fmt(float(row["knee_abatement_t"]) / config.study_light_count, 2),
                    _fmt_int(
                        float(row["knee_delta_cost"])
                        / config.ntd_per_usd
                        / config.study_light_count
                    ),
                ]
            )
        )
    blocks["table.s12_rows"] = "\n".join(s12_rows)
    low_degradation = _row_by(degradation, "scenario", "low_degradation")
    high_degradation = _row_by(degradation, "scenario", "high_degradation")
    base_degradation = _row_by(degradation, "scenario", "base_degradation")
    degradation_pv_kw, degradation_battery_kwh = _capacity_from_factor(
        float(base_degradation["knee_solar_panel_factor"]),
        float(base_degradation["knee_battery_factor"]),
        contract,
        config,
    )
    tokens.update(
        {
            "degradation.low_pv_fade_pct_1": _fmt(
                float(low_degradation["pv_degradation_rate_per_year"]) * 100.0, 1
            ),
            "degradation.low_usable_capacity_pct_int": _fmt_int(
                float(low_degradation["battery_usable_capacity_multiplier"]) * 100.0
            ),
            "degradation.high_pv_fade_pct_1": _fmt(
                float(high_degradation["pv_degradation_rate_per_year"]) * 100.0, 1
            ),
            "degradation.high_usable_capacity_pct_int": _fmt_int(
                float(high_degradation["battery_usable_capacity_multiplier"]) * 100.0
            ),
            "degradation.high_replacement_year_int": _fmt_int(
                float(high_degradation["battery_replacement_year"])
            ),
            "degradation.knee_pv_kw_2": _fmt(degradation_pv_kw, 2),
            "degradation.knee_battery_kwh_2": _fmt(degradation_battery_kwh, 2),
        }
    )
    values["degradation"] = {
        "scenarios": degradation.to_dict(orient="records"),
        "knee_pv_kw_per_streetlight": degradation_pv_kw,
        "knee_battery_kwh_per_streetlight": degradation_battery_kwh,
    }

    mc = {str(row["metric"]): row for _, row in f8.iterrows()}
    mc_cost = mc["delta_cost_usd_per_streetlight"]
    mc_macc = mc["macc_usd_per_tco2e"]
    n_samples = len(uncertainty_samples)
    tokens.update(
        {
            "monte_carlo.n_samples_int": f"{n_samples:,}",
            "monte_carlo.seed": str(int(uncertainty_metadata["seed"])),
            "monte_carlo.cost_p05_usd_int": _fmt_int(float(mc_cost["p05"])),
            "monte_carlo.cost_p50_usd_int": _fmt_int(float(mc_cost["p50"])),
            "monte_carlo.cost_p95_usd_int": _fmt_int(float(mc_cost["p95"])),
            "monte_carlo.macc_p05_usd_int": _fmt_int(float(mc_macc["p05"])),
            "monte_carlo.macc_p50_usd_int": _fmt_int(float(mc_macc["p50"])),
            "monte_carlo.macc_p95_usd_int": _fmt_int(float(mc_macc["p95"])),
            "monte_carlo.macc_min_usd_int": _fmt_int(
                float(uncertainty_samples["macc_usd_per_tco2e"].min())
            ),
            "monte_carlo.cost_saving_count_int": str(
                int(round(float(mc_cost["probability_below_zero"]) * n_samples))
            ),
            "monte_carlo.battery_cost_low_pct_int": _fmt_int(
                float(config.economic_scenarios["optimistic"]["cost"]["battery_capex_ntd_per_kwh"])
                / float(config.economic_costs["battery_capex_ntd_per_kwh"])
                * 100.0
            ),
            "monte_carlo.battery_cost_mid_pct_int": "100",
            "monte_carlo.battery_cost_high_pct_int": _fmt_int(
                float(config.economic_scenarios["conservative"]["cost"]["battery_capex_ntd_per_kwh"])
                / float(config.economic_costs["battery_capex_ntd_per_kwh"])
                * 100.0
            ),
            "monte_carlo.replacement_prob_low_2": _fmt(
                config.battery_replacement_year_probabilities[0], 2
            ),
            "monte_carlo.replacement_prob_mid_2": _fmt(
                config.battery_replacement_year_probabilities[1], 2
            ),
            "monte_carlo.replacement_prob_high_2": _fmt(
                config.battery_replacement_year_probabilities[2], 2
            ),
        }
    )
    values["monte_carlo"] = {
        "metadata": uncertainty_metadata,
        "summary": _records(f8),
        "minimum_sampled_macc_usd_per_tco2e": float(
            uncertainty_samples["macc_usd_per_tco2e"].min()
        ),
    }

    # Calibration values used in methods/SI.
    alpha_cal = float(contract["effective_par_to_kw_factor"]) / float(
        contract["economic_costs"]["pv_capacity_kw_per_factor"]
    )
    yields = per_city_calibration["annual_par_integral"].astype(float) * alpha_cal
    annual_loads = per_city_calibration["annual_load_kwh"].astype(float)
    annual_loads_per_light = annual_loads / config.lights_per_city
    k_city_mean_per_light = (
        float(calibration["mean_city_parity_factor"]) / config.lights_per_city
    )
    rounded_baseline_per_light = (
        float(calibration["rounded_paper_baseline"]) / config.lights_per_city
    )
    raw_photon_energy_j = 0.219
    raw_par_shortwave_ratio = 0.46
    raw_pvwatts_derate = 0.86
    raw_stc_irradiance_w_m2 = 1000.0
    alpha_raw = raw_photon_energy_j * raw_pvwatts_derate / (
        raw_par_shortwave_ratio * raw_stc_irradiance_w_m2
    )
    tokens.update(
        {
            "calibration.alpha_cal_sci": _fmt_sci_latex(alpha_cal, 2),
            "calibration.load_min_per_light_int": _fmt_int(
                float(annual_loads_per_light.min())
            ),
            "calibration.load_max_per_light_int": _fmt_int(
                float(annual_loads_per_light.max())
            ),
            "calibration.k_city_mean_per_light_sci": _fmt_sci_latex(
                k_city_mean_per_light, 2
            ),
            "calibration.rounded_baseline_per_light_sci": _fmt_sci_latex(
                rounded_baseline_per_light, 2
            ),
            "calibration.raw_photon_energy_j_3": _fmt(raw_photon_energy_j, 3),
            "calibration.raw_par_shortwave_ratio_2": _fmt(raw_par_shortwave_ratio, 2),
            "calibration.raw_pvwatts_derate_2": _fmt(raw_pvwatts_derate, 2),
            "calibration.raw_stc_irradiance_int": _fmt_int(raw_stc_irradiance_w_m2),
            "calibration.alpha_raw_sci": _fmt_sci_latex(alpha_raw, 2),
            "calibration.yield_min_int": _fmt_int(float(yields.min())),
            "calibration.yield_max_int": _fmt_int(float(yields.max())),
            "calibration.yield_mean_int": _fmt_int(float(yields.mean())),
            "calibration.yield_median_int": _fmt_int(float(yields.median())),
        }
    )
    values["calibration"] = {
        "alpha_cal": alpha_cal,
        "alpha_raw": alpha_raw,
        "annual_load_kwh": {
            "min": float(annual_loads.min()),
            "max": float(annual_loads.max()),
            "mean": float(annual_loads.mean()),
            "median": float(annual_loads.median()),
        },
        "annual_load_kwh_per_streetlight": {
            "min": float(annual_loads_per_light.min()),
            "max": float(annual_loads_per_light.max()),
            "mean": float(annual_loads_per_light.mean()),
            "median": float(annual_loads_per_light.median()),
        },
        "mean_city_parity_factor_per_streetlight": k_city_mean_per_light,
        "rounded_paper_baseline_per_streetlight": rounded_baseline_per_light,
        "annual_yield_kwh_per_kwp": {
            "min": float(yields.min()),
            "max": float(yields.max()),
            "mean": float(yields.mean()),
            "median": float(yields.median()),
        },
    }

    # Compact, reproducible recognized-model comparison; weather stays outside Git.
    # Build the upstream values first, then PV and conditional uncertainty.
    # Never bootstrap these calculations from the published answer snapshot.
    if include_extensions:
        values["pv_model_comparison"] = json.loads(
            (results_dir / "pv_model_comparison.json").read_text(encoding="utf-8")
        )
        values["conditional_uncertainty"] = json.loads(
            (results_dir / "conditional_uncertainty.json").read_text(encoding="utf-8")
        )

    values["aef_audit"] = {
        "data_quality": build_aef_quality(config.canonical_inputs_dir),
        "storage": build_aef_storage_audit(config.canonical_inputs_dir),
    }

    return {
        "schema_version": 1,
        "source_config": config.config_path.relative_to(ROOT).as_posix(),
        "source_results_dir": str(results_dir.relative_to(ROOT)),
        "source_figure_data_dir": str(figure_data_dir.relative_to(ROOT)),
        "values": values,
        "tokens": tokens,
        "blocks": blocks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figure-data-dir", type=Path, default=DEFAULT_FIGURE_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG.config_path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-only", action="store_true",
                        help="Build upstream values before PV/conditional analyses")
    args = parser.parse_args()

    result = build_values(args.results_dir, args.figure_data_dir, args.config,
                          include_extensions=not args.base_only)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        display_path = output_path.relative_to(ROOT)
    except ValueError:
        display_path = output_path
    print(f"Wrote {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
