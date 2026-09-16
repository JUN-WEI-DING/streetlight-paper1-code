from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from streetlight.config import get_config
from streetlight.simulation.streetlight_simulation import (
    load_par_wide,
    load_region_city_map,
    simulate_from_par_and_aef,
)


IPP_LNG_PLANTS = ("嘉惠", "國光", "新桃", "星元", "星彰", "海湖", "豐德")
IPP_COAL_PLANTS = ("和平", "麥寮")
EXCLUDED_MAIN_GRID_MERIT_KEYWORDS = ("金門", "馬祖", "離島", "澎湖", "核三Gas", "核二Gas")

# 台電 113 上半年電價費率檢討方案－自發及購入電力燃料成本
# 單位：NTD/kWh
MERIT_COST_PROXY = {
    "Nuclear": 0.4,
    "Coal": 1.8,
    "IPP-Coal": 1.9,
    "LNG": 3.2,
    "IPP-LNG": 3.3,
    "LNG-GT": 4.45,
    "Oil": 5.4,
    "Diesel": 10.7,
    "Co-Gen": 3.3,
}

# 與現有 AEF 流程一致的燃料碳排係數；這裡作為 operational/marginal proxy 使用
# 單位：kgCO2e/kWh
EMISSION_PROXY = {
    "Coal": 1.02,
    "IPP-Coal": 1.02,
    "LNG": 0.48,
    "IPP-LNG": 0.567,
    "LNG-GT": 0.48,
    "Oil": 1.08,
    "Diesel": 0.511,
    "Co-Gen": 0.549,
    "Nuclear": 0.0,
}

MARGINAL_TRANCHE_SHARE = 0.05


def _normalize_wide_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.columns = [html.unescape(str(col)) for col in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.columns.has_duplicates:
        df = df.T.groupby(level=0).sum(min_count=1).T
    return df


def _classify_unit(unit: str) -> dict[str, Any] | None:
    if unit.startswith("Coal-"):
        dispatch_group = "IPP-Coal" if any(name in unit for name in IPP_COAL_PLANTS) else "Coal"
    elif unit.startswith("LNG-"):
        if any(name in unit for name in IPP_LNG_PLANTS):
            dispatch_group = "IPP-LNG"
        elif "GT" in unit:
            dispatch_group = "LNG-GT"
        else:
            dispatch_group = "LNG"
    elif unit.startswith("Oil-"):
        dispatch_group = "Oil"
    elif unit.startswith("Diesel-"):
        dispatch_group = "Diesel"
    elif unit.startswith("Co-Gen-"):
        dispatch_group = "Co-Gen"
    elif unit.startswith("Nuclear-"):
        dispatch_group = "Nuclear"
    else:
        return None

    return {
        "unit": unit,
        "dispatch_group": dispatch_group,
        "cost_proxy_ntd_per_kwh": MERIT_COST_PROXY[dispatch_group],
        "emission_proxy_kgco2e_per_kwh": EMISSION_PROXY[dispatch_group],
    }


def _build_unit_meta(unit_generation: pd.DataFrame, unit_capacity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in unit_generation.columns:
        meta = _classify_unit(col)
        if meta is None:
            continue
        series = pd.to_numeric(unit_generation[col], errors="coerce").fillna(0.0)
        cap_series = (
            pd.to_numeric(unit_capacity[col], errors="coerce").fillna(0.0)
            if col in unit_capacity.columns
            else pd.Series(dtype=float)
        )
        capacity_mw = float(cap_series.max()) if not cap_series.empty else np.nan
        online_mask = series > 0
        online_hours = float(online_mask.sum()) / 6.0
        annual_generation_mwh = float(series.sum()) / 6.0
        avg_output_when_online_mw = float(series[online_mask].mean()) if online_mask.any() else 0.0
        avg_load_ratio_when_online = (
            avg_output_when_online_mw / capacity_mw if capacity_mw and capacity_mw > 0 else np.nan
        )
        rows.append(
            {
                **meta,
                "capacity_mw": capacity_mw,
                "online_hours_2024": online_hours,
                "annual_generation_mwh_2024": annual_generation_mwh,
                "avg_output_when_online_mw": avg_output_when_online_mw,
                "avg_load_ratio_when_online": avg_load_ratio_when_online,
                "annual_cost_proxy_ntd_2024": annual_generation_mwh * 1000.0 * meta["cost_proxy_ntd_per_kwh"],
                "annual_emission_proxy_tco2e_2024": annual_generation_mwh * meta["emission_proxy_kgco2e_per_kwh"],
                "included_in_merit_order": not any(
                    keyword in col for keyword in EXCLUDED_MAIN_GRID_MERIT_KEYWORDS
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["cost_proxy_ntd_per_kwh", "dispatch_group", "unit"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def _build_marginal_timeseries(unit_generation: pd.DataFrame, unit_meta: pd.DataFrame) -> pd.DataFrame:
    merit_meta = unit_meta[unit_meta["included_in_merit_order"]].reset_index(drop=True)
    grouped = []
    for dispatch_group, group_df in merit_meta.groupby("dispatch_group", sort=False):
        units = group_df["unit"].tolist()
        grouped.append(
            {
                "dispatch_group": dispatch_group,
                "cost_proxy_ntd_per_kwh": float(group_df["cost_proxy_ntd_per_kwh"].iloc[0]),
                "emission_proxy_kgco2e_per_kwh": float(group_df["emission_proxy_kgco2e_per_kwh"].iloc[0]),
                "generation_mw": unit_generation[units].fillna(0.0).sum(axis=1).to_numpy(dtype=float),
            }
        )
    grouped = sorted(grouped, key=lambda row: row["cost_proxy_ntd_per_kwh"], reverse=True)

    marginal_cost: list[float | None] = []
    marginal_carbon: list[float | None] = []
    marginal_group: list[str | None] = []
    marginal_gen: list[float] = []
    thermal_total: list[float] = []

    row_count = len(unit_generation.index)
    for idx in range(row_count):
        total_generation = float(sum(row["generation_mw"][idx] for row in grouped))
        thermal_total.append(total_generation)
        if total_generation <= 0:
            marginal_cost.append(None)
            marginal_carbon.append(None)
            marginal_group.append(None)
            marginal_gen.append(0.0)
            continue

        tranche_target = total_generation * MARGINAL_TRANCHE_SHARE
        remaining = tranche_target
        carbon_num = 0.0
        carbon_den = 0.0
        active_groups: list[str] = []
        first_cost: float | None = None

        for row in grouped:
            generation = float(row["generation_mw"][idx])
            if generation <= 0:
                continue
            take = min(generation, remaining)
            if take <= 0:
                continue
            if first_cost is None:
                first_cost = float(row["cost_proxy_ntd_per_kwh"])
            active_groups.append(str(row["dispatch_group"]))
            carbon_num += take * float(row["emission_proxy_kgco2e_per_kwh"])
            carbon_den += take
            remaining -= take
            if remaining <= 1e-9:
                break

        marginal_cost.append(first_cost)
        marginal_carbon.append(carbon_num / carbon_den if carbon_den > 0 else None)
        marginal_group.append("+".join(active_groups) if active_groups else None)
        marginal_gen.append(carbon_den)

    return pd.DataFrame(
        {
            "timestamp": unit_generation.index,
            "marginal_cost_ntd_per_kwh": marginal_cost,
            "marginal_carbon_kgco2e_per_kwh": marginal_carbon,
            "marginal_dispatch_group": marginal_group,
            "marginal_unit_count": None,
            "thermal_generation_mw": thermal_total,
            "marginal_generation_mw": marginal_gen,
        }
    )


def _average_operational_ef(category_generation: pd.DataFrame) -> float:
    ef_map = {
        "Co-Gen": EMISSION_PROXY["Co-Gen"],
        "Coal": EMISSION_PROXY["Coal"],
        "Diesel": EMISSION_PROXY["Diesel"],
        "IPP-Coal": EMISSION_PROXY["IPP-Coal"],
        "IPP-LNG": EMISSION_PROXY["IPP-LNG"],
        "LNG": EMISSION_PROXY["LNG"],
        "Nuclear": EMISSION_PROXY["Nuclear"],
        "Oil": EMISSION_PROXY["Oil"],
    }
    num = 0.0
    den = 0.0
    for col, factor in ef_map.items():
        if col not in category_generation.columns:
            continue
        gen = pd.to_numeric(category_generation[col], errors="coerce").fillna(0.0)
        num += float((gen * factor).sum())
        den += float(gen.sum())
    return num / den if den > 0 else float("nan")


def _load_effective_par_to_kw_factor(output_dir: Path, cfg) -> float:
    contract_path = output_dir / "paper_contract.json"
    if not contract_path.exists():
        return float(cfg.par_to_kw_factor)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return float(contract.get("effective_par_to_kw_factor", cfg.par_to_kw_factor))


def _load_design_points(output_dir: Path) -> list[dict[str, Any]]:
    design_table = pd.read_csv(output_dir / "closing_analyses" / "design_decision_table.csv")
    cols = ["design_label", "solar_panel_factor", "battery_factor"]
    return design_table[cols].to_dict(orient="records")


def _simulate_designs_under_merit_mcf(
    cfg,
    output_dir: Path,
    merit_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    par_df = load_par_wide(cfg.paper_par_path)
    region_map_path = cfg.paper_region_map_path
    region_city_map = load_region_city_map(region_map_path)

    effective_par_to_kw_factor = _load_effective_par_to_kw_factor(output_dir, cfg)
    design_points = _load_design_points(output_dir)

    storage_params = dict(cfg.storage_params)
    base_capacity_kwh = float(cfg.storage_capacity_kwh)
    base_power_kw = float(cfg.storage_power_kw)
    eta_roundtrip = float(cfg.storage_eta_roundtrip)
    init_soc = storage_params.pop("init_soc", None)
    if init_soc is not None and "soc0_kwh" not in storage_params:
        storage_params["soc0_kwh"] = float(init_soc) * base_capacity_kwh

    installation_load_kw = float(cfg.n_lights) * float(cfg.light_power_kw)
    battery_power_cap_multiplier = cfg.battery_power_cap_multiplier_of_load
    power_cap_kw = (
        None
        if battery_power_cap_multiplier is None
        else float(battery_power_cap_multiplier) * installation_load_kw
    )

    merit_series = pd.Series(
        pd.to_numeric(merit_df["marginal_carbon_kgco2e_per_kwh"], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(merit_df["timestamp"]),
        name="merit_marginal_carbon",
    )
    aef_by_region = {region: merit_series for region in region_city_map.keys()}

    baseline_design_table = pd.read_csv(output_dir / "closing_analyses" / "design_decision_table.csv")
    baseline_lookup = baseline_design_table.set_index("design_label")

    rows: list[dict[str, Any]] = []
    city_rows: list[dict[str, Any]] = []
    for point in design_points:
        label = str(point["design_label"])
        solar = float(point["solar_panel_factor"])
        battery = float(point["battery_factor"])
        requested_power_kw = base_power_kw * battery
        storage = {
            **storage_params,
            "capacity_kwh": base_capacity_kwh * battery,
            "power_kw": requested_power_kw if power_cap_kw is None else min(requested_power_kw, power_cap_kw),
            "eta_roundtrip": eta_roundtrip,
        }
        city_df, _, _ = simulate_from_par_and_aef(
            par_df=par_df,
            aef_by_region=aef_by_region,
            region_city_map=region_city_map,
            par_to_kw_factor=effective_par_to_kw_factor * solar,
            storage_params=storage,
            load_mode=cfg.load_mode,
            n_lights=cfg.n_lights,
            light_power_kw=cfg.light_power_kw,
            analysis_years=cfg.analysis_years,
            lighting_threshold_par=cfg.lighting_threshold_par,
            align_strategy=cfg.paper_align_strategy,
        )
        totals = city_df[["baseline_emission_kg", "scenario_emission_kg", "emission_reduction_kg"]].sum()
        city_rows.extend(
            {
                "design_label": label,
                "solar_panel_factor": solar,
                "battery_factor": battery,
                "city": row["city"],
                "region": row["region"],
                "grid_only_emission_t_merit_order_mcf": float(row["baseline_emission_kg"]) / 1000.0,
                "pv_storage_emission_t_merit_order_mcf": float(row["scenario_emission_kg"]) / 1000.0,
                "abatement_t_merit_order_mcf": float(row["emission_reduction_kg"]) / 1000.0,
            }
            for row in city_df.to_dict(orient="records")
        )
        baseline = baseline_lookup.loc[label]
        rows.append(
            {
                "design_label": label,
                "solar_panel_factor": solar,
                "battery_factor": battery,
                "grid_only_emission_t_average_aef": float(baseline["grid_only_emission_t"]),
                "pv_storage_emission_t_average_aef": float(baseline["pv_storage_emission_t"]),
                "abatement_t_average_aef": float(baseline["abatement_t"]),
                "grid_only_emission_t_merit_order_mcf": float(totals["baseline_emission_kg"]) / 1000.0,
                "pv_storage_emission_t_merit_order_mcf": float(totals["scenario_emission_kg"]) / 1000.0,
                "abatement_t_merit_order_mcf": float(totals["emission_reduction_kg"]) / 1000.0,
                "abatement_delta_t_merit_minus_average": float(
                    (totals["emission_reduction_kg"] / 1000.0) - baseline["abatement_t"]
                ),
                "abatement_ratio_merit_over_average": (
                    float((totals["emission_reduction_kg"] / 1000.0) / baseline["abatement_t"])
                    if float(baseline["abatement_t"]) != 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(city_rows)


def main() -> None:
    cfg = get_config()
    output_dir = cfg.paper_output_dir
    analysis_dir = output_dir / "merit_order_marginal"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    power_dir = Path(cfg.power_dir)

    unit_generation = _normalize_wide_csv(power_dir / "unit_generation.csv")
    unit_capacity = _normalize_wide_csv(power_dir / "unit_capacity.csv")
    category_generation = _normalize_wide_csv(power_dir / "category_generation.csv")

    unit_meta = _build_unit_meta(unit_generation, unit_capacity)
    unit_meta.to_csv(analysis_dir / "annual_unit_cost_carbon_proxy.csv", index=False)

    merit_df = _build_marginal_timeseries(unit_generation, unit_meta)
    merit_df.to_csv(analysis_dir / "marginal_carbon_timeseries.csv", index=False)

    design_comparison, city_design_comparison = _simulate_designs_under_merit_mcf(cfg, output_dir, merit_df)
    design_comparison.to_csv(analysis_dir / "representative_design_mcf_comparison.csv", index=False)
    city_design_comparison.to_csv(analysis_dir / "representative_design_city_mcf_comparison.csv", index=False)

    average_operational_ef = _average_operational_ef(category_generation)
    merit_mean = float(pd.to_numeric(merit_df["marginal_carbon_kgco2e_per_kwh"], errors="coerce").mean())
    group_share = (
        merit_df["marginal_dispatch_group"]
        .fillna("None")
        .value_counts(normalize=True, dropna=False)
        .rename_axis("marginal_dispatch_group")
        .reset_index(name="share_of_timestamps")
    )
    group_share.to_csv(analysis_dir / "marginal_dispatch_group_share.csv", index=False)

    summary = {
        "interpretation": (
            "Proxy merit-order marginal carbon analysis using Taipower 113 fuel-cost ordering "
            "and per-unit generation outputs. This is an alternative operational-carbon lens and "
            "does not replace the paper's regional average-emission-factor framework."
        ),
        "average_operational_ef_kgco2e_per_kwh": average_operational_ef,
        "average_proxy_marginal_carbon_kgco2e_per_kwh": merit_mean,
        "marginal_over_average_ratio": merit_mean / average_operational_ef if average_operational_ef > 0 else None,
        "marginal_tranche_share_assumption": MARGINAL_TRANCHE_SHARE,
        "timestamp_count": int(len(merit_df)),
        "unit_count_in_merit_stack": int(unit_meta["included_in_merit_order"].sum()),
        "excluded_from_main_grid_merit_keywords": list(EXCLUDED_MAIN_GRID_MERIT_KEYWORDS),
        "dispatch_group_share": group_share.to_dict(orient="records"),
        "cost_proxy_ntd_per_kwh": MERIT_COST_PROXY,
        "emission_proxy_kgco2e_per_kwh": EMISSION_PROXY,
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
