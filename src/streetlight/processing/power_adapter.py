"""
Power data adapter for converting parquet format to regional CSV format.

This module converts the new parquet-based power data format
(from taipower_gen/powerInfo-consolidator) to the regional CSV format
expected by the AEF pipeline.

New format (parquet):
    - 80M+ records with unit-level data
    - Columns: timestamp, plant_name, unit_id, energy_type, used_mw, etc.
    - mapping_names column contains plant/location names

Old format (CSV):
    - 5 CSV files: north_unit_generation.csv, central_unit_generation.csv, etc.
    - Rows: timestamps
    - Columns: Each generation unit (e.g., "Coal-林口#1", "Hydro-石門#1")
"""

import json
import re

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings

TAIPOWER_SOURCE_TZ = "Asia/Taipei"
CANONICAL_TIMESTAMP_TZ = "UTC"


# Storage actor classification: aggregate Taipower storage units into the
# two grid-side actors (PHS, BESS) used by the shared carbon-energy pool.
# See docs/engineering/grid_storage_pool_design.md.
PHS_PLANT_KEYWORDS = ("明潭", "大觀二", "大觀")  # pumped hydro plants
BESS_PLANT_KEYWORDS = ("電池", "儲能")  # Taipower aggregate lithium battery


def _classify_storage_column(colname: str) -> Optional[str]:
    """Return 'PHS' / 'BESS' if a Storage-* column name matches a known actor."""
    if not colname.startswith("Storage-"):
        return None
    suffix = colname[len("Storage-"):]
    plant = re.split(r"#", suffix, maxsplit=1)[0]
    if any(keyword in plant for keyword in PHS_PLANT_KEYWORDS):
        return "PHS"
    if any(keyword in plant for keyword in BESS_PLANT_KEYWORDS):
        return "BESS"
    return None


# Option Y: distributed energy types from 'other' region get redistributed
# across mainland regions (north/central/south/east). Source-backed fixed
# shares are used where Taipower publishes a better regional allocation basis;
# otherwise the allocator falls back to same-fuel named capacity.
DISTRIBUTED_ENERGY_TYPES = ("Solar", "Wind", "Co-Gen", "Biomass")

# Fixed regional shares for nationally aggregated fuels where Taipower reports
# capacity distribution outside the unit-level generation feed. For Co-Gen, the
# 2024 purchased-power capacity distribution is encoded here so the pipeline no
# longer depends on the archaeological spreadsheet. The Mailiao Plastic
# cogeneration exception follows Taipower's regional load-area note: it is tied
# to the central accounting boundary, while the Mailiao power plant remains
# south.
_COGEN_CAPACITY_KW_2024: Dict[str, float] = {
    "north": 994_287.0,
    "central": 2_779_958.0,
    "south": 1_156_443.0,
    "east": 8_900.0,
}
_COGEN_CAPACITY_TOTAL_KW_2024 = sum(_COGEN_CAPACITY_KW_2024.values())

# Taipower "各縣市再生能源別購入情形" reports county-level purchased renewable
# energy and feed-in capacity. The unresolved "Solar-其它購電太陽能" bucket is
# purchased solar generation, so the active allocation uses annual purchased
# solar kWh rather than installed capacity. Capacity is retained in the audit
# basis as a cross-check. Counties are aggregated to the same mainland
# grid-accounting regions used for regional generation balance. Hsinchu and
# Miaoli are assigned central at this county resolution because Taipower's
# regional boundary is near Hsinchu's Fengshan River; Yunlin is assigned south
# because the south block begins below the Zhuoshui River. Offshore island
# counties are retained as three disconnected load-serving zones in the audit
# basis but excluded from the mainland allocation denominator.
_SOLAR_PURCHASED_ENERGY_2024_BY_COUNTY: Dict[str, Dict[str, float]] = {
    "基隆市": {"feed_in_capacity_kw": 26_713.565, "purchased_kwh": 20_810_961.0},
    "台北市": {"feed_in_capacity_kw": 80_241.981, "purchased_kwh": 75_332_892.0},
    "新北市": {"feed_in_capacity_kw": 183_827.418, "purchased_kwh": 158_347_782.0},
    "桃園市": {"feed_in_capacity_kw": 803_635.667, "purchased_kwh": 778_462_889.0},
    "新竹市": {"feed_in_capacity_kw": 47_235.031, "purchased_kwh": 55_143_333.0},
    "新竹縣": {"feed_in_capacity_kw": 208_747.674, "purchased_kwh": 223_306_042.0},
    "苗栗縣": {"feed_in_capacity_kw": 376_371.969, "purchased_kwh": 425_549_875.0},
    "台中市": {"feed_in_capacity_kw": 801_052.016, "purchased_kwh": 868_797_806.0},
    "彰化縣": {"feed_in_capacity_kw": 1_752_994.998, "purchased_kwh": 1_822_288_953.0},
    "南投縣": {"feed_in_capacity_kw": 267_429.660, "purchased_kwh": 279_187_828.0},
    "雲林縣": {"feed_in_capacity_kw": 1_568_305.632, "purchased_kwh": 1_872_386_817.0},
    "嘉義市": {"feed_in_capacity_kw": 49_947.363, "purchased_kwh": 54_764_277.0},
    "嘉義縣": {"feed_in_capacity_kw": 1_100_497.819, "purchased_kwh": 1_113_421_503.0},
    "台南市": {"feed_in_capacity_kw": 2_668_601.893, "purchased_kwh": 3_040_073_673.0},
    "高雄市": {"feed_in_capacity_kw": 1_394_797.480, "purchased_kwh": 1_429_354_811.0},
    "屏東縣": {"feed_in_capacity_kw": 1_384_555.355, "purchased_kwh": 1_417_217_446.0},
    "宜蘭縣": {"feed_in_capacity_kw": 202_860.940, "purchased_kwh": 172_978_120.0},
    "花蓮縣": {"feed_in_capacity_kw": 200_849.619, "purchased_kwh": 198_777_740.0},
    "台東縣": {"feed_in_capacity_kw": 87_039.255, "purchased_kwh": 81_652_812.0},
    "澎湖縣": {"feed_in_capacity_kw": 71_970.660, "purchased_kwh": 53_034_394.0},
    "金門縣": {"feed_in_capacity_kw": 25_421.310, "purchased_kwh": 26_185_786.0},
    "連江縣": {"feed_in_capacity_kw": 70.290, "purchased_kwh": 73_681.0},
}
_SOLAR_PURCHASED_COUNTIES_BY_GRID_REGION: Dict[str, List[str]] = {
    "north": ["基隆市", "台北市", "新北市", "桃園市", "宜蘭縣"],
    "central": ["新竹市", "新竹縣", "苗栗縣", "台中市", "彰化縣", "南投縣"],
    "south": ["雲林縣", "嘉義市", "嘉義縣", "台南市", "高雄市", "屏東縣"],
    "east": ["花蓮縣", "台東縣"],
    "island_penghu": ["澎湖縣"],
    "island_kinmen": ["金門縣"],
    "island_lienchiang": ["連江縣"],
}
_MAINLAND_GRID_REGIONS = ("north", "central", "south", "east")
_OUTLYING_ISLAND_GRID_REGIONS = (
    "island_penghu",
    "island_kinmen",
    "island_lienchiang",
)
_SOLAR_PURCHASED_KWH_2024_BY_GRID_REGION: Dict[str, float] = {
    region: sum(
        _SOLAR_PURCHASED_ENERGY_2024_BY_COUNTY[county]["purchased_kwh"]
        for county in counties
    )
    for region, counties in _SOLAR_PURCHASED_COUNTIES_BY_GRID_REGION.items()
}
_SOLAR_PURCHASED_CAPACITY_KW_2024_BY_GRID_REGION: Dict[str, float] = {
    region: sum(
        _SOLAR_PURCHASED_ENERGY_2024_BY_COUNTY[county]["feed_in_capacity_kw"]
        for county in counties
    )
    for region, counties in _SOLAR_PURCHASED_COUNTIES_BY_GRID_REGION.items()
}
_SOLAR_PURCHASED_MAINLAND_KWH_2024 = sum(
    _SOLAR_PURCHASED_KWH_2024_BY_GRID_REGION[region]
    for region in _MAINLAND_GRID_REGIONS
)
_SOLAR_PURCHASED_OUTLYING_ISLAND_KWH_2024 = sum(
    _SOLAR_PURCHASED_KWH_2024_BY_GRID_REGION[region]
    for region in _OUTLYING_ISLAND_GRID_REGIONS
)
_SOLAR_PURCHASED_OUTLYING_ISLAND_CAPACITY_KW_2024 = sum(
    _SOLAR_PURCHASED_CAPACITY_KW_2024_BY_GRID_REGION[region]
    for region in _OUTLYING_ISLAND_GRID_REGIONS
)
_FIXED_DISTRIBUTED_RATIOS: Dict[str, Dict[str, float]] = {
    "Solar": {
        region: _SOLAR_PURCHASED_KWH_2024_BY_GRID_REGION[region]
        / _SOLAR_PURCHASED_MAINLAND_KWH_2024
        for region in _MAINLAND_GRID_REGIONS
    },
    "Co-Gen": {
        region: capacity_kw / _COGEN_CAPACITY_TOTAL_KW_2024
        for region, capacity_kw in _COGEN_CAPACITY_KW_2024.items()
    },
}
_FIXED_DISTRIBUTED_RATIO_SOURCE: Dict[str, str] = {
    "Solar": "fixed_solar_county_purchased_generation_share_2024",
    "Co-Gen": "fixed_cogeneration_capacity_share_2024",
}
_FIXED_DISTRIBUTED_RATIO_BASIS: Dict[str, dict] = {
    "Solar": {
        "basis": "taipower_2024_county_purchased_solar_generation_kwh",
        "source_url": "https://data.gov.tw/dataset/29936",
        "source_data_url": "https://service.taipower.com.tw/data/opendata/apply/file/d007011/001.csv",
        "source_note": (
            "Taipower county renewable-energy purchase table; year 113 rows, "
            "solar purchased kWh used for the active allocation and feed-in "
            "capacity retained as provenance."
        ),
        "source_year_roc": 113,
        "source_fields": [
            "太陽光電躉購容量(KW)",
            "太陽光電本年累計購電度數(度)",
        ],
        "counties_by_grid_region": _SOLAR_PURCHASED_COUNTIES_BY_GRID_REGION,
        "purchased_kwh_by_grid_region": _SOLAR_PURCHASED_KWH_2024_BY_GRID_REGION,
        "capacity_kw_by_grid_region": _SOLAR_PURCHASED_CAPACITY_KW_2024_BY_GRID_REGION,
        "mainland_purchased_kwh": _SOLAR_PURCHASED_MAINLAND_KWH_2024,
        "excluded_island_purchased_kwh": _SOLAR_PURCHASED_OUTLYING_ISLAND_KWH_2024,
        "excluded_island_capacity_kw": _SOLAR_PURCHASED_OUTLYING_ISLAND_CAPACITY_KW_2024,
        "excluded_island_purchased_kwh_by_load_serving_zone": {
            region: _SOLAR_PURCHASED_KWH_2024_BY_GRID_REGION[region]
            for region in _OUTLYING_ISLAND_GRID_REGIONS
        },
        "excluded_island_capacity_kw_by_load_serving_zone": {
            region: _SOLAR_PURCHASED_CAPACITY_KW_2024_BY_GRID_REGION[region]
            for region in _OUTLYING_ISLAND_GRID_REGIONS
        },
    },
    "Co-Gen": {
        "basis": "taipower_2024_purchased_power_cogeneration_capacity_kw",
        "source_url": "https://www.taipower.com.tw/2289/2363/2380/2385/10623/normalPost",
        "capacity_kw_by_region": _COGEN_CAPACITY_KW_2024,
    },
}

# Fallback share ratios for fuels with no mainland capacity and no stronger
# source-backed distribution. Biomass remains a fallback case: if
# `regional_demand_path` is supplied, observed 2024 regional load share is used
# first; otherwise this industrial-distribution heuristic is used.
_FALLBACK_DISTRIBUTED_RATIOS: Dict[str, Dict[str, float]] = {
    "Biomass": {"north": 0.20, "central": 0.30, "south": 0.40, "east": 0.10},
}


def taipower_local_to_utc_naive(timestamps: pd.Series) -> pd.Series:
    """
    Convert Taipower local-naive timestamps to the pipeline's UTC-naive index.

    Taipower generation and regional-demand archives label timestamps in Taiwan
    local time, while PAR handoff files are UTC. Downstream CSVs remain
    timezone-naive for compatibility, but their timestamp basis is UTC.
    """
    parsed = pd.to_datetime(timestamps, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        localized = parsed.dt.tz_localize(TAIPOWER_SOURCE_TZ)
    else:
        localized = parsed
    return localized.dt.tz_convert(CANONICAL_TIMESTAMP_TZ).dt.tz_localize(None)


def _compute_regional_load_share(
    regional_demand_path: Path,
    *,
    year: int = 2024,
    source_filter: str = "power_demand",
) -> Optional[Dict[str, float]]:
    """
    Derive regional load share ratios from Taipower regional_power_load.parquet.

    Returns {north, central, south, east} → ratio summing to 1.0, or None if
    the data is unavailable. Used as the objective fallback for distributed
    fuels that lack both mainland capacity and a stronger fixed capacity
    distribution.
    """
    regional_demand_path = Path(regional_demand_path)
    if not regional_demand_path.exists():
        return None
    try:
        df = pd.read_parquet(regional_demand_path)
    except Exception:
        return None

    required = {"date", "time", "source", "north_load", "central_load", "south_load", "east_load"}
    if not required.issubset(df.columns):
        return None

    ts = _parse_regional_demand_timestamp(df)
    df = df.assign(ts=ts)
    df = df[(df["ts"] >= f"{year}-01-01") & (df["ts"] < f"{year + 1}-01-01")]
    if source_filter is not None:
        df = df[df["source"] == source_filter]
    if df.empty:
        return None

    loads: Dict[str, float] = {}
    for region in ("north", "central", "south", "east"):
        col = f"{region}_load"
        loads[region] = float(pd.to_numeric(df[col], errors="coerce").mean())

    total = sum(loads.values())
    if total <= 0:
        return None
    return {r: loads[r] / total for r in loads}


def _is_regional_generation_column(colname: str) -> bool:
    """
    Return True for plant/fuel generation columns used in balance checks.

    Storage is counted through the de-duplicated PHS/BESS aggregate columns.
    The raw Storage-* unit columns remain in regional CSVs for audit, but are
    excluded here to avoid double-counting.
    """
    return (
        colname != "timestamp"
        and not colname.startswith("F_")
        and not colname.startswith("Storage-")
    )


def _regional_generation_balance_diagnostics(
    regional_pivots: Dict[str, pd.DataFrame],
    regional_demand_path: Optional[Path],
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    source_filter: str = "power_demand",
    unit_scale: float = 10.0,
) -> dict:
    """
    Compare reconstructed regional generation with Taipower regional totals.

    The flow proxy is reconstructed from the same regional generation-load
    archive. Keeping this diagnostic in the generation conversion report makes
    plant-to-region mapping drift visible before a full Paper 1 rerun.
    """
    if regional_demand_path is None:
        return {"status": "skipped", "reason": "no regional_demand_path"}
    regional_demand_path = Path(regional_demand_path)
    if not regional_demand_path.exists():
        return {"status": "skipped", "reason": "regional_demand_path missing"}

    regions = ("north", "central", "south", "east")
    required = {"date", "time", "source", *(f"{region}_gen" for region in regions)}
    try:
        demand = pd.read_parquet(regional_demand_path)
    except Exception as exc:
        return {"status": "skipped", "reason": f"regional demand unreadable: {exc}"}

    missing = sorted(required - set(demand.columns))
    if missing:
        return {
            "status": "skipped",
            "reason": "regional demand missing required columns",
            "missing_columns": missing,
        }

    demand = demand[list(required)].copy()
    demand["timestamp"] = _parse_regional_demand_timestamp(demand)
    if source_filter is not None:
        demand = demand[demand["source"] == source_filter]
    if start_date is not None:
        demand = demand[demand["timestamp"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        demand = demand[demand["timestamp"] <= pd.Timestamp(end_date)]
    if demand.empty:
        return {"status": "skipped", "reason": "no regional demand rows in requested window"}

    target = demand.set_index("timestamp")[[f"{region}_gen" for region in regions]]
    target = target.apply(pd.to_numeric, errors="coerce") * unit_scale
    target.columns = list(regions)

    model_parts = {}
    for region in regions:
        pivot = regional_pivots.get(region)
        if pivot is None or pivot.empty:
            continue
        if "timestamp" in pivot.columns:
            values = pivot.set_index("timestamp")
        else:
            values = pivot.copy()
        values.index = pd.to_datetime(values.index)
        gen_cols = [col for col in values.columns if _is_regional_generation_column(str(col))]
        if not gen_cols:
            continue
        model_parts[region] = values[gen_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    if set(model_parts) != set(regions):
        return {
            "status": "skipped",
            "reason": "missing reconstructed regional generation",
            "available_regions": sorted(model_parts),
        }

    model = pd.DataFrame(model_parts)
    common = model.index.intersection(target.index)
    if len(common) == 0:
        return {"status": "skipped", "reason": "no common timestamps"}

    by_region = {}
    for region in regions:
        diff = model.loc[common, region] - target.loc[common, region]
        by_region[region] = {
            "model_mean_mw": float(model.loc[common, region].mean()),
            "taipower_mean_mw": float(target.loc[common, region].mean()),
            "mean_error_mw": float(diff.mean()),
            "mean_abs_error_mw": float(diff.abs().mean()),
            "p95_abs_error_mw": float(diff.abs().quantile(0.95)),
        }

    return {
        "status": "computed",
        "source_path": str(regional_demand_path),
        "source_filter": source_filter,
        "target_unit_scale": unit_scale,
        "generation_column_policy": {
            "included_storage_aggregates": ["PHS", "BESS"],
            "excluded_prefixes": ["F_", "Storage-"],
        },
        "aligned_timestamps": int(len(common)),
        "by_region": by_region,
    }


def _allocate_other_to_mainland(
    regional_pivots: Dict[str, pd.DataFrame],
    capacity_pivots: Dict[str, pd.DataFrame],
    *,
    demand_share: Optional[Dict[str, float]] = None,
) -> Dict[str, dict]:
    """
    Redistribute 'other' region generation of distributed energy types into
    mainland regions by capacity-share ratio.

    For each energy type in DISTRIBUTED_ENERGY_TYPES:
      1. Compute mainland share ratio = mean capacity of that fuel per region / total
      2. Sum 'other' columns of that fuel across plants → distributed_total(t)
      3. For each mainland region, add column `{ET}-_other_allocated` =
         distributed_total(t) × ratio[region]
      4. Drop redistributed columns from 'other' pivot (they're now elsewhere)

    Modifies regional_pivots in place. Returns audit dict.
    """
    other = regional_pivots.get("other")
    if other is None or other.empty:
        return {}

    mainland = ["north", "central", "south", "east"]
    audit: Dict[str, dict] = {}

    for et in DISTRIBUTED_ENERGY_TYPES:
        # Mainland capacity share for this energy type
        cap_per_region = {}
        for region in mainland:
            cap_df = capacity_pivots.get(region)
            if cap_df is None or cap_df.empty:
                cap_per_region[region] = 0.0
                continue
            cols = [c for c in cap_df.columns if c.startswith(f"{et}-")]
            if cols:
                cap_per_region[region] = float(cap_df[cols].mean().sum())
            else:
                cap_per_region[region] = 0.0

        total_cap = sum(cap_per_region.values())

        # 'other' columns of this fuel
        other_cols = [c for c in other.columns if c.startswith(f"{et}-")]
        if not other_cols:
            audit[et] = {"status": "skipped", "reason": "no_other_columns"}
            continue

        # Prefer source-backed fixed distributions where available.
        # Otherwise use same-fuel named mainland capacity, then regional
        # electricity load share derived from Taipower demand data, then the
        # fallback industrial-distribution heuristic.
        ratio_basis = None
        fixed = _FIXED_DISTRIBUTED_RATIOS.get(et)
        if fixed is not None:
            ratios = dict(fixed)
            ratio_source = _FIXED_DISTRIBUTED_RATIO_SOURCE.get(et, "fixed_capacity_share")
            ratio_basis = _FIXED_DISTRIBUTED_RATIO_BASIS.get(et)
        elif total_cap > 0:
            ratios = {r: cap_per_region[r] / total_cap for r in mainland}
            ratio_source = "capacity_share"
            ratio_basis = {
                "basis": "same_fuel_named_mainland_capacity_mw",
                "capacity_mw_by_region": cap_per_region,
                "mainland_capacity_mw": round(total_cap, 2),
            }
        elif demand_share is not None:
            ratios = dict(demand_share)
            ratio_source = "regional_load_share"
            ratio_basis = {"basis": "taipower_regional_load_share"}
        else:
            fb = _FALLBACK_DISTRIBUTED_RATIOS.get(et)
            if fb is None:
                audit[et] = {"status": "skipped", "reason": "no_mainland_capacity"}
                continue
            ratios = dict(fb)
            ratio_source = "fallback_industrial_share"
            ratio_basis = {"basis": "fallback_industrial_distribution_share"}

        # Sum 'other' generation per timestamp across plants of this fuel
        # (preserve sign: cogen never goes negative; solar/wind also typically positive)
        other_total = other[other_cols].sum(axis=1, min_count=1)
        other_total_numeric = pd.to_numeric(other_total, errors="coerce")

        for region, ratio in ratios.items():
            if ratio <= 0 or region not in regional_pivots:
                continue
            alloc_col = f"{et}-_other_allocated"
            target = regional_pivots[region]
            if alloc_col in target.columns:
                target[alloc_col] = target[alloc_col].fillna(0.0) + other_total * ratio
            else:
                target[alloc_col] = other_total * ratio

        # Drop redistributed columns from 'other'
        regional_pivots["other"] = other.drop(columns=other_cols)
        other = regional_pivots["other"]

        audit_entry = {
            "status": "redistributed",
            "ratio_source": ratio_source,
            "ratios": ratios,
            "other_total_mean_mw": float(other_total_numeric.mean()),
            "other_total_p95_mw": float(other_total_numeric.quantile(0.95)),
            "other_total_max_mw": float(other_total_numeric.max()),
            "allocated_mean_mw_by_region": {
                region: float(other_total_numeric.mean() * ratios.get(region, 0.0))
                for region in mainland
            },
            "other_columns_dropped": other_cols,
            "other_columns_count": int(len(other_cols)),
        }
        if ratio_basis is not None:
            audit_entry["ratio_basis"] = ratio_basis
        audit[et] = audit_entry

    return audit


def _attach_phs_bess_columns(region_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate unit-level Storage-* columns into a single PHS column and a
    single BESS column, preserving the underlying unit columns for audit.

    Sign convention preserved: positive = grid-side discharge, negative =
    charge from grid (matches taipower storage / storage_load).
    """
    out = region_pivot.copy()
    phs_cols = [c for c in out.columns if _classify_storage_column(c) == "PHS"]
    bess_cols = [c for c in out.columns if _classify_storage_column(c) == "BESS"]

    out["PHS"] = (
        out[phs_cols].sum(axis=1, min_count=1) if phs_cols else 0.0
    )
    out["BESS"] = (
        out[bess_cols].sum(axis=1, min_count=1) if bess_cols else 0.0
    )
    return out


def _aggregate_unit_values(
    df: pd.DataFrame,
    value_col: str,
    *,
    within_source: str,
    across_sources: str = "mean",
) -> pd.DataFrame:
    """
    Collapse rows onto one value per normalized timestamp and unit.

    Taipower can provide multiple scrape records within the same 10-minute
    bucket. Those are repeated observations and should be averaged. Storage is
    different: `storage` and `storage_load` are distinct rows in the same scrape
    and must be summed first to preserve net grid exchange.
    """
    required = {"timestamp", "unit_name", value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns for unit aggregation: {sorted(missing)}")

    if "_source_timestamp" not in df.columns:
        return (
            df.groupby(["timestamp", "unit_name"], dropna=False)[value_col]
            .agg(across_sources)
            .reset_index()
        )

    if within_source == "net_generation" and "energy_type" in df.columns:
        per_energy = (
            df.groupby(
                ["timestamp", "_source_timestamp", "unit_name", "energy_type"],
                dropna=False,
            )[value_col]
            .mean()
            .reset_index()
        )
        by_source = (
            per_energy.groupby(
                ["timestamp", "_source_timestamp", "unit_name"],
                dropna=False,
            )[value_col]
            .sum()
            .reset_index()
        )
    else:
        by_source = (
            df.groupby(["timestamp", "_source_timestamp", "unit_name"], dropna=False)[value_col]
            .agg(within_source)
            .reset_index()
        )
    return (
        by_source.groupby(["timestamp", "unit_name"], dropna=False)[value_col]
        .agg(across_sources)
        .reset_index()
    )


def _category_generation_from_unit_values(unit_values: pd.DataFrame) -> pd.DataFrame:
    """Build fuel-category generation from already de-duplicated unit values."""
    if unit_values.empty:
        return pd.DataFrame(columns=["timestamp"])

    df = unit_values.copy()
    known_types = sorted({str(value) for value in ENERGY_TYPE_MAP.values()}, key=len, reverse=True)

    def category_from_unit(unit_name: object) -> str:
        text = str(unit_name)
        for energy_type in known_types:
            if text.startswith(f"{energy_type}-"):
                return energy_type
        return text.split("-", 1)[0]

    df["energy_type"] = df["unit_name"].map(category_from_unit)
    category = (
        df.groupby(["timestamp", "energy_type"], dropna=False)["used_mw"]
        .sum()
        .reset_index()
        .pivot(index="timestamp", columns="energy_type", values="used_mw")
        .reset_index()
    )
    category.columns.name = None
    return category


# Mapping from plant names to the regional accounting boundary used by the AEF
# pipeline. Taipower's area-load page defines regional generation primarily by
# north/central/south/east generation area, with explicit EHV interconnection
# exceptions for Heping, Bihai, Mailiao cogeneration, and Mailiao power plant.
# This mapping therefore follows the regional generation-accounting boundary,
# not county labels alone.
_PLANT_TO_REGION_RAW: Dict[str, str] = {                                                                                                         
      # North (北部)
      '核一': 'north',
      '核二': 'north',
      '林口': 'north',
      '大潭': 'north',
      '協和': 'north',
      '新桃': 'north',
      '石門': 'north',
      '桂山': 'north',
      '翡翠': 'north',
      '蘭陽': 'north',
      '粗坑': 'north',
      '烏來': 'north',
      '義興': 'north',
      '國光': 'north',
      '海湖': 'north',
      '觀園': 'north',
      '觀威': 'north',
      '桃威': 'north',
      '新屋': 'north',
      '觀音': 'north',
      '北部小水力': 'north',
      '恆水創電': 'north',
      '台電自有地熱': 'north',
      '購電地熱': 'north',
      '碧海': 'north',
      '核一Gas1': 'north',
      '核一Gas2': 'north',
      '核二Gas1': 'north',
      '核二Gas2': 'north',
      '和平': 'north',

      # Named wind farms follow grid-accounting evidence, not county labels
      # alone. Specific Miaoli wind projects are tied to the New-Taoyuan /
      # Yingpan side of the grid and reconcile against Taipower's regional
      # generation archive as north. This exception is intentionally narrow:
      # Tongxiao remains central, and Changhua offshore wind remains central.
      '海洋竹南': 'north',
      '苗栗大鵬': 'north',
      '苗栗竹南': 'north',
      '苗栗通苑': 'north',
      '海能風': 'north',
      '龍威後龍': 'north',
      '崎威崎頂': 'north',

      # Changhua offshore wind remains central in the regional archive.
      '離岸一期': 'central',
      '離岸二期': 'central',
      '中能風': 'central',
      '芳一風': 'central',
      '芳二風': 'central',
      '彰品風': 'central',
      '沃南風': 'central',
      '沃四風': 'central',
      '鹿威鹿港': 'central',

      # Central (中部)
      '台中': 'central',
      '通霄': 'central',
      '大甲溪': 'central',
      '明潭': 'central',
      '大觀': 'central',
      '萬大': 'central',
      '德基': 'central',
      '谷關': 'central',
      '青山': 'central',
      '天輪': 'central',
      '馬鞍': 'central',
      '卓蘭': 'central',
      '水裡': 'central',
      '水里': 'central',
      '后里': 'central',
      '東勢': 'central',
      '中部小水力': 'central',
      '達觀': 'central',
      '松林': 'central',
      '武界': 'central',
      '梨山': 'central',
      '名間': 'central',
      '彰工': 'central',
      '彰濱光': 'central',
      '鹿威彰濱': 'central',
      '星彰': 'central',
      '王功': 'central',
      '中威大安': 'central',
      '星元': 'central',
      '大觀一': 'central',
      '大觀二': 'central',
      '鉅工': 'central',  # 集集鉅工水力電廠

      # Greater Changhua offshore wind remains central in the regional archive.
      # Yunlin offshore wind (允湖/允西) is assigned in the south block below.
      '沃一風': 'central',
      '沃二風': 'central',

      # South (南部)
      '興達': 'south',
      '大林': 'south',
      '南部': 'south',
      '嘉義': 'south',
      '嘉惠': 'south',
      '豐德': 'south',
      '麥寮': 'south',
      '創維風': 'south',
      '創維麥寮': 'south',
      '禾風麥寮': 'south',
      '四湖': 'south',
      '雲麥': 'south',
      '新源崙背': 'south',
      '允湖': 'south',
      '允西': 'south',
      '核三': 'south',
      '核三Gas1': 'south',
      '核三Gas2': 'south',
      '南部小水力': 'south',
      '高屏': 'south',
      '高雄': 'south',
      '興達新': 'south',
      '舊興達': 'south',
      '南鹽': 'south',
      '台中龍井': 'central',
      '嘉南': 'south',
      '曾文': 'south',

      # East (東部)
      '東部': 'east',
      '立霧': 'east',
      '東部小水力': 'east',
      '清水地熱': 'north',
      '東興': 'east',
      '龍澗': 'east',
      '卑南': 'east',
      '捷祥關山': 'east',

      # Disconnected outlying-island load-serving zones. These are not
      # assigned by administrative broad labels; they follow the island grid
      # served by the named generation proxy visible in Taipower data.
      '澎湖': 'island_penghu',
      '七美': 'island_penghu',
      '望安': 'island_penghu',
      '虎井': 'island_penghu',
      '湖西': 'island_penghu',
      '七美二期': 'island_penghu',
      '澎湖尖山': 'island_penghu',
      '澎湖湖西': 'island_penghu',
      '中屯': 'island_penghu',

      '金門': 'island_kinmen',
      '金門塔山': 'island_kinmen',
      '金門金沙': 'island_kinmen',

      '連江': 'island_lienchiang',
      '馬祖': 'island_lienchiang',
      '東引': 'island_lienchiang',
      '馬祖珠山': 'island_lienchiang',
      '珠山': 'island_lienchiang',
      '南竿': 'island_lienchiang',
      '北竿': 'island_lienchiang',

      # Lanyu and Green Island are Taitung sub-county microgrids. Paper 1 has
      # no township-level load/PAR rows, so keep them out of the county-level
      # modeled zones unless a future sub-county workflow is added.
      '蘭嶼': 'island_other',
      '綠島': 'island_other',
      '旭光': 'island_other',
      '離島': 'island_other',
      '離島其他': 'island_other',

      # Grid-side aggregate batteries (台電鋰電池聚合代號) — virtually attached
      # to central region for AEF accounting; see grid_storage_pool_design.md
      '電池': 'central',
      '儲能': 'central',

      # Other
      '汽電共生': 'other',
      '太陽能': 'other',
      '風力': 'other',
      '地熱': 'other',
      '生質能': 'other',
  }

# Validate no duplicate keys exist at module load time
def _validate_no_duplicates(mapping: Dict[str, str], source_name: str) -> Dict[str, str]:
    """Validate that the mapping has no duplicate keys and warn if found."""
    seen = {}
    duplicates = {}
    for key, value in mapping.items():
        if key in seen:
            if key not in duplicates:
                duplicates[key] = [seen[key]]
            duplicates[key].append(value)
        else:
            seen[key] = value

    if duplicates:
        dup_list = [f"'{k}': {v}" for k, v in duplicates.items()]
        warnings.warn(
            f"Duplicate keys found in {source_name}: {dup_list}. "
            f"Later values override earlier ones. Using last occurrence.",
            RuntimeWarning,
            stacklevel=2
        )

    # Return the mapping as-is (Python keeps last occurrence)
    return mapping

PLANT_TO_REGION: Dict[str, str] = _validate_no_duplicates(_PLANT_TO_REGION_RAW, "PLANT_TO_REGION")

# List of special categories that should go to other_unit_generation.csv
OTHER_CATEGORIES = [
    '汽電共生',
    '太陽能',
    '購電',
    '地熱',
    '生質能',
    '儲能',
    '電池',
]


def infer_region_from_plant_name(plant_name: str) -> str:
    """
    Infer region from plant name using keyword matching.

    Args:
        plant_name: Plant name from mapping_names column

    Returns:
        Region code: mainland, split outlying-island, island_other, or other.
    """
    # Direct mapping first
    if plant_name in PLANT_TO_REGION:
        return PLANT_TO_REGION[plant_name]

    # Keyword matching
    for keyword, region in PLANT_TO_REGION.items():
        if keyword in plant_name:
            return region

    # Special patterns
    if any(x in plant_name for x in ['北部', 'North', 'north']):
        return 'north'
    if any(x in plant_name for x in ['中部', 'Central', 'central']):
        return 'central'
    if any(x in plant_name for x in ['南部', 'South', 'south']):
        return 'south'
    if any(x in plant_name for x in ['東部', 'East', 'east']):
        return 'east'
    if any(x in plant_name for x in ['澎湖']):
        return 'island_penghu'
    if any(x in plant_name for x in ['金門']):
        return 'island_kinmen'
    if any(x in plant_name for x in ['連江', '馬祖']):
        return 'island_lienchiang'
    if any(x in plant_name for x in ['離島', 'Island', 'island', '蘭嶼', '綠島']):
        return 'island_other'

    # Default to other for renewable / distributed generation
    return 'other'


# Normalize energy type names — keys must match the EF dictionary in
# streetlight.aef.calculator.AEFCalculator.default_emission_factors.
# IPP plants get distinct IPP-* names for traceability; under IPCC AR5 Annex III
# lifecycle medians IPP-Coal/IPP-LNG share the same EF as their utility-side
# counterparts (PC 0.820, CCGT 0.490). See config/config.yaml for citations and
# docs/engineering/grid_storage_pool_design.md for storage handling.
ENERGY_TYPE_MAP = {
    'coal': 'Coal',
    'ipp_coal': 'IPP-Coal',
    'lng': 'LNG',
    'ipp_lng': 'IPP-LNG',
    'gas': 'LNG',  # legacy energy_type 'gas' folded into LNG
    'oil': 'Oil',
    'hydro': 'Hydro',
    'wind': 'Wind',
    'solar': 'Solar',
    'geothermal': 'Geothermal',
    'biofuel': 'Biomass',
    'biomass': 'Biomass',
    'other_renewable': 'Biomass',
    'nuclear': 'Nuclear',
    'pumped': 'PHS',  # legacy pumped storage rows fold into PHS pool
    'pumping_gen': 'PHS',
    'pumping_load': 'PHS',
    'diesel': 'Diesel',
    'cogen': 'Co-Gen',  # was 'Cogen' which mismatched EF dict — fixed
    'storage': 'Storage',  # gets re-aggregated into PHS / BESS columns downstream
    'storage_load': 'Storage',
    'other': 'Other',  # not in EF dict; intentionally dropped from AEF
}


def normalize_energy_type(energy_type: str) -> str:
    """Normalize energy type to match original format."""
    if pd.isna(energy_type):
        return 'Other'
    et = str(energy_type).lower()
    return ENERGY_TYPE_MAP.get(et, energy_type)


def clean_plant_name(plant_name: str) -> str:
    """
    Clean plant name by removing footnote markers like (註4), (註10), etc.

    This ensures that "金門塔山(註4)" and "金門塔山" are treated as the same unit.
    Also handles HTML entities like &amp; -> &.

    Args:
        plant_name: Raw plant name from the data source

    Returns:
        Cleaned plant name without footnote markers
    """
    if pd.isna(plant_name):
        return 'Unknown'

    # Convert to string and fix HTML entities
    name = str(plant_name).replace('&amp;', '&')

    # Remove footnote patterns like (註4), (註10), (注4), etc.
    import re
    cleaned = re.sub(r'[（(][註注][0-9]+[）)]', '', name)
    return cleaned.strip()


def create_unit_name(plant_name: str, unit_id: str, energy_type: str) -> str:
    """
    Create unit identifier in the format expected by AEF pipeline.

    Args:
        plant_name: Plant name
        unit_id: Unit identifier
        energy_type: Energy type (e.g., 'Coal', 'LNG', 'Hydro')

    Returns:
        Unit name string (e.g., 'Coal-林口#1')
    """
    # Normalize energy type (capitalize first letter)
    et_normalized = normalize_energy_type(energy_type)

    # Clean unit_id (remove .0 suffix if present from float conversion)
    if unit_id and str(unit_id) != 'nan' and str(unit_id) != 'None':
        unit_clean = str(unit_id).rstrip('.0')
        if unit_clean:
            return f"{et_normalized}-{plant_name}#{unit_clean}"

    return f"{et_normalized}-{plant_name}"


def _expected_time_index(
    start_date: Optional[str],
    end_date: Optional[str],
    *,
    freq: str = "10min",
) -> pd.DatetimeIndex:
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date are required for gap-controlled output")
    return pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq=freq)


def _interpolate_short_numeric_gaps(
    wide_df: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
    *,
    max_run: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Reindex a numeric wide table, fill only short internal gaps, and flag fills.

    Rows that remain entirely missing after imputation are removed from the
    returned value table so downstream AEF keeps using intersection timestamps.
    """
    if "timestamp" in wide_df.columns:
        df = wide_df.set_index("timestamp")
    else:
        df = wide_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    value_cols = list(df.columns)

    original_index = pd.DatetimeIndex(df.index.unique()).sort_values()
    missing_before = expected_index.difference(original_index)

    df = df.reindex(expected_index)
    before_missing = df.isna()
    filled = df.interpolate(
        method="time",
        axis=0,
        limit=max_run,
        limit_direction="both",
        limit_area="inside",
    )

    for col in value_cols:
        missing = before_missing[col]
        if not missing.any():
            continue
        group_id = (missing != missing.shift(fill_value=False)).cumsum()
        run_lengths = missing.groupby(group_id).transform("sum")
        long_missing = missing & (run_lengths > max_run)
        filled.loc[long_missing, col] = pd.NA

    flags = before_missing & filled.notna()
    remaining_missing = expected_index.difference(filled.dropna(how="all").index)
    filled = filled.dropna(how="all").reset_index().rename(columns={"index": "timestamp"})
    flags = flags.reset_index().rename(columns={"index": "timestamp"})
    return filled, flags, missing_before, remaining_missing


def _fill_short_categorical_gaps(
    wide_df: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
    *,
    max_run: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill short categorical gaps only when both neighboring values agree.
    """
    if "timestamp" in wide_df.columns:
        df = wide_df.set_index("timestamp")
    else:
        df = wide_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().reindex(expected_index)
    before_missing = df.isna()

    fwd = df.ffill()
    bwd = df.bfill()
    filled = df.copy()
    for col in df.columns:
        missing = before_missing[col]
        if not missing.any():
            continue
        group_id = (missing != missing.shift(fill_value=False)).cumsum()
        run_lengths = missing.groupby(group_id).transform("sum")
        can_fill = missing & (run_lengths <= max_run) & fwd[col].notna() & (fwd[col] == bwd[col])
        filled.loc[can_fill, col] = fwd.loc[can_fill, col]

    flags = before_missing & filled.notna()
    filled = filled.dropna(how="all").reset_index().rename(columns={"index": "timestamp"})
    flags = flags.reset_index().rename(columns={"index": "timestamp"})
    return filled, flags


def _write_gap_reports(
    output_dir: Path,
    prefix: str,
    missing_before: pd.DatetimeIndex,
    missing_after: pd.DatetimeIndex,
) -> tuple[Path, Path]:
    before_path = output_dir / f"{prefix}_gap_report.csv"
    after_path = output_dir / f"{prefix}_remaining_gap_report.csv"
    _build_missing_run_report(missing_before).to_csv(before_path, index=False)
    _build_missing_run_report(missing_after).to_csv(after_path, index=False)
    return before_path, after_path


def convert_parquet_to_regional_csv(
    parquet_path: Path,
    output_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_date: str = '2020-01-01',
    *,
    normalize_off_grid: bool = True,
    max_offset_minutes: int = 2,
    impute_short_gaps: bool = True,
    max_impute_run: int = 6,
    regional_demand_path: Optional[Path] = None,
    min_valid_total_generation_mw: Optional[float] = None,
) -> Dict[str, Path]:
    """
    Convert new parquet format to regional CSV format.

    Args:
        parquet_path: Path to input parquet file
        output_dir: Directory to save CSV files
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        min_date: Minimum date to include (default: 2020-01-01)

    Returns:
        Dictionary mapping region names to output file paths
    """
    print(f"Loading parquet data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Convert Taipower local timestamps to the canonical UTC-naive basis, then
    # normalize near-grid scrape offsets such as 15:21.
    parsed_timestamp = taipower_local_to_utc_naive(df['timestamp'])
    df["_source_timestamp"] = parsed_timestamp
    if normalize_off_grid:
        df['timestamp'], normalized_audit = _normalize_to_10min_grid(
            parsed_timestamp,
            max_offset_minutes=max_offset_minutes,
        )
    else:
        df['timestamp'] = parsed_timestamp
        normalized_audit = pd.DataFrame(
            columns=["original_timestamp", "normalized_timestamp", "offset_minutes"]
        )

    # Filter by date range
    if min_date:
        df = df[df['timestamp'] >= min_date]
    if start_date:
        df = df[df['timestamp'] >= start_date]
    if end_date:
        df = df[df['timestamp'] <= end_date]
    if not normalized_audit.empty:
        normalized_audit = normalized_audit.loc[
            normalized_audit.index.intersection(df.index)
        ].copy()

    print(f"Loaded {len(df)} records ({df['timestamp'].min()} to {df['timestamp'].max()})")

    expected_index = None
    if impute_short_gaps and start_date is not None and end_date is not None:
        expected_index = _expected_time_index(start_date, end_date)

    # Fill missing mapping_names with plant_name
    df['mapping_names'] = df['mapping_names'].fillna(df['plant_name'])

    # Clean plant names to remove footnote markers
    df['plant_name_clean'] = df['plant_name'].apply(clean_plant_name)

    # Infer region per record. Try plant_name_clean first because mapping_names
    # is sometimes a Taipower aggregate label (e.g., '其它購電風力') that hides the
    # individual plant's geography. Fall back to mapping_names only if the
    # plant-name lookup yields 'other'. Vectorized via unique pair lookup to
    # avoid 80M-row apply.
    print("Mapping plants to regions...")
    pairs = df[['plant_name_clean', 'mapping_names']].drop_duplicates().copy()

    def _resolve_region(plant_clean: str, mapping_name: str) -> str:
        r1 = infer_region_from_plant_name(plant_clean) if pd.notna(plant_clean) else 'other'
        if r1 != 'other':
            return r1
        return infer_region_from_plant_name(mapping_name) if pd.notna(mapping_name) else 'other'

    pairs['region'] = pairs.apply(
        lambda r: _resolve_region(r['plant_name_clean'], r['mapping_names']),
        axis=1,
    )
    df = df.merge(pairs, on=['plant_name_clean', 'mapping_names'], how='left')

    # Create unit identifier - use cleaned plant_name
    df['unit_name'] = df.apply(
        lambda row: (
            f"{normalize_energy_type(row['energy_type'])}-{row['plant_name_clean']}"
            if pd.notna(row['plant_name']) else 'Other-Unknown'
        ),
        axis=1
    )

    invalid_generation_totals = pd.DataFrame(
        columns=["timestamp", "total_generation_mw", "min_valid_total_generation_mw"]
    )
    invalid_raw_rows_dropped = 0
    if min_valid_total_generation_mw is not None:
        validation_unit_values = _aggregate_unit_values(
            df,
            "used_mw",
            within_source="net_generation",
        )
        validation_totals = (
            validation_unit_values.groupby("timestamp", dropna=False)["used_mw"]
            .sum(min_count=1)
            .sort_index()
        )
        invalid_totals = validation_totals[
            validation_totals < float(min_valid_total_generation_mw)
        ]
        if not invalid_totals.empty:
            invalid_generation_totals = invalid_totals.reset_index(
                name="total_generation_mw"
            )
            invalid_generation_totals["min_valid_total_generation_mw"] = float(
                min_valid_total_generation_mw
            )
            invalid_timestamps = pd.DatetimeIndex(invalid_totals.index)
            before_rows = len(df)
            df = df.loc[~df["timestamp"].isin(invalid_timestamps)].copy()
            invalid_raw_rows_dropped = before_rows - len(df)
            if not normalized_audit.empty:
                normalized_audit = normalized_audit.loc[
                    normalized_audit.index.intersection(df.index)
                ].copy()
            print(
                "Dropped "
                f"{len(invalid_timestamps)} invalid generation timestamps "
                f"below {float(min_valid_total_generation_mw):g} MW "
                f"({invalid_raw_rows_dropped} source rows)"
            )

    # Aggregate by timestamp and unit
    print("Aggregating data...")
    unit_values = _aggregate_unit_values(
        df,
        "used_mw",
        within_source="net_generation",
    )
    agg_df = unit_values
    agg_df = agg_df.pivot(index='timestamp', columns='unit_name', values='used_mw')
    agg_df = agg_df.reset_index()
    agg_df.columns.name = None

    capacity_df = None
    if "capacity_mw" in df.columns:
        capacity_df = _aggregate_unit_values(
            df,
            "capacity_mw",
            within_source="mean",
        )
        capacity_df = capacity_df.pivot(index='timestamp', columns='unit_name', values='capacity_mw')
        capacity_df = capacity_df.reset_index()
        capacity_df.columns.name = None

    anno_df = None
    if "status" in df.columns:
        anno_df = df.groupby(['timestamp', 'unit_name'])['status'].first().reset_index()
        anno_df = anno_df.pivot(index='timestamp', columns='unit_name', values='status')
        anno_df = anno_df.reset_index()
        anno_df.columns.name = None

    category_df = _category_generation_from_unit_values(unit_values)

    # Also aggregate by region (used_mw + capacity_mw both pivoted regionally)
    regional_dfs = {}
    regional_capacity_dfs: Dict[str, pd.DataFrame] = {}
    for region in df['region'].unique():
        region_df = df[df['region'] == region].copy()

        if len(region_df) == 0:
            continue

        # Pivot for this region
        region_agg = _aggregate_unit_values(
            region_df,
            "used_mw",
            within_source="net_generation",
        )
        region_pivot = region_agg.pivot(index='timestamp', columns='unit_name', values='used_mw')
        region_pivot = region_pivot.reset_index()
        region_pivot.columns.name = None

        # Capacity pivot (used by Option Y allocator to compute share ratios)
        if "capacity_mw" in region_df.columns:
            cap_agg = _aggregate_unit_values(
                region_df,
                "capacity_mw",
                within_source="mean",
            )
            cap_pivot = cap_agg.pivot(index='timestamp', columns='unit_name', values='capacity_mw')
            regional_capacity_dfs[region] = cap_pivot

        regional_dfs[region] = region_pivot

    # Option Y: redistribute 'other' region distributed-fuel generation to
    # mainland by capacity-weighted ratio. Co-Gen uses the source-backed 2024
    # purchased-power capacity distribution; Biomass falls back to Taipower
    # regional load share when capacity is unavailable.
    demand_share = (
        _compute_regional_load_share(regional_demand_path)
        if regional_demand_path is not None
        else None
    )
    other_alloc_audit = _allocate_other_to_mainland(
        regional_dfs, regional_capacity_dfs, demand_share=demand_share
    )

    # Sort columns + attach PHS/BESS actor aggregates (after allocator so storage
    # additions to central aren't smeared to other regions)
    for region in list(regional_dfs.keys()):
        region_pivot = regional_dfs[region]
        if 'timestamp' not in region_pivot.columns:
            region_pivot = region_pivot.reset_index()
        region_pivot.columns.name = None
        region_pivot = _attach_phs_bess_columns(region_pivot)
        cols = ['timestamp'] + sorted([c for c in region_pivot.columns if c != 'timestamp'])
        regional_dfs[region] = region_pivot[cols]

    # Save to CSV files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "generation_timestamp_normalization.csv"
    normalized_audit.to_csv(normalized_path, index=False)
    invalid_total_path = output_dir / "generation_invalid_total_report.csv"
    invalid_generation_totals.to_csv(invalid_total_path, index=False)

    region_to_filename = {
        'north': 'north_unit_generation.csv',
        'central': 'central_unit_generation.csv',
        'south': 'south_unit_generation.csv',
        'east': 'east_unit_generation.csv',
        'island_penghu': 'island_penghu_unit_generation.csv',
        'island_kinmen': 'island_kinmen_unit_generation.csv',
        'island_lienchiang': 'island_lienchiang_unit_generation.csv',
        'island_other': 'island_other_unit_generation.csv',
        'other': 'other_unit_generation.csv',
    }

    output_paths = {
        "timestamp_normalization": normalized_path,
        "invalid_total_report": invalid_total_path,
    }
    region_quality = {}
    for region, region_df in regional_dfs.items():
        if region in region_to_filename:
            flags_path = None
            if expected_index is not None:
                region_df, region_flags, region_missing_before, region_missing_after = (
                    _interpolate_short_numeric_gaps(
                        region_df,
                        expected_index,
                        max_run=max_impute_run,
                    )
                )
                flags_path = output_dir / f"{region}_unit_generation_imputed_flags.csv"
                region_flags.to_csv(flags_path, index=False)
                _write_gap_reports(
                    output_dir,
                    f"{region}_unit_generation",
                    region_missing_before,
                    region_missing_after,
                )
                region_quality[region] = {
                    "missing_before_imputation": int(len(region_missing_before)),
                    "missing_after_imputation": int(len(region_missing_after)),
                    "imputed_cells": int(
                        region_flags.drop(columns=["timestamp"], errors="ignore")
                        .to_numpy(dtype=bool)
                        .sum()
                    ),
                    "rows_after_imputation": int(len(region_df)),
                }
            filename = region_to_filename[region]
            output_path = output_dir / filename
            region_df.to_csv(output_path, index=False)
            regional_dfs[region] = region_df
            output_paths[region] = output_path
            if flags_path is not None:
                output_paths[f"{region}_imputed_flags"] = flags_path
            print(f"Saved {region}: {len(region_df)} rows, {len(region_df.columns)-1} units -> {output_path}")

    # Also save the combined data
    combined_missing_before = combined_missing_after = pd.DatetimeIndex([])
    combined_imputed_cells = 0
    if expected_index is not None:
        agg_df, agg_flags, combined_missing_before, combined_missing_after = (
            _interpolate_short_numeric_gaps(
                agg_df,
                expected_index,
                max_run=max_impute_run,
            )
        )
        combined_flags_path = output_dir / 'unit_generation_imputed_flags.csv'
        agg_flags.to_csv(combined_flags_path, index=False)
        output_paths['combined_imputed_flags'] = combined_flags_path
        before_path, after_path = _write_gap_reports(
            output_dir,
            "unit_generation",
            combined_missing_before,
            combined_missing_after,
        )
        output_paths['combined_gap_report'] = before_path
        output_paths['combined_remaining_gap_report'] = after_path
        combined_imputed_cells = int(
            agg_flags.drop(columns=["timestamp"], errors="ignore").to_numpy(dtype=bool).sum()
        )

    combined_path = output_dir / 'unit_generation.csv'
    agg_df.to_csv(combined_path, index=False)
    print(f"Saved combined: {len(agg_df)} rows, {len(agg_df.columns)-1} units -> {combined_path}")
    output_paths['combined'] = combined_path

    if capacity_df is not None:
        capacity_imputed_cells = 0
        if expected_index is not None:
            capacity_df, capacity_flags, _, _ = _interpolate_short_numeric_gaps(
                capacity_df,
                expected_index,
                max_run=max_impute_run,
            )
            capacity_flags_path = output_dir / 'unit_capacity_imputed_flags.csv'
            capacity_flags.to_csv(capacity_flags_path, index=False)
            output_paths['capacity_imputed_flags'] = capacity_flags_path
            capacity_imputed_cells = int(
                capacity_flags.drop(columns=["timestamp"], errors="ignore")
                .to_numpy(dtype=bool)
                .sum()
            )
        capacity_path = output_dir / 'unit_capacity.csv'
        capacity_df.to_csv(capacity_path, index=False)
        output_paths['capacity'] = capacity_path
        print(
            f"Saved capacity: {len(capacity_df)} rows, "
            f"{len(capacity_df.columns)-1} units -> {capacity_path}"
        )

    if anno_df is not None:
        annotation_imputed_cells = 0
        if expected_index is not None:
            anno_df, anno_flags = _fill_short_categorical_gaps(
                anno_df,
                expected_index,
                max_run=max_impute_run,
            )
            anno_flags_path = output_dir / 'unit_anno_imputed_flags.csv'
            anno_flags.to_csv(anno_flags_path, index=False)
            output_paths['annotation_imputed_flags'] = anno_flags_path
            annotation_imputed_cells = int(
                anno_flags.drop(columns=["timestamp"], errors="ignore")
                .to_numpy(dtype=bool)
                .sum()
            )
        anno_path = output_dir / 'unit_anno.csv'
        anno_df.to_csv(anno_path, index=False)
        output_paths['annotation'] = anno_path
        print(
            f"Saved annotation: {len(anno_df)} rows, "
            f"{len(anno_df.columns)-1} units -> {anno_path}"
        )

    category_path = output_dir / 'category_generation.csv'
    category_imputed_cells = 0
    if expected_index is not None:
        category_df, category_flags, _, _ = _interpolate_short_numeric_gaps(
            category_df,
            expected_index,
            max_run=max_impute_run,
        )
        category_flags_path = output_dir / 'category_generation_imputed_flags.csv'
        category_flags.to_csv(category_flags_path, index=False)
        output_paths['category_imputed_flags'] = category_flags_path
        category_imputed_cells = int(
            category_flags.drop(columns=["timestamp"], errors="ignore")
            .to_numpy(dtype=bool)
            .sum()
        )
    category_df.to_csv(category_path, index=False)
    output_paths['category'] = category_path
    print(
        f"Saved category: {len(category_df)} rows, "
        f"{len(category_df.columns)-1} categories -> {category_path}"
    )

    quality = {
        "source_path": str(parquet_path),
        "source_timestamp_timezone": TAIPOWER_SOURCE_TZ,
        "output_timestamp_timezone": CANONICAL_TIMESTAMP_TZ,
        "output_timestamp_storage": "timezone-naive UTC",
        "start": str(pd.Timestamp(start_date)) if start_date is not None else None,
        "end": str(pd.Timestamp(end_date)) if end_date is not None else None,
        "normalize_off_grid": normalize_off_grid,
        "max_offset_minutes": max_offset_minutes,
        "min_valid_total_generation_mw": min_valid_total_generation_mw,
        "invalid_total_generation": {
            "timestamps": int(len(invalid_generation_totals)),
            "source_rows_dropped": int(invalid_raw_rows_dropped),
            "report": str(invalid_total_path),
        },
        "normalized_rows": int(len(normalized_audit)),
        "normalized_unique_timestamps": int(
            normalized_audit["original_timestamp"].nunique()
            if "original_timestamp" in normalized_audit.columns else 0
        ),
        "impute_short_gaps": bool(expected_index is not None),
        "max_impute_run": max_impute_run if expected_index is not None else None,
        "expected_timestamps": int(len(expected_index)) if expected_index is not None else None,
        "combined": {
            "rows_after_imputation": int(len(agg_df)),
            "missing_before_imputation": int(len(combined_missing_before)),
            "missing_after_imputation": int(len(combined_missing_after)),
            "imputed_cells": combined_imputed_cells,
        },
        "regions": region_quality,
        "other_region_redistribution": other_alloc_audit,
        "regional_generation_balance": _regional_generation_balance_diagnostics(
            regional_dfs,
            regional_demand_path,
            start_date=start_date,
            end_date=end_date,
        ),
    }
    if capacity_df is not None:
        quality["capacity"] = {
            "rows_after_imputation": int(len(capacity_df)),
            "imputed_cells": capacity_imputed_cells,
        }
    if anno_df is not None:
        quality["annotation"] = {
            "rows_after_imputation": int(len(anno_df)),
            "imputed_cells": annotation_imputed_cells,
        }
    quality["category"] = {
        "rows_after_imputation": int(len(category_df)),
        "imputed_cells": category_imputed_cells,
    }

    quality_path = output_dir / "generation_quality_report.json"
    quality_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["quality_report"] = quality_path

    return output_paths


def create_load_and_flow_from_parquet(
    parquet_path: Path,
    output_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Create load.csv and flow.csv from parquet data.

    The load is the total power consumption (sum of all used_mw plus imports/exports).
    Flow represents power transfers between regions.

    Args:
        parquet_path: Path to input parquet file
        output_dir: Directory to save CSV files
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Dictionary with paths to load.csv and flow.csv
    """
    print(f"Creating load and flow data from {parquet_path}...")

    df = pd.read_parquet(parquet_path)
    df['timestamp'] = taipower_local_to_utc_naive(df['timestamp'])

    if start_date:
        df = df[df['timestamp'] >= start_date]
    if end_date:
        df = df[df['timestamp'] <= end_date]

    # Fill missing mapping_names
    df['mapping_names'] = df['mapping_names'].fillna(df['plant_name'])

    # Infer region
    df['region'] = df['mapping_names'].apply(infer_region_from_plant_name)

    # Aggregate load by region and timestamp
    load_df = df.groupby(['timestamp', 'region'])['used_mw'].sum().reset_index()
    load_pivot = load_df.pivot(index='timestamp', columns='region', values='used_mw').fillna(0)

    # Calculate total load (sum of all regions)
    load_pivot['total'] = load_pivot.sum(axis=1)
    load_pivot = load_pivot.reset_index()
    load_pivot.columns.name = None

    # Save load.csv
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_path = output_dir / 'load.csv'
    load_pivot.to_csv(load_path, index=False)
    print(f"Saved load.csv: {len(load_pivot)} rows")

    # For flow.csv, we need to estimate inter-regional flows
    # This is a simplified version - actual flows would need network topology data
    regions = [
        'north',
        'central',
        'south',
        'east',
        'island_penghu',
        'island_kinmen',
        'island_lienchiang',
    ]
    flow_df = load_pivot[['timestamp']].copy()

    # Initialize flow columns (would need actual topology data for accurate flows)
    for i, r1 in enumerate(regions):
        for r2 in regions[i+1:]:
            flow_df[f'F_{r1}_{r2}'] = 0.0

    flow_path = output_dir / 'flow.csv'
    flow_df.to_csv(flow_path, index=False)
    print(f"Saved flow.csv: {len(flow_df)} rows")

    return {'load': load_path, 'flow': flow_path}


def _parse_regional_demand_timestamp(df: pd.DataFrame) -> pd.Series:
    """Parse Taipower regional demand local timestamps onto the UTC-naive basis."""
    date = pd.to_datetime(df["date"], errors="coerce")
    time = df["time"].astype(str).str.strip()
    normalized_time = time.where(time.str.contains(":", regex=False), time + ":00")
    local_timestamp = pd.to_datetime(
        date.dt.strftime("%Y-%m-%d") + " " + normalized_time,
        errors="coerce",
    )
    return taipower_local_to_utc_naive(local_timestamp)


def _normalize_to_10min_grid(
    timestamps: pd.Series,
    *,
    max_offset_minutes: int = 2,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Normalize near-grid timestamps to the nearest 10-minute slot.

    Returns normalized timestamps plus a small audit table for changed rows.
    """
    rounded = timestamps.dt.round("10min")
    offset_minutes = (timestamps - rounded).abs().dt.total_seconds().div(60)
    can_normalize = timestamps.notna() & (offset_minutes > 0) & (
        offset_minutes <= max_offset_minutes
    )
    normalized = timestamps.copy()
    normalized.loc[can_normalize] = rounded.loc[can_normalize]

    audit = pd.DataFrame(
        {
            "original_timestamp": timestamps.loc[can_normalize],
            "normalized_timestamp": normalized.loc[can_normalize],
            "offset_minutes": offset_minutes.loc[can_normalize],
        }
    )
    return normalized, audit


def create_flow_from_regional_demand(
    demand_path: Path,
    output_dir: Path,
    start_date: str = "2024-01-01 00:00",
    end_date: str = "2024-12-31 23:50",
    *,
    source: str = "power_demand",
    unit_scale: float = 10.0,
    normalize_off_grid: bool = True,
    max_offset_minutes: int = 2,
    reference_flow_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Create signed C2N/C2E/C2S flow.csv from Taipower regional demand balance.

    The adopted Paper 1 convention is:
    - F_CN = (north_load - north_gen) * 10, Central to North
    - F_CE = (east_load - east_gen) * 10, Central to East
    - F_CS = (south_load - south_gen) * 10, Central to South
    """
    demand_path = Path(demand_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(demand_path)
    required = {
        "date",
        "time",
        "source",
        "north_gen",
        "north_load",
        "central_gen",
        "central_load",
        "south_gen",
        "south_load",
        "east_gen",
        "east_load",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Regional demand data is missing columns: {', '.join(missing)}")

    parsed_ts = _parse_regional_demand_timestamp(df)
    if normalize_off_grid:
        df["timestamp"], normalized_audit = _normalize_to_10min_grid(
            parsed_ts,
            max_offset_minutes=max_offset_minutes,
        )
    else:
        df["timestamp"] = parsed_ts
        normalized_audit = pd.DataFrame(
            columns=["original_timestamp", "normalized_timestamp", "offset_minutes"]
        )

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    df = df[
        (df["source"] == source)
        & (df["timestamp"] >= start)
        & (df["timestamp"] <= end)
    ].copy()

    numeric_cols = [
        "north_gen",
        "north_load",
        "central_gen",
        "central_load",
        "south_gen",
        "south_load",
        "east_gen",
        "east_load",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_numeric_rows = df[numeric_cols].isna().any(axis=1)
    all_zero_regional_rows = df[numeric_cols].eq(0).all(axis=1)
    invalid_rows = invalid_numeric_rows | all_zero_regional_rows
    invalid_zero_timestamps = pd.DatetimeIndex(
        df.loc[all_zero_regional_rows, "timestamp"].dropna().unique()
    ).sort_values()
    duplicate_input_timestamps = int(df["timestamp"].duplicated().sum())
    valid_df = df.loc[~invalid_rows].copy()

    balance = valid_df.groupby("timestamp", sort=True)[numeric_cols].mean()
    flow = pd.DataFrame(index=balance.index)
    flow["F_CN"] = (balance["north_load"] - balance["north_gen"]) * unit_scale
    flow["F_CE"] = (balance["east_load"] - balance["east_gen"]) * unit_scale
    flow["F_CS"] = (balance["south_load"] - balance["south_gen"]) * unit_scale
    flow = flow.sort_index()

    expected_index = pd.date_range(start, end, freq="10min")
    extra_index = flow.index.difference(expected_index)
    if len(extra_index):
        flow = flow.loc[flow.index.intersection(expected_index)].sort_index()
    flow_nan_rows = flow[["F_CN", "F_CE", "F_CS"]].isna().any(axis=1)
    if flow_nan_rows.any():
        flow = flow.loc[~flow_nan_rows].copy()
    missing_index = expected_index.difference(flow.index)

    flow_path = output_dir / "flow.csv"
    flow.to_csv(flow_path, index=True)

    gap_report = _build_missing_run_report(missing_index)
    gap_path = output_dir / "flow_gap_report.csv"
    gap_report.to_csv(gap_path, index=False)

    normalized_path = output_dir / "flow_timestamp_normalization.csv"
    normalized_audit.to_csv(normalized_path, index=False)

    quality = {
        "source_path": str(demand_path),
        "source_filter": source,
        "source_timestamp_timezone": TAIPOWER_SOURCE_TZ,
        "output_timestamp_timezone": CANONICAL_TIMESTAMP_TZ,
        "output_timestamp_storage": "timezone-naive UTC",
        "unit_scale": unit_scale,
        "start": str(start),
        "end": str(end),
        "expected_timestamps": int(len(expected_index)),
        "observed_timestamps": int(len(flow.index)),
        "missing_timestamps": int(len(missing_index)),
        "extra_timestamps": int(len(extra_index)),
        "normalized_timestamps": int(len(normalized_audit)),
        "input_rows_after_filter": int(len(df)),
        "duplicate_input_timestamps": duplicate_input_timestamps,
        "invalid_numeric_rows": int(invalid_numeric_rows.sum()),
        "all_zero_regional_rows": int(all_zero_regional_rows.sum()),
        "all_zero_regional_timestamps": int(len(invalid_zero_timestamps)),
        "flow_nan_rows": int(flow_nan_rows.sum()),
        "dropped_extra_timestamps": int(len(extra_index)),
        "columns": ["F_CN", "F_CE", "F_CS"],
        "direction_convention": {
            "F_CN": "C2N",
            "F_CE": "C2E",
            "F_CS": "C2S",
        },
    }

    if reference_flow_path is not None and Path(reference_flow_path).exists():
        ref = pd.read_csv(reference_flow_path, index_col=0, parse_dates=True)
        common = flow.index.intersection(ref.index)
        comparisons = {}
        for col in ["F_CN", "F_CE", "F_CS"]:
            if col in ref.columns:
                both = pd.concat(
                    [flow.loc[common, col].rename("new"), ref.loc[common, col].rename("reference")],
                    axis=1,
                ).dropna()
                if not both.empty:
                    diff = both["new"] - both["reference"]
                    comparisons[col] = {
                        "n": int(len(both)),
                        "correlation": float(both["new"].corr(both["reference"])),
                        "mae": float(diff.abs().mean()),
                        "bias_new_minus_reference": float(diff.mean()),
                    }
        quality["reference_flow_path"] = str(reference_flow_path)
        quality["reference_comparison"] = comparisons

    import json

    quality_path = output_dir / "flow_quality_report.json"
    quality_path.write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "flow": flow_path,
        "gap_report": gap_path,
        "timestamp_normalization": normalized_path,
        "quality_report": quality_path,
    }


def split_signed_flow_for_aef(flow_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split signed C2N/C2E/C2S flow into nonnegative regional AEF columns.

    Positive canonical columns represent Central to other regions. Negative
    values are reverse flows back to Central and are written to F_NC/F_EC/F_SC.
    """
    required = {"F_CN", "F_CE", "F_CS"}
    missing = sorted(required - set(flow_df.columns))
    if missing:
        raise ValueError(f"Flow table is missing columns: {', '.join(missing)}")

    idx = flow_df.index
    return {
        "north": pd.DataFrame({"F_CN": flow_df["F_CN"].clip(lower=0)}, index=idx),
        "east": pd.DataFrame({"F_CE": flow_df["F_CE"].clip(lower=0)}, index=idx),
        "south": pd.DataFrame({"F_CS": flow_df["F_CS"].clip(lower=0)}, index=idx),
        "central": pd.DataFrame(
            {
                "F_NC": (-flow_df["F_CN"]).clip(lower=0),
                "F_EC": (-flow_df["F_CE"]).clip(lower=0),
                "F_SC": (-flow_df["F_CS"]).clip(lower=0),
            },
            index=idx,
        ),
    }


def _build_missing_run_report(missing_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a missing-run table for a sorted 10-minute timestamp index."""
    missing_index = pd.DatetimeIndex(sorted(missing_index))
    if len(missing_index) == 0:
        return pd.DataFrame(columns=["start", "end", "n_timestamps", "duration_minutes"])

    runs = []
    start = prev = missing_index[0]
    n = 1
    for ts in missing_index[1:]:
        if ts - prev == pd.Timedelta(minutes=10):
            prev = ts
            n += 1
        else:
            runs.append((start, prev, n))
            start = prev = ts
            n = 1
    runs.append((start, prev, n))

    return pd.DataFrame(
        [
            {
                "start": run_start,
                "end": run_end,
                "n_timestamps": count,
                "duration_minutes": count * 10,
            }
            for run_start, run_end, count in runs
        ]
    )


def merge_flow_data_to_regional_csvs(
    output_dir: Path,
    flow_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Merge flow.csv data into regional CSV files.

    The AEF pipeline expects flow columns in each regional CSV:
    - north: F_CN (Flow from Central to North)
    - central: F_NC, F_SC, F_EC (Flows from North, South, East to Central)
    - south: F_CS (Flow from Central to South)
    - east: F_CE (Flow from Central to East)

    Args:
        output_dir: Directory containing regional CSV files
        flow_path: Path to flow.csv (defaults to output_dir/flow.csv)

    Returns:
        Dictionary mapping region names to updated file paths
    """
    output_dir = Path(output_dir)
    flow_path = flow_path or output_dir / 'flow.csv'

    if not flow_path.exists():
        print(f"Warning: flow.csv not found at {flow_path}")
        return {}

    print(f"Merging flow data from {flow_path}...")

    # Load flow data
    flow_df = pd.read_csv(flow_path, index_col=0, parse_dates=True)

    region_flow_data = split_signed_flow_for_aef(flow_df)

    region_files = {
        'north': 'north_unit_generation.csv',
        'central': 'central_unit_generation.csv',
        'south': 'south_unit_generation.csv',
        'east': 'east_unit_generation.csv',
    }

    updated_paths = {}

    for region, filename in region_files.items():
        file_path = output_dir / filename
        if not file_path.exists():
            continue

        # Load regional CSV
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # Merge nonnegative regional flow columns derived from signed canonical
        # flow. Missing flow rows are dropped rather than silently interpreted as
        # zero imports/exports; the AEF stage should keep only timestamps with a
        # complete physical time basis.
        flow_cols = region_flow_data.get(region, pd.DataFrame(index=flow_df.index))
        if not flow_cols.empty:
            flow_cols = flow_cols.reindex(df.index)
            missing_flow_rows = flow_cols.isna().any(axis=1)
            if missing_flow_rows.any():
                df = df.loc[~missing_flow_rows].copy()
                flow_cols = flow_cols.loc[df.index]
        for col in flow_cols.columns:
            df[col] = flow_cols[col]

        # Save updated CSV
        df.to_csv(file_path)
        updated_paths[region] = file_path
        print(f"  Updated {region}: added {list(flow_cols.columns)}")

    return updated_paths


def main():
    """CLI entry point for power data conversion."""
    import argparse

    from streetlight.config import get_config

    cfg = get_config()
    parser = argparse.ArgumentParser(description='Convert power data from parquet to CSV format')
    parser.add_argument('parquet_path', type=Path, help='Path to input parquet file')
    parser.add_argument('-o', '--output-dir', type=Path, default=cfg.power_dir,
                        help='Output directory for CSV files')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--create-load-flow', action='store_true',
                        help='Also create load.csv and flow.csv')

    args = parser.parse_args()

    # Convert regional CSVs
    convert_parquet_to_regional_csv(
        args.parquet_path,
        args.output_dir,
        args.start_date,
        args.end_date
    )

    # Create load and flow if requested
    if args.create_load_flow:
        create_load_and_flow_from_parquet(
            args.parquet_path,
            args.output_dir,
            args.start_date,
            args.end_date
        )


if __name__ == '__main__':
    main()
