"""
Pareto analysis module for cost-emission optimization.

This module implements battery storage simulation, emission calculation,
Pareto frontier extraction, and MACC analysis for streetlight systems.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        # Fallback decorator: return the function unchanged
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def _wrap(fn):
            return fn
        return _wrap


@njit(cache=True)
def _dispatch_inner_njit(
    values: np.ndarray,
    capacity_kwh: float,
    power_kw: float,
    eta_c: float,
    eta_d: float,
    sd_factor: float,
    dt_h: float,
    soc0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inner dispatch loop, JIT-compiled when numba is available.

    Operates entirely on numpy arrays + scalars. The wrapper function
    ``storage_dispatch`` handles pandas Series I/O and parameter resolution.
    """
    n = values.shape[0]
    soc_list = np.empty(n, dtype=np.float64)
    grid_import = np.zeros(n, dtype=np.float64)
    net_after = np.zeros(n, dtype=np.float64)
    pw_step_limit_kwh = power_kw * dt_h
    soc = soc0

    for i in range(n):
        p_kw = values[i]
        # Apply self-discharge first
        soc *= sd_factor
        if soc < 0.0:
            soc = 0.0
        elif soc > capacity_kwh:
            soc = capacity_kwh

        if p_kw >= 0.0:
            # Charging
            e_surplus = p_kw * dt_h
            e_ch_in = e_surplus
            if e_ch_in > pw_step_limit_kwh:
                e_ch_in = pw_step_limit_kwh
            capacity_limit = (capacity_kwh - soc) / eta_c
            if e_ch_in > capacity_limit:
                e_ch_in = capacity_limit
            soc += eta_c * e_ch_in
            e_export = e_surplus - e_ch_in
            net_after[i] = e_export / dt_h
            grid_import[i] = 0.0
        else:
            # Discharging
            e_def = (-p_kw) * dt_h
            e_can_ac_from_power = pw_step_limit_kwh
            e_can_ac_from_soc = eta_d * soc
            e_supplied_ac = e_def
            if e_supplied_ac > e_can_ac_from_power:
                e_supplied_ac = e_can_ac_from_power
            if e_supplied_ac > e_can_ac_from_soc:
                e_supplied_ac = e_can_ac_from_soc
            soc -= (e_supplied_ac / eta_d)
            if soc < 0.0:
                soc = 0.0
            e_grid = e_def - e_supplied_ac
            grid_import[i] = e_grid / dt_h
            net_after[i] = -(e_grid / dt_h)

        if soc < 0.0:
            soc = 0.0
        elif soc > capacity_kwh:
            soc = capacity_kwh
        soc_list[i] = soc

    return soc_list, grid_import, net_after


def _resolve_economic_costs(economic_costs: Optional[dict]) -> dict:
    """Resolve economic costs from input or global config and validate required keys."""
    if economic_costs is None:
        # Local import avoids coupling at module import time.
        from streetlight.config import get_config
        economic_costs = get_config().economic_costs

    if not isinstance(economic_costs, dict):
        raise ValueError("economic_costs must be a mapping")

    physical_proxy_keys = {
        "pv_capacity_kw_per_factor",
        "pv_capex_ntd_per_kw",
        "pv_om_ntd_per_kw_year",
        "battery_capex_ntd_per_kwh",
        "battery_power_capex_ntd_per_kw",
        "battery_om_fraction_of_capex_per_year",
        "other_capex",
        "other_om",
        "other_eol",
        "grid_fixed_cost",
    }
    if physical_proxy_keys.issubset(economic_costs.keys()):
        resolved = {
            "model": "physical_proxy",
            "pv_capacity_kw_per_factor": float(economic_costs["pv_capacity_kw_per_factor"]),
            "pv_capex_ntd_per_kw": float(economic_costs["pv_capex_ntd_per_kw"]),
            "pv_om_ntd_per_kw_year": float(economic_costs["pv_om_ntd_per_kw_year"]),
            "pv_eol_ntd_per_kw": float(economic_costs.get("pv_eol_ntd_per_kw", 0.0)),
            "battery_capex_ntd_per_kwh": float(economic_costs["battery_capex_ntd_per_kwh"]),
            "battery_power_capex_ntd_per_kw": float(economic_costs["battery_power_capex_ntd_per_kw"]),
            "battery_om_fraction_of_capex_per_year": float(economic_costs["battery_om_fraction_of_capex_per_year"]),
            "battery_replacement_year": float(economic_costs.get("battery_replacement_year", 0.0)),
            "battery_replacement_energy_capex_ntd_per_kwh": float(
                economic_costs.get(
                    "battery_replacement_energy_capex_ntd_per_kwh",
                    economic_costs["battery_capex_ntd_per_kwh"],
                )
            ),
            "battery_replacement_power_capex_ntd_per_kw": float(
                economic_costs.get("battery_replacement_power_capex_ntd_per_kw", 0.0)
            ),
            "battery_eol": float(economic_costs.get("battery_eol", 0.0)),
            "other_capex": float(economic_costs["other_capex"]),
            "other_om": float(economic_costs["other_om"]),
            "other_eol": float(economic_costs["other_eol"]),
            "grid_fixed_cost": float(economic_costs["grid_fixed_cost"]),
        }
        if resolved["pv_capacity_kw_per_factor"] <= 0:
            raise ValueError("pv_capacity_kw_per_factor must be positive")
        return resolved

    required_keys = {
        "solar_capex",
        "battery_capex",
        "other_capex",
        "solar_om",
        "battery_om",
        "other_om",
        "solar_eol",
        "battery_eol",
        "other_eol",
        "grid_fixed_cost",
    }
    missing = sorted(required_keys - set(economic_costs.keys()))
    if missing:
        raise ValueError(f"Missing economics.cost keys in config: {', '.join(missing)}")
    resolved = {k: float(economic_costs[k]) for k in required_keys}
    resolved["model"] = "legacy_flat"
    return resolved


def _resolve_financial_params(financial_params: Optional[dict]) -> dict:
    """Resolve financial assumptions for discounted lifecycle costing."""
    if financial_params is None:
        from streetlight.config import get_config
        financial_params = get_config().financial_params

    if financial_params is None:
        financial_params = {}
    if not isinstance(financial_params, dict):
        raise ValueError("financial_params must be a mapping")

    return {
        "discount_rate": float(financial_params.get("discount_rate", 0.0)),
        "pv_salvage_fraction": float(financial_params.get("pv_salvage_fraction", 0.0)),
        "battery_salvage_fraction": float(financial_params.get("battery_salvage_fraction", 0.0)),
        "other_salvage_fraction": float(financial_params.get("other_salvage_fraction", 0.0)),
    }


def _present_worth_factor(years: float, discount_rate: float) -> float:
    """Present-worth factor for a uniform annual series."""
    if years <= 0:
        return 0.0
    if abs(discount_rate) < 1e-12:
        return float(years)
    return (1.0 - (1.0 + discount_rate) ** (-float(years))) / discount_rate


def _discount_lump(value: float, year: float, discount_rate: float) -> float:
    """Discount a lump-sum payment to present value."""
    if value == 0.0:
        return 0.0
    if abs(discount_rate) < 1e-12:
        return float(value)
    return float(value) / ((1.0 + discount_rate) ** float(year))


def infer_dt_hours(idx: pd.DatetimeIndex) -> float:
    """
    Infer time step in hours from datetime index.

    Parameters
    ----------
    idx : pd.DatetimeIndex
        Datetime index to infer timestep from

    Returns
    -------
    float
        Time step in hours (default: 1.0 if insufficient data)
    """
    if len(idx) <= 1:
        return 1.0
    # Use timedelta to be unit-aware; pd.to_timedelta handles the conversion correctly
    return pd.to_timedelta(pd.Series(idx).diff()).median().total_seconds() / 3600.0


def storage_dispatch(
    net_kw: pd.Series,
    capacity_kwh: float = 10.0,
    power_kw: float = 5.0,
    eta_roundtrip: float = 0.90,
    eta_c: Optional[float] = None,
    eta_d: Optional[float] = None,
    soc0_kwh: float = 0.0,
    self_discharge_per_hour: float = 0.0,
    dt_hours: Optional[float] = None,
) -> dict:
    """
    Simulate battery charge/discharge dispatch.

    Parameters
    ----------
    net_kw : pd.Series
        Net power (kW); positive = surplus (charge), negative = deficit (discharge)
    capacity_kwh : float
        Battery capacity in kWh
    power_kw : float
        Charge/discharge power limit (kW)
    eta_roundtrip, eta_c, eta_d : float
        Round-trip, charge, discharge efficiencies
    soc0_kwh : float
        Initial state of charge (kWh)
    self_discharge_per_hour : float
        Self-discharge rate per hour (0-1)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'grid_import_kw': Series of grid import needed (kW)
        - 'net_with_storage_kw': Series of net power after storage (kW)
        - 'soc_kwh': Series of battery state of charge (kWh)

    Notes
    -----
    Positive net_kw: Surplus energy available for charging
    Negative net_kw: Energy deficit requiring discharge or grid import

    Self-discharge is applied BEFORE processing each timestep.
    """
    net_kw = net_kw.astype(float, copy=False)

    # Get time step (hours)
    if dt_hours is not None:
        dt_h = float(dt_hours)
    elif net_kw.index.freq is not None:
        dt_h = pd.Timedelta(net_kw.index.freq).total_seconds() / 3600.0
    else:
        dt_h = pd.to_timedelta(pd.Series(net_kw.index).diff().median()).total_seconds() / 3600.0
    if not np.isfinite(dt_h) or dt_h <= 0:
        raise ValueError(
            "Cannot infer timestep from index. Ensure uniform frequency "
            "or preprocess manually."
        )

    # Split efficiency
    if eta_c is None or eta_d is None:
        # Symmetric charge/discharge from round-trip: eta_c * eta_d = eta_roundtrip
        base = np.sqrt(eta_roundtrip)
        if eta_c is None:
            eta_c = base
        if eta_d is None:
            eta_d = base

    # Initialize
    idx = net_kw.index
    values = np.ascontiguousarray(net_kw.to_numpy(copy=False), dtype=np.float64)
    soc0 = float(np.clip(soc0_kwh, 0.0, capacity_kwh))

    # Self-discharge factor per step (energy basis)
    sd_factor = max(0.0, min(1.0, 1.0 - self_discharge_per_hour * dt_h))

    soc_list, grid_import, net_after = _dispatch_inner_njit(
        values=values,
        capacity_kwh=float(capacity_kwh),
        power_kw=float(power_kw),
        eta_c=float(eta_c),
        eta_d=float(eta_d),
        sd_factor=float(sd_factor),
        dt_h=float(dt_h),
        soc0=soc0,
    )

    return {
        "grid_import_kw": pd.Series(grid_import, index=idx, name="grid_import_kw"),
        "net_with_storage_kw": pd.Series(net_after, index=idx, name="net_with_storage_kw"),
        "soc_kwh": pd.Series(soc_list, index=idx, name="soc_kwh"),
    }


def compute_city_emission(
    city_power_kw: pd.Series,
    aef: pd.Series,
    dt_hours: float,
    years: float,
    solar_panel_factor: float = 1.0,
    battery_factor: float = 1.0,
    elec_price: float = 3.7556,
    params: dict = None,
    economic_costs: Optional[dict] = None,
    financial_params: Optional[dict] = None,
    analysis_scaling_factor: Optional[float] = None,
    _resolved_costs: Optional[dict] = None,
    _resolved_financial: Optional[dict] = None,
) -> dict:
    """
    Calculate emissions and costs for grid-only vs PV+battery scenarios.

    Parameters
    ----------
    city_power_kw : pd.Series
        City power profile (kW); >0 = export to grid, <0 = import
    aef : pd.Series
        Average emission factor (kg CO2e/kWh) indexed by time
    dt_hours : float
        Time step size in hours (from index frequency)
    years : float
        Analysis period metadata, retained for compatibility
    solar_panel_factor, battery_factor : float
        Capacity scaling factors
    elec_price : float
        Electricity price (NTD/kWh)
    params : dict
        Storage parameters for storage_dispatch()
    economic_costs : dict
        Economic cost coefficients. If None, defaults are used.
    analysis_scaling_factor : float | None
        Multiplier that scales the modeled observation window to the target
        analysis horizon. If None, falls back to the legacy `years` multiplier.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'grid_only_emission_t': Grid-only emissions (tonnes CO2e)
        - 'pv_storage_emission_t': PV+battery emissions (tonnes CO2e)
        - 'delta_t': Emission reduction (tonnes, negative = reduction)
        - 'grid_energy_kwh_20y': Grid energy consumption (kWh)
        - 'pv_storage_energy_kwh_20y': PV+battery energy consumption (kWh)
        - 'grid_only_cost': Grid-only cost (NTD)
        - 'pv_storage_cost': PV+battery cost (NTD)
        - 'delta_cost': Cost difference (NTD, negative = savings)

    Notes
    -----
    Grid-only scenario: All net imports come from grid
    PV+battery scenario: Net imports reduced by PV+battery system

    Cost includes:
    - Electricity cost (energy × price)
    - Capital cost (amortized CAPEX: CAPEX + O&M + EOL)
    """
    if params is None:
        params = {}
    costs = _resolved_costs if _resolved_costs is not None else _resolve_economic_costs(economic_costs)
    financial = (
        _resolved_financial if _resolved_financial is not None else _resolve_financial_params(financial_params)
    )
    discount_rate = financial["discount_rate"]
    pw_factor = _present_worth_factor(float(years), discount_rate)
    scale = float(analysis_scaling_factor) if analysis_scaling_factor is not None else float(years)

    # Reindex power to AEF timeline
    if city_power_kw.index.equals(aef.index):
        city_power_aligned = city_power_kw
    else:
        city_power_aligned = city_power_kw.reindex(aef.index).fillna(0.0)
    aef_values = aef.to_numpy(copy=False)
    city_power_values = city_power_aligned.to_numpy(copy=False)

    # Scenario A: Grid-only (no PV/storage)
    import_kw = np.maximum(-city_power_values, 0.0)
    import_kwh = import_kw * dt_hours  # Observation period energy (kWh)
    grid_energy_kwh_20y = float(import_kwh.sum()) * scale
    grid_emission_t = float((import_kwh * aef_values).sum()) * scale / 1000.0  # tonnes CO2e

    # Scenario B: PV+storage -> still needs grid
    res = storage_dispatch(city_power_aligned, dt_hours=dt_hours, **params)
    need_grid_values = res["grid_import_kw"].to_numpy(copy=False)
    need_grid_kwh = need_grid_values * dt_hours
    pv_storage_energy_kwh_20y = float(need_grid_kwh.sum()) * scale
    pv_storage_emission_t = float((need_grid_kwh * aef_values).sum()) * scale / 1000.0

    # Cost calculation
    annual_grid_energy_kwh = grid_energy_kwh_20y / float(years)
    grid_only_energy_cost = annual_grid_energy_kwh * elec_price * pw_factor
    grid_only_fixed_cost = (costs["grid_fixed_cost"] / float(years)) * pw_factor
    grid_only_cost = grid_only_energy_cost + grid_only_fixed_cost

    if costs.get("model") == "physical_proxy":
        pv_capacity_kw = costs["pv_capacity_kw_per_factor"] * solar_panel_factor
        battery_capacity_kwh = float(params.get("capacity_kwh", 0.0))
        battery_power_kw = float(params.get("power_kw", 0.0))

        battery_init_cost = (
            battery_capacity_kwh * costs["battery_capex_ntd_per_kwh"]
            + battery_power_kw * costs["battery_power_capex_ntd_per_kw"]
        )
        battery_om_cost = battery_init_cost * costs["battery_om_fraction_of_capex_per_year"] * pw_factor

        replacement_cost = 0.0
        replacement_year = costs["battery_replacement_year"]
        if replacement_year > 0:
            n_replacements = int(np.floor((float(years) - 1e-9) / replacement_year))
            replacement_unit_cost = (
                battery_capacity_kwh * costs["battery_replacement_energy_capex_ntd_per_kwh"]
                + battery_power_kw * costs["battery_replacement_power_capex_ntd_per_kw"]
            )
            for i in range(1, n_replacements + 1):
                replacement_cost += _discount_lump(
                    replacement_unit_cost,
                    replacement_year * i,
                    discount_rate,
                )

        s_init_cost = (
            pv_capacity_kw * costs["pv_capex_ntd_per_kw"]
            + battery_init_cost
            + costs["other_capex"]
        )
        s_om_cost = (
            pv_capacity_kw * costs["pv_om_ntd_per_kw_year"] * pw_factor
            + battery_om_cost
            + (costs["other_om"] / float(years)) * pw_factor
        )
        pv_eol_cost = _discount_lump(
            pv_capacity_kw * costs["pv_eol_ntd_per_kw"],
            float(years),
            discount_rate,
        )
        battery_eol_cost = _discount_lump(costs["battery_eol"], float(years), discount_rate)
        other_eol_cost = _discount_lump(costs["other_eol"], float(years), discount_rate)
        salvage_credit = _discount_lump(
            pv_capacity_kw * costs["pv_capex_ntd_per_kw"] * financial["pv_salvage_fraction"]
            + battery_init_cost * financial["battery_salvage_fraction"]
            + costs["other_capex"] * financial["other_salvage_fraction"],
            float(years),
            discount_rate,
        )
        s_eol_cost = pv_eol_cost + battery_eol_cost + other_eol_cost + replacement_cost - salvage_credit
    else:
        s_init_cost = (
            costs["solar_capex"] * solar_panel_factor
            + costs["battery_capex"] * battery_factor
            + costs["other_capex"]
        )
        s_om_cost = (
            ((costs["solar_om"] * solar_panel_factor) / float(years)) * pw_factor
            + ((costs["battery_om"] * battery_factor) / float(years)) * pw_factor
            + (costs["other_om"] / float(years)) * pw_factor
        )
        salvage_credit = _discount_lump(
            costs["solar_capex"] * solar_panel_factor * financial["pv_salvage_fraction"]
            + costs["battery_capex"] * battery_factor * financial["battery_salvage_fraction"]
            + costs["other_capex"] * financial["other_salvage_fraction"],
            float(years),
            discount_rate,
        )
        s_eol_cost = (
            _discount_lump(costs["solar_eol"] * solar_panel_factor, float(years), discount_rate)
            + _discount_lump(costs["battery_eol"] * battery_factor, float(years), discount_rate)
            + _discount_lump(costs["other_eol"], float(years), discount_rate)
            - salvage_credit
        )

    annual_pv_storage_energy_kwh = pv_storage_energy_kwh_20y / float(years)
    pv_storage_energy_cost = annual_pv_storage_energy_kwh * elec_price * pw_factor
    pv_storage_cost = pv_storage_energy_cost + (s_init_cost + s_om_cost + s_eol_cost)
    delta_cost = pv_storage_cost - grid_only_cost

    return {
        "grid_only_emission_t": grid_emission_t,
        "pv_storage_emission_t": pv_storage_emission_t,
        "delta_t": pv_storage_emission_t - grid_emission_t,  # Negative = reduction
        "grid_energy_kwh_20y": grid_energy_kwh_20y,
        "pv_storage_energy_kwh_20y": pv_storage_energy_kwh_20y,
        "grid_only_cost": grid_only_cost,
        "pv_storage_cost": pv_storage_cost,
        "grid_only_energy_cost": grid_only_energy_cost,
        "grid_only_fixed_cost": grid_only_fixed_cost,
        "pv_storage_energy_cost": pv_storage_energy_cost,
        "pv_storage_capex_cost": s_init_cost,
        "pv_storage_om_cost": s_om_cost,
        "pv_storage_eol_cost": s_eol_cost,
        "delta_cost": delta_cost  # Negative = savings
    }


def recompute_costs_from_results(
    results_df: pd.DataFrame,
    *,
    years: float,
    elec_price: float,
    economic_costs: Optional[dict] = None,
    financial_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Recompute lifecycle cost columns from precomputed energy/sizing results.

    This is intended for scenario analysis where dispatch/emissions remain
    unchanged and only economics/financial assumptions vary.
    """
    costs = _resolve_economic_costs(economic_costs)
    financial = _resolve_financial_params(financial_params)
    discount_rate = financial["discount_rate"]
    pw_factor = _present_worth_factor(float(years), discount_rate)

    out = results_df.copy()

    annual_grid_energy_kwh = out["grid_energy_kwh_20y"].astype(float) / float(years)
    annual_pv_storage_energy_kwh = out["pv_storage_energy_kwh_20y"].astype(float) / float(years)
    out["grid_only_energy_cost"] = annual_grid_energy_kwh * float(elec_price) * pw_factor
    out["pv_storage_energy_cost"] = annual_pv_storage_energy_kwh * float(elec_price) * pw_factor
    out["grid_only_fixed_cost"] = (float(costs["grid_fixed_cost"]) / float(years)) * pw_factor
    out["grid_only_cost"] = out["grid_only_energy_cost"] + out["grid_only_fixed_cost"]

    if costs.get("model") == "physical_proxy":
        pv_capacity_kw = costs["pv_capacity_kw_per_factor"] * out["solar_panel_factor"].astype(float)
        battery_capacity_kwh = out["battery_capacity_kwh"].astype(float)
        battery_power_kw = out["battery_power_kw"].astype(float)

        battery_init_cost = (
            battery_capacity_kwh * float(costs["battery_capex_ntd_per_kwh"])
            + battery_power_kw * float(costs["battery_power_capex_ntd_per_kw"])
        )
        battery_om_cost = battery_init_cost * float(costs["battery_om_fraction_of_capex_per_year"]) * pw_factor

        replacement_cost = np.zeros(len(out), dtype=float)
        replacement_year = float(costs["battery_replacement_year"])
        if replacement_year > 0:
            n_replacements = int(np.floor((float(years) - 1e-9) / replacement_year))
            if n_replacements > 0:
                replacement_unit_cost = (
                    battery_capacity_kwh * float(costs["battery_replacement_energy_capex_ntd_per_kwh"])
                    + battery_power_kw * float(costs["battery_replacement_power_capex_ntd_per_kw"])
                )
                discount_sum = 0.0
                for i in range(1, n_replacements + 1):
                    discount_sum += 1.0 if abs(discount_rate) < 1e-12 else 1.0 / (
                        (1.0 + discount_rate) ** (replacement_year * i)
                    )
                replacement_cost = replacement_unit_cost * discount_sum

        s_init_cost = (
            pv_capacity_kw * float(costs["pv_capex_ntd_per_kw"])
            + battery_init_cost
            + float(costs["other_capex"])
        )
        s_om_cost = (
            pv_capacity_kw * float(costs["pv_om_ntd_per_kw_year"]) * pw_factor
            + battery_om_cost
            + (float(costs["other_om"]) / float(years)) * pw_factor
        )
        pv_eol_cost = _discount_lump(
            1.0,
            float(years),
            discount_rate,
        ) * (pv_capacity_kw * float(costs["pv_eol_ntd_per_kw"]))
        battery_eol_cost = _discount_lump(float(costs["battery_eol"]), float(years), discount_rate)
        other_eol_cost = _discount_lump(float(costs["other_eol"]), float(years), discount_rate)
        salvage_credit = _discount_lump(
            1.0,
            float(years),
            discount_rate,
        ) * (
            pv_capacity_kw * float(costs["pv_capex_ntd_per_kw"]) * float(financial["pv_salvage_fraction"])
            + battery_init_cost * float(financial["battery_salvage_fraction"])
            + float(costs["other_capex"]) * float(financial["other_salvage_fraction"])
        )
        s_eol_cost = pv_eol_cost + battery_eol_cost + other_eol_cost + replacement_cost - salvage_credit
    else:
        solar_panel_factor = out["solar_panel_factor"].astype(float)
        battery_factor = out["battery_factor"].astype(float)
        s_init_cost = (
            float(costs["solar_capex"]) * solar_panel_factor
            + float(costs["battery_capex"]) * battery_factor
            + float(costs["other_capex"])
        )
        s_om_cost = (
            ((float(costs["solar_om"]) * solar_panel_factor) / float(years)) * pw_factor
            + ((float(costs["battery_om"]) * battery_factor) / float(years)) * pw_factor
            + (float(costs["other_om"]) / float(years)) * pw_factor
        )
        salvage_credit = _discount_lump(
            1.0,
            float(years),
            discount_rate,
        ) * (
            float(costs["solar_capex"]) * solar_panel_factor * float(financial["pv_salvage_fraction"])
            + float(costs["battery_capex"]) * battery_factor * float(financial["battery_salvage_fraction"])
            + float(costs["other_capex"]) * float(financial["other_salvage_fraction"])
        )
        s_eol_cost = (
            _discount_lump(1.0, float(years), discount_rate)
            * (
                float(costs["solar_eol"]) * solar_panel_factor
                + float(costs["battery_eol"]) * battery_factor
                + float(costs["other_eol"])
            )
            - salvage_credit
        )

    out["pv_storage_capex_cost"] = s_init_cost
    out["pv_storage_om_cost"] = s_om_cost
    out["pv_storage_eol_cost"] = s_eol_cost
    out["pv_storage_cost"] = out["pv_storage_energy_cost"] + out["pv_storage_capex_cost"] + out["pv_storage_om_cost"] + out["pv_storage_eol_cost"]
    out["delta_cost"] = out["pv_storage_cost"] - out["grid_only_cost"]
    return out


def run_five_regions_split(
    power_by_region: dict[str, pd.DataFrame],
    AEF_by_region: dict[str, pd.Series],
    region_cols: dict[str, list[str]],
    years: float,
    solar_panel_factor: float,
    battery_factor: float,
    elec_price: float,
    params: dict,
    economic_costs: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Process all regions and cities for emission analysis.

    Parameters
    ----------
    power_by_region : dict
        Dictionary mapping region names to power DataFrames (kW)
    AEF_by_region : dict
        Dictionary mapping region names to AEF Series (kg CO2e/kWh)
    region_cols : dict
        Dictionary mapping region names to lists of city columns
    years : float
        Analysis period in years
    solar_panel_factor, battery_factor : float
        Capacity scaling factors
    elec_price : float
        Electricity price (NTD/kWh)
    params : dict
        Storage parameters
    economic_costs : dict
        Economic cost coefficients for cost model.

    Returns
    -------
    df_city : pd.DataFrame
        City-level results with MultiIndex (region, city)
    df_region : pd.DataFrame
        Region-level aggregated results
    total_row : pd.DataFrame
        Total across all regions
    dt_hours : float
        Time step in hours
    """
    # Find common time index
    idx_list = [s.dropna().index for s in AEF_by_region.values() if isinstance(s, pd.Series)]
    for r_df in power_by_region.values():
        idx_list.append(r_df.index)
    if not idx_list:
        raise ValueError("No available time indices.")
    common_index = idx_list[0]
    for idx in idx_list[1:]:
        common_index = common_index.intersection(idx)
    if len(common_index) == 0:
        raise ValueError(
            "No time index intersection. Check AEF and power_df time ranges."
        )

    # Reindex to common timeline
    power_by_region = {
        r: df.sort_index().reindex(common_index).fillna(0.0)
        for r, df in power_by_region.items()
    }
    AEF_by_region = {
        r: s.sort_index().reindex(common_index).ffill().bfill()
        for r, s in AEF_by_region.items()
    }

    dt_hours = infer_dt_hours(common_index)

    city_records = []
    missing_aef_regions, missing_city_map = [], {}

    for region, df in power_by_region.items():
        if region not in AEF_by_region:
            missing_aef_regions.append(region)
            continue
        aef = AEF_by_region[region]

        if region_cols and region in region_cols:
            cities = [c for c in region_cols[region] if c in df.columns]
            not_found = [c for c in region_cols[region] if c not in df.columns]
            if not_found:
                missing_city_map.setdefault(region, []).extend(not_found)
        else:
            cities = list(df.columns)

        for city in cities:
            out = compute_city_emission(
                df[city], aef, dt_hours, years,
                solar_panel_factor, battery_factor, elec_price, params, economic_costs
            )
            out.update({"region": region, "city": city})
            city_records.append(out)

    if not city_records:
        raise ValueError("No cities calculated. Check input data.")

    df_city = (pd.DataFrame.from_records(city_records)
               .set_index(["region", "city"])
               .sort_index())

    # Regional aggregation
    sum_cols = [
        "grid_only_emission_t", "pv_storage_emission_t", "delta_t",
        "grid_energy_kwh_20y", "pv_storage_energy_kwh_20y",
        "grid_only_cost", "pv_storage_cost", "delta_cost"
    ]
    df_region = df_city.groupby(level="region")[sum_cols].sum().sort_index()
    df_region["n_cities"] = df_city.groupby(level="region").size()

    # Total across all regions
    total_row = pd.DataFrame({
        "grid_only_emission_t": [df_city["grid_only_emission_t"].sum()],
        "pv_storage_emission_t": [df_city["pv_storage_emission_t"].sum()],
        "delta_t": [df_city["delta_t"].sum()],
        "grid_energy_kwh_20y": [df_city["grid_energy_kwh_20y"].sum()],
        "pv_storage_energy_kwh_20y": [df_city["pv_storage_energy_kwh_20y"].sum()],
        "grid_only_cost": [df_city["grid_only_cost"].sum()],
        "pv_storage_cost": [df_city["pv_storage_cost"].sum()],
        "delta_cost": [df_city["delta_cost"].sum()],
        "n_cities": [df_city.shape[0]]
    }, index=["全台合計"])

    if missing_aef_regions or missing_city_map:
        print("⚠️ 注意：")
        if missing_aef_regions:
            print("  - 下列區域缺少 AEF，已略過：", ", ".join(missing_aef_regions))
        for r, lst in missing_city_map.items():
            print(f"  - 區域「{r}」中以下縣市在該區 power_df 沒找到，未計算：{', '.join(lst)}")

    return df_city, df_region, total_row, dt_hours


def pareto_front(
    df: pd.DataFrame,
    col_x: str = "abatement_t",
    col_y: str = "delta_cost"
) -> pd.DataFrame:
    """
    Extract Pareto frontier (dominated solutions).

    Parameters
    ----------
    df : pd.DataFrame
        Data with cost and emission reduction columns
    col_x : str
        Column for x-axis (emission reduction, should be maximized)
    col_y : str
        Column for y-axis (cost difference, should be minimized)

    Returns
    -------
    pd.DataFrame
        Dataframe containing only Pareto-optimal points

    Notes
    -----
    A point (x1, y1) dominates (x2, y2) if:
    - x1 >= x2 AND y1 <= y2 (better or equal reduction at lower cost)
    """
    if df.empty:
        return df.copy()

    xy = df[[col_x, col_y]].to_numpy()
    n = len(xy)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        # Check if point i is dominated by any other point
        dom = (xy[:, 0] >= xy[i, 0]) & (xy[:, 1] <= xy[i, 1]) & \
              ((xy[:, 0] > xy[i, 0]) | (xy[:, 1] < xy[i, 1]))
        if np.any(dom):
            keep[i] = False

    return df.iloc[keep].copy()


def pareto_frontier_min_cost_max_abatement(
    df: pd.DataFrame,
    col_x: str = "abatement_t",
    col_y: str = "delta_cost",
    epsilon: float = 0.0
) -> pd.DataFrame:
    """
    Extract Pareto frontier ensuring maximum abatement for each cost level.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with cost and emission columns
    col_x, col_y : str
        Column names for emission reduction and cost difference
    epsilon : float
        Tolerance for treating near-equal abatement values

    Returns
    -------
    pd.DataFrame
        Pareto frontier sorted by cost

    Notes
    -----
    For each cost level, select the configuration with maximum abatement.
    """
    d = df[[col_x, col_y, "solar_panel_factor", "battery_factor"]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_y, col_x])
    d = d.sort_values(by=[col_y, col_x], ascending=[True, False]).reset_index(drop=True)

    frontier_rows, best_abatement = [], -np.inf
    for _, row in d.iterrows():
        if row[col_x] >= best_abatement + epsilon:
            frontier_rows.append(row)
            best_abatement = row[col_x]

    if not frontier_rows:
        return pd.DataFrame()

    df_frontier = pd.DataFrame(frontier_rows)
    return pareto_front(df_frontier, col_x=col_x, col_y=col_y)


def compute_macc_from_frontier(
    frontier: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate Marginal Abatement Cost (MACC) from Pareto frontier.

    Parameters
    ----------
    frontier : pd.DataFrame
        Pareto frontier DataFrame sorted by cost (ascending)

    Returns
    -------
    pd.DataFrame
        DataFrame with MACC segments:
        - abate_left: Left cumulative abatement (t)
        - abate_right: Right cumulative abatement (t)
        - delta_abate: Emission reduction in segment (t)
        - mac: Marginal abatement cost ($/t)
        - solar_panel_factor, battery_factor: Configuration markers

    Notes
    -----
    MACC_i = Δcost_i / Δabatement_i

    Segments computed between consecutive points on frontier.
    """
    if frontier.shape[0] < 2:
        return pd.DataFrame()

    f = frontier.sort_values("delta_cost").reset_index(drop=True)

    # Cumulative abatement as MACC x-axis
    cum_abate = f["abatement_t"].values
    costs = f["delta_cost"].values

    rows = []
    for i in range(len(f) - 1):
        da = cum_abate[i + 1] - cum_abate[i]
        dc = costs[i + 1] - costs[i]
        if da <= 0:
            # Should not happen in proper frontier; skip if occurs
            continue
        mac = dc / da
        rows.append({
            "abate_left": cum_abate[i],
            "abate_right": cum_abate[i + 1],
            "delta_abate": da,
            "mac": mac,
            "solar_panel_factor": f.loc[i + 1, "solar_panel_factor"],
            "battery_factor": f.loc[i + 1, "battery_factor"]
        })
    return pd.DataFrame(rows)


def find_knee_point(
    frontier: pd.DataFrame
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Find knee point on Pareto frontier using the normalized geometric method.

    Parameters
    ----------
    frontier : pd.DataFrame
        Pareto frontier DataFrame with abatement_t and cost columns

    Returns
    -------
    Tuple[int, float, float]
        - Index of knee point (None if insufficient points)
        - x-coordinate (abatement at knee, None if insufficient)
        - y-coordinate (cost at knee, None if insufficient)

    Notes
    -----
    Knee point is the point farthest from the line connecting the
    minimum-cost and maximum-abatement endpoints after min-max normalizing
    abatement and cost. The returned index is a positional index into the
    input DataFrame, matching callers that use ``frontier.iloc[idx]``.
    """
    if len(frontier) < 3:
        return None, None, None

    x = frontier["abatement_t"].to_numpy(dtype=float)
    y = frontier["delta_cost"].to_numpy(dtype=float)

    # Endpoints are defined by the decision rule, not by row order.
    min_cost_idx = int(np.lexsort((-x, y))[0])
    max_abatement_idx = int(np.lexsort((y, -x))[0])

    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    x_norm = np.zeros_like(x) if x_span == 0 else (x - x.min()) / x_span
    y_norm = np.zeros_like(y) if y_span == 0 else (y - y.min()) / y_span

    p1 = np.array([x_norm[min_cost_idx], y_norm[min_cost_idx]])
    p2 = np.array([x_norm[max_abatement_idx], y_norm[max_abatement_idx]])

    # Convert to vector form, compute distance from the endpoint chord.
    line_vec = p2 - p1
    line_norm = np.linalg.norm(line_vec)
    if line_norm == 0:
        idx_knee = len(frontier) // 2
        return idx_knee, x[idx_knee], y[idx_knee]

    line_vec_norm = line_vec / line_norm
    point_vecs = np.vstack([x_norm, y_norm]).T - p1
    proj = np.dot(point_vecs, line_vec_norm)
    proj_point = np.outer(proj, line_vec_norm)
    dist = np.linalg.norm(point_vecs - proj_point, axis=1)

    max_dist = float(np.max(dist))
    tied = np.flatnonzero(np.isclose(dist, max_dist, rtol=1e-12, atol=1e-15))
    midpoint = (len(frontier) - 1) / 2
    idx_knee = int(tied[np.argmin(np.abs(tied - midpoint))])
    return idx_knee, x[idx_knee], y[idx_knee]


def build_city_metrics(
    city_results_df: pd.DataFrame,
    region: Optional[str] = None
) -> pd.DataFrame:
    """
    Build summary metrics for all cities.

    Parameters
    ----------
    city_results_df : pd.DataFrame
        DataFrame with results from multiple configurations
    region : str, optional
        Filter to specific region

    Returns
    -------
    pd.DataFrame
        Summary table with:
        - max_abatement_t: Maximum emission reduction potential
        - cost_at_max_abatement: Cost at max reduction
        - knee_abatement_t: Knee point abatement (t)
        - cost_at_knee: Cost at knee point
        - knee_share_of_max: Knee as share of max potential
        - s, b: Solar and battery factors for max/knee points

    Notes
    -----
    Used for ranking cities by abatement potential and cost-effectiveness.
    """
    df = city_results_df.copy()

    # Reset index if needed
    for name in ["city", "region"]:
        if name in (df.index.names or []):
            df = df.reset_index(level=name)

    # Add abatement if not present
    if "abatement_t" not in df.columns and "delta_t" in df.columns:
        df["abatement_t"] = -df["delta_t"]

    # Add cost per tonne if not present
    if "cpt" not in df.columns:
        df["cpt"] = df["delta_cost"] / df["abatement_t"]

    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["abatement_t", "delta_cost"])

    # Ensure numeric types
    for col in ["solar_panel_factor", "battery_factor", "abatement_t", "delta_cost"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Filter by region if specified
    if region is not None and "region" in df.columns:
        df = df[df["region"] == region].copy()

    if "city" not in df.columns:
        raise KeyError("City column not found in DataFrame")

    rows = []
    for city_name, d_city in df.groupby("city", dropna=True):
        if d_city.empty:
            continue

        # Get region for this city
        if "region" in d_city.columns:
            reg_series = d_city["region"].dropna()
            reg_raw = reg_series.mode().iloc[0] if not reg_series.empty else "Unknown"
        else:
            reg_raw = "Unknown"

        # Extract Pareto frontier for this city
        d_pf = pareto_front(d_city, col_x="abatement_t", col_y="delta_cost").sort_values("abatement_t")
        if d_pf.empty:
            continue

        # Maximum abatement
        idx_max = d_pf["abatement_t"].idxmax()
        x_max = float(d_pf.loc[idx_max, "abatement_t"])
        y_at_max = float(d_pf.loc[idx_max, "delta_cost"])

        # Knee point
        idx_knee, x_knee, y_knee = find_knee_point(d_pf)
        if x_knee is None:
            x_knee, y_knee, knee_ratio = np.nan, np.nan, np.nan
        else:
            x_knee = float(x_knee)
            y_knee = float(y_knee)
            knee_ratio = (x_knee / x_max) if x_max > 0 else np.nan

        rows.append({
            "city": city_name,
            "region": reg_raw,
            "max_abatement_t": x_max,
            "cost_at_max_abatement": y_at_max,
            "knee_abatement_t": x_knee,
            "cost_at_knee": y_knee,
            "knee_share_of_max": knee_ratio,
        })

    metrics = pd.DataFrame(rows).sort_values("max_abatement_t", ascending=False).reset_index(drop=True)
    return metrics
