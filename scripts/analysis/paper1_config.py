from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "paper_baseline.yaml"
DEFAULT_LOAD_SIDE_REGIONS = [
    "north",
    "central",
    "south",
    "east",
    "island_penghu",
    "island_kinmen",
    "island_lienchiang",
]
REGION_LABELS = {
    "north": "north",
    "central": "central",
    "south": "south",
    "east": "east",
    "island_penghu": "Penghu island load-serving zone",
    "island_kinmen": "Kinmen island load-serving zone",
    "island_lienchiang": "Lienchiang island load-serving zone",
}
CITY_LABELS = {
    "臺北市": "Taipei",
    "新北市": "New Taipei",
    "基隆市": "Keelung",
    "桃園市": "Taoyuan",
    "新竹市": "Hsinchu City",
    "新竹縣": "Hsinchu County",
    "苗栗縣": "Miaoli",
    "臺中市": "Taichung",
    "彰化縣": "Changhua",
    "南投縣": "Nantou",
    "雲林縣": "Yunlin",
    "嘉義市": "Chiayi City",
    "嘉義縣": "Chiayi County",
    "臺南市": "Tainan",
    "高雄市": "Kaohsiung",
    "屏東縣": "Pingtung",
    "宜蘭縣": "Yilan",
    "花蓮縣": "Hualien",
    "臺東縣": "Taitung",
    "澎湖縣": "Penghu",
    "金門縣": "Kinmen",
    "連江縣": "Lienchiang",
}


def _deep_get(data: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _resolve_repo_path(value: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _fmt_int(value: float) -> str:
    return f"{int(round(value)):,}"


def _fmt_1(value: float) -> str:
    return f"{value:.1f}"


def _fmt_2(value: float) -> str:
    return f"{value:.2f}"


def _fmt_3(value: float) -> str:
    return f"{value:.3f}"


def _fmt_4(value: float) -> str:
    return f"{value:.4f}"


def _fmt_5(value: float) -> str:
    return f"{value:.5f}"


def _pct_int(value: float) -> str:
    return f"{int(round(value * 100.0))}"


def _range_values(config: dict[str, Any]) -> list[float]:
    start = float(config.get("min", 0.5))
    stop = float(config.get("max", 3.0))
    step = float(config.get("step", 0.5))
    count = int(round((stop - start) / step)) + 1
    return [start + i * step for i in range(count)]


def _present_worth_factor(years: float, discount_rate: float) -> float:
    if abs(discount_rate) < 1e-12:
        return float(years)
    return (1.0 - (1.0 + float(discount_rate)) ** (-float(years))) / float(discount_rate)


def _md_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


@dataclass(frozen=True)
class Paper1Config:
    config_path: Path
    raw: dict[str, Any]

    @property
    def workflow(self) -> dict[str, Any]:
        return dict(_deep_get(self.raw, ("paper", "workflow"), {}))

    @property
    def manuscript(self) -> dict[str, Any]:
        return dict(_deep_get(self.raw, ("paper", "manuscript"), {}))

    @property
    def paper_inputs(self) -> dict[str, Any]:
        return dict(_deep_get(self.raw, ("paper", "inputs"), {}))

    @property
    def canonical_results_dir(self) -> Path:
        return _resolve_repo_path(
            self.workflow.get("canonical_results_dir", "outputs/final_runs/paper1_canonical_results")
        )

    @property
    def canonical_inputs_dir(self) -> Path:
        return _resolve_repo_path(
            self.workflow.get("canonical_inputs_dir", "outputs/final_runs/paper1_canonical_inputs")
        )

    @property
    def paper_assets_dir(self) -> Path:
        return _resolve_repo_path(self.workflow.get("paper_assets_dir", "outputs/paper_assets/paper1"))

    @property
    def figures_dir(self) -> Path:
        return _resolve_repo_path(
            self.workflow.get("figures_dir", "outputs/paper_assets/paper1/figures")
        )

    @property
    def si_figures_dir(self) -> Path:
        return _resolve_repo_path(
            self.workflow.get("si_figures_dir", "outputs/paper_assets/paper1/si_figures")
        )

    @property
    def figure_data_dir(self) -> Path:
        return _resolve_repo_path(
            self.workflow.get("figure_data_dir", "outputs/paper_assets/paper1/figure_data")
        )

    @property
    def manuscript_values_path(self) -> Path:
        return _resolve_repo_path(
            self.workflow.get(
                "manuscript_values_path",
                "outputs/final_runs/paper1_canonical_results/paper1_manuscript_values.json",
            )
        )

    @property
    def region_map_path(self) -> Path:
        return _resolve_repo_path(
            self.paper_inputs.get("region_map", "config/region_city_map.json")
        )

    @property
    def region_city_map(self) -> dict[str, list[str]]:
        data = json.loads(self.region_map_path.read_text(encoding="utf-8"))
        return {
            str(region): [str(city) for city in cities]
            for region, cities in data.items()
        }

    @property
    def selected_design(self) -> dict[str, Any]:
        return dict(_deep_get(self.raw, ("paper", "selected_design"), {}))

    @property
    def selected_design_label(self) -> str:
        return str(self.selected_design.get("label", "knee"))

    @property
    def selected_solar_panel_factor(self) -> float:
        return float(self.selected_design.get("solar_panel_factor", 1.85))

    @property
    def selected_battery_factor(self) -> float:
        return float(self.selected_design.get("battery_factor", 8.45))

    @property
    def study_city_count(self) -> int:
        return int(_deep_get(self.raw, ("paper", "study", "city_count"), 22))

    @property
    def load_side_region_count(self) -> int:
        return len(self.load_side_regions)

    @property
    def load_side_regions(self) -> list[str]:
        return list(_deep_get(self.raw, ("regions", "list"), DEFAULT_LOAD_SIDE_REGIONS))

    @property
    def par_source_spatial_resolution_km(self) -> int:
        return int(_deep_get(self.raw, ("par", "upsample_factor"), 5))

    @property
    def light_power_kw(self) -> float:
        return float(_deep_get(self.raw, ("simulation", "standardized_installation", "light_power_kw"), 0.1))

    @property
    def solar_zenith_deg(self) -> float:
        return float(
            _deep_get(self.raw, ("simulation", "standardized_installation", "solar_zenith_deg"), 90.833)
        )

    @property
    def lights_per_city(self) -> int:
        return int(_deep_get(self.raw, ("simulation", "standardized_installation", "n_lights"), 64))

    @property
    def study_light_count(self) -> int:
        return self.study_city_count * self.lights_per_city

    @property
    def connected_load_kw_per_city(self) -> float:
        return self.light_power_kw * self.lights_per_city

    @property
    def battery_power_cap_multiplier_of_load(self) -> float:
        return float(_deep_get(self.raw, ("economics", "pareto", "battery", "power_cap_multiplier_of_load"), 1.5))

    @property
    def battery_power_cap_kw_per_city(self) -> float:
        return self.connected_load_kw_per_city * self.battery_power_cap_multiplier_of_load

    @property
    def ntd_per_usd(self) -> float:
        return float(_deep_get(self.raw, ("paper", "currency", "ntd_per_usd"), 32.108))

    @property
    def usd_per_ntd(self) -> float:
        return 1.0 / self.ntd_per_usd

    @property
    def storage_capacity_kwh_per_battery_factor(self) -> float:
        return float(_deep_get(self.raw, ("storage_dispatch", "default", "capacity_kwh"), 10.0))

    @property
    def storage_power_kw_per_battery_factor(self) -> float:
        return float(_deep_get(self.raw, ("storage_dispatch", "default", "power_kw"), 5.0))

    @property
    def storage_power_kw_per_kwh(self) -> float:
        return self.storage_power_kw_per_battery_factor / self.storage_capacity_kwh_per_battery_factor

    @property
    def storage_roundtrip_efficiency(self) -> float:
        return float(_deep_get(self.raw, ("storage_dispatch", "default", "eta_roundtrip"), 0.90))

    @property
    def storage_initial_soc_fraction(self) -> float:
        return float(_deep_get(self.raw, ("storage_dispatch", "default", "init_soc"), 0.0))

    @property
    def pareto_solar_cfg(self) -> dict[str, Any]:
        return dict(_deep_get(self.raw, ("economics", "pareto", "solar"), {}))

    @property
    def pareto_battery_cfg(self) -> dict[str, Any]:
        return dict(_deep_get(self.raw, ("economics", "pareto", "battery"), {}))

    @property
    def pareto_solar_factors(self) -> list[float]:
        return _range_values(self.pareto_solar_cfg)

    @property
    def pareto_battery_factors(self) -> list[float]:
        return _range_values(self.pareto_battery_cfg)

    @property
    def pareto_allocation_count(self) -> int:
        return len(self.pareto_solar_factors) * len(self.pareto_battery_factors)

    @property
    def pv_capacity_kw_per_solar_factor(self) -> float:
        return float(self.economic_costs.get("pv_capacity_kw_per_factor", 18.0))

    @property
    def pareto_pv_kw_min_per_light(self) -> float:
        return min(self.pareto_solar_factors) * self.pv_capacity_kw_per_solar_factor / self.lights_per_city

    @property
    def pareto_pv_kw_max_per_light(self) -> float:
        return max(self.pareto_solar_factors) * self.pv_capacity_kw_per_solar_factor / self.lights_per_city

    @property
    def pareto_battery_kwh_min_per_light(self) -> float:
        return (
            min(self.pareto_battery_factors)
            * self.storage_capacity_kwh_per_battery_factor
            / self.lights_per_city
        )

    @property
    def pareto_battery_kwh_max_per_light(self) -> float:
        return (
            max(self.pareto_battery_factors)
            * self.storage_capacity_kwh_per_battery_factor
            / self.lights_per_city
        )

    @property
    def analysis_years(self) -> float:
        return float(_deep_get(self.raw, ("economics", "analysis_years"), 20.0))

    @property
    def electricity_price_ntd_per_kwh(self) -> float:
        return float(_deep_get(self.raw, ("economics", "electricity_price"), 3.7556))

    @property
    def electricity_price_usd_per_kwh(self) -> float:
        return self.electricity_price_ntd_per_kwh / self.ntd_per_usd

    @property
    def economic_costs(self) -> dict[str, Any]:
        return dict(_deep_get(self.raw, ("economics", "cost"), {}))

    @property
    def financial_params(self) -> dict[str, Any]:
        return dict(_deep_get(self.raw, ("economics", "financial"), {}))

    @property
    def economic_scenarios(self) -> dict[str, Any]:
        return dict(_deep_get(self.raw, ("economics", "scenarios"), {}))

    @property
    def discount_rate(self) -> float:
        return float(self.financial_params.get("discount_rate", 0.05))

    @property
    def optimistic_discount_rate(self) -> float:
        return float(
            _deep_get(self.raw, ("economics", "scenarios", "optimistic", "financial", "discount_rate"), 0.03)
        )

    @property
    def conservative_discount_rate(self) -> float:
        return float(
            _deep_get(self.raw, ("economics", "scenarios", "conservative", "financial", "discount_rate"), 0.08)
        )

    @property
    def sensitivity(self) -> dict[str, Any]:
        return dict(_deep_get(self.raw, ("paper", "sensitivity"), {}))

    @property
    def uncertainty(self) -> dict[str, Any]:
        return dict(_deep_get(self.raw, ("paper", "uncertainty"), {}))

    @property
    def electricity_price_multipliers(self) -> list[float]:
        return [float(v) for v in self.sensitivity.get("electricity_price_multipliers", [0.8, 1.0, 1.2])]

    @property
    def battery_replacement_year_values(self) -> list[float]:
        return [float(v) for v in self.uncertainty.get("battery_replacement_year_values", [10, 12, 14])]

    @property
    def battery_replacement_year_probabilities(self) -> list[float]:
        return [
            float(v)
            for v in self.uncertainty.get("battery_replacement_year_probabilities", [0.25, 0.5, 0.25])
        ]

    @property
    def city_name_overrides(self) -> dict[str, str]:
        return {
            str(key): str(value)
            for key, value in dict(self.manuscript.get("city_name_overrides", {})).items()
        }

    @property
    def future_grid_scenario_labels(self) -> dict[str, str]:
        return {
            str(key): str(value)
            for key, value in dict(self.manuscript.get("future_grid_scenario_labels", {})).items()
        }

    @property
    def sensitivity_labels(self) -> dict[str, str]:
        sensitivity = dict(self.manuscript.get("sensitivity", {}))
        return {str(key): str(value) for key, value in dict(sensitivity.get("labels", {})).items()}

    @property
    def sensitivity_tested_values(self) -> dict[str, str]:
        sensitivity = dict(self.manuscript.get("sensitivity", {}))
        return {
            str(key): str(value)
            for key, value in dict(sensitivity.get("tested_values", {})).items()
        }

    def public_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path.relative_to(ROOT).as_posix(),
            "workflow": {
                "canonical_results_dir": self.canonical_results_dir.relative_to(ROOT).as_posix(),
                "canonical_inputs_dir": self.canonical_inputs_dir.relative_to(ROOT).as_posix(),
                "paper_assets_dir": self.paper_assets_dir.relative_to(ROOT).as_posix(),
                "figures_dir": self.figures_dir.relative_to(ROOT).as_posix(),
                "si_figures_dir": self.si_figures_dir.relative_to(ROOT).as_posix(),
                "figure_data_dir": self.figure_data_dir.relative_to(ROOT).as_posix(),
                "manuscript_values_path": self.manuscript_values_path.relative_to(ROOT).as_posix(),
            },
            "study": {
                "city_count": self.study_city_count,
                "load_side_regions": self.load_side_regions,
                "light_power_kw": self.light_power_kw,
                "solar_zenith_deg": self.solar_zenith_deg,
                "lights_per_city": self.lights_per_city,
                "study_light_count": self.study_light_count,
                "connected_load_kw_per_city": self.connected_load_kw_per_city,
            },
            "selected_design": {
                "label": self.selected_design_label,
                "solar_panel_factor": self.selected_solar_panel_factor,
                "battery_factor": self.selected_battery_factor,
            },
            "currency": {
                "ntd_per_usd": self.ntd_per_usd,
                "usd_per_ntd": self.usd_per_ntd,
            },
            "economics": {
                "analysis_years": self.analysis_years,
                "electricity_price_ntd_per_kwh": self.electricity_price_ntd_per_kwh,
                "electricity_price_usd_per_kwh": self.electricity_price_usd_per_kwh,
                "discount_rate": self.discount_rate,
                "optimistic_discount_rate": self.optimistic_discount_rate,
                "conservative_discount_rate": self.conservative_discount_rate,
                "cost": self.economic_costs,
                "pareto": {
                    "solar": self.pareto_solar_cfg,
                    "battery": self.pareto_battery_cfg,
                    "solar_level_count": len(self.pareto_solar_factors),
                    "battery_level_count": len(self.pareto_battery_factors),
                    "allocation_count": self.pareto_allocation_count,
                    "pv_kw_min_per_light": self.pareto_pv_kw_min_per_light,
                    "pv_kw_max_per_light": self.pareto_pv_kw_max_per_light,
                    "battery_kwh_min_per_light": self.pareto_battery_kwh_min_per_light,
                    "battery_kwh_max_per_light": self.pareto_battery_kwh_max_per_light,
                },
            },
            "storage_dispatch": {
                "capacity_kwh_per_battery_factor": self.storage_capacity_kwh_per_battery_factor,
                "power_kw_per_battery_factor": self.storage_power_kw_per_battery_factor,
                "power_kw_per_kwh": self.storage_power_kw_per_kwh,
                "roundtrip_efficiency": self.storage_roundtrip_efficiency,
                "initial_soc_fraction": self.storage_initial_soc_fraction,
                "initial_soc_fraction_scope": (
                    "Standalone streetlight_sim diagnostic only; "
                    "config.storage.initial_soc_pct_int is its legacy display token, not the Pareto sweep initial state."
                ),
                "pareto_sweep_initial_soc_kwh": 0.0,
                "pareto_sweep_initial_soc_source": (
                    "storage.storage_dispatch default soc0_kwh; "
                    "the Pareto sweep does not pass the diagnostic init_soc setting."
                ),
            },
            "manuscript": {
                "city_name_overrides": self.city_name_overrides,
                "future_grid_scenario_labels": self.future_grid_scenario_labels,
                "sensitivity": {
                    "labels": self.sensitivity_labels,
                    "tested_values": self.sensitivity_tested_values,
                },
            },
        }


def load_paper1_config(config_path: str | Path | None = None) -> Paper1Config:
    path = Path(config_path or os.environ.get("STREETLIGHT_CONFIG", DEFAULT_CONFIG_PATH))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Paper1Config(config_path=path, raw=dict(raw))


def build_config_render_tokens(config: Paper1Config) -> dict[str, str]:
    electricity_values = [
        config.electricity_price_usd_per_kwh * multiplier
        for multiplier in config.electricity_price_multipliers
    ]
    cost = config.economic_costs
    scenarios = config.economic_scenarios
    optimistic_cost = dict(_deep_get(scenarios, ("optimistic", "cost"), {}))
    conservative_cost = dict(_deep_get(scenarios, ("conservative", "cost"), {}))

    def usd_cost(name: str, source: dict[str, Any] | None = None) -> float:
        values = cost if source is None else source
        return float(values[name]) / config.ntd_per_usd

    grid_pv_usd = (
        (float(cost.get("grid_fixed_cost", 95000.0)) / config.analysis_years)
        * _present_worth_factor(config.analysis_years, config.discount_rate)
        / config.ntd_per_usd
    )
    grid_fixed_usd = float(cost.get("grid_fixed_cost", 95000.0)) / config.ntd_per_usd

    return {
        "config.study.city_count": str(config.study_city_count),
        "config.study.load_side_region_count_int": str(config.load_side_region_count),
        "config.study.light_power_kw_1": f"{config.light_power_kw:.1f}",
        "config.study.solar_zenith_deg_3": f"{config.solar_zenith_deg:.3f}",
        "config.study.lights_per_city": str(config.lights_per_city),
        "config.study.study_light_count": str(config.study_light_count),
        "config.study.connected_load_kw_per_light_1": _fmt_1(config.light_power_kw),
        "config.par.source_spatial_resolution_km_int": str(config.par_source_spatial_resolution_km),
        "config.pareto.solar_level_count_int": _fmt_int(len(config.pareto_solar_factors)),
        "config.pareto.battery_level_count_int": _fmt_int(len(config.pareto_battery_factors)),
        "config.pareto.allocation_count_int": _fmt_int(config.pareto_allocation_count),
        "config.pareto.pv_kw_min_per_light_2": f"{config.pareto_pv_kw_min_per_light:.2f}",
        "config.pareto.pv_kw_max_per_light_2": f"{config.pareto_pv_kw_max_per_light:.2f}",
        "config.pareto.battery_kwh_min_per_light_2": f"{config.pareto_battery_kwh_min_per_light:.2f}",
        "config.pareto.battery_kwh_max_per_light_2": f"{config.pareto_battery_kwh_max_per_light:.2f}",
        "config.storage.power_cap_multiplier_of_load_1": _fmt_1(
            config.battery_power_cap_multiplier_of_load
        ),
        "config.storage.power_cap_kw_per_light_2": _fmt_2(
            config.battery_power_cap_kw_per_city / config.lights_per_city
        ),
        "config.storage.power_kw_per_kwh_1": _fmt_1(config.storage_power_kw_per_kwh),
        "config.storage.roundtrip_efficiency_2": f"{config.storage_roundtrip_efficiency:.2f}",
        "config.storage.initial_soc_pct_int": _fmt_int(config.storage_initial_soc_fraction * 100.0),
        "config.selected_design.label": config.selected_design_label,
        "config.selected_design.solar_panel_factor_2": (
            f"{config.selected_solar_panel_factor:.2f}"
        ),
        "config.selected_design.battery_factor_2": f"{config.selected_battery_factor:.2f}",
        "config.currency.ntd_per_usd_3": f"{config.ntd_per_usd:.3f}",
        "config.economics.analysis_years_int": str(int(round(config.analysis_years))),
        "config.economics.discount_rate_decimal_2": f"{config.discount_rate:.2f}",
        "config.economics.discount_rate_pct_int": _pct_int(config.discount_rate),
        "config.economics.discount_rate_low_pct_int": _pct_int(config.optimistic_discount_rate),
        "config.economics.discount_rate_high_pct_int": _pct_int(config.conservative_discount_rate),
        "config.economics.pv_capacity_kw_per_solar_factor_per_light_2": _fmt_2(
            config.pv_capacity_kw_per_solar_factor / config.lights_per_city
        ),
        "config.economics.pv_capacity_kw_per_solar_factor_per_light_5": _fmt_5(
            config.pv_capacity_kw_per_solar_factor / config.lights_per_city
        ),
        "config.economics.electricity_price_ntd_per_kwh_4": (
            _fmt_4(config.electricity_price_ntd_per_kwh)
        ),
        "config.economics.electricity_price_usd_per_kwh_3": (
            _fmt_3(config.electricity_price_usd_per_kwh)
        ),
        "config.economics.electricity_price_usd_low_3": _fmt_3(electricity_values[0]),
        "config.economics.electricity_price_usd_mid_3": _fmt_3(electricity_values[1]),
        "config.economics.electricity_price_usd_high_3": _fmt_3(electricity_values[2]),
        "config.cost.pv_capex_usd_per_kw_int": _fmt_int(usd_cost("pv_capex_ntd_per_kw")),
        "config.cost.pv_capex_usd_low_int": _fmt_int(
            usd_cost("pv_capex_ntd_per_kw", optimistic_cost)
        ),
        "config.cost.pv_capex_usd_high_int": _fmt_int(
            usd_cost("pv_capex_ntd_per_kw", conservative_cost)
        ),
        "config.cost.pv_om_usd_per_kw_year_1": _fmt_1(usd_cost("pv_om_ntd_per_kw_year")),
        "config.cost.battery_capex_usd_per_kwh_int": _fmt_int(
            usd_cost("battery_capex_ntd_per_kwh")
        ),
        "config.cost.battery_power_capex_usd_per_kw_int": _fmt_int(
            usd_cost("battery_power_capex_ntd_per_kw")
        ),
        "config.cost.battery_om_pct_1": _fmt_1(
            float(cost.get("battery_om_fraction_of_capex_per_year", 0.025)) * 100.0
        ),
        "config.cost.battery_replacement_year": str(
            int(round(float(cost.get("battery_replacement_year", 12))))
        ),
        "config.cost.battery_replacement_year_low": str(
            int(round(config.battery_replacement_year_values[0]))
        ),
        "config.cost.battery_replacement_year_mid": str(
            int(round(config.battery_replacement_year_values[1]))
        ),
        "config.cost.battery_replacement_year_high": str(
            int(round(config.battery_replacement_year_values[2]))
        ),
        "config.cost.replacement_energy_capex_usd_per_kwh_int": _fmt_int(
            usd_cost("battery_replacement_energy_capex_ntd_per_kwh")
        ),
        "config.cost.grid_fixed_total_usd_per_light_1": _fmt_1(
            grid_fixed_usd / config.lights_per_city
        ),
        "config.cost.grid_fixed_present_value_usd_per_light_int": _fmt_int(
            grid_pv_usd / config.lights_per_city
        ),
        "config.cost.grid_fixed_present_value_usd_per_light_1": _fmt_1(
            grid_pv_usd / config.lights_per_city
        ),
    }


def build_config_render_blocks(config: Paper1Config) -> dict[str, str]:
    tokens = build_config_render_tokens(config)
    cost = config.economic_costs
    years = int(round(config.analysis_years))
    region_city_map = config.region_city_map

    def usd(value_ntd: float) -> float:
        return float(value_ntd) / config.ntd_per_usd

    def per_streetlight_fixed_cost(
        value_ntd: float, note: str = "undiscounted allocation"
    ) -> str:
        value_usd = usd(value_ntd) / config.lights_per_city
        return f"{_fmt_1(value_usd)} USD/streetlight ({note})"

    grid_fixed_usd = usd(float(cost.get("grid_fixed_cost", 95000.0)))
    grid_pv_usd = (
        (float(cost.get("grid_fixed_cost", 95000.0)) / config.analysis_years)
        * _present_worth_factor(config.analysis_years, config.discount_rate)
        / config.ntd_per_usd
    )

    discount_range = (
        f"{tokens['config.economics.discount_rate_low_pct_int']} / "
        f"{tokens['config.economics.discount_rate_pct_int']} / "
        f"{tokens['config.economics.discount_rate_high_pct_int']}% per year"
    )
    electricity_range = (
        f"{tokens['config.economics.electricity_price_usd_low_3']} / "
        f"{tokens['config.economics.electricity_price_usd_mid_3']} / "
        f"{tokens['config.economics.electricity_price_usd_high_3']} USD/kWh"
    )
    pv_capex_range = (
        f"{tokens['config.cost.pv_capex_usd_low_int']} / "
        f"{tokens['config.cost.pv_capex_usd_per_kw_int']} / "
        f"{tokens['config.cost.pv_capex_usd_high_int']} USD/kW"
    )
    replacement_year_range = (
        f"{tokens['config.cost.battery_replacement_year_low']} / "
        f"{tokens['config.cost.battery_replacement_year_mid']} / "
        f"{tokens['config.cost.battery_replacement_year_high']}"
    )

    s4_rows = [
        _md_table_row([
            "Analysis horizon",
            f"{years} years",
            "not varied",
            (
                "Streetlight life-cycle assessment (LCA) and life-cycle cost (LCC) literature "
                "[@tahkamo2015lca; @casamayor2018led; @tahkamo2016lcc]"
            ),
        ]),
        _md_table_row([
                "Discount rate",
                f"{tokens['config.economics.discount_rate_pct_int']}% per year",
                discount_range,
                (
                    "Study-defined public-investment LCC scenario range, informed by public-project "
                    "discounting guidance [@omb_a94_2023]"
                ),
            ]),
        _md_table_row([
            "Electricity price",
            f"{tokens['config.economics.electricity_price_usd_per_kwh_3']} USD/kWh",
            electricity_range,
            (
                "Taipower post-October 2024 average tariff level used as retail-price proxy "
                "[@taipower_2024_average_tariff]"
            ),
        ]),
        _md_table_row([
            "PV CAPEX",
            f"{tokens['config.cost.pv_capex_usd_per_kw_int']} USD/kW",
            f"one-way: ±20%; Monte Carlo: {pv_capex_range}",
            "PV technology-cost benchmarks [@nrel_atb_2024; @irena2023costs]",
        ]),
        _md_table_row([
            "PV operation and maintenance",
            f"{tokens['config.cost.pv_om_usd_per_kw_year_1']} USD/kW-year",
            "not varied",
            "PV operation and maintenance benchmark [@nrel_atb_2024]",
        ]),
        _md_table_row([
            "Battery energy CAPEX",
            f"{tokens['config.cost.battery_capex_usd_per_kwh_int']} USD/kWh",
            "one-way: 80 / 100 / 120%; Monte Carlo: 85 / 100 / 115% of baseline battery CAPEX bundle",
            "Storage technology-cost benchmarks [@nrel_atb_2024; @irena2023costs]",
        ]),
        _md_table_row([
            "Battery power CAPEX",
            f"{tokens['config.cost.battery_power_capex_usd_per_kw_int']} USD/kW",
            "one-way: 80 / 100 / 120%; Monte Carlo: 85 / 100 / 115% of baseline battery CAPEX bundle",
            "Storage technology-cost benchmarks [@nrel_atb_2024; @irena2023costs]",
        ]),
        _md_table_row([
            "Battery operation and maintenance",
            f"{tokens['config.cost.battery_om_pct_1']}% of battery CAPEX bundle per year",
            f"not varied; {tokens['config.cost.battery_om_pct_1']}% of battery CAPEX bundle per year",
            "Storage operation and maintenance benchmark [@nrel_atb_2024]",
        ]),
        _md_table_row([
            "Replacement year",
            tokens["config.cost.battery_replacement_year"],
            replacement_year_range,
            "Battery life-cycle literature [@hiremath2015battery; @yudhistira2022liion]",
        ]),
        _md_table_row([
            "Replacement energy CAPEX",
            f"{tokens['config.cost.replacement_energy_capex_usd_per_kwh_int']} USD/kWh",
            "one-way: 80 / 100 / 120%; Monte Carlo: 85 / 100 / 115% of baseline battery CAPEX bundle",
            "Storage cost benchmarks [@nrel_atb_2024; @irena2023costs]",
        ]),
        _md_table_row([
            "Balance-of-system, mounting, installation",
            per_streetlight_fixed_cost(float(cost.get("other_capex", 44583.0))),
            "not varied",
            (
                "Study-defined supporting-hardware allocation informed by Taiwan solar-streetlight "
                "product specifications and public lighting-work cost context "
                "[@tht_solar_led_streetlight; @xtg_solar_ms4500; @hengs_solar_lighting; "
                "@kaohsiung_gushan_wanshou_road_proposal]"
            ),
        ]),
        _md_table_row([
            f"Supporting operation and maintenance, {years}-year aggregate",
            per_streetlight_fixed_cost(
                float(cost.get("other_om", 46097.0)),
                f"undiscounted {years}-year aggregate allocation",
            ),
            "not varied",
            "Study-defined fixed service allocation informed by Taiwan public lighting-work cost context [@kaohsiung_gushan_wanshou_road_proposal]",
        ]),
        _md_table_row([
            "Non-PV, non-battery end-of-life treatment",
            per_streetlight_fixed_cost(float(cost.get("other_eol", 6300.0))),
            "not varied",
            "Study-defined fixed end-of-life allocation; no row-level public quote is used",
        ]),
        _md_table_row([
            "Grid-connection equipment",
            (
                f"{_fmt_1(grid_fixed_usd / config.lights_per_city)} USD/streetlight "
                f"(undiscounted {years}-year service allocation)"
            ),
            "not varied",
            (
                "Taiwan public-project grid-interface cost context and municipal infrastructure "
                "drawings [@kaohsiung_gushan_wanshou_road_proposal; "
                "@taipei_streetlight_base_drawing; @taipei_streetlight_pole_drawing; "
                "@chungkung_lighting_pole]"
            ),
        ]),
    ]

    s4a_rows = [
        _md_table_row([
            "Electricity price",
            "Taipower reported post-October 2024 average tariff level",
            (
                f"{tokens['config.economics.electricity_price_ntd_per_kwh_4']} NTD/kWh, "
                f"converted to {tokens['config.economics.electricity_price_usd_per_kwh_3']} USD/kWh"
            ),
            "Retail-price proxy; not a streetlight-specific annual tariff",
        ]),
        _md_table_row([
            "PV CAPEX",
            "NREL ATB technology-cost structure; IRENA installed-cost range check",
            (
                f"{tokens['config.cost.pv_capex_usd_per_kw_int']} USD/kW baseline with "
                f"{pv_capex_range} Monte Carlo values"
            ),
            "Benchmark proxy including installed PV cost structure; not a vendor bid",
        ]),
        _md_table_row([
            "PV operation and maintenance",
            "NREL ATB PV operation and maintenance benchmark",
            f"{tokens['config.cost.pv_om_usd_per_kw_year_1']} USD/kW-year",
            "Applied as annual operation-and-maintenance (O&M) stream",
        ]),
        _md_table_row([
            "Battery energy CAPEX",
            "NREL ATB battery-storage cost structure; IRENA storage-cost range check",
            f"{tokens['config.cost.battery_capex_usd_per_kwh_int']} USD/kWh baseline",
            "Energy component of the battery CAPEX bundle",
        ]),
        _md_table_row([
            "Battery power CAPEX",
            "NREL ATB battery-storage cost structure",
            f"{tokens['config.cost.battery_power_capex_usd_per_kw_int']} USD/kW baseline",
            "Power component of the battery CAPEX bundle",
        ]),
        _md_table_row([
            "Battery replacement energy CAPEX",
            "Storage cost benchmarks",
            f"{tokens['config.cost.replacement_energy_capex_usd_per_kwh_int']} USD/kWh",
            "Replacement cost is treated as an energy-capacity replacement stream, not as a battery-carbon-footprint value",
        ]),
        _md_table_row([
            "Battery replacement timing",
            "Battery life-cycle literature",
            (
                f"year {tokens['config.cost.battery_replacement_year_low']} / "
                f"{tokens['config.cost.battery_replacement_year_mid']} / "
                f"{tokens['config.cost.battery_replacement_year_high']} categorical Monte Carlo values, "
                f"centered on year {tokens['config.cost.battery_replacement_year']}"
            ),
            "Timing scenario for the discounted replacement stream and B4 hardware-cycle burden",
        ]),
        _md_table_row([
            "Discount rate",
            "Study-defined public-investment LCC scenario range",
            discount_range,
            (
                "OMB A-94 supports public-project discounting practice but is not the direct "
                f"source of the central {tokens['config.economics.discount_rate_pct_int']}% value"
            ),
        ]),
    ]

    s1_region_rows = [
        _md_table_row([
            REGION_LABELS.get(region, region.replace("_", " ")),
            ", ".join(CITY_LABELS.get(city, city) for city in region_city_map.get(region, [])),
        ])
        for region in config.load_side_regions
    ]

    return {
        "table.s1_region_rows": "\n".join(s1_region_rows),
        "table.s4_rows": "\n".join(s4_rows),
        "table.s4a_rows": "\n".join(s4a_rows),
    }
