"""
Streetlight configuration management.

Provides unified configuration loading from YAML files with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from streetlight import paths as sl_paths
import yaml


class Config:
    """
    Unified configuration management class.

    Loads configuration from YAML file and allows environment variable overrides.
    Implements singleton pattern for consistent access across the application.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._config = self._load_config()
        # Ensure defaults are merged (deeply for nested mappings)
        self._config = self._deep_merge(self._default_config(), self._config)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries.

        - Scalars/lists in override replace base.
        - Nested dicts are merged key-by-key.
        """
        out: Dict[str, Any] = dict(base)
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = Config._deep_merge(out[k], v)  # type: ignore[arg-type]
            else:
                out[k] = v
        return out

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(os.getenv(
            'STREETLIGHT_CONFIG',
            Path(__file__).parent.parent.parent.parent / 'config' / 'config.yaml'
        ))

        if not config_path.exists():
            return self._default_config()

        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        return config

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'data': {
                'root_dir': 'data',
                'power_dir': 'data/power',
                'wind_unit_classification_path': 'data/power/wind_unit_classification.csv',
            },
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs',
            },
            'cache': {
                'enabled': True,
                'dir': 'cache',
            },
            'outputs': {
                'base_dir': 'outputs',
                'aef': {'dir': 'outputs/AEF'},
                'par': {'dir': 'outputs/par'},
                'pareto': {'dir': 'outputs/pareto'},
                'streetlight_sim': {'dir': 'outputs/streetlight_sim'},
            },
            # Lifecycle EFs (kg CO2e/kWh) — see config/config.yaml for source notes.
            # IPCC AR5 WG3 Annex III medians for fuels in the IPCC scope; ecoinvent
            # 3.x via SimaPro 10.2.0.3 (IPCC 2021 GWP100 V1.03) for Oil/Diesel/Co-Gen.
            'emission_factors': {
                'Coal': 0.820,
                'IPP-Coal': 0.820,
                'LNG': 0.490,
                'IPP-LNG': 0.490,
                'Oil': 0.815,
                'Diesel': 0.880,
                'Nuclear': 0.012,
                'Hydro': 0.024,
                'Wind': 0.011,
                'Offshore Wind': 0.012,
                'Solar': 0.048,
                'Geothermal': 0.038,
                'Biomass': 0.230,
                'Co-Gen': 0.460,
                'BESS': 3.1e-8,
                'PHS': 3.1e-8,
            },
            'storage': {
                'bess': {
                    'charge_efficiency': 0.9487,
                    'discharge_efficiency': 0.9487,
                },
                'phs': {
                    'charge_efficiency': 0.8832,
                    'discharge_efficiency': 0.8832,
                },
            },
            'regions': {
                'list': [
                    'north',
                    'central',
                    'south',
                    'east',
                    'island_penghu',
                    'island_kinmen',
                    'island_lienchiang',
                ],
                'mapping': {
                    'north': 'N',
                    'central': 'C',
                    'south': 'S',
                    'east': 'E',
                    'island_penghu': 'IP',
                    'island_kinmen': 'IK',
                    'island_lienchiang': 'IL',
                },
            },
            'simulation': {
                'standardized_installation': {
                    'n_lights': 64,
                    'light_power_kw': 0.1,
                    'par_to_kw_factor': 0.0081,
                    'par_to_kw_factor_status': 'internally_calibrated',
                    'load_mode': 'solar_zenith',
                    'solar_zenith_deg': 90.833,
                    'lighting_threshold_par': 1.0,
                },
            },
            'economics': {
                'analysis_years': 20.0,
                'electricity_price': 3.7556,
                'cost': {
                    'pv_capacity_kw_per_factor': 18.0,
                    'pv_capex_ntd_per_kw': 42880,
                    'pv_om_ntd_per_kw_year': 704,
                    'pv_eol_ntd_per_kw': 0.0,
                    'battery_capex_ntd_per_kwh': 8096,
                    'battery_power_capex_ntd_per_kw': 30976,
                    'battery_om_fraction_of_capex_per_year': 0.025,
                    'battery_replacement_year': 12,
                    'battery_replacement_energy_capex_ntd_per_kwh': 4896,
                    'battery_replacement_power_capex_ntd_per_kw': 0.0,
                    'battery_eol': 0.0,
                    'other_capex': 44583,
                    'other_om': 46097,
                    'other_eol': 6300,
                    'grid_fixed_cost': 95000,
                },
                'financial': {
                    'discount_rate': 0.05,
                    'pv_salvage_fraction': 0.0,
                    'battery_salvage_fraction': 0.0,
                    'other_salvage_fraction': 0.0,
                },
                'scenarios': {
                    'conservative': {
                        'financial': {
                            'discount_rate': 0.08,
                        },
                        'cost': {
                            'pv_capex_ntd_per_kw': 48320,
                            'battery_capex_ntd_per_kwh': 9310,
                            'battery_power_capex_ntd_per_kw': 35622,
                            'battery_replacement_year': 10,
                            'battery_replacement_energy_capex_ntd_per_kwh': 4896,
                        },
                    },
                    'optimistic': {
                        'financial': {
                            'discount_rate': 0.03,
                        },
                        'cost': {
                            'pv_capex_ntd_per_kw': 35840,
                            'battery_capex_ntd_per_kwh': 6882,
                            'battery_power_capex_ntd_per_kw': 26330,
                            'battery_replacement_year': 15,
                            'battery_replacement_energy_capex_ntd_per_kwh': 3917,
                        },
                    },
                },
            },
            'storage_dispatch': {
                'default': {
                    'capacity_kwh': 10.0,
                    'power_kw': 5.0,
                    'eta_roundtrip': 0.90,
                    'init_soc': 0.5,
                    'self_discharge_per_hour': 0.0,
                },
            },
        }

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        if 'STREETLIGHT_DATA_DIR' in os.environ:
            config.setdefault('data', {})
            config['data']['root_dir'] = os.environ['STREETLIGHT_DATA_DIR']

        if 'STREETLIGHT_OUTPUTS_DIR' in os.environ:
            config.setdefault('outputs', {})
            config['outputs']['base_dir'] = os.environ['STREETLIGHT_OUTPUTS_DIR']

        if 'STREETLIGHT_CACHE_DIR' in os.environ:
            config.setdefault('cache', {})
            config['cache']['dir'] = os.environ['STREETLIGHT_CACHE_DIR']

        if 'STREETLIGHT_LOGS_DIR' in os.environ:
            config.setdefault('logging', {})
            config['logging']['log_dir'] = os.environ['STREETLIGHT_LOGS_DIR']

        return config

    def _resolve_path(self, value: Any, *, base: Optional[Path] = None) -> Path:
        """
        Resolve a config path value.

        - Absolute paths are returned as-is.
        - Relative paths are resolved relative to repo root (not CWD).
        """
        if value is None:
            raise ValueError("Path config value is None")
        p = Path(str(value))
        if p.is_absolute():
            return p
        base_dir = base or sl_paths.project_root()
        return (base_dir / p).resolve()

    def resolve_path(self, value: Any, *, base: Optional[Path] = None) -> Path:
        """Public wrapper for config path resolution."""
        return self._resolve_path(value, base=base)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Configuration key in dot notation (e.g., 'data.root_dir')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def require(self, key: str) -> Any:
        """Get required configuration value, raising if missing."""
        value = self.get(key, None)
        if value is None:
            raise KeyError(f"Missing required config key: {key}")
        return value

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self._resolve_path(self.get('data.root_dir', 'data'))

    @property
    def power_dir(self) -> Path:
        """Get power data directory path."""
        return self._resolve_path(self.get('data.power_dir', self.data_dir / 'power'), base=sl_paths.project_root())

    @property
    def wind_unit_classification_path(self) -> Optional[Path]:
        """Get optional wind unit onshore/offshore classification table path."""
        value = self.get('data.wind_unit_classification_path', None)
        if not value:
            return None
        return self._resolve_path(value, base=sl_paths.project_root())

    @property
    def outputs_dir(self) -> Path:
        """Get outputs base directory."""
        return self._resolve_path(self.get('outputs.base_dir', 'outputs'))

    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        return self._resolve_path(self.get('logging.log_dir', 'logs'))

    @property
    def par_zarr_path(self) -> Path:
        """Get PAR zarr dataset path."""
        return self._resolve_path(
            self.get('data.par_zarr_path', self.data_dir / 'zarr' / 'himawari_par_2024.zarr'),
            base=sl_paths.project_root()
        )

    @property
    def outputs_aef_dir(self) -> Path:
        return self._resolve_path(self.get('outputs.aef.dir', self.outputs_dir / 'AEF'), base=sl_paths.project_root())

    @property
    def outputs_par_dir(self) -> Path:
        return self._resolve_path(self.get('outputs.par.dir', self.outputs_dir / 'par'), base=sl_paths.project_root())

    @property
    def outputs_pareto_dir(self) -> Path:
        return self._resolve_path(self.get('outputs.pareto.dir', self.outputs_dir / 'pareto'), base=sl_paths.project_root())

    @property
    def outputs_streetlight_sim_dir(self) -> Path:
        return self._resolve_path(
            self.get('outputs.streetlight_sim.dir', self.outputs_dir / 'streetlight_sim'),
            base=sl_paths.project_root(),
        )

    @property
    def emission_factors(self) -> Dict[str, float]:
        """Get emission factors mapping."""
        return self.get('emission_factors', {})

    @property
    def regions(self) -> list[str]:
        """Get list of regions."""
        return self.get('regions.list', [
            'north',
            'central',
            'south',
            'east',
            'island_penghu',
            'island_kinmen',
            'island_lienchiang',
        ])

    @property
    def region_mapping(self) -> Dict[str, str]:
        """Get mapping of region names to codes."""
        return self.get('regions.mapping', {
            'north': 'N',
            'central': 'C',
            'south': 'S',
            'east': 'E',
            'island_penghu': 'IP',
            'island_kinmen': 'IK',
            'island_lienchiang': 'IL',
        })

    @property
    def economics(self) -> Dict[str, Any]:
        """Get economics configuration."""
        return self.get('economics', {})

    @property
    def analysis_years(self) -> float:
        """Get analysis period in years."""
        return float(self.get('economics.analysis_years', 20.0))

    @property
    def standardized_installation(self) -> Dict[str, Any]:
        """Get standardized installation parameters."""
        value = self.get('simulation.standardized_installation', {
            'n_lights': 64,
            'light_power_kw': 0.1,
            'par_to_kw_factor': 0.0081,
            'par_to_kw_factor_status': 'internally_calibrated',
            'load_mode': 'solar_zenith',
            'solar_zenith_deg': 90.833,
            'lighting_threshold_par': 1.0,
        })
        if not isinstance(value, dict):
            raise TypeError("Config key simulation.standardized_installation must be a mapping")
        return value

    @property
    def n_lights(self) -> int:
        """Get number of lights in the standardized installation."""
        return int(self.get('simulation.standardized_installation.n_lights', 64))

    @property
    def light_power_kw(self) -> float:
        """Get rated power per light in kW."""
        return float(self.get('simulation.standardized_installation.light_power_kw', 0.1))

    @property
    def par_to_kw_factor(self) -> float:
        """Get the provisional PAR-to-PV conversion factor."""
        return float(self.get('simulation.standardized_installation.par_to_kw_factor', 0.0081))

    @property
    def par_to_kw_factor_status(self) -> str:
        """Get status of the PAR-to-PV conversion factor."""
        return str(self.get('simulation.standardized_installation.par_to_kw_factor_status', 'internally_calibrated'))

    @property
    def load_mode(self) -> str:
        """Get the standardized installation load mode."""
        return str(self.get('simulation.standardized_installation.load_mode', 'solar_zenith'))

    @property
    def solar_zenith_deg(self) -> float:
        """Get the solar zenith threshold used to determine lighting demand."""
        return float(self.get('simulation.standardized_installation.solar_zenith_deg', 90.833))

    @property
    def lighting_threshold_par(self) -> float:
        """Get the PAR threshold used to determine lighting demand."""
        return float(self.get('simulation.standardized_installation.lighting_threshold_par', 1.0))

    @property
    def electricity_price(self) -> float:
        """Get electricity price (NTD/kWh)."""
        return float(self.get('economics.electricity_price', 3.0))

    @property
    def economic_costs(self) -> Dict[str, Any]:
        """Get economic cost parameters."""
        value = self.get('economics.cost', {
            'streetlight': {
                'installation': 30000,
                'maintenance': 1000,
            },
            'pv': {
                'installation_per_kw': 30000,
                'maintenance_per_kw': 300,
            },
            'battery': {
                'installation_per_kwh': 10000,
                'maintenance_per_kwh': 100,
            },
        })
        if not isinstance(value, dict):
            raise TypeError("Config key economics.cost must be a mapping")
        return value

    @property
    def pareto_solar_cfg(self) -> Dict[str, Any]:
        """Get Pareto solar sweep configuration."""
        value = self.get('economics.pareto.solar', {'min': 0.5, 'max': 3.0, 'step': 0.5})
        if not isinstance(value, dict):
            raise TypeError("Config key economics.pareto.solar must be a mapping")
        return value

    @property
    def pareto_battery_cfg(self) -> Dict[str, Any]:
        """Get Pareto battery sweep configuration."""
        value = self.get('economics.pareto.battery', {'min': 0.5, 'max': 3.0, 'step': 0.5})
        if not isinstance(value, dict):
            raise TypeError("Config key economics.pareto.battery must be a mapping")
        return value

    @property
    def battery_power_cap_multiplier_of_load(self) -> float | None:
        """Get the battery power cap multiplier relative to installation load."""
        value = self.pareto_battery_cfg.get('power_cap_multiplier_of_load')
        return None if value is None else float(value)

    @property
    def financial_params(self) -> Dict[str, Any]:
        """Get financial parameters for discounted lifecycle costing."""
        value = self.get('economics.financial', {
            'discount_rate': 0.05,
            'pv_salvage_fraction': 0.0,
            'battery_salvage_fraction': 0.0,
            'other_salvage_fraction': 0.0,
        })
        if not isinstance(value, dict):
            raise TypeError("Config key economics.financial must be a mapping")
        return value

    @property
    def storage_params(self) -> Dict[str, Any]:
        """Get storage dispatch parameters."""
        return self.get('storage_dispatch.default', {})

    @property
    def storage_capacity_kwh(self) -> float:
        """Get default storage capacity in kWh."""
        return float(self.storage_params.get('capacity_kwh', 10.0))

    @property
    def storage_power_kw(self) -> float:
        """Get default storage power in kW."""
        return float(self.storage_params.get('power_kw', 5.0))

    @property
    def storage_eta_roundtrip(self) -> float:
        """Get default round-trip efficiency for storage dispatch."""
        return float(self.storage_params.get('eta_roundtrip', 0.90))

    @property
    def storage_init_soc(self) -> float:
        """Get default initial state of charge fraction."""
        return float(self.storage_params.get('init_soc', 0.5))

    @property
    def storage_soc0_kwh(self) -> float:
        """Get default initial state of charge in kWh for direct dispatch runs."""
        value = self.storage_params.get('soc0_kwh')
        if value is not None:
            return float(value)
        return self.storage_capacity_kwh * self.storage_init_soc

    @property
    def storage_self_discharge_per_hour(self) -> float:
        """Get default self-discharge per hour."""
        return float(self.storage_params.get('self_discharge_per_hour', 0.0))

    @property
    def paper(self) -> Dict[str, Any]:
        """Get paper workflow configuration."""
        value = self.get('paper', {})
        if not isinstance(value, dict):
            raise TypeError("Config key paper must be a mapping")
        return value

    def paper_settings(self, key: str, default: Any | None = None) -> Any:
        """Get a namespaced paper workflow setting."""
        return self.get(f'paper.{key}', default)

    @property
    def paper_inputs(self) -> Dict[str, Any]:
        """Get paper workflow input paths and options."""
        value = self.paper_settings('inputs', {})
        if not isinstance(value, dict):
            raise TypeError("Config key paper.inputs must be a mapping")
        return value

    @property
    def paper_outputs(self) -> Dict[str, Any]:
        """Get paper workflow output paths."""
        value = self.paper_settings('outputs', {})
        if not isinstance(value, dict):
            raise TypeError("Config key paper.outputs must be a mapping")
        return value

    @property
    def paper_output_dir(self) -> Path:
        """Get root output directory for the paper workflow."""
        if 'PAPER_OUTPUT_DIR' in os.environ:
            return self._resolve_path(os.environ['PAPER_OUTPUT_DIR'], base=sl_paths.project_root())
        return self._resolve_path(
            self.paper_outputs.get('dir', 'outputs/paper_baseline'),
            base=sl_paths.project_root(),
        )

    @property
    def paper_par_path(self) -> Path:
        """Get canonical PAR input path for the paper workflow."""
        if 'PAPER_PAR_PATH' in os.environ:
            return self._resolve_path(os.environ['PAPER_PAR_PATH'], base=sl_paths.project_root())
        return self._resolve_path(
            self.paper_inputs.get('par_path', 'outputs/par/par_wide.csv'),
            base=sl_paths.project_root(),
        )

    @property
    def paper_aef_dir(self) -> Path:
        """Get AEF input directory for the paper workflow."""
        if 'PAPER_AEF_DIR' in os.environ:
            return self._resolve_path(os.environ['PAPER_AEF_DIR'], base=sl_paths.project_root())
        return self._resolve_path(
            self.paper_inputs.get('aef_dir', 'outputs/AEF'),
            base=sl_paths.project_root(),
        )

    @property
    def paper_region_map_path(self) -> Path:
        """Get city-to-region mapping file for the paper workflow."""
        return self._resolve_path(
            self.paper_inputs.get('region_map', 'config/region_city_map.json'),
            base=sl_paths.project_root(),
        )

    @property
    def paper_aef_column(self) -> str:
        """Get AEF column name for paper workflow analyses."""
        return str(os.getenv('PAPER_AEF_COLUMN', self.paper_inputs.get('aef_column', 'FLOW_UNIT_FINAL_AEF')))

    @property
    def paper_align_strategy(self) -> str:
        """Get PAR/AEF time-alignment strategy for paper workflow analyses."""
        return str(os.getenv('PAPER_ALIGN_STRATEGY', self.paper_inputs.get('align_strategy', 'ffill')))


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config
