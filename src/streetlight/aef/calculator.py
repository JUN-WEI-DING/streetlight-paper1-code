"""
AEF (Average Emission Factor) calculation module.

This module provides core functions for computing average emission factors
from power generation data and emission factors by fuel type.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd


class AEFCalculator:
    """
    Calculator for Average Emission Factor (AEF).

    Computes AEF from power generation data and fuel-specific emission factors.
    """

    # Small epsilon value for numerical stability
    EPS = 1e-12

    def __init__(
        self,
        emission_factors: Optional[Dict[str, float]] = None,
        unit_fuel_map: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the AEF calculator.

        Args:
            emission_factors: Dictionary mapping fuel names to emission factors
                             (kg CO2e per kWh). Uses default if None.
            unit_fuel_map: Optional exact column-name mapping from unit-level
                           generation columns to fuel categories.
        """
        self.emission_factors = emission_factors or self.default_emission_factors()
        self.unit_fuel_map = dict(unit_fuel_map or {})

    @staticmethod
    def default_emission_factors() -> Dict[str, float]:
        """
        Get default emission factors for Taiwan power generation.

        Lifecycle kg CO2e/kWh. IPCC AR5 WG3 Annex III medians for fuels in the
        IPCC scope; ecoinvent 3.x via SimaPro 10.2.0.3 (IPCC 2021 GWP100 V1.03)
        for Oil/Diesel/Co-Gen which IPCC does not tabulate. PHS/BESS are
        placeholders resolved dynamically by the storage carbon-pool model.
        See config/config.yaml inline comments for per-fuel source pointers.

        Returns:
            Dictionary mapping fuel types to emission factors (kg CO2e/kWh)
        """
        return {
            "Coal": 0.820,
            "IPP-Coal": 0.820,
            "LNG": 0.490,
            "IPP-LNG": 0.490,
            "Oil": 0.815,
            "Diesel": 0.880,
            "Nuclear": 0.012,
            "Hydro": 0.024,
            "Wind": 0.011,
            "Offshore Wind": 0.012,
            "Solar": 0.048,
            "Geothermal": 0.038,
            "Biomass": 0.230,
            "Co-Gen": 0.460,
            "BESS": 3.1e-8,
            "PHS": 3.1e-8,
        }

    def compute_row_aef(
        self,
        row: pd.Series,
        ef_dict: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Compute AEF for a single time row.

        Args:
            row: Series with generation values by fuel type
            ef_dict: Emission factors dictionary (uses instance default if None)

        Returns:
            Dictionary with:
                - total_gen: Total generation (MWh)
                - total_emis: Total emissions (kg CO2e)
                - aef: Average emission factor (kg CO2e/kWh)
                - {fuel}_emis: Emissions by fuel type
        """
        if ef_dict is None:
            ef_dict = self.emission_factors

        total_gen = 0.0
        total_emis = 0.0
        emis_breakdown = {}

        for col, val in row.items():
            if col not in ef_dict:
                continue
            if pd.isna(val):
                continue
            g = max(float(val), 0.0)
            emis = g * ef_dict[col]
            total_gen += g
            total_emis += emis
            emis_breakdown[col] = emis

        if total_gen == 0:
            result = {
                "total_gen": np.nan,
                "total_emis": np.nan,
                "aef": np.nan,
                **{col: np.nan for col in ef_dict.keys()}
            }
        else:
            aef = total_emis / total_gen
            result = {
                "total_gen": total_gen,
                "total_emis": total_emis,
                "aef": aef,
                **emis_breakdown
            }

        return result

    def compute_aef_dataframe(
        self,
        df: pd.DataFrame,
        fuel_cols: Optional[Sequence[str]] = None,
        ef_dict: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Compute AEF for an entire DataFrame.

        Args:
            df: DataFrame with generation columns (MWh)
            fuel_cols: List of fuel column names (auto-detected if None)
            ef_dict: Emission factors (uses instance default if None)

        Returns:
            DataFrame with AEF results including:
                - Total_Gen (MWh): Total generation
                - Total_Emis: Total emissions (kg CO2e)
                - AEF: Average emission factor (kg CO2e/kWh)
                - {fuel}_Emis: Emissions by fuel type
        """
        if ef_dict is None:
            ef_dict = self.emission_factors

        if fuel_cols is None:
            fuel_cols = [col for col in df.columns if col in ef_dict]

        results = []
        for idx in df.index:
            row = df[fuel_cols].loc[idx]
            result = self.compute_row_aef(row, ef_dict)
            results.append(result)

        result_df = pd.DataFrame(results, index=df.index)

        # Rename columns to match expected format
        result_df = result_df.rename(columns={
            "total_gen": "Total_Gen (MWh)",
            "total_emis": "Total_Emis",
            "aef": "AEF"
        })

        # Rename emission columns
        for col in result_df.columns:
            if col in ef_dict and col not in ["Total_Gen (MWh)", "Total_Emis", "AEF"]:
                result_df = result_df.rename(columns={col: f"{col}_Emis"})

        return result_df

    @staticmethod
    def extract_prefix(colname: str) -> str:
        """
        Extract fuel prefix from column name.

        For example, "Coal-Unit1" -> "Coal", "LNG-CC1" -> "LNG"

        Args:
            colname: Column name to parse

        Returns:
            Fuel prefix or original column name if no dash
        """
        parts = colname.split('-')
        return '-'.join(parts[:2]) if len(parts) > 2 else parts[0]

    @staticmethod
    def load_unit_fuel_map(path: Optional[Union[Path, str]]) -> Dict[str, str]:
        """
        Load exact unit-column fuel overrides from a CSV classification table.

        The current table is used to split ``Wind-*`` columns into onshore
        ``Wind`` and ``Offshore Wind``. Unknown classifications are skipped so
        callers can keep the ordinary prefix fallback for unresolved rows.
        """
        if path is None:
            return {}

        csv_path = Path(path)
        if not csv_path.exists():
            return {}

        df = pd.read_csv(csv_path)
        required = {"raw_unit_column", "classification"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Unit fuel mapping {csv_path} missing required columns: {sorted(missing)}"
            )

        class_to_fuel = {
            "onshore": "Wind",
            "offshore": "Offshore Wind",
        }
        mapping: Dict[str, str] = {}
        for _, row in df.iterrows():
            raw_col = str(row["raw_unit_column"]).strip()
            classification = str(row["classification"]).strip().lower()
            fuel = class_to_fuel.get(classification)
            if raw_col and fuel:
                mapping[raw_col] = fuel
        return mapping

    def aggregate_by_fuel(
        self,
        df: pd.DataFrame,
        prefix_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate generation columns by fuel type.

        Args:
            df: DataFrame with unit-level generation columns
            prefix_map: Optional mapping from column names to fuel types

        Returns:
            DataFrame with columns aggregated by fuel type
        """
        if prefix_map is None:
            prefix_map = df.columns.to_series().apply(self.extract_prefix)
            if self.unit_fuel_map:
                prefix_map = prefix_map.copy()
                for col in df.columns:
                    if col in self.unit_fuel_map:
                        prefix_map.loc[col] = self.unit_fuel_map[col]

        # Use T.groupby to avoid FutureWarning about axis=1
        return df.T.groupby(prefix_map).sum(min_count=1).T

    def safe_value(self, x) -> float:
        """
        Convert value to float, returning 0.0 for NaN/None.

        Args:
            x: Value to convert

        Returns:
            Float value or 0.0 if NaN/None
        """
        try:
            return 0.0 if pd.isna(x) else float(x)
        except Exception:
            return 0.0


def compute_aef_row(row: pd.Series, ef_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Standalone function to compute AEF for a single row.

    Args:
        row: Series with generation values by fuel type
        ef_dict: Emission factors dictionary

    Returns:
        Dictionary with total_gen, total_emis, aef, and fuel-wise emissions
    """
    calc = AEFCalculator(ef_dict)
    return calc.compute_row_aef(row, ef_dict)
