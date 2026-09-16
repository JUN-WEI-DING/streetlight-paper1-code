"""
Grid storage state management for PHS (pumped hydro) and BESS (battery).

This module manages a SHARED carbon-energy pool that both PHS and BESS
operate on. Each physical object has its own round-trip efficiency, but
the pool itself is fungible — once stored on the grid, energy origin
cannot be physically traced. See docs/engineering/grid_storage_pool_design.md.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from streetlight.config import get_config


@dataclass
class StorageConfig:
    """Configuration for a storage actor (PHS or BESS)."""

    charge_efficiency: float = 0.9487
    discharge_efficiency: float = 0.9487


@dataclass
class StorageState:
    """
    Shared grid storage carbon-energy pool.

    Tracks pool energy (MWh) and accumulated carbon (kg CO2e) plus per-actor
    histories of {energy, carbon, deficit, discharge_aef}. Both PHS and BESS
    contribute to / draw from the same pool.

    Attributes:
        energy_mwh: Stored energy currently in pool (MWh)
        carbon_kg: Stored carbon currently in pool (kg CO2e)
        history_energy: end-of-step pool energy
        history_carbon: end-of-step pool carbon
        phs_deficit_history: PHS unmet discharge per step
        bess_deficit_history: BESS unmet discharge per step
        phs_discharge_aef_history: PHS aef_to_grid per step (NaN if not discharged)
        bess_discharge_aef_history: BESS aef_to_grid per step
    """

    energy_mwh: float = 0.0
    carbon_kg: float = 0.0
    history_energy: list = field(default_factory=list)
    history_carbon: list = field(default_factory=list)
    phs_deficit_history: list = field(default_factory=list)
    bess_deficit_history: list = field(default_factory=list)
    phs_discharge_aef_history: list = field(default_factory=list)
    bess_discharge_aef_history: list = field(default_factory=list)

    @property
    def aef(self) -> Optional[float]:
        """Pool's current average emission factor (kg CO2e/kWh) or None if empty."""
        if self.energy_mwh > 1e-12:
            return self.carbon_kg / self.energy_mwh
        return None

    def charge(
        self,
        grid_kwh: float,
        current_aef: float,
        charge_efficiency: float,
    ) -> None:
        """
        Charge the pool from grid.

        Args:
            grid_kwh: Energy drawn from grid (positive value, MWh)
            current_aef: Grid AEF at this moment (kg CO2e/kWh)
            charge_efficiency: Round-trip-aware charge efficiency for this actor
        """
        if grid_kwh <= 0:
            return
        actual_stored = grid_kwh * charge_efficiency
        self.energy_mwh += actual_stored
        # Carbon uses GROSS grid kWh (includes the part lost to inefficiency)
        # to preserve carbon conservation across charge/discharge cycles.
        self.carbon_kg += grid_kwh * current_aef

    def request_discharge(
        self,
        request_grid_kwh: float,
        discharge_efficiency: float,
    ) -> Tuple[float, Optional[float], float]:
        """
        Request discharge to grid.

        Args:
            request_grid_kwh: Requested grid-side delivery (MWh)
            discharge_efficiency: Round-trip-aware discharge efficiency for this actor

        Returns:
            (delivered_grid_kwh, aef_to_grid, deficit)
        """
        if self.energy_mwh <= 1e-12 or request_grid_kwh <= 0:
            deficit = max(request_grid_kwh, 0.0)
            return 0.0, None, deficit

        pool_aef = self.carbon_kg / max(self.energy_mwh, 1e-12)
        deliverable = min(
            request_grid_kwh, self.energy_mwh * discharge_efficiency
        )
        from_pool = deliverable / max(discharge_efficiency, 1e-12)

        self.energy_mwh = max(self.energy_mwh - from_pool, 0.0)
        self.carbon_kg = max(self.carbon_kg - pool_aef * from_pool, 0.0)

        aef_to_grid = pool_aef / max(discharge_efficiency, 1e-12)
        deficit = request_grid_kwh - deliverable
        return deliverable, aef_to_grid, deficit

    def record_state(
        self,
        phs_deficit: float = 0.0,
        bess_deficit: float = 0.0,
        phs_discharge_aef: Optional[float] = None,
        bess_discharge_aef: Optional[float] = None,
    ) -> None:
        """Record current state plus per-actor metrics for this step."""
        self.history_energy.append(self.energy_mwh)
        self.history_carbon.append(self.carbon_kg)
        self.phs_deficit_history.append(phs_deficit)
        self.bess_deficit_history.append(bess_deficit)
        self.phs_discharge_aef_history.append(
            float(phs_discharge_aef) if phs_discharge_aef is not None else np.nan
        )
        self.bess_discharge_aef_history.append(
            float(bess_discharge_aef) if bess_discharge_aef is not None else np.nan
        )


class StorageManager:
    """
    Manager for grid storage with one shared pool and two actors (PHS + BESS).

    Each actor has its own round-trip efficiency but charges/discharges
    against the same pool. The pool's AEF is the volume-weighted average of
    all carbon ever stored (less what has been discharged proportionally).
    """

    def __init__(
        self,
        bess_config: Optional[StorageConfig] = None,
        phs_config: Optional[StorageConfig] = None,
    ):
        config = get_config()

        if bess_config is None:
            bess_config = StorageConfig(
                charge_efficiency=config.get("storage.bess.charge_efficiency", 0.9487),
                discharge_efficiency=config.get("storage.bess.discharge_efficiency", 0.9487),
            )

        if phs_config is None:
            phs_config = StorageConfig(
                charge_efficiency=config.get("storage.phs.charge_efficiency", 0.8832),
                discharge_efficiency=config.get("storage.phs.discharge_efficiency", 0.8832),
            )

        self.bess_config = bess_config
        self.phs_config = phs_config

        # Single shared pool
        self.grid_pool_state = StorageState()

    def reset(self) -> None:
        """Reset pool state to empty."""
        self.grid_pool_state = StorageState()

    def charge_phs(self, grid_kwh: float, current_aef: float) -> None:
        self.grid_pool_state.charge(
            grid_kwh, current_aef, self.phs_config.charge_efficiency
        )

    def charge_bess(self, grid_kwh: float, current_aef: float) -> None:
        self.grid_pool_state.charge(
            grid_kwh, current_aef, self.bess_config.charge_efficiency
        )

    def discharge_phs(
        self, request_grid_kwh: float
    ) -> Tuple[float, Optional[float], float]:
        return self.grid_pool_state.request_discharge(
            request_grid_kwh, self.phs_config.discharge_efficiency
        )

    def discharge_bess(
        self, request_grid_kwh: float
    ) -> Tuple[float, Optional[float], float]:
        return self.grid_pool_state.request_discharge(
            request_grid_kwh, self.bess_config.discharge_efficiency
        )

    def get_pool_dataframe(self, index: pd.Index) -> pd.DataFrame:
        """
        Build pool history DataFrame.

        Returns columns:
            GRID_POOL_E (MWh), GRID_POOL_C (kgCO2), GRID_POOL_AEF (kg/kWh),
            PHS_DISCHARGE_AEF (kg/kWh), BESS_DISCHARGE_AEF (kg/kWh),
            PHS_DEFICIT (MWh), BESS_DEFICIT (MWh)
        """
        df = pd.DataFrame(index=index)
        idx_len = len(index)

        def pad(arr, n):
            arr = list(arr)
            while len(arr) < n:
                arr.append(np.nan)
            return arr[:n]

        df["GRID_POOL_E (MWh)"] = pad(self.grid_pool_state.history_energy, idx_len)
        df["GRID_POOL_C (kgCO2)"] = pad(self.grid_pool_state.history_carbon, idx_len)
        df["PHS_DEFICIT (MWh)"] = pad(self.grid_pool_state.phs_deficit_history, idx_len)
        df["BESS_DEFICIT (MWh)"] = pad(self.grid_pool_state.bess_deficit_history, idx_len)
        df["PHS_DISCHARGE_AEF (kg/kWh)"] = pad(
            self.grid_pool_state.phs_discharge_aef_history, idx_len
        )
        df["BESS_DISCHARGE_AEF (kg/kWh)"] = pad(
            self.grid_pool_state.bess_discharge_aef_history, idx_len
        )

        df["GRID_POOL_AEF (kg/kWh)"] = np.where(
            df["GRID_POOL_E (MWh)"] > 1e-12,
            df["GRID_POOL_C (kgCO2)"] / df["GRID_POOL_E (MWh)"],
            np.nan,
        )
        return df
