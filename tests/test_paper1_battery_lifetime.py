"""Battery replacement counts, accounting boundaries, and fixed-design results."""

import importlib.util
import sys
from pathlib import Path

import pytest

from streetlight.lca import hardware

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("paper1_battery_builder", SCRIPTS / "build_paper1_manuscript_values.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


@pytest.mark.parametrize("lifetime,expected", [(8, [8, 16]), (10, [10]), (12, [12]), (15, [15]), (20, [])])
def test_replacement_schedule_excludes_service_endpoint(lifetime, expected):
    assert BUILDER._battery_replacement_years(lifetime, 20) == expected


def test_extra_battery_adds_freight_and_treatment_but_no_extra_mppt():
    pv, batt = 1.85, 8.45
    original = hardware.compute_lca_at_knee(pv, batt)
    assert original == hardware.compute_lca_at_knee(pv, batt, battery_replacements=1)
    extra = hardware.compute_lca_at_knee(pv, batt, battery_replacements=2)
    cells = batt * hardware.CELLS_PER_POLE_PER_FACTOR
    mass = cells * hardware.BATT_SYSTEM_MASS_KG
    assert extra["SOLAR"]["B"] - original["SOLAR"]["B"] == pytest.approx(
        cells * hardware.G_BATTERY_PER_UNIT + mass * hardware.G_FREIGHT_PER_KG
    )
    assert extra["SOLAR"]["B"] == pytest.approx(2 * original["SOLAR"]["B"] - hardware.G_MPPT)
    assert extra["SOLAR"]["C"] - original["SOLAR"]["C"] == pytest.approx(
        mass * hardware.G_C_PER_KG_OF_EXTRA_MASS
    )
    assert extra["TRAD"] == original["TRAD"]
    for stage in ("A1_A3", "A4", "A5"):
        assert extra["SOLAR"][stage] == original["SOLAR"][stage]
