"""Allocation weighting, independent frontier selection, and carbon boundaries."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("paper1_selection_builder", SCRIPTS / "build_paper1_manuscript_values.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


@pytest.fixture
def panel():
    return pd.DataFrame([
        {"city": city, "solar_panel_factor": design, "battery_factor": 1.0,
         "abatement_t": operational, "delta_cost": cost, "functional_unit_lights": 10}
        for design in (1.0, 2.0, 3.0)
        for city, operational, cost in (("A", design * 20, design * 4000), ("B", design * 60, design * 8000))
    ])


def test_population_aggregation_uses_normalized_weights_and_per_light_units(panel):
    weights = pd.Series({"B": 3.0, "A": 1.0})
    result = BUILDER._aggregate_selection_panel(panel, weights, n_lights=10, ntd_per_usd=40)
    # First design: (20 + 3*60)/4/10 = 5 t; (4000 + 3*8000)/4/10/40 = 17.5 USD.
    assert result["operational_abatement_t_per_streetlight"].tolist() == pytest.approx([5, 10, 15])
    assert result["delta_cost_usd_per_streetlight"].tolist() == pytest.approx([17.5, 35, 52.5])
    pd.testing.assert_frame_equal(result, BUILDER._aggregate_selection_panel(
        panel.sample(frac=1, random_state=3), weights * 7, n_lights=10, ntd_per_usd=40,
    ))
    only_a = BUILDER._aggregate_selection_panel(panel, pd.Series({"A": 1.0, "B": 0.0}), n_lights=10, ntd_per_usd=40)
    assert only_a.iloc[0]["operational_abatement_t_per_streetlight"] == 2


@pytest.mark.parametrize("weights", [
    {"A": -1, "B": 3}, {"A": np.nan, "B": 1}, {"A": np.inf, "B": 1},
    {"A": 0, "B": 0}, {"A": 1}, {"A": 1, "B": 1, "C": 1},
])
def test_invalid_or_unmatched_weights_are_rejected(panel, weights):
    with pytest.raises(ValueError, match="weights"):
        BUILDER._aggregate_selection_panel(panel, pd.Series(weights), n_lights=10, ntd_per_usd=40)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unit", "nonfinite"])
def test_incomplete_panel_and_mismatched_units_are_rejected(panel, change):
    if change == "missing":
        panel = panel.iloc[1:]
    elif change == "duplicate":
        panel = pd.concat([panel, panel.iloc[:1]])
    elif change == "unit":
        panel.loc[0, "functional_unit_lights"] = 64
    else:
        panel.loc[0, "delta_cost"] = np.nan
    with pytest.raises(ValueError):
        BUILDER._aggregate_selection_panel(panel, pd.Series({"A": 1, "B": 1}), n_lights=10, ntd_per_usd=40)


def test_each_objective_rebuilds_frontier_and_geometry_from_all_candidates():
    points = pd.DataFrame({
        "solar_panel_factor": [1., 2., 3., 4., 5., 6.],
        "battery_factor": [1.] * 6,
        "delta_cost_usd_per_streetlight": [1., 2., 3., 4., 5., 6.],
        "operational": [1., 5., 4., 7., 8., 9.],
        "net": [0., 1., 3.7, 4., 3., 2.],
    })
    selected = {}
    for objective in ("operational", "net"):
        frontier, knee_index = BUILDER._selection_frontier(points, objective)
        # Independent pairwise dominance check, including a design dominated
        # operationally (factor 3) that becomes nondominated after netting.
        survivors = []
        for i, point in points.iterrows():
            dominates = ((points[objective] >= point[objective])
                         & (points["delta_cost_usd_per_streetlight"] <= point["delta_cost_usd_per_streetlight"])
                         & ((points[objective] > point[objective])
                            | (points["delta_cost_usd_per_streetlight"] < point["delta_cost_usd_per_streetlight"])))
            if not dominates.any():
                survivors.append(i)
        expected = points.loc[survivors].sort_values("delta_cost_usd_per_streetlight")
        assert frontier["solar_panel_factor"].tolist() == expected["solar_panel_factor"].tolist()
        x = expected[objective].to_numpy()
        y = expected["delta_cost_usd_per_streetlight"].to_numpy()
        # Ordered monotone frontier has normalized endpoints (0,0) and (1,1).
        distance = np.abs((x - x.min()) / np.ptp(x) - (y - y.min()) / np.ptp(y)) / np.sqrt(2)
        assert distance[knee_index] == pytest.approx(distance.max())
        selected[objective] = float(frontier.iloc[knee_index]["solar_panel_factor"])
    assert selected == {"operational": 2., "net": 3.}
    assert 3. not in BUILDER._selection_frontier(points, "operational")[0]["solar_panel_factor"].tolist()
    assert 3. in BUILDER._selection_frontier(points, "net")[0]["solar_panel_factor"].tolist()
