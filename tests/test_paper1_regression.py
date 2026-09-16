"""Focused checks for the descriptive EIAR regression and region sensitivities."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("paper1_regression_builder", SCRIPTS / "build_paper1_manuscript_values.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_orthogonal_predictors_have_known_standardized_fit_and_partition():
    aef = np.array([-1.0, -1.0, 1.0, 1.0])
    par = np.array([-1.0, 1.0, -1.0, 1.0])
    frame = pd.DataFrame({"aef": aef, "mean_par": par, "abatement_t": 2 * aef + par})
    values, _ = BUILDER._build_regression_tokens(frame)
    assert values["aef_par"]["aef_coefficient"] == pytest.approx(2 / np.sqrt(5))
    assert values["aef_par"]["par_coefficient"] == pytest.approx(1 / np.sqrt(5))
    assert values["aef_only"]["r2"] == pytest.approx(0.8)
    assert values["par_only"]["r2"] == pytest.approx(0.2)
    assert values["aef_par"]["r2"] == pytest.approx(1)
    assert values["incremental_r2"] == pytest.approx({"aef_given_par": 0.8, "par_given_aef": 0.2})
    assert values["partial_r2"] == pytest.approx({"aef_given_par": 1, "par_given_aef": 1})
    assert values["r2_partition"]["shared"] == pytest.approx(0)


def test_spearman_uses_average_ties_despite_floating_representation():
    frame = pd.DataFrame({
        "aef": [0.3, 0.5, 0.88, 0.8800000000000003],
        "mean_par": [1, 3, 2, 4],
        "abatement_t": [1, 2, 3, 4],
    })
    values, _ = BUILDER._build_regression_tokens(frame)
    # Average ranks [1, 2, 3.5, 3.5] versus [1, 2, 3, 4].
    assert values["spearman"]["aef"] == pytest.approx(np.sqrt(0.9))
    assert values["pearson"]["aef"] == pytest.approx(frame["aef"].corr(frame["abatement_t"]))


def test_negative_shared_component_is_retained_for_suppression():
    aef = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    par = np.array([-1.5, -1.2, 0.4, 0.8, 2.3])
    values, _ = BUILDER._build_regression_tokens(pd.DataFrame({
        "aef": aef, "mean_par": par, "abatement_t": aef - par,
    }))
    partition = values["r2_partition"]
    assert partition["shared"] < 0
    assert sum(partition.values()) == pytest.approx(1)
    assert partition["unique_aef"] + partition["unique_par"] + partition["shared"] == pytest.approx(values["aef_par"]["r2"])


def test_perfect_reduced_fit_has_undefined_partial_r2():
    values, _ = BUILDER._build_regression_tokens(pd.DataFrame({
        "aef": [-1, -1, 1, 1], "mean_par": [-1, 1, -1, 1], "abatement_t": [-2, -2, 2, 2],
    }))
    assert values["partial_r2"]["par_given_aef"] is None
