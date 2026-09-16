"""Independent small-grid cases for the staged AEF quality audit."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/paper1_aef_quality.py"
SPEC = importlib.util.spec_from_file_location("paper1_aef_quality", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def staged_inputs(tmp_path):
    grid = pd.date_range("2024-01-01", periods=20, freq="10min")
    generation_index = grid.delete(range(5, 12))
    flow_index = grid.delete([11, 12])
    common = generation_index.intersection(flow_index)
    original = Path("/original/staging")

    def write(name, df):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)

    def write_report(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    par = pd.DataFrame({"city_a": 1.0, "city_b": 2.0}, index=grid)
    par.index.name = "timestamp"
    write("par/par_wide.csv", par)
    par_flags = par.astype(bool) & False
    par_flags.loc[grid[[0, 12]], "city_a"] = True
    write("par/par_imputed_flags.csv", par_flags)
    write("power/unit_generation.csv", pd.DataFrame({"unit_a": 1.0, "unit_b": 2.0}, index=generation_index.rename("timestamp")))
    gen_flags = pd.DataFrame(False, index=grid.rename("timestamp"), columns=["unit_a", "unit_b"])
    gen_flags.loc[grid[2]] = True  # Restored whole timestamp.
    gen_flags.loc[grid[3], "unit_a"] = True  # Imputed cell in an already present row.
    write("power/unit_generation_imputed_flags.csv", gen_flags)
    for name, intervals in (
        ("unit_generation_gap_report.csv", [(2, 2), (5, 11)]),
        ("unit_generation_remaining_gap_report.csv", [(5, 11)]),
        ("flow_gap_report.csv", [(11, 12)]),
    ):
        write("power/" + name, pd.DataFrame([
            {"start": grid[first], "end": grid[last], "n_timestamps": last - first + 1}
            for first, last in intervals]))
    write("power/flow.csv", pd.DataFrame({"F_CN": 1.0, "F_CE": -2.0, "F_CS": 3.0}, index=flow_index.rename("timestamp")))
    for region in MODULE.REGIONS:
        index = generation_index if region.startswith("island") else common
        df = pd.DataFrame({"Total_Gen (MWh)": 10.0, "Total_Emis": 8.8,
                           "FLOW_UNIT_FINAL_AEF": 0.88}, index=index.rename("timestamp"))
        if region == "island_lienchiang":
            df.loc[grid[1], ["Total_Gen (MWh)", "Total_Emis"]] = np.nan
        write(f"aef/{region}.csv", df)
        power = pd.DataFrame({"Coal-unit_a": 10.0, "Coal-unit_b": 1.0}, index=index.rename("timestamp"))
        power.loc[grid[1], "Coal-unit_b"] = -1.0
        if region == "central":
            power["F_NC"], power["F_EC"], power["F_SC"] = 0.0, 2.0, 0.0
        elif region in ("north", "east", "south"):
            name, amount = {"north": ("F_CN", 1.0), "east": ("F_CE", 0.0), "south": ("F_CS", 3.0)}[region]
            power[name] = amount
        write(f"power/{region}_unit_generation.csv", power)
    write("power/generation_invalid_total_report.csv", pd.DataFrame(
        {"total_generation_mw": [0.0], "min_valid_total_generation_mw": [10000.0]},
        index=grid[[5]].rename("timestamp")))
    for kind in ("flow", "generation"):
        write(f"power/{kind}_timestamp_normalization.csv", pd.DataFrame(columns=["original", "normalized"]))
    write_report("power/generation_quality_report.json", {
        "normalize_off_grid": True, "max_offset_minutes": 2,
        "min_valid_total_generation_mw": 10000.0, "normalized_rows": 0,
        "normalized_unique_timestamps": 0, "impute_short_gaps": True, "max_impute_run": 6,
        "combined": {"missing_before_imputation": 8, "missing_after_imputation": 7},
        "invalid_total_generation": {"timestamps": 1, "source_rows_dropped": 100},
    })
    write_report("power/flow_quality_report.json", {
        "unit_scale": 10.0, "input_rows_after_filter": 20, "duplicate_input_timestamps": 0,
        "invalid_numeric_rows": 0, "all_zero_regional_rows": 2, "all_zero_regional_timestamps": 2,
        "normalized_timestamps": 0, "flow_nan_rows": 0, "dropped_extra_timestamps": 0,
    })
    entries = [{"path": str(original / path.relative_to(tmp_path)), "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in sorted(tmp_path.rglob("*")) if path.is_file()]
    write_report("manifest/final_run_manifest.json", {
        "run_id": "fixture", "staging_root": str(original),
        "expected_time_range": {"start": str(grid[0]), "end": str(grid[-1]), "freq": "10min",
                                "timezone": "UTC", "timestamp_storage": "timezone-naive UTC"},
    })
    pd.DataFrame(entries).to_csv(tmp_path / "manifest/final_output_hashes.csv", index=False)
    return tmp_path


def test_missing_union_and_imputation_units(staged_inputs):
    result = MODULE.build_aef_quality(staged_inputs)
    sample = result["intersection"]
    assert sample["timestamps"] == 12
    assert sample["excluded_expected_timestamps"] == 8
    assert sample["generation_missing"] + sample["flow_missing"] - sample["missing_both"] == 8
    assert sample["missing_both"] == 1
    assert sample["generation_only_removed"] == 1
    assert sample["finite_aef_all_regions"] == 12
    assert sample["finite_aef_index_matches_input_intersection"]
    generation = result["generation"]
    assert generation["restored_whole_gap_timestamps"] == 1
    assert generation["imputed_partial_row_timestamps"] == 1
    assert generation["imputed_timestamps"] == 2
    assert generation["imputed_cells"] == 3
    assert sample["par_imputed_cells"] == 1
    assert sample["par_cell_denominator"] == 24


def test_nan_generation_uses_island_median_fallback(staged_inputs):
    result = MODULE.build_aef_quality(staged_inputs)
    island = result["aef"]["island_lienchiang"]
    assert island["local_aef_fallback_rows"] == 1
    assert island["local_aef_fallback_rows_in_sample"] == 1
    assert island["local_aef_fallback_value"] == pytest.approx(0.88)
    assert island["fallback_matches_local_median"]
    assert island["nonfinite_final_aef"] == 0
    assert result["aef"]["island_kinmen"]["local_aef_fallback_rows"] == 0


def test_provenance_is_relocated_and_deterministic(staged_inputs):
    first = MODULE.build_aef_quality(staged_inputs)
    second = MODULE.build_aef_quality(staged_inputs)
    assert first == second
    serialized = json.dumps(first)
    assert str(staged_inputs) not in serialized
    assert "/original/staging" not in serialized
    provenance = first["provenance"]
    assert provenance["original_output_hashes_verified"] == provenance["original_output_hash_entries"]
    assert all(not Path(row["path"]).is_absolute() for row in provenance["source_files"])
    assert first["reported_preparation"]["generation_invalid_total"]["source_rows_dropped"] == 100


def test_modified_source_fails_original_hash_check(staged_inputs):
    path = staged_inputs / "par/par_wide.csv"
    path.write_text(path.read_text().replace("1.0", "3.0"))
    with pytest.raises(ValueError, match="original output hash: par/par_wide.csv"):
        MODULE.build_aef_quality(staged_inputs)


def test_signed_flow_and_fuel_aggregation_checks(staged_inputs):
    result = MODULE.build_aef_quality(staged_inputs)
    link = result["flow"]["by_link"]["F_CE"]
    assert link["negative_rows"] == 18
    assert link["negative_rows_in_sample"] == 12
    assert link["positive_rows"] == 0
    assert link["split_nonnegative"]
    assert link["split_mutually_exclusive"]
    assert link["split_matches_signed_flow"]
    fuel = result["generation"]["regional_fuel_checks"]["north"]
    assert fuel["negative_unit_cells"] == 1
    assert fuel["negative_fuel_aggregate_cells"] == 0
