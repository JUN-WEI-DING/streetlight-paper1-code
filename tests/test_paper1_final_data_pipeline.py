import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from streetlight.par.handoff import interpolate_short_gaps
from streetlight.processing.power_adapter import (
    convert_parquet_to_regional_csv,
    create_flow_from_regional_demand,
    infer_region_from_plant_name,
    merge_flow_data_to_regional_csvs,
    split_signed_flow_for_aef,
    taipower_local_to_utc_naive,
)

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPTS = ROOT / "scripts" / "analysis"


def _load_analysis_script(module_name: str, filename: str):
    if str(ANALYSIS_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS_SCRIPTS))
    spec = importlib.util.spec_from_file_location(module_name, ANALYSIS_SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_interpolate_short_gaps_flags_only_short_runs():
    idx = pd.date_range("2024-01-01 00:00", periods=10, freq="10min")
    df = pd.DataFrame({"A": range(10), "B": range(10)}, index=idx).astype(float)
    df.loc[idx[2:4], "A"] = np.nan
    df.loc[idx[2:9], "B"] = np.nan

    filled, flags = interpolate_short_gaps(df, max_run=2)

    assert filled.loc[idx[2], "A"] == pytest.approx(2.0)
    assert filled.loc[idx[3], "A"] == pytest.approx(3.0)
    assert flags.loc[idx[2], "A"]
    assert flags.loc[idx[3], "A"]
    assert filled.loc[idx[2:8], "B"].isna().all()
    assert not flags["B"].any()




def test_repo_display_path_normalizes_repo_clones_and_keeps_external_paths(tmp_path: Path):
    module = _load_analysis_script("paper1_provenance_test", "paper1_provenance.py")
    repo_root = tmp_path / "streetlight-paper1-reproducibility"
    repo_root.mkdir()

    assert module.repo_display_path(
        repo_root / "outputs/final_runs/paper1_canonical_inputs/par/par_wide.csv",
        repo_root=repo_root,
    ) == "outputs/final_runs/paper1_canonical_inputs/par/par_wide.csv"
    assert module.repo_display_path(
        Path("/tmp/old-proof/streetlight/outputs/par/par_wide.csv"),
        repo_root=repo_root,
    ) == "outputs/par/par_wide.csv"
    assert module.repo_display_path(
        Path("/external/power/taipower_generation.parquet"),
        repo_root=repo_root,
    ) == "/external/power/taipower_generation.parquet"


def test_taipower_local_timestamps_are_converted_to_utc_naive():
    converted = taipower_local_to_utc_naive(
        pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:10"]))
    )

    assert converted.tolist() == [
        pd.Timestamp("2023-12-31 16:00"),
        pd.Timestamp("2024-01-01 00:10"),
    ]
    assert converted.dt.tz is None


def test_canonical_inputs_preflight_reports_missing_inputs(tmp_path: Path):
    module = _load_analysis_script(
        "check_paper1_canonical_inputs_preflight_test",
        "check_paper1_canonical_inputs_preflight.py",
    )
    args = SimpleNamespace(
        demand_path=tmp_path / "missing_demand.parquet",
        generation_path=tmp_path / "missing_generation.parquet",
        par_weekly_root=tmp_path / "PAR/weekly",
        par_metadata="none",
        staging_root=tmp_path / "outputs/final_runs/paper1_canonical_inputs",
        min_free_multiplier=1.5,
    )

    failures = {check.name for check in module.run_checks(args) if not check.ok}

    assert "demand.exists" in failures
    assert "generation.exists" in failures
    assert "par_weekly_root.expected_stores" in failures


def test_canonical_inputs_preflight_rejects_missing_demand_columns(tmp_path: Path):
    module = _load_analysis_script(
        "check_paper1_canonical_inputs_preflight_columns_test",
        "check_paper1_canonical_inputs_preflight.py",
    )
    demand_path = tmp_path / "regional_power_load.parquet"
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "time": ["00"],
            "source": ["power_demand"],
        }
    ).to_parquet(demand_path)

    failures = {
        check.name: check.detail
        for check in module.check_demand_parquet(demand_path)
        if not check.ok
    }

    assert "demand.required_columns" in failures
    assert "central_gen" in failures["demand.required_columns"]
    assert "central_load" in failures["demand.required_columns"]


def test_power_plant_region_mapping_matches_regional_accounting_boundary():
    assert infer_region_from_plant_name("通霄") == "central"
    assert infer_region_from_plant_name("嘉惠") == "south"
    assert infer_region_from_plant_name("豐德") == "south"
    assert infer_region_from_plant_name("嘉南西口、烏山頭和八田") == "south"
    assert infer_region_from_plant_name("台中龍井") == "central"
    assert infer_region_from_plant_name("清水地熱") == "north"

    # These large plants are grid-accounting exceptions in the Taipower
    # regional generation-balance archive, so they are intentionally not moved
    # by administrative-county location alone.
    assert infer_region_from_plant_name("和平") == "north"
    assert infer_region_from_plant_name("碧海") == "north"
    assert infer_region_from_plant_name("麥寮") == "south"
    assert infer_region_from_plant_name("麥寮#1") == "south"

    # Named wind farms follow the grid-accounting boundary, not a broad county
    # bucket. Specific Miaoli wind projects align with the New-Taoyuan/Yingpan
    # side, Tongxiao remains central, and named Yunlin coastal/offshore wind
    # projects align with the south regional archive.
    assert infer_region_from_plant_name("海洋竹南") == "north"
    assert infer_region_from_plant_name("海能風") == "north"
    assert infer_region_from_plant_name("苗栗大鵬") == "north"
    assert infer_region_from_plant_name("苗栗竹南") == "north"
    assert infer_region_from_plant_name("苗栗通苑") == "north"
    assert infer_region_from_plant_name("龍威後龍") == "north"
    assert infer_region_from_plant_name("崎威崎頂") == "north"
    assert infer_region_from_plant_name("離岸一期") == "central"
    assert infer_region_from_plant_name("中能風") == "central"
    assert infer_region_from_plant_name("芳一風") == "central"
    assert infer_region_from_plant_name("創維風") == "south"
    assert infer_region_from_plant_name("創維麥寮") == "south"
    assert infer_region_from_plant_name("禾風麥寮") == "south"
    assert infer_region_from_plant_name("四湖") == "south"
    assert infer_region_from_plant_name("雲麥") == "south"
    assert infer_region_from_plant_name("新源崙背") == "south"
    assert infer_region_from_plant_name("允湖") == "south"
    assert infer_region_from_plant_name("允西") == "south"
    assert infer_region_from_plant_name("澎湖尖山") == "island_penghu"
    assert infer_region_from_plant_name("七美二期") == "island_penghu"
    assert infer_region_from_plant_name("金門塔山") == "island_kinmen"
    assert infer_region_from_plant_name("馬祖珠山") == "island_lienchiang"
    assert infer_region_from_plant_name("東引") == "island_lienchiang"
    assert infer_region_from_plant_name("蘭嶼") == "island_other"
    assert infer_region_from_plant_name("綠島") == "island_other"
    assert infer_region_from_plant_name("離島其他") == "island_other"


def test_stage_manifest_serializes_repo_local_paths_as_relative(tmp_path: Path):
    module = _load_analysis_script(
        "run_paper1_final_data_pipeline_manifest_paths_test",
        "run_paper1_final_data_pipeline.py",
    )
    staging_root = tmp_path / "outputs/final_runs/paper1_canonical_inputs"
    par_dir = staging_root / "par"
    par_dir.mkdir(parents=True)
    (par_dir / "par_wide.csv").write_text(
        "time,Keelung\n2024-01-01 00:00,1.0\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        run_id="paper1_canonical_inputs_test",
        stage="flow",
        staging_root=staging_root,
        demand_path=Path("/external/power/taipower_demand/current/regional_power_load.parquet"),
        generation_path=Path("/external/power/taipower_gen/current/taipower_generation.parquet"),
        par_weekly_root=Path("/external/satellite/PAR/weekly"),
    )

    result = module.stage_manifest(
        args,
        cfg=None,
        repo_root=tmp_path,
        step_results=[
            {
                "stage": "flow",
                "elapsed_seconds": 1.0,
                "result": {
                    "source_path": args.demand_path,
                    "reference_flow_path": tmp_path / "data/power/flow.csv",
                },
            }
        ],
    )
    manifest = json.loads(
        (staging_root / "manifest/final_run_manifest.json").read_text(encoding="utf-8")
    )
    data_quality = pd.read_csv(staging_root / "manifest/data_quality_summary.csv")

    assert result == {
        "manifest": "outputs/final_runs/paper1_canonical_inputs/manifest/final_run_manifest.json",
        "quality": "outputs/final_runs/paper1_canonical_inputs/manifest/data_quality_summary.csv",
    }
    assert manifest["staging_root"] == "outputs/final_runs/paper1_canonical_inputs"
    assert (
        manifest["quality_summary"]
        == "outputs/final_runs/paper1_canonical_inputs/manifest/data_quality_summary.csv"
    )
    assert manifest["config"]["demand_path"] == str(args.demand_path)
    assert manifest["steps"][0]["result"]["source_path"] == str(args.demand_path)
    assert manifest["steps"][0]["result"]["reference_flow_path"] == "data/power/flow.csv"
    assert manifest["outputs"][0]["path"] == "outputs/final_runs/paper1_canonical_inputs/par/par_wide.csv"
    assert data_quality.loc[0, "path"] == "outputs/final_runs/paper1_canonical_inputs/par/par_wide.csv"


def test_stage_manifest_preserves_prior_steps_when_rerun_after_baseline(tmp_path: Path):
    module = _load_analysis_script(
        "run_paper1_final_data_pipeline_manifest_merge_test",
        "run_paper1_final_data_pipeline.py",
    )
    staging_root = tmp_path / "staging"

    for rel in [
        "manifest/snapshot_current_inputs_hashes.csv",
        "par/par_wide.csv",
        "power/unit_generation.csv",
        "power/central_unit_generation.csv",
        "aef/central.csv",
        "baseline/paper_contract.json",
    ]:
        path = staging_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        else:
            path.write_text("timestamp,value\n2024-01-01 00:00,1.0\n", encoding="utf-8")
    (staging_root / "power/flow_quality_report.json").write_text("{}", encoding="utf-8")
    (staging_root / "manifest/final_run_manifest.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "stage": "baseline",
                        "elapsed_seconds": 12.3,
                        "result": {"baseline": "completed"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        run_id="paper1_manifest_merge_test",
        stage="manifest",
        staging_root=staging_root,
        demand_path=Path("/external/power/taipower_demand/current/regional_power_load.parquet"),
        generation_path=Path("/external/power/taipower_gen/current/taipower_generation.parquet"),
        par_weekly_root=Path("/external/satellite/PAR/weekly"),
    )

    module.stage_manifest(args, cfg=None, repo_root=tmp_path, step_results=[])

    manifest = json.loads(
        (staging_root / "manifest/final_run_manifest.json").read_text(encoding="utf-8")
    )
    assert [step["stage"] for step in manifest["steps"]] == [
        "snapshot",
        "flow",
        "par",
        "power",
        "merge-flow",
        "aef",
        "baseline",
    ]
    assert manifest["steps"][-1]["elapsed_seconds"] == 12.3
    assert manifest["steps"][0]["result"] == {"recovered_from_existing_outputs": True}


def test_daylight_alignment_diagnostics_catches_local_utc_mismatch(tmp_path: Path):
    module = _load_analysis_script(
        "run_paper1_final_data_pipeline_daylight_test",
        "run_paper1_final_data_pipeline.py",
    )
    staging_root = tmp_path / "staging"
    (staging_root / "par").mkdir(parents=True)
    (staging_root / "power").mkdir(parents=True)
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    par_profile = pd.Series(0.0, index=idx)
    par_profile.loc[pd.Timestamp("2024-01-01 04:00")] = 100.0
    pd.DataFrame({"Keelung": par_profile}, index=idx).to_csv(
        staging_root / "par/par_wide.csv",
        index_label="time_utc",
    )
    pd.DataFrame({"timestamp": idx, "Solar": par_profile.to_numpy()}).to_csv(
        staging_root / "power/category_generation.csv",
        index=False,
    )

    aligned = module.daylight_alignment_diagnostics(staging_root)
    assert aligned["status"] == "pass"
    assert aligned["par_peak_hour_utc_label"] == 4
    assert aligned["taipower_solar_peak_hour_utc_label"] == 4

    mismatched_solar = np.roll(par_profile.to_numpy(), 8)
    pd.DataFrame({"timestamp": idx, "Solar": mismatched_solar}).to_csv(
        staging_root / "power/category_generation.csv",
        index=False,
    )

    mismatched = module.daylight_alignment_diagnostics(staging_root)
    assert mismatched["status"] == "fail"
    assert mismatched["peak_hour_circular_distance"] == 8


def test_canonical_inputs_preflight_rejects_generation_without_2024_rows(tmp_path: Path):
    module = _load_analysis_script(
        "check_paper1_canonical_inputs_preflight_generation_test",
        "check_paper1_canonical_inputs_preflight.py",
    )
    generation_path = tmp_path / "taipower_generation.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-12-31 23:50"]),
            "plant_name": ["plant"],
            "mapping_names": ["plant"],
            "energy_type": ["solar"],
            "used_mw": [1.0],
            "capacity_mw": [2.0],
            "status": ["online"],
        }
    ).to_parquet(generation_path)

    failures = {
        check.name: check.detail
        for check in module.check_generation_parquet(generation_path)
        if not check.ok
    }

    assert failures["generation.2024_rows"] == "rows_in_window=0"


def test_canonical_inputs_preflight_rejects_low_disk_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_analysis_script(
        "check_paper1_canonical_inputs_preflight_disk_test",
        "check_paper1_canonical_inputs_preflight.py",
    )
    staging_root = tmp_path / "inputs"
    staging_root.mkdir()
    (staging_root / "input.csv").write_text("x" * 20, encoding="utf-8")
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1))

    check = module.check_disk_space(staging_root, min_free_multiplier=2.0)

    assert not check.ok
    assert "required_bytes=40" in check.detail


def test_promote_dry_run_does_not_write(tmp_path: Path):
    module = _load_analysis_script(
        "run_paper1_final_data_pipeline_promote_test",
        "run_paper1_final_data_pipeline.py",
    )
    staging_root = tmp_path / "staging"
    for subdir in ("par", "power", "aef"):
        path = staging_root / subdir
        path.mkdir(parents=True)
        (path / "input.csv").write_text("value\n1\n", encoding="utf-8")
    args = SimpleNamespace(
        staging_root=staging_root,
        promote_dry_run=True,
        allow_stale_promote=False,
    )

    result = module.stage_promote(args, cfg=None, repo_root=tmp_path)

    assert result["promote_dry_run"]
    assert not (tmp_path / "outputs/par").exists()
    assert not (tmp_path / "data/power").exists()
    assert not (tmp_path / "outputs/AEF").exists()


def test_promote_plan_detects_stale_destination(tmp_path: Path):
    module = _load_analysis_script(
        "run_paper1_final_data_pipeline_plan_test",
        "run_paper1_final_data_pipeline.py",
    )
    (tmp_path / "staging/par").mkdir(parents=True)
    (tmp_path / "staging/par/par_wide.csv").write_text("value\n1\n", encoding="utf-8")
    (tmp_path / "outputs/par").mkdir(parents=True)
    (tmp_path / "outputs/par/stale.csv").write_text("old\n", encoding="utf-8")

    plan = module.build_promote_plan(tmp_path / "staging", tmp_path)
    par_plan = next(item for item in plan if item["name"] == "par")

    assert Path("stale.csv") in par_plan["stale"]


def test_stage_baseline_writes_inside_staging_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_analysis_script(
        "run_paper1_final_data_pipeline_baseline_env_test",
        "run_paper1_final_data_pipeline.py",
    )
    captured = {}

    def fake_run(cmd, *, cwd, env, check):
        del cmd, cwd, check
        captured.update(env)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = SimpleNamespace(staging_root=tmp_path / "staging")

    result = module.stage_baseline(args, cfg=None, repo_root=tmp_path)

    assert result == {
        "baseline": "completed",
        "output_dir": tmp_path / "staging" / "baseline",
    }
    assert captured["PAPER_PAR_PATH"] == str(tmp_path / "staging/par/par_wide.csv")
    assert captured["PAPER_AEF_DIR"] == str(tmp_path / "staging/aef")
    assert captured["PAPER_OUTPUT_DIR"] == str(tmp_path / "staging/baseline")


def test_create_flow_from_regional_demand_uses_confirmed_formula(tmp_path: Path):
    demand = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "time": ["00", "00:10", "00:21"],
            "source": ["power_demand", "areas", "power_demand"],
            "north_gen": [10.0, np.nan, 20.0],
            "north_load": [15.0, 15.0, 25.0],
            "central_gen": [50.0, np.nan, 50.0],
            "central_load": [40.0, 40.0, 40.0],
            "south_gen": [30.0, np.nan, 40.0],
            "south_load": [25.0, 25.0, 45.0],
            "east_gen": [1.0, np.nan, 2.0],
            "east_load": [4.0, 4.0, 5.0],
        }
    )
    demand_path = tmp_path / "regional_power_load.parquet"
    demand.to_parquet(demand_path)

    result = create_flow_from_regional_demand(
        demand_path,
        tmp_path,
        start_date="2023-12-31 16:00",
        end_date="2023-12-31 16:20",
    )

    flow = pd.read_csv(result["flow"], index_col=0, parse_dates=True)
    assert flow.loc[pd.Timestamp("2023-12-31 16:00"), "F_CN"] == pytest.approx(50.0)
    assert flow.loc[pd.Timestamp("2023-12-31 16:00"), "F_CE"] == pytest.approx(30.0)
    assert flow.loc[pd.Timestamp("2023-12-31 16:00"), "F_CS"] == pytest.approx(-50.0)
    assert flow.loc[pd.Timestamp("2023-12-31 16:20"), "F_CN"] == pytest.approx(50.0)


def test_create_flow_drops_invalid_zero_and_off_grid_rows(tmp_path: Path):
    demand = pd.DataFrame(
        {
            "date": ["2024-01-01"] * 4,
            "time": ["00", "00:10", "00:21", "00:25"],
            "source": ["power_demand"] * 4,
            "north_gen": [10.0, 0.0, 20.0, 30.0],
            "north_load": [15.0, 0.0, 25.0, 35.0],
            "central_gen": [50.0, 0.0, 50.0, 50.0],
            "central_load": [40.0, 0.0, 40.0, 40.0],
            "south_gen": [30.0, 0.0, 40.0, 40.0],
            "south_load": [25.0, 0.0, 45.0, 45.0],
            "east_gen": [1.0, 0.0, 2.0, 2.0],
            "east_load": [4.0, 0.0, 5.0, 5.0],
        }
    )
    demand_path = tmp_path / "regional_power_load.parquet"
    demand.to_parquet(demand_path)

    result = create_flow_from_regional_demand(
        demand_path,
        tmp_path,
        start_date="2023-12-31 16:00",
        end_date="2023-12-31 16:30",
    )

    flow = pd.read_csv(result["flow"], index_col=0, parse_dates=True)
    assert flow.index.tolist() == [
        pd.Timestamp("2023-12-31 16:00"),
        pd.Timestamp("2023-12-31 16:20"),
    ]
    quality = json.loads(Path(result["quality_report"]).read_text(encoding="utf-8"))
    assert quality["all_zero_regional_rows"] == 1
    assert quality["all_zero_regional_timestamps"] == 1
    assert quality["extra_timestamps"] == 1
    assert quality["dropped_extra_timestamps"] == 1
    assert quality["observed_timestamps"] == 2
    assert quality["missing_timestamps"] == 2


def test_split_signed_flow_for_aef_creates_reverse_central_columns():
    idx = pd.date_range("2024-01-01", periods=2, freq="10min")
    flow = pd.DataFrame(
        {
            "F_CN": [10.0, -3.0],
            "F_CE": [-2.0, 4.0],
            "F_CS": [5.0, -6.0],
        },
        index=idx,
    )

    split = split_signed_flow_for_aef(flow)

    assert split["north"].loc[idx[0], "F_CN"] == pytest.approx(10.0)
    assert split["north"].loc[idx[1], "F_CN"] == pytest.approx(0.0)
    assert split["central"].loc[idx[1], "F_NC"] == pytest.approx(3.0)
    assert split["central"].loc[idx[0], "F_EC"] == pytest.approx(2.0)
    assert split["central"].loc[idx[1], "F_SC"] == pytest.approx(6.0)


def test_merge_flow_drops_missing_flow_rows_instead_of_zero_filling(tmp_path: Path):
    idx = pd.date_range("2024-01-01", periods=2, freq="10min")
    for region in ("north", "central", "south", "east"):
        pd.DataFrame({"Coal-Unit1": [1.0, 2.0]}, index=idx).to_csv(
            tmp_path / f"{region}_unit_generation.csv"
        )
    pd.DataFrame(
        {"F_CN": [10.0], "F_CE": [20.0], "F_CS": [30.0]},
        index=pd.DatetimeIndex([idx[0]]),
    ).to_csv(tmp_path / "flow.csv")

    merge_flow_data_to_regional_csvs(tmp_path)

    north = pd.read_csv(tmp_path / "north_unit_generation.csv", index_col=0, parse_dates=True)
    assert north.index.tolist() == [idx[0]]
    assert north.loc[idx[0], "F_CN"] == pytest.approx(10.0)


def test_convert_generation_normalizes_and_imputes_short_gaps(tmp_path: Path):
    generation = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 00:10",
                    "2024-01-01 00:31",
                    "2024-01-01 00:40",
                ]
            ),
            "plant_name": ["林口#1"] * 4,
            "unit_id": [1.0] * 4,
            "energy_type": ["coal"] * 4,
            "capacity_mw": [100.0] * 4,
            "used_mw": [10.0, 20.0, 40.0, 50.0],
            "percent": [10.0, 20.0, 40.0, 50.0],
            "is_gov": [True] * 4,
            "status": ["online"] * 4,
            "note": [None] * 4,
            "note_id": [np.nan] * 4,
            "mapping_names": ["林口"] * 4,
            "plant_name_has_note": [False] * 4,
            "plant_name_note_number": [np.nan] * 4,
        }
    )
    generation_path = tmp_path / "taipower_generation.parquet"
    generation.to_parquet(generation_path)

    result = convert_parquet_to_regional_csv(
        generation_path,
        tmp_path,
        start_date="2023-12-31 16:00",
        end_date="2023-12-31 16:50",
        min_date="2023-12-31",
        max_impute_run=1,
    )

    unit_generation = pd.read_csv(result["combined"], parse_dates=["timestamp"])
    assert unit_generation["timestamp"].tolist() == list(
        pd.date_range("2023-12-31 16:00", "2023-12-31 16:40", freq="10min")
    )
    assert unit_generation.loc[
        unit_generation["timestamp"] == pd.Timestamp("2023-12-31 16:20"),
        "Coal-林口#1",
    ].iloc[0] == pytest.approx(30.0)

    flags = pd.read_csv(result["combined_imputed_flags"], parse_dates=["timestamp"])
    assert flags.loc[
        flags["timestamp"] == pd.Timestamp("2023-12-31 16:20"),
        "Coal-林口#1",
    ].iloc[0]
    assert not (
        unit_generation["timestamp"] == pd.Timestamp("2023-12-31 16:50")
    ).any()

    normalized = pd.read_csv(result["timestamp_normalization"])
    assert len(normalized) == 1


def test_generation_conversion_averages_scrape_duplicates_but_sums_storage_net(
    tmp_path: Path,
):
    generation = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 08:00",
                    "2024-01-01 08:01",
                    "2024-01-01 08:00",
                    "2024-01-01 08:00",
                    "2024-01-01 08:01",
                    "2024-01-01 08:01",
                    "2024-01-01 08:00",
                ]
            ),
            "plant_name": [
                "林口#1",
                "林口#1",
                "大觀二#1",
                "大觀二#1",
                "大觀二#1",
                "大觀二#1",
                "汽電共生",
            ],
            "unit_id": [1.0] * 7,
            "energy_type": [
                "coal",
                "coal",
                "storage",
                "storage_load",
                "storage",
                "storage_load",
                "cogen",
            ],
            "capacity_mw": [100.0, 100.0, 120.0, 120.0, 120.0, 120.0, 50.0],
            "used_mw": [10.0, 14.0, 100.0, -40.0, 80.0, -20.0, 5.0],
            "percent": [10.0, 14.0, 83.3, -33.3, 66.7, -16.7, 10.0],
            "is_gov": [True] * 7,
            "status": ["online"] * 7,
            "note": [None] * 7,
            "note_id": [np.nan] * 7,
            "mapping_names": ["林口", "林口", "大觀", "大觀", "大觀", "大觀", "汽電共生"],
            "plant_name_has_note": [False] * 7,
            "plant_name_note_number": [np.nan] * 7,
        }
    )
    generation_path = tmp_path / "taipower_generation.parquet"
    generation.to_parquet(generation_path)

    result = convert_parquet_to_regional_csv(
        generation_path,
        tmp_path,
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:00",
        min_date="2024-01-01",
    )

    unit_generation = pd.read_csv(result["combined"], parse_dates=["timestamp"])
    row = unit_generation.iloc[0]
    assert row["Coal-林口#1"] == pytest.approx(12.0)
    assert row["Storage-大觀二#1"] == pytest.approx(60.0)

    category_generation = pd.read_csv(result["category"], parse_dates=["timestamp"])
    category_row = category_generation.iloc[0]
    assert category_row["Coal"] == pytest.approx(12.0)
    assert category_row["Co-Gen"] == pytest.approx(5.0)
    assert "Co" not in category_generation.columns
    assert category_row["Storage"] == pytest.approx(60.0)

    central_generation = pd.read_csv(result["central"], parse_dates=["timestamp"])
    central_row = central_generation.iloc[0]
    assert central_row["Storage-大觀二#1"] == pytest.approx(60.0)
    assert central_row["PHS"] == pytest.approx(60.0)


def test_generation_conversion_drops_invalid_low_total_timestamps(tmp_path: Path):
    generation = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 08:00",
                    "2024-01-01 08:10",
                    "2024-01-01 08:20",
                    "2024-01-01 08:30",
                    "2024-01-01 08:40",
                ]
            ),
            "plant_name": ["林口#1"] * 5,
            "unit_id": [1.0] * 5,
            "energy_type": ["coal"] * 5,
            "capacity_mw": [200.0] * 5,
            "used_mw": [150.0, 160.0, 1.0, 2.0, 170.0],
            "percent": [75.0, 80.0, 0.5, 1.0, 85.0],
            "is_gov": [True] * 5,
            "status": ["online"] * 5,
            "note": [None] * 5,
            "note_id": [np.nan] * 5,
            "mapping_names": ["林口"] * 5,
            "plant_name_has_note": [False] * 5,
            "plant_name_note_number": [np.nan] * 5,
        }
    )
    generation_path = tmp_path / "taipower_generation.parquet"
    generation.to_parquet(generation_path)

    result = convert_parquet_to_regional_csv(
        generation_path,
        tmp_path,
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:40",
        min_date="2024-01-01",
        max_impute_run=1,
        min_valid_total_generation_mw=100.0,
    )

    unit_generation = pd.read_csv(result["combined"], parse_dates=["timestamp"])
    assert unit_generation["timestamp"].tolist() == [
        pd.Timestamp("2024-01-01 00:00"),
        pd.Timestamp("2024-01-01 00:10"),
        pd.Timestamp("2024-01-01 00:40"),
    ]
    assert not (
        unit_generation["timestamp"].isin(
            [pd.Timestamp("2024-01-01 00:20"), pd.Timestamp("2024-01-01 00:30")]
        )
    ).any()

    invalid = pd.read_csv(result["invalid_total_report"], parse_dates=["timestamp"])
    assert invalid["timestamp"].tolist() == [
        pd.Timestamp("2024-01-01 00:20"),
        pd.Timestamp("2024-01-01 00:30"),
    ]
    assert invalid["total_generation_mw"].tolist() == pytest.approx([1.0, 2.0])

    quality = json.loads(Path(result["quality_report"]).read_text(encoding="utf-8"))
    assert quality["invalid_total_generation"]["timestamps"] == 2
    assert quality["invalid_total_generation"]["source_rows_dropped"] == 2
    assert quality["combined"]["missing_after_imputation"] == 2


def test_generation_conversion_reports_regional_balance_diagnostics(tmp_path: Path):
    generation = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 08:00"] * 5),
            "plant_name": ["林口", "台中", "大觀二", "興達", "東部小水力"],
            "unit_id": [1.0, 1.0, 1.0, 1.0, 1.0],
            "energy_type": ["coal", "coal", "storage", "coal", "hydro"],
            "capacity_mw": [100.0, 100.0, 100.0, 100.0, 100.0],
            "used_mw": [10.0, 0.0, 6.0, 0.0, 0.0],
            "percent": [10.0, 0.0, 6.0, 0.0, 0.0],
            "is_gov": [True] * 5,
            "status": ["online"] * 5,
            "note": [None] * 5,
            "note_id": [np.nan] * 5,
            "mapping_names": ["林口", "台中", "大觀二", "興達", "東部小水力"],
            "plant_name_has_note": [False] * 5,
            "plant_name_note_number": [np.nan] * 5,
        }
    )
    generation_path = tmp_path / "taipower_generation.parquet"
    generation.to_parquet(generation_path)

    demand = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "time": ["08:00"],
            "source": ["power_demand"],
            "north_gen": [1.0],
            "north_load": [1.0],
            "central_gen": [0.6],
            "central_load": [0.0],
            "south_gen": [0.0],
            "south_load": [0.0],
            "east_gen": [0.0],
            "east_load": [0.0],
        }
    )
    demand_path = tmp_path / "regional_power_load.parquet"
    demand.to_parquet(demand_path)

    result = convert_parquet_to_regional_csv(
        generation_path,
        tmp_path,
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:00",
        min_date="2024-01-01",
        regional_demand_path=demand_path,
    )

    quality = json.loads(Path(result["quality_report"]).read_text(encoding="utf-8"))
    balance = quality["regional_generation_balance"]

    assert balance["status"] == "computed"
    assert balance["aligned_timestamps"] == 1
    assert balance["by_region"]["north"]["mean_error_mw"] == pytest.approx(0.0)
    assert balance["by_region"]["central"]["mean_error_mw"] == pytest.approx(0.0)
    assert balance["generation_column_policy"]["included_storage_aggregates"] == [
        "PHS",
        "BESS",
    ]


def test_solar_other_uses_taipower_county_purchased_generation_distribution(
    tmp_path: Path,
):
    generation = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 08:00"] * 5),
            "plant_name": ["恆水創電", "彰濱光", "興達", "東部小水力", "太陽能"],
            "unit_id": [1.0, 1.0, 1.0, 1.0, 1.0],
            "energy_type": ["solar", "solar", "coal", "hydro", "solar"],
            "capacity_mw": [100.0, 100.0, 100.0, 100.0, 100.0],
            "used_mw": [0.0, 0.0, 0.0, 0.0, 30.0],
            "percent": [0.0, 0.0, 0.0, 0.0, 30.0],
            "is_gov": [True] * 5,
            "status": ["online"] * 5,
            "note": [None] * 5,
            "note_id": [np.nan] * 5,
            "mapping_names": ["恆水創電", "彰濱光", "興達", "東部小水力", "太陽能"],
            "plant_name_has_note": [False] * 5,
            "plant_name_note_number": [np.nan] * 5,
        }
    )
    generation_path = tmp_path / "taipower_generation.parquet"
    generation.to_parquet(generation_path)

    result = convert_parquet_to_regional_csv(
        generation_path,
        tmp_path,
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:00",
        min_date="2024-01-01",
    )

    quality = json.loads(Path(result["quality_report"]).read_text(encoding="utf-8"))
    solar = quality["other_region_redistribution"]["Solar"]
    basis = solar["ratio_basis"]
    purchased_kwh = basis["purchased_kwh_by_grid_region"]
    mainland_total = basis["mainland_purchased_kwh"]

    assert solar["status"] == "redistributed"
    assert solar["ratio_source"] == "fixed_solar_county_purchased_generation_share_2024"
    assert "total_mainland_capacity_mw" not in solar
    assert basis["basis"] == "taipower_2024_county_purchased_solar_generation_kwh"
    assert basis["source_year_roc"] == 113
    assert basis["counties_by_grid_region"]["south"] == [
        "雲林縣",
        "嘉義市",
        "嘉義縣",
        "台南市",
        "高雄市",
        "屏東縣",
    ]
    assert basis["counties_by_grid_region"]["island_penghu"] == ["澎湖縣"]
    assert basis["counties_by_grid_region"]["island_kinmen"] == ["金門縣"]
    assert basis["counties_by_grid_region"]["island_lienchiang"] == ["連江縣"]
    assert basis["excluded_island_purchased_kwh"] == pytest.approx(79_293_861.0)
    assert basis["excluded_island_capacity_kw"] == pytest.approx(97_462.26)
    assert basis["excluded_island_purchased_kwh_by_load_serving_zone"] == {
        "island_penghu": pytest.approx(53_034_394.0),
        "island_kinmen": pytest.approx(26_185_786.0),
        "island_lienchiang": pytest.approx(73_681.0),
    }
    assert solar["ratios"]["north"] == pytest.approx(purchased_kwh["north"] / mainland_total)
    assert solar["ratios"]["central"] == pytest.approx(
        purchased_kwh["central"] / mainland_total
    )
    assert solar["ratios"]["south"] == pytest.approx(purchased_kwh["south"] / mainland_total)
    assert solar["ratios"]["east"] == pytest.approx(purchased_kwh["east"] / mainland_total)
    assert solar["other_total_mean_mw"] == pytest.approx(30.0)
    assert solar["other_total_p95_mw"] == pytest.approx(30.0)
    assert solar["other_total_max_mw"] == pytest.approx(30.0)
    assert solar["allocated_mean_mw_by_region"]["north"] == pytest.approx(
        30.0 * solar["ratios"]["north"]
    )
    assert solar["allocated_mean_mw_by_region"]["central"] == pytest.approx(
        30.0 * solar["ratios"]["central"]
    )
    assert solar["other_columns_count"] == 1

    north = pd.read_csv(result["north"])
    central = pd.read_csv(result["central"])
    south = pd.read_csv(result["south"])
    east = pd.read_csv(result["east"])
    assert north["Solar-_other_allocated"].iloc[0] == pytest.approx(
        30.0 * solar["ratios"]["north"]
    )
    assert central["Solar-_other_allocated"].iloc[0] == pytest.approx(
        30.0 * solar["ratios"]["central"]
    )
    assert south["Solar-_other_allocated"].iloc[0] == pytest.approx(
        30.0 * solar["ratios"]["south"]
    )
    assert east["Solar-_other_allocated"].iloc[0] == pytest.approx(
        30.0 * solar["ratios"]["east"]
    )


def test_generation_conversion_splits_named_outlying_island_grids(tmp_path: Path):
    generation = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 08:00"] * 4),
            "plant_name": ["澎湖尖山", "金門塔山", "馬祖珠山", "離島其他"],
            "unit_id": [1.0, 1.0, 1.0, 1.0],
            "energy_type": ["oil", "diesel", "diesel", "diesel"],
            "capacity_mw": [100.0, 100.0, 100.0, 100.0],
            "used_mw": [4.0, 5.0, 6.0, 0.0],
            "percent": [4.0, 5.0, 6.0, 0.0],
            "is_gov": [True] * 4,
            "status": ["online"] * 4,
            "note": [None] * 4,
            "note_id": [np.nan] * 4,
            "mapping_names": ["澎湖尖山", "金門塔山", "馬祖珠山", "離島其他"],
            "plant_name_has_note": [False] * 4,
            "plant_name_note_number": [np.nan] * 4,
        }
    )
    generation_path = tmp_path / "taipower_generation.parquet"
    generation.to_parquet(generation_path)

    result = convert_parquet_to_regional_csv(
        generation_path,
        tmp_path,
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:00",
        min_date="2024-01-01",
    )

    penghu = pd.read_csv(result["island_penghu"])
    kinmen = pd.read_csv(result["island_kinmen"])
    lienchiang = pd.read_csv(result["island_lienchiang"])
    island_other = pd.read_csv(result["island_other"])

    assert "island" not in result
    assert penghu["Oil-澎湖尖山"].iloc[0] == pytest.approx(4.0)
    assert kinmen["Diesel-金門塔山"].iloc[0] == pytest.approx(5.0)
    assert lienchiang["Diesel-馬祖珠山"].iloc[0] == pytest.approx(6.0)
    assert island_other["Diesel-離島其他"].iloc[0] == pytest.approx(0.0)


def test_wind_other_still_uses_named_capacity_distribution(tmp_path: Path):
    generation = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 08:00"] * 3),
            "plant_name": ["觀音", "彰工", "風力"],
            "unit_id": [1.0, 1.0, 1.0],
            "energy_type": ["wind", "wind", "wind"],
            "capacity_mw": [30.0, 70.0, 100.0],
            "used_mw": [0.0, 0.0, 50.0],
            "percent": [0.0, 0.0, 50.0],
            "is_gov": [True] * 3,
            "status": ["online"] * 3,
            "note": [None] * 3,
            "note_id": [np.nan] * 3,
            "mapping_names": ["觀音", "彰工", "風力"],
            "plant_name_has_note": [False] * 3,
            "plant_name_note_number": [np.nan] * 3,
        }
    )
    generation_path = tmp_path / "taipower_generation.parquet"
    generation.to_parquet(generation_path)

    result = convert_parquet_to_regional_csv(
        generation_path,
        tmp_path,
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:00",
        min_date="2024-01-01",
    )

    quality = json.loads(Path(result["quality_report"]).read_text(encoding="utf-8"))
    wind = quality["other_region_redistribution"]["Wind"]

    assert wind["status"] == "redistributed"
    assert wind["ratio_source"] == "capacity_share"
    assert wind["ratio_basis"]["basis"] == "same_fuel_named_mainland_capacity_mw"
    assert wind["ratio_basis"]["mainland_capacity_mw"] == pytest.approx(100.0)
    assert wind["ratios"]["north"] == pytest.approx(0.3)
    assert wind["ratios"]["central"] == pytest.approx(0.7)
    north = pd.read_csv(result["north"])
    central = pd.read_csv(result["central"])
    assert north["Wind-_other_allocated"].iloc[0] == pytest.approx(15.0)
    assert central["Wind-_other_allocated"].iloc[0] == pytest.approx(35.0)


def test_cogeneration_other_uses_fixed_capacity_distribution(tmp_path: Path):
    generation = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 08:00"] * 5),
            "plant_name": ["林口", "台中", "興達", "東部小水力", "汽電共生"],
            "unit_id": [1.0, 1.0, 1.0, 1.0, np.nan],
            "energy_type": ["coal", "coal", "coal", "hydro", "cogen"],
            "capacity_mw": [100.0, 100.0, 100.0, 100.0, 0.0],
            "used_mw": [0.0, 0.0, 0.0, 0.0, 100.0],
            "percent": [0.0, 0.0, 0.0, 0.0, 100.0],
            "is_gov": [True] * 5,
            "status": ["online"] * 5,
            "note": [None] * 5,
            "note_id": [np.nan] * 5,
            "mapping_names": ["林口", "台中", "興達", "東部小水力", "汽電共生"],
            "plant_name_has_note": [False] * 5,
            "plant_name_note_number": [np.nan] * 5,
        }
    )
    generation_path = tmp_path / "taipower_generation.parquet"
    generation.to_parquet(generation_path)

    demand = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "time": ["08:00"],
            "source": ["power_demand"],
            "north_gen": [0.0],
            "north_load": [1000.0],
            "central_gen": [0.0],
            "central_load": [1.0],
            "south_gen": [0.0],
            "south_load": [1.0],
            "east_gen": [0.0],
            "east_load": [1.0],
        }
    )
    demand_path = tmp_path / "regional_power_load.parquet"
    demand.to_parquet(demand_path)

    result = convert_parquet_to_regional_csv(
        generation_path,
        tmp_path,
        start_date="2024-01-01 00:00",
        end_date="2024-01-01 00:00",
        min_date="2024-01-01",
        regional_demand_path=demand_path,
    )

    quality = json.loads(Path(result["quality_report"]).read_text(encoding="utf-8"))
    cogen = quality["other_region_redistribution"]["Co-Gen"]
    total_kw = 4_939_588.0
    expected = {
        "north": 994_287.0 / total_kw,
        "central": 2_779_958.0 / total_kw,
        "south": 1_156_443.0 / total_kw,
        "east": 8_900.0 / total_kw,
    }

    assert cogen["status"] == "redistributed"
    assert cogen["ratio_source"] == "fixed_cogeneration_capacity_share_2024"
    assert cogen["ratio_basis"]["basis"] == "taipower_2024_purchased_power_cogeneration_capacity_kw"
    for region, ratio in expected.items():
        assert cogen["ratios"][region] == pytest.approx(ratio)
        regional = pd.read_csv(result[region])
        assert regional["Co-Gen-_other_allocated"].iloc[0] == pytest.approx(100.0 * ratio)


def test_area_weighted_county_par_with_manual_weights():
    from streetlight.par.par_aggregation import compute_area_weighted_county_par

    par = np.array([[10.0, 20.0], [30.0, 40.0]])
    weights = pd.DataFrame(
        {
            "__cid__": [1, 1, 2],
            "cell_index": [0, 1, 3],
            "overlap_area": [1.0, 3.0, 2.0],
        }
    )

    out = compute_area_weighted_county_par(par, weights, np.array([1, 2], dtype=np.int32))

    county1 = out[out["__cid__"] == 1].iloc[0]
    county2 = out[out["__cid__"] == 2].iloc[0]
    assert county1["mean_PAR_umol_m2_s"] == pytest.approx((10.0 * 1.0 + 20.0 * 3.0) / 4.0)
    assert county2["mean_PAR_umol_m2_s"] == pytest.approx(40.0)


def test_probabilistic_uncertainty_scales_pv_om_by_deployment_count(monkeypatch):
    script_path = Path(__file__).resolve().parents[1] / "scripts/analysis/run_probabilistic_uncertainty.py"
    spec = importlib.util.spec_from_file_location("run_probabilistic_uncertainty_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module,
        "_sample_triangular",
        lambda _rng, _low, mode, _high, size: np.full(size, mode, dtype=float),
    )
    monkeypatch.setattr(
        module,
        "_sample_replacement_year",
        lambda _rng, size: np.full(size, 12.0, dtype=float),
    )

    row = pd.Series(
        {
            "solar_panel_factor": 2.0,
            "battery_capacity_kwh": 5.0,
            "battery_power_kw": 1.0,
            "grid_energy_kwh_20y": 1000.0,
            "pv_storage_energy_kwh_20y": 400.0,
            "abatement_t": 10.0,
        }
    )
    years = 20.0
    n_deployments = 3
    pv_capacity_kw = 10.0 * row["solar_panel_factor"]
    discount_rate = 0.05
    pw_factor = module._present_worth_factor(years, discount_rate)

    result = module._compute_cost_distribution(
        row=row,
        years=years,
        pv_capacity_kw_per_factor=10.0,
        n_deployments=n_deployments,
        n_samples=1,
        seed=1,
    ).iloc[0]

    annual_grid_energy = row["grid_energy_kwh_20y"] / years
    annual_pv_grid_energy = row["pv_storage_energy_kwh_20y"] / years
    grid_only_cost = (
        annual_grid_energy * 3.7556 * pw_factor
        + (95000.0 * n_deployments / years) * pw_factor
    )
    battery_init_cost = n_deployments * (
        row["battery_capacity_kwh"] * 8096.0
        + row["battery_power_kw"] * 30976.0
    )
    init_cost = (
        n_deployments * pv_capacity_kw * 42880.0
        + battery_init_cost
        + 44583.0 * n_deployments
    )
    om_cost = (
        n_deployments * pv_capacity_kw * 704.0 * pw_factor
        + battery_init_cost * 0.025 * pw_factor
        + (46097.0 * n_deployments / years) * pw_factor
    )
    replacement_cost = module._discount_lump(
        n_deployments * row["battery_capacity_kwh"] * 4896.0,
        12.0,
        discount_rate,
    )
    eol_cost = module._discount_lump(6300.0 * n_deployments, years, discount_rate) + replacement_cost
    pv_storage_cost = annual_pv_grid_energy * 3.7556 * pw_factor + init_cost + om_cost + eol_cost

    assert result["delta_cost"] == pytest.approx(pv_storage_cost - grid_only_cost)

    missing_if_unscaled = (n_deployments - 1) * pv_capacity_kw * 704.0 * pw_factor
    faulty_delta = result["delta_cost"] - missing_if_unscaled
    assert result["delta_cost"] - faulty_delta == pytest.approx(missing_if_unscaled)
