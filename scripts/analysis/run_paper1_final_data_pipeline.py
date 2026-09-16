from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from paper1_provenance import normalize_repo_paths, repo_display_path

from streetlight.aef import AEFPipeline
from streetlight.config import get_config
from streetlight.par import (
    PARZarrConfig,
    PARZarrPipeline,
    interpolate_short_gaps,
    par_aggregated_to_wide,
    save_par_wide,
)
from streetlight.processing.power_adapter import (
    CANONICAL_TIMESTAMP_TZ,
    TAIPOWER_SOURCE_TZ,
    convert_parquet_to_regional_csv,
    create_flow_from_regional_demand,
    merge_flow_data_to_regional_csvs,
)

DEFAULT_RUN_ID = "paper1_canonical_inputs"
EXPECTED_START = "2024-01-01 00:00"
EXPECTED_END = "2024-12-31 23:50"
EXPECTED_TIMEZONE = CANONICAL_TIMESTAMP_TZ
MIN_VALID_TOTAL_GENERATION_MW = 10_000.0
PROMOTE_TREES = (
    ("par", "par", Path("outputs/par")),
    ("power", "power", Path("data/power")),
    ("aef", "aef", Path("outputs/AEF")),
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        path = Path(path)
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    yield child


def write_hash_manifest(paths: Iterable[Path], output_path: Path, root: Path) -> pd.DataFrame:
    rows = []
    for path in iter_files(paths):
        rows.append(
            {
                "path": repo_display_path(path, repo_root=root),
                "relative_path": repo_display_path(path, repo_root=root),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    df = pd.DataFrame(rows).sort_values("path") if rows else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def git_sha(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def git_status_short(repo_root: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return [line for line in result.stdout.splitlines() if line]
    except Exception:
        return None


def expected_2024_index() -> pd.DatetimeIndex:
    return pd.date_range(EXPECTED_START, EXPECTED_END, freq="10min")


def timestamp_quality(path: Path, *, index_col: int | str | None = 0, repo_root: Path | None = None) -> dict:
    df = pd.read_csv(path, index_col=index_col)
    if index_col is None:
        ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    else:
        ts = pd.to_datetime(df.index, errors="coerce")
    ts = pd.DatetimeIndex(ts.dropna().unique()).sort_values()
    expected = expected_2024_index()
    in_range = ts[(ts >= expected[0]) & (ts <= expected[-1])]
    missing = expected.difference(in_range)
    return {
        "path": repo_display_path(path, repo_root=repo_root or path.parent),
        "timestamp_timezone": EXPECTED_TIMEZONE,
        "timestamp_storage": "timezone-naive UTC",
        "observed_timestamps": int(len(in_range)),
        "expected_timestamps": int(len(expected)),
        "missing_timestamps": int(len(missing)),
        "coverage": float(len(in_range) / len(expected)),
        "first_timestamp": str(in_range[0]) if len(in_range) else None,
        "last_timestamp": str(in_range[-1]) if len(in_range) else None,
    }


def circular_hour_distance(a: int, b: int) -> int:
    return int(abs((a - b + 12) % 24 - 12))


def daylight_alignment_diagnostics(staging_root: Path) -> dict:
    """
    Check that UTC-labeled PAR daylight and Taipower solar generation peak together.

    A large peak-hour separation is a strong signal that PAR is UTC while power
    data are still local-naive. The check is intentionally simple and runs from
    workflow outputs so the rebuilt canonical input snapshot carries its own
    timestamp-basis evidence.
    """
    par_path = staging_root / "par/par_wide.csv"
    category_path = staging_root / "power/category_generation.csv"
    if not par_path.exists() or not category_path.exists():
        return {
            "status": "skipped",
            "reason": "missing par_wide.csv or category_generation.csv",
        }

    par = pd.read_csv(par_path, index_col=0, parse_dates=True)
    category = pd.read_csv(category_path, parse_dates=["timestamp"])
    if "Solar" not in category.columns:
        return {"status": "skipped", "reason": "category_generation.csv has no Solar column"}

    par_mean = par.apply(pd.to_numeric, errors="coerce").mean(axis=1)
    solar = pd.to_numeric(category["Solar"], errors="coerce")
    solar.index = pd.DatetimeIndex(category["timestamp"])

    par_hourly = par_mean.groupby(par_mean.index.hour).mean()
    solar_hourly = solar.groupby(solar.index.hour).mean()
    if par_hourly.empty or solar_hourly.empty or solar_hourly.max() <= 0:
        return {"status": "skipped", "reason": "insufficient positive PAR/solar signal"}

    par_peak_hour = int(par_hourly.idxmax())
    solar_peak_hour = int(solar_hourly.idxmax())
    peak_hour_distance = circular_hour_distance(par_peak_hour, solar_peak_hour)
    max_allowed_distance = 2
    status = "pass" if peak_hour_distance <= max_allowed_distance else "fail"
    return {
        "status": status,
        "par_peak_hour_utc_label": par_peak_hour,
        "taipower_solar_peak_hour_utc_label": solar_peak_hour,
        "peak_hour_circular_distance": peak_hour_distance,
        "max_allowed_peak_hour_distance": max_allowed_distance,
        "par_time_basis": "UTC",
        "taipower_source_time_basis": TAIPOWER_SOURCE_TZ,
        "taipower_output_time_basis": EXPECTED_TIMEZONE,
    }


def stage_snapshot(args, cfg, repo_root: Path) -> dict:
    del cfg
    manifest_dir = args.staging_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        repo_root / "data/par/processed/par_wide.csv",
        repo_root / "outputs/par/par_wide.csv",
        repo_root / "data/power",
        repo_root / "outputs/AEF",
        repo_root / "config/region_city_map.json",
        repo_root / "config/paper_baseline.yaml",
    ]
    existing = [p for p in paths if p.exists()]
    hash_df = write_hash_manifest(
        existing,
        manifest_dir / "snapshot_current_inputs_hashes.csv",
        repo_root,
    )
    return {"files": int(len(hash_df))}


def stage_flow(args, cfg, repo_root: Path) -> dict:
    del cfg
    output_dir = args.staging_root / "power"
    result = create_flow_from_regional_demand(
        args.demand_path,
        output_dir,
        start_date=EXPECTED_START,
        end_date=EXPECTED_END,
    )
    return {key: str(value) for key, value in result.items()}


def _par_weekly_paths(root: Path) -> list[Path]:
    return sorted((root / "2024").glob("2024W*.zarr")) + [root / "2025" / "2025W01.zarr"]


def stage_par(args, cfg, repo_root: Path) -> dict:
    del repo_root
    par_dir = args.staging_root / "par"
    par_dir.mkdir(parents=True, exist_ok=True)

    weekly_root = args.par_weekly_root
    stores = [p for p in _par_weekly_paths(weekly_root) if p.exists()]
    if not stores:
        raise FileNotFoundError(f"No weekly PAR zarr stores found under {weekly_root}")

    long_parts = []
    for idx, store in enumerate(stores, start=1):
        print(f"[PAR] {idx}/{len(stores)} start {store}", flush=True)
        config = PARZarrConfig.from_runtime_config(
            cfg,
            zarr_path=store,
            output_dir=par_dir,
            time_range=(EXPECTED_START, EXPECTED_END),
        )
        config.aggregation_method = "area_weighted"
        config.upsample_factor = 1
        pipeline = PARZarrPipeline(config=config)
        part = pipeline.run(save=False, show_progress=args.show_progress)
        part_path = par_dir / f"par_aggregated_{store.parent.name}_{store.stem}.csv"
        part.to_csv(part_path, index=False)
        print(f"[PAR] {idx}/{len(stores)} done rows={len(part)} -> {part_path}", flush=True)
        long_parts.append(part)

    long_df = pd.concat(long_parts, ignore_index=True)
    long_df = long_df[
        (pd.to_datetime(long_df["time_utc"]) >= pd.Timestamp(EXPECTED_START))
        & (pd.to_datetime(long_df["time_utc"]) <= pd.Timestamp(EXPECTED_END))
    ].copy()
    long_df = long_df.drop_duplicates(subset=["time_utc", "county"], keep="last")
    long_path = par_dir / "par_aggregated_area_weighted.csv"
    long_df.to_csv(long_path, index=False)

    wide = par_aggregated_to_wide(long_df)
    filled, flags = interpolate_short_gaps(wide, max_run=6, freq="10min")
    wide_path = par_dir / "par_wide.csv"
    flags_path = par_dir / "par_imputed_flags.csv"
    save_par_wide(filled, wide_path)
    flags.to_csv(flags_path, index=True, index_label="time_utc")

    return {
        "stores": len(stores),
        "long_rows": int(len(long_df)),
        "wide_path": str(wide_path),
        "flags_path": str(flags_path),
    }


def stage_power(args, cfg, repo_root: Path) -> dict:
    del cfg, repo_root
    power_dir = args.staging_root / "power"
    power_dir.mkdir(parents=True, exist_ok=True)
    paths = convert_parquet_to_regional_csv(
        args.generation_path,
        power_dir,
        start_date=EXPECTED_START,
        end_date=EXPECTED_END,
        min_date=EXPECTED_START,
        regional_demand_path=args.demand_path,
        min_valid_total_generation_mw=MIN_VALID_TOTAL_GENERATION_MW,
    )
    return {key: str(value) for key, value in paths.items()}


def stage_merge_flow(args, cfg, repo_root: Path) -> dict:
    del cfg, repo_root
    power_dir = args.staging_root / "power"
    flow_path = power_dir / "flow.csv"
    updated = merge_flow_data_to_regional_csvs(power_dir, flow_path)
    return {key: str(value) for key, value in updated.items()}


def stage_aef(args, cfg, repo_root: Path) -> dict:
    del cfg, repo_root
    power_dir = args.staging_root / "power"
    aef_dir = args.staging_root / "aef"
    pipeline = AEFPipeline()
    region_data, pool_df = pipeline.run(data_dir=power_dir, scale=1.0)
    pipeline.save_results(aef_dir, pool_df)
    return {
        "regions": sorted(region_data),
        "output_dir": str(aef_dir),
    }


def stage_manifest(args, cfg, repo_root: Path, step_results: list[dict]) -> dict:
    del cfg
    manifest_dir = args.staging_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "final_run_manifest.json"
    previous_steps = []
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_steps = previous.get("steps", [])
        except Exception:
            previous_steps = []
    recovery_checks = {
        "snapshot": args.staging_root / "manifest/snapshot_current_inputs_hashes.csv",
        "flow": args.staging_root / "power/flow_quality_report.json",
        "par": args.staging_root / "par/par_wide.csv",
        "power": args.staging_root / "power/unit_generation.csv",
        "merge-flow": args.staging_root / "power/central_unit_generation.csv",
        "aef": args.staging_root / "aef/central.csv",
        "baseline": args.staging_root / "baseline/paper_contract.json",
    }
    recovered = []
    for name, path in recovery_checks.items():
        if path.exists():
            recovered.append(
                {
                    "stage": name,
                    "elapsed_seconds": None,
                    "result": {"recovered_from_existing_outputs": True},
                }
            )
    stage_order = {
        stage: idx
        for idx, stage in enumerate(
            ["snapshot", "flow", "par", "power", "merge-flow", "aef", "promote", "baseline"]
        )
    }
    if step_results and args.stage != "all":
        merged_steps = {}
        for source in (recovered, previous_steps, step_results):
            for step in source:
                stage = step.get("stage")
                if not stage:
                    continue
                merged_steps[stage] = step
        step_results = sorted(
            merged_steps.values(),
            key=lambda step: stage_order.get(step.get("stage"), len(stage_order)),
        )
    elif not step_results:
        step_results = previous_steps
        existing_stages = {step.get("stage") for step in step_results}
        missing_recovered = [
            step for step in recovered if step.get("stage") not in existing_stages
        ]
        if missing_recovered:
            step_results = sorted(
                [*missing_recovered, *step_results],
                key=lambda step: stage_order.get(step.get("stage"), len(stage_order)),
            )
        if not step_results:
            step_results = recovered

    quality_paths = [
        args.staging_root / "par/par_wide.csv",
        args.staging_root / "power/category_generation.csv",
        args.staging_root / "power/flow.csv",
        args.staging_root / "power/unit_anno.csv",
        args.staging_root / "power/unit_capacity.csv",
        args.staging_root / "power/unit_generation.csv",
    ]
    quality_paths.extend(sorted((args.staging_root / "power").glob("*_unit_generation.csv")))
    quality_rows = []
    for path in quality_paths:
        if path.exists():
            quality_rows.append(timestamp_quality(path, repo_root=repo_root))
    for path in sorted((args.staging_root / "aef").glob("*.csv")):
        quality_rows.append(timestamp_quality(path, repo_root=repo_root))
    quality_df = pd.DataFrame(quality_rows)
    quality_path = manifest_dir / "data_quality_summary.csv"
    quality_df.to_csv(quality_path, index=False)

    alignment_checks = daylight_alignment_diagnostics(args.staging_root)
    if alignment_checks.get("status") == "fail":
        raise RuntimeError(
            "PAR and Taipower solar daylight profiles are not aligned on the "
            "canonical UTC timestamp basis: "
            + json.dumps(alignment_checks, ensure_ascii=False)
        )

    output_hashes = write_hash_manifest(
        [args.staging_root / "par", args.staging_root / "power", args.staging_root / "aef"],
        manifest_dir / "final_output_hashes.csv",
        repo_root,
    )
    output_records = output_hashes.to_dict(orient="records") if not output_hashes.empty else []
    git_status = git_status_short(repo_root)

    run_manifest = {
        "run_id": args.run_id,
        "git_sha": git_sha(repo_root),
        "git_dirty": bool(git_status) if git_status is not None else None,
        "git_status_short": git_status,
        "staging_root": args.staging_root,
        "created_at_unix": time.time(),
        "expected_time_range": {
            "start": EXPECTED_START,
            "end": EXPECTED_END,
            "freq": "10min",
            "timezone": EXPECTED_TIMEZONE,
            "timestamp_storage": "timezone-naive UTC",
        },
        "time_basis": {
            "canonical_outputs": EXPECTED_TIMEZONE,
            "par_source": "UTC",
            "taipower_source": TAIPOWER_SOURCE_TZ,
            "taipower_conversion": "local Asia/Taipei -> UTC before filtering and AEF",
        },
        "time_alignment_checks": alignment_checks,
        "config": {
            "streetlight_config": os.getenv("STREETLIGHT_CONFIG", "config/config.yaml"),
            "demand_path": args.demand_path,
            "generation_path": args.generation_path,
            "par_weekly_root": args.par_weekly_root,
        },
        "steps": step_results,
        "quality_summary": quality_path,
        "output_hash_count": int(len(output_hashes)),
        "outputs": output_records,
    }
    run_manifest = normalize_repo_paths(run_manifest, repo_root=repo_root)
    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return normalize_repo_paths({"manifest": manifest_path, "quality": quality_path}, repo_root=repo_root)


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def relative_file_sizes(root: Path) -> dict[Path, int]:
    if not root.exists() or not root.is_dir():
        return {}
    return {
        child.relative_to(root): child.stat().st_size
        for child in sorted(root.rglob("*"))
        if child.is_file()
    }


def build_promote_plan(staging_root: Path, repo_root: Path) -> list[dict]:
    plan = []
    for name, src_name, dst_rel in PROMOTE_TREES:
        src = staging_root / src_name
        dst = repo_root / dst_rel
        source_files = relative_file_sizes(src)
        destination_files = relative_file_sizes(dst)
        source_set = set(source_files)
        destination_set = set(destination_files)
        plan.append(
            {
                "name": name,
                "source": src,
                "destination": dst,
                "source_exists": src.exists() and src.is_dir(),
                "destination_is_file": dst.exists() and not dst.is_dir(),
                "create": sorted(source_set - destination_set),
                "overwrite": sorted(source_set & destination_set),
                "stale": sorted(destination_set - source_set),
                "bytes": sum(source_files.values()),
            }
        )
    return plan


def print_promote_plan(plan: list[dict]) -> None:
    print("Paper 1 promote plan:", flush=True)
    for item in plan:
        print(
            "[PROMOTE] "
            f"{item['name']}: {item['source']} -> {item['destination']} "
            f"source_exists={item['source_exists']} "
            f"create={len(item['create'])} overwrite={len(item['overwrite'])} "
            f"stale={len(item['stale'])} bytes={item['bytes']}",
            flush=True,
        )
        if item["destination_is_file"]:
            print("  BLOCKED destination is an existing file", flush=True)
        if item["stale"]:
            for rel_path in item["stale"][:20]:
                print(f"  stale: {rel_path}", flush=True)
            if len(item["stale"]) > 20:
                print(f"  ... {len(item['stale']) - 20} more stale files", flush=True)


def stage_promote(args, cfg, repo_root: Path) -> dict:
    del cfg
    plan = build_promote_plan(args.staging_root, repo_root)
    print_promote_plan(plan)
    missing_sources = [item["name"] for item in plan if not item["source_exists"]]
    file_collisions = [item["name"] for item in plan if item["destination_is_file"]]
    stale_destinations = [item["name"] for item in plan if item["stale"]]

    if missing_sources:
        raise FileNotFoundError(
            "Missing promote source directories: " + ", ".join(sorted(missing_sources))
        )
    if file_collisions:
        raise RuntimeError(
            "Promote destination path is an existing file: "
            + ", ".join(sorted(file_collisions))
        )
    if stale_destinations and not args.allow_stale_promote:
        raise RuntimeError(
            "Promote destination has stale files; inspect dry-run output or rerun "
            "with --allow-stale-promote: "
            + ", ".join(sorted(stale_destinations))
        )
    summary = {
        item["name"]: {
            "create": len(item["create"]),
            "overwrite": len(item["overwrite"]),
            "stale": len(item["stale"]),
            "bytes": item["bytes"],
        }
        for item in plan
    }
    if args.promote_dry_run:
        return {"promote_dry_run": True, "plan": summary}

    copy_tree_contents(args.staging_root / "par", repo_root / "outputs/par")
    copy_tree_contents(args.staging_root / "power", repo_root / "data/power")
    copy_tree_contents(args.staging_root / "aef", repo_root / "outputs/AEF")
    return {"promoted": True, "plan": summary}


def stage_baseline(args, cfg, repo_root: Path) -> dict:
    del cfg
    baseline_output_dir = args.staging_root / "baseline"
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    env["STREETLIGHT_CONFIG"] = str(repo_root / "config/paper_baseline.yaml")
    env["PAPER_PAR_PATH"] = str(args.staging_root / "par/par_wide.csv")
    env["PAPER_AEF_DIR"] = str(args.staging_root / "aef")
    env["PAPER_OUTPUT_DIR"] = str(baseline_output_dir)
    env.setdefault("PAPER_ALIGN_STRATEGY", "intersection")
    subprocess.run(
        [sys.executable, str(repo_root / "scripts/analysis/run_paper_baseline.py")],
        cwd=repo_root,
        env=env,
        check=True,
    )
    return {"baseline": "completed", "output_dir": baseline_output_dir}


STAGES = {
    "snapshot": stage_snapshot,
    "flow": stage_flow,
    "par": stage_par,
    "power": stage_power,
    "merge-flow": stage_merge_flow,
    "aef": stage_aef,
    "promote": stage_promote,
    "baseline": stage_baseline,
}


def resolve_stage_list(stage: str, *, include_baseline: bool, promote: bool) -> list[str]:
    if stage != "all":
        return [stage]
    stages = ["snapshot", "flow", "par", "power", "merge-flow", "aef", "manifest"]
    if promote:
        stages.append("promote")
    if include_baseline:
        stages.append("baseline")
    return stages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper 1 final data pipeline")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--stage", default="all", choices=["all", *STAGES.keys(), "manifest"])
    parser.add_argument("--staging-root", type=Path, default=None)
    parser.add_argument("--demand-path", type=Path, required=True)
    parser.add_argument("--generation-path", type=Path, required=True)
    parser.add_argument("--par-weekly-root", type=Path, required=True)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--promote-dry-run", action="store_true")
    parser.add_argument("--allow-stale-promote", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()
    if args.staging_root is None:
        args.staging_root = Path("outputs/final_runs") / args.run_id
    return args


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    args.staging_root = (repo_root / args.staging_root).resolve() if not args.staging_root.is_absolute() else args.staging_root
    stages = resolve_stage_list(
        args.stage,
        include_baseline=args.include_baseline,
        promote=args.promote,
    )
    if not (args.promote_dry_run and stages == ["promote"]):
        args.staging_root.mkdir(parents=True, exist_ok=True)
    cfg = get_config()

    step_results: list[dict] = []
    for stage_name in stages:
        started = time.perf_counter()
        print(f"[START] {stage_name}", flush=True)
        if stage_name == "manifest":
            result = stage_manifest(args, cfg, repo_root, step_results)
        else:
            result = STAGES[stage_name](args, cfg, repo_root)
        step_results.append(
            {
                "stage": stage_name,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "result": result,
            }
        )
        print(f"[OK] {stage_name}: {result}", flush=True)

    if args.stage != "all" and args.stage != "manifest" and not (
        args.promote_dry_run and args.stage == "promote"
    ):
        stage_manifest(args, cfg, repo_root, step_results)


if __name__ == "__main__":
    main()
