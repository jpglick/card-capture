"""v4 regression harness CLI.

Provides commands for running the harness, managing baselines, and
validating truth files.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

import click

from harness.baseline import freeze_baseline, get_baseline, persist_run
from harness.config import load_pipeline_config
from harness.metrics.types import MetricResult
from harness.runner import run_metrics
from harness.validator import validate_file


@click.group()
def harness():
    """v4 regression harness commands."""
    pass


@harness.command()
@click.option("--baseline", required=True, help="Baseline name to compare against.")
@click.option(
    "--db",
    type=click.Path(exists=True, path_type=Path),
    default=Path("cards.sqlite"),
    help="Path to cards.sqlite database.",
)
@click.option(
    "--truth-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing ground-truth files.",
)
@click.option(
    "--videos",
    help="Comma-separated list of video IDs to run. Defaults to all in truth-dir.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    help="Optional path to save the full JSON report.",
)
def run(
    baseline: str,
    db: Path,
    truth_dir: Path,
    videos: Optional[str],
    out: Optional[Path],
):
    """Run regression metrics and compare against a baseline."""
    if videos:
        video_ids = [v.strip() for v in videos.split(",") if v.strip()]
    else:
        # Fallback: find all .truth.json files in truth_dir
        video_ids = [
            p.name.replace(".truth.json", "")
            for p in truth_dir.glob("*.truth.json")
        ]
        # Also check subdirectories
        video_ids.extend([
            p.parent.name
            for p in truth_dir.glob("**/truth.json")
            if p.parent != truth_dir
        ])
        video_ids = sorted(list(set(video_ids)))

    if not video_ids:
        click.echo("Error: No videos specified and none found in truth-dir.", err=True)
        return

    click.echo(f"Running metrics for {len(video_ids)} videos...")
    report = run_metrics(db_path=db, truth_dir=truth_dir, videos=video_ids)

    # Compare to baseline
    b = get_baseline(db_path=db, name=baseline)
    deltas = _compute_deltas(report.metrics, b.metrics)

    # Load current pipeline config
    config = load_pipeline_config()

    # Persist the run
    code_sha = _get_git_sha()
    run_id = persist_run(
        db_path=db,
        baseline_name=baseline,
        code_sha=code_sha,
        config=config,
        metrics=_to_dict(report.metrics),
        per_video=[_to_dict(pv) for pv in report.per_video],
    )

    result = {
        "run_id": run_id,
        "baseline": baseline,
        "metrics": report.metrics,
        "deltas": deltas,
        "per_video": [_to_dict(pv) for pv in report.per_video],
    }

    if out:
        out.write_text(json.dumps(result, indent=2, default=str))
        click.echo(f"Report saved to {out}")

    # Print summary
    click.echo("\nRegression Summary vs. " + baseline)
    click.echo("-" * 40)
    for k, v in report.metrics.items():
        if isinstance(v, MetricResult):
            if v.value is not None:
                delta = deltas.get(k)
                delta_val = delta.value if isinstance(delta, MetricResult) else None
                delta_str = f" ({delta_val:+.4f})" if delta_val is not None else ""
                click.echo(f"{k:20}: {v.value:.4f}{delta_str}")
            elif v.breakdown:
                click.echo(f"{k:20}: {v.breakdown}")


@harness.group()
def baseline():
    """Manage regression baselines."""
    pass


@baseline.command("freeze")
@click.option("--name", required=True, help="Unique name for the baseline.")
@click.option(
    "--db",
    type=click.Path(exists=True, path_type=Path),
    default=Path("cards.sqlite"),
)
@click.option(
    "--truth-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--videos", help="Comma-separated list of video IDs.")
def freeze(name: str, db: Path, truth_dir: Path, videos: Optional[str]):
    """Run metrics on the current pipeline and freeze as a baseline."""
    if videos:
        video_ids = [v.strip() for v in videos.split(",") if v.strip()]
    else:
        video_ids = [
            p.name.replace(".truth.json", "")
            for p in truth_dir.glob("*.truth.json")
        ]
        video_ids = sorted(list(set(video_ids)))

    click.echo(f"Freezing baseline '{name}' using {len(video_ids)} videos...")
    report = run_metrics(db_path=db, truth_dir=truth_dir, videos=video_ids)

    # Load current pipeline config
    config = load_pipeline_config()

    code_sha = _get_git_sha()
    freeze_baseline(
        db_path=db,
        name=name,
        code_sha=code_sha,
        config=config,
        metrics=_to_dict(report.metrics),
        per_video=[_to_dict(pv) for pv in report.per_video],
    )
    click.echo(f"Baseline '{name}' frozen @ {code_sha}")


@harness.command()
@click.option(
    "--db",
    type=click.Path(exists=True, path_type=Path),
    default=Path("cards.sqlite"),
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to save exported truth.json files.",
)
@click.option("--videos", help="Comma-separated video IDs to export.")
def export(db: Path, out_dir: Path, videos: Optional[str]):
    """Export truth.json files from the database."""
    import sqlite3
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    video_ids = [v.strip() for v in videos.split(",")] if videos else []
    
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        query = "SELECT video_id, payload_json FROM truth_files"
        params = []
        if video_ids:
            query += " WHERE video_id IN (" + ",".join(["?"] * len(video_ids)) + ")"
            params = video_ids
            
        rows = conn.execute(query, params).fetchall()
        
        for r in rows:
            vid = r["video_id"]
            payload = r["payload_json"]
            
            # Save as <vid>.truth.json
            p = out_dir / f"{vid}.truth.json"
            p.write_text(payload)
            click.echo(f"Exported {p}")


def _compute_deltas(current: dict, baseline: dict) -> dict:
    deltas = {}
    for k, v in current.items():
        bv = baseline.get(k)
        if not isinstance(v, MetricResult) or not isinstance(bv, MetricResult):
            continue
        if v.value is not None and bv.value is not None:
            deltas[k] = MetricResult(value=v.value - bv.value)
        elif v.breakdown and bv.breakdown:
            delta_breakdown = {}
            for sub_k in v.breakdown:
                a = v.breakdown.get(sub_k)
                b = bv.breakdown.get(sub_k)
                delta_breakdown[sub_k] = (a - b) if (a is not None and b is not None) else None
            deltas[k] = MetricResult(breakdown=delta_breakdown)
    return deltas


def _get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return {k: _to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


if __name__ == "__main__":
    harness()
