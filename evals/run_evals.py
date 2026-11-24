"""Runnable evaluation harness supporting deterministic (offline) and live LLM modes.

Usage:
    python -m evals.run_evals --mode deterministic
    python -m evals.run_evals --mode live
"""

from __future__ import annotations

import json
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evals.scenarios import EvalScenario, build_eval_scenarios
from evals.scoring import EvalScore, score_scenario
from fleet_copilot.agent.service import FleetCopilotService
from fleet_copilot.config import PROJECT_ROOT, Settings, get_settings
from fleet_copilot.storage.database import FleetDatabase
from fleet_copilot.storage.ingestion import ingest_ndjson_file, ingest_snapshot
from tests.conftest import ScriptedChatModel

app = typer.Typer(add_completion=False)
console = Console()


def _run_single_scenario(
    scenario: EvalScenario,
    *,
    mode: Literal["deterministic", "live"],
    base_settings: Settings,
    tmp_dir: Path,
) -> EvalScore:
    scen_dir = tmp_dir / scenario.scenario_id
    scen_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        llm_provider=base_settings.llm_provider,
        llm_model=base_settings.llm_model,
        llm_temperature=0.0,
        google_api_key=base_settings.google_api_key,
        openai_api_key=base_settings.openai_api_key,
        anthropic_api_key=base_settings.anthropic_api_key,
        telemetry_dataset_path=base_settings.telemetry_dataset_path,
        telemetry_db_path=scen_dir / "fleet.db",
        checkpoint_db_path=scen_dir / "checkpoints.db",
        audit_log_jsonl_path=scen_dir / "audit.jsonl",
    )

    llm = (
        ScriptedChatModel(responses=list(scenario.deterministic_responses))
        if mode == "deterministic"
        else None
    )

    svc = FleetCopilotService(settings=settings, llm=llm, auto_ingest_default_dataset=True)
    try:
        if scenario.inject_realtime_snapshot:
            ingest_snapshot(svc.db, scenario.inject_realtime_snapshot)

        tid = f"eval-{scenario.scenario_id}"
        initial = svc.run_turn(
            thread_id=tid,
            company_id=scenario.tenant_id,
            user_message=scenario.prompt,
            actor_id="eval-runner",
        )

        final = initial
        if initial.status == "awaiting_approval" and initial.pending_proposals:
            decisions = [
                {
                    "proposal_id": p["proposal_id"],
                    "approved": scenario.hitl_approve,
                    "actor_id": "eval-reviewer",
                    "comment": "Automated eval decision",
                }
                for p in initial.pending_proposals
            ]
            final = svc.resume_approval(
                thread_id=tid,
                company_id=scenario.tenant_id,
                decisions=decisions,
            )

        return score_scenario(scenario, initial, final)
    finally:
        svc.close()


def _write_reports(
    scores: list[EvalScore],
    *,
    mode: str,
    model_label: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    total = len(scores)
    passed = sum(1 for s in scores if s.passed)
    pass_rate = round((passed / total) * 100.0, 1) if total else 0.0

    by_cat: dict[str, list[EvalScore]] = defaultdict(list)
    for s in scores:
        by_cat[s.category].append(s)

    report_dict = {
        "timestamp": ts,
        "mode": mode,
        "model": model_label,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate_pct": pass_rate,
        },
        "by_category": {
            cat: {
                "passed": sum(1 for s in items if s.passed),
                "total": len(items),
            }
            for cat, items in by_cat.items()
        },
        "scenarios": [
            {
                "scenario_id": s.scenario_id,
                "category": s.category,
                "passed": s.passed,
                "checks": s.checks,
                "tools_used": s.tools_used,
                "cited_count": s.cited_count,
                "notes": s.notes,
            }
            for s in scores
        ],
    }

    json_path = output_dir / "latest_report.json"
    json_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    md_lines = [
        f"# Fleet Copilot Evaluation Report ({mode.upper()})",
        "",
        f"- **Timestamp:** `{ts}`",
        f"- **Model:** `{model_label}`",
        f"- **Overall Score:** **{passed}/{total} ({pass_rate}%)**",
        "",
        "## Category Breakdown",
        "",
        "| Category | Passed | Total |",
        "| --- | ---: | ---: |",
    ]
    for cat, items in by_cat.items():
        md_lines.append(f"| `{cat}` | {sum(1 for s in items if s.passed)} | {len(items)} |")

    md_lines.extend(
        [
            "",
            "## Scenario Details",
            "",
            "| ID | Category | Result | Tools Used | Checks | Notes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for s in scores:
        status_str = "PASS" if s.passed else "FAIL"
        checks_str = ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in s.checks.items())
        notes_str = "; ".join(s.notes) if s.notes else "-"
        md_lines.append(
            f"| `{s.scenario_id}` | `{s.category}` | **{status_str}** | "
            f"`{', '.join(s.tools_used) or '-'}` | {checks_str} | {notes_str} |"
        )

    md_path = output_dir / "latest_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path


@app.command()
def main(
    mode: str = typer.Option(
        "deterministic",
        "--mode",
        "-m",
        help="Execution mode: 'deterministic' (offline scripted LLM) or 'live' (configured Gemini/OpenAI/Anthropic).",
    ),
    report_dir: Path = typer.Option(
        PROJECT_ROOT / "evals" / "reports",
        "--report-dir",
        help="Directory where JSON and Markdown eval reports are saved.",
    ),
) -> None:
    """Run the full 14-scenario Fleet Copilot evaluation suite."""
    if mode not in ("deterministic", "live"):
        raise typer.BadParameter("mode must be 'deterministic' or 'live'")

    base_settings = get_settings()

    with tempfile.TemporaryDirectory(prefix="rayda_eval_") as td:
        tmp_path = Path(td)
        # Load baseline DB once to compute dynamic ground-truth scenarios
        gt_db = FleetDatabase(tmp_path / "ground_truth.db")
        ingest_ndjson_file(gt_db, base_settings.telemetry_dataset_path)
        scenarios = build_eval_scenarios(gt_db)

        scores: list[EvalScore] = []
        total_scens = len(scenarios)
        for idx, scen in enumerate(scenarios, start=1):
            t0 = time.time()
            try:
                score = _run_single_scenario(
                    scen,
                    mode=mode,  # type: ignore[arg-type]
                    base_settings=base_settings,
                    tmp_dir=tmp_path,
                )
            except Exception as exc:
                score = EvalScore(
                    scenario_id=scen.scenario_id,
                    category=scen.category,
                    passed=False,
                    checks={"runtime_error": False},
                    notes=[f"{type(exc).__name__}: {exc}"],
                )
            elapsed = time.time() - t0
            status_tag = "[green]PASS[/green]" if score.passed else "[red]FAIL[/red]"
            console.print(
                f"[{idx}/{total_scens}] {scen.scenario_id} ({scen.category}) -> "
                f"{status_tag} ({elapsed:.1f}s) | tools={score.tools_used}"
            )
            if not score.passed and score.notes:
                console.print(f"    [dim]Notes: {'; '.join(score.notes)}[/dim]")
            scores.append(score)

    model_label = (
        "scripted-deterministic"
        if mode == "deterministic"
        else f"{base_settings.llm_provider}:{base_settings.llm_model}"
    )
    json_path, md_path = _write_reports(
        scores, mode=mode, model_label=model_label, output_dir=report_dir
    )

    table = Table(title=f"Rayda Fleet Copilot Evaluation Scorecard ({mode.upper()})")
    table.add_column("ID", style="bold")
    table.add_column("Category")
    table.add_column("Result")
    table.add_column("Tools")
    table.add_column("Notes")
    for s in scores:
        res_fmt = "[green]PASS[/green]" if s.passed else "[red]FAIL[/red]"
        table.add_row(
            s.scenario_id,
            s.category,
            res_fmt,
            ", ".join(s.tools_used) or "-",
            "; ".join(s.notes) or "All checks passed",
        )
    console.print(table)

    passed = sum(1 for s in scores if s.passed)
    total = len(scores)
    console.print(
        Panel(
            f"Overall: [bold]{passed}/{total} ({round(passed / total * 100, 1)}%)[/bold]\n"
            f"JSON Report: {json_path}\n"
            f"Markdown Report: {md_path}",
            border_style="green" if passed == total else "red",
        )
    )

    if passed < total:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
