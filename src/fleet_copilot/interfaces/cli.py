"""Interactive Typer + Rich CLI for Rayda Fleet Copilot."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from fleet_copilot.agent.service import FleetCopilotService, TurnResult

app = typer.Typer(
    name="fleet-copilot",
    help="Rayda Agentic AI Fleet Copilot CLI.",
    no_args_is_help=True,
)
console = Console()


def _render_turn(result: TurnResult) -> None:
    if result.plan:
        console.print(
            Panel(
                "\n".join(result.plan),
                title=f"Execution Plan ({result.intent})",
                border_style="cyan",
            )
        )

    if result.tool_traces:
        trace_table = Table(title="Tool Execution Trace", show_lines=False)
        trace_table.add_column("Step", style="dim", width=5)
        trace_table.add_column("Tool", style="bold")
        trace_table.add_column("Status")
        trace_table.add_column("Summary")
        for tr in result.tool_traces:
            trace_table.add_row(
                str(tr.get("step", "")),
                str(tr.get("tool_name", "")),
                str(tr.get("status", "")),
                str(tr.get("summary", "")),
            )
        console.print(trace_table)

    if result.answer:
        console.print(
            Panel(
                Markdown(result.answer),
                title=f"Fleet Copilot [{result.company_id}]",
                border_style="green" if not result.cross_tenant_blocked else "red",
            )
        )

    if result.cited_evidence_ids:
        ev_table = Table(title="Verified Telemetry Citations")
        ev_table.add_column("ID", style="bold cyan", width=6)
        ev_table.add_column("Device / Entity")
        ev_table.add_column("Metric")
        ev_table.add_column("Evidence Summary")
        for eid in result.cited_evidence_ids:
            cit = result.evidence_store.get(eid)
            if cit:
                ev_table.add_row(
                    eid,
                    str(cit.get("device_id") or cit.get("employee_id") or "-"),
                    str(cit.get("metric", "")),
                    str(cit.get("summary", "")),
                )
        console.print(ev_table)


@app.command()
def companies() -> None:
    """List companies dynamically discovered in the telemetry database."""
    svc = FleetCopilotService()
    try:
        table = Table(title="Discovered Tenants")
        table.add_column("Company ID", style="bold")
        table.add_column("Devices", justify="right")
        table.add_column("Snapshots", justify="right")
        table.add_column("Latest Telemetry")
        for cid in svc.list_companies():
            ov = svc.get_repository(cid).get_fleet_overview()
            table.add_row(
                cid,
                str(ov["device_count"]),
                str(ov["snapshot_count"]),
                str(ov["latest_snapshot"]),
            )
        console.print(table)
    finally:
        svc.close()


@app.command()
def ingest(
    file_path: Path = typer.Argument(..., exists=True, help="Path to NDJSON or JSON telemetry file."),
    actor: str = typer.Option("cli-admin", help="Actor ID recorded in audit log."),
) -> None:
    """Ingest telemetry snapshots (idempotent upsert) into the database."""
    svc = FleetCopilotService(auto_ingest_default_dataset=False)
    try:
        report = svc.ingest_file(file_path, actor_id=actor)
        console.print(
            Panel(
                f"Processed: {report.processed}\n"
                f"Inserted:  {report.inserted}\n"
                f"Updated:   {report.updated}\n"
                f"Companies: {', '.join(report.companies_seen)}\n"
                f"Errors:    {len(report.errors)}",
                title="Ingestion Report",
                border_style="green" if not report.errors else "yellow",
            )
        )
        for err in report.errors[:10]:
            console.print(f"[red]- {err}[/red]")
    finally:
        svc.close()


@app.command()
def chat(
    tenant: str = typer.Option(..., "--tenant", "-t", help="Active company_id scoping this session."),
    actor: str = typer.Option("it-admin", "--actor", "-a", help="Authenticated IT admin identifier."),
    thread_id: str | None = typer.Option(None, help="Optional LangGraph thread_id to resume."),
) -> None:
    """Start an interactive tenant-scoped Fleet Copilot chat session with HITL approvals."""
    svc = FleetCopilotService()
    tid = thread_id or f"{tenant}-{uuid.uuid4().hex[:8]}"
    console.print(
        Panel(
            f"Tenant: [bold cyan]{tenant}[/bold cyan]\n"
            f"Actor:  [bold]{actor}[/bold]\n"
            f"Thread: [dim]{tid}[/dim]\n"
            f"Model:  {svc.settings.llm_provider}:{svc.settings.llm_model}\n"
            "Type 'exit' or 'quit' to end the session.",
            title="Rayda Agentic AI Fleet Copilot",
            border_style="blue",
        )
    )

    try:
        while True:
            user_input = Prompt.ask(f"\n[bold cyan]{actor}@{tenant}[/bold cyan]")
            if user_input.strip().lower() in {"exit", "quit", "q"}:
                break
            if not user_input.strip():
                continue

            result = svc.run_turn(
                thread_id=tid,
                company_id=tenant,
                user_message=user_input,
                actor_id=actor,
            )
            _render_turn(result)

            while result.status == "awaiting_approval" and result.pending_proposals:
                console.print(
                    Panel(
                        "One or more state-changing actions passed the Policy & Evidence Gate and "
                        "require your explicit approval before execution.",
                        title="Human Approval Required (LangGraph interrupt)",
                        border_style="yellow",
                    )
                )
                decisions = []
                for prop in result.pending_proposals:
                    target = prop.get("target_device_id") or prop.get("target_employee_id")
                    cits_str = "\n".join(
                        f"  - [{c['evidence_id']}] {c['summary']}"
                        for c in prop.get("citations", [])
                    )
                    console.print(
                        Panel(
                            f"Proposal ID: [bold]{prop['proposal_id']}[/bold]\n"
                            f"Action:      [bold yellow]{prop['action_type']}[/bold yellow] (risk: {prop['risk_level']})\n"
                            f"Target:      {target}\n"
                            f"Parameters:  {json.dumps(prop['parameters'])}\n"
                            f"Rationale:   {prop['rationale']}\n"
                            f"Evidence:\n{cits_str}",
                            border_style="yellow",
                        )
                    )
                    approved = Confirm.ask(f"Approve {prop['proposal_id']}?", default=False)
                    comment = Prompt.ask("Optional reviewer comment", default="")
                    decisions.append(
                        {
                            "proposal_id": prop["proposal_id"],
                            "approved": approved,
                            "actor_id": actor,
                            "comment": comment,
                        }
                    )

                result = svc.resume_approval(
                    thread_id=tid,
                    company_id=tenant,
                    decisions=decisions,
                )
                _render_turn(result)
    finally:
        svc.close()


@app.command()
def actions(
    tenant: str = typer.Option(..., "--tenant", "-t", help="Tenant company_id."),
    limit: int = typer.Option(25, help="Max rows to display."),
) -> None:
    """View persisted operational actions for a tenant."""
    svc = FleetCopilotService()
    try:
        rows = svc.action_executor.list_actions(tenant, limit=limit)
        table = Table(title=f"Persisted Actions [{tenant}]")
        table.add_column("Action ID", style="bold")
        table.add_column("Type")
        table.add_column("Target")
        table.add_column("Status")
        table.add_column("Approved By")
        table.add_column("Created At")
        for r in rows:
            table.add_row(
                r.action_id,
                r.action_type.value,
                str(r.target_device_id or r.target_employee_id or "-"),
                r.status.value,
                str(r.approved_by or "-"),
                r.created_at,
            )
        console.print(table)
    finally:
        svc.close()


@app.command()
def audit(
    tenant: str = typer.Option(..., "--tenant", "-t", help="Tenant company_id."),
    limit: int = typer.Option(30, help="Max events to display."),
) -> None:
    """View immutable audit log events for a tenant."""
    svc = FleetCopilotService()
    try:
        events = svc.audit.list_events(tenant, limit=limit)
        table = Table(title=f"Audit Trail [{tenant}]")
        table.add_column("Timestamp", style="dim")
        table.add_column("Event Type", style="bold")
        table.add_column("Status")
        table.add_column("Actor")
        table.add_column("Target")
        for ev in events:
            table.add_row(
                ev.timestamp,
                ev.event_type,
                ev.status,
                ev.actor_id,
                str(ev.target_entity or "-"),
            )
        console.print(table)
    finally:
        svc.close()


if __name__ == "__main__":
    app()
