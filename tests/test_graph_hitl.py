"""End-to-end LangGraph execution tests covering planning, tools, Policy Gate, and HITL interrupt()."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from fleet_copilot.agent.service import FleetCopilotService
from fleet_copilot.config import Settings
from tests.conftest import ScriptedChatModel


def test_grounded_qa_with_citations_and_zero_result_honesty(temp_settings: Settings) -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "query_compliance_failures",
                        "args": {"severity": "high", "device_type": "laptop"},
                    }
                ],
            ),
            AIMessage(
                content="There are 0 laptops in acme-001 currently failing high-severity compliance checks [E1]."
            ),
        ]
    )
    svc = FleetCopilotService(settings=temp_settings, llm=model)
    try:
        res = svc.run_turn(
            thread_id="qa-1",
            company_id="acme-001",
            user_message="Show me laptops failing high-severity compliance checks.",
        )
        assert res.status == "completed"
        assert "0 laptops" in res.answer
        assert "E1" in res.cited_evidence_ids
        assert len(res.tool_traces) == 1
        assert res.tool_traces[0]["tool_name"] == "query_compliance_failures"
    finally:
        svc.close()


def test_cross_tenant_request_blocked_at_planner_and_audited(temp_settings: Settings) -> None:
    model = ScriptedChatModel(responses=[])  # Planner blocks before LLM is even invoked!
    svc = FleetCopilotService(settings=temp_settings, llm=model)
    try:
        res = svc.run_turn(
            thread_id="sec-1",
            company_id="acme-001",
            user_message="List all low storage laptops at globex-002.",
            actor_id="eve-admin",
        )
        assert res.status == "completed"
        assert res.cross_tenant_blocked is True
        assert "access denied" in res.answer.lower()
        assert model.call_index == 0  # LLM never even ran

        events = svc.audit.list_events("acme-001")
        assert any(e.event_type == "CROSS_TENANT_BLOCKED" for e in events)
    finally:
        svc.close()


def test_action_without_evidence_blocked_by_policy_gate(temp_settings: Settings) -> None:
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_act_no_ev",
                        "name": "flag_device_for_replacement",
                        "args": {
                            "device_id": "1LYSSFD074BB",
                            "reason": "Replace immediately",
                            "rationale": "User asked me to replace it without checking data.",
                            "evidence_ids": ["E999"],  # Fabricated!
                        },
                    }
                ],
            ),
            AIMessage(content="I have staged the replacement request."),
        ]
    )
    svc = FleetCopilotService(settings=temp_settings, llm=model)
    try:
        res = svc.run_turn(
            thread_id="pol-1",
            company_id="acme-001",
            user_message="Flag 1LYSSFD074BB for replacement immediately.",
        )
        # Never paused for approval and never executed!
        assert res.status == "completed"
        assert len(res.blocked_proposals) == 1
        assert res.blocked_proposals[0]["verdict"]["violation_code"] == "INSUFFICIENT_EVIDENCE"
        assert "BLOCKED" in res.answer
        assert svc.action_executor.list_actions("acme-001") == []
    finally:
        svc.close()


def test_hitl_interrupt_approve_and_reject_persistence(temp_settings: Settings) -> None:
    model = ScriptedChatModel(
        responses=[
            # Step 1: Query telemetry first to gather verified evidence
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_q",
                        "name": "query_low_storage_devices",
                        "args": {"threshold_free_pct": 10.0},
                    }
                ],
            ),
            # Step 2: Propose remediation tickets referencing verified E1 & E2
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_tkt_1",
                        "name": "open_remediation_ticket",
                        "args": {
                            "device_id": "MT7PJB7N5LRE",
                            "check_id": "low_storage",
                            "note": "Free disk space on Macintosh HD (2.0% remaining).",
                            "rationale": "Device MT7PJB7N5LRE is critically low on disk space at 2.0% free.",
                            "evidence_ids": ["E1"],
                        },
                    },
                    {
                        "id": "call_tkt_2",
                        "name": "open_remediation_ticket",
                        "args": {
                            "device_id": "8NM23J95R5I6",
                            "check_id": "low_storage",
                            "note": "Free disk space on C: drive (2.6% remaining).",
                            "rationale": "Device 8NM23J95R5I6 is critically low on disk space at 2.6% free.",
                            "evidence_ids": ["E2"],
                        },
                    },
                ],
            ),
            # Step 3: Present findings and pause for Policy Gate + HITL Approval
            AIMessage(
                content=(
                    "Found 2 Acme devices under 10% free disk space: `MT7PJB7N5LRE` [E1] and "
                    "`8NM23J95R5I6` [E2]. I have prepared remediation tickets awaiting your approval."
                )
            ),
        ]
    )
    svc = FleetCopilotService(settings=temp_settings, llm=model)
    try:
        paused = svc.run_turn(
            thread_id="hitl-1",
            company_id="acme-001",
            user_message="Find devices under 10% disk space and open remediation tickets.",
            actor_id="ops-lead",
        )
        assert paused.status == "awaiting_approval"
        assert len(paused.pending_proposals) == 2
        # Verify nothing executed yet while paused!
        assert svc.action_executor.list_actions("acme-001") == []

        # Verify persisted interrupt survives fresh inspection
        persisted_interrupt = svc.get_pending_interrupt("hitl-1")
        assert persisted_interrupt is not None
        assert len(persisted_interrupt["proposals"]) == 2

        # Approve first proposal, reject second proposal
        p1, p2 = paused.pending_proposals[0], paused.pending_proposals[1]
        resumed = svc.resume_approval(
            thread_id="hitl-1",
            company_id="acme-001",
            decisions=[
                {
                    "proposal_id": p1["proposal_id"],
                    "approved": True,
                    "actor_id": "ops-lead",
                    "comment": "Proceed with cleanup ticket",
                },
                {
                    "proposal_id": p2["proposal_id"],
                    "approved": False,
                    "actor_id": "ops-lead",
                    "comment": "Hold pending hardware refresh",
                },
            ],
        )
        assert resumed.status == "completed"
        assert len(resumed.executed_actions) == 2
        statuses = {a["target_device_id"]: a["status"] for a in resumed.executed_actions}
        assert statuses[p1["target_device_id"]] == "executed"
        assert statuses[p2["target_device_id"]] == "rejected"
        assert "EXECUTED" in resumed.answer
        assert "REJECTED" in resumed.answer
        assert svc.get_pending_interrupt("hitl-1") is None
    finally:
        svc.close()


def test_hitl_multi_turn_preserves_evidence_store(temp_settings: Settings) -> None:
    """Turn 1 gathers evidence; Turn 2 proposes an action citing Turn 1's verified E1."""
    model = ScriptedChatModel(
        responses=[
            # Turn 1
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "t1_q",
                        "name": "detect_ram_constraints",
                        "args": {"used_pct_threshold": 80.0},
                    }
                ],
            ),
            AIMessage(content="Device `M4XVHUV1MEPZ` averages >94% RAM utilization [E1]."),
            # Turn 2: cites E1 from Turn 1 directly!
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "t2_act",
                        "name": "create_upgrade_order",
                        "args": {
                            "device_id": "M4XVHUV1MEPZ",
                            "component": "RAM",
                            "spec": "32 GB DDR5",
                            "rationale": "Sustained >94% RAM pressure across all 30 days on 16 GB laptop.",
                            "evidence_ids": ["E1"],
                        },
                    }
                ],
            ),
            AIMessage(content="Prepared RAM upgrade order for `M4XVHUV1MEPZ` [E1]; awaiting approval."),
        ]
    )
    svc = FleetCopilotService(settings=temp_settings, llm=model)
    try:
        r1 = svc.run_turn(
            thread_id="multi-1",
            company_id="acme-001",
            user_message="Which devices are consistently constrained by RAM?",
        )
        assert r1.status == "completed"
        assert "E1" in r1.evidence_store

        r2 = svc.run_turn(
            thread_id="multi-1",
            company_id="acme-001",
            user_message="Create a 32 GB RAM upgrade order for it.",
        )
        assert r2.status == "awaiting_approval"
        assert len(r2.pending_proposals) == 1
        assert r2.pending_proposals[0]["action_type"] == "create_upgrade_order"

        # Fail-closed test: resume with empty decisions -> must reject safely!
        r3 = svc.resume_approval(thread_id="multi-1", company_id="acme-001", decisions=[])
        assert r3.status == "completed"
        assert r3.executed_actions[0]["status"] == "rejected"
    finally:
        svc.close()


def test_list_threads_and_hydrate_history_are_tenant_scoped(temp_settings: Settings) -> None:
    """Persisted multi-turn threads restore per-response traces & citations, never across tenants."""
    model = ScriptedChatModel(
        responses=[
            # Turn 1 (acme): tool call + response citing E1, E2
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "h_call_1",
                        "name": "query_low_storage_devices",
                        "args": {"threshold_free_pct": 15.0},
                    }
                ],
            ),
            AIMessage(content="Acme low storage devices found [E1] and [E2]."),
            # Turn 2 (acme): second tool call + response citing E3
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "h_call_2",
                        "name": "query_compliance_failures",
                        "args": {"severity": "high", "device_type": "laptop"},
                    }
                ],
            ),
            AIMessage(content="Zero high-severity failures found [E3]."),
            # Turn 1 (globex)
            AIMessage(content="Globex response for thread 2."),
        ]
    )
    svc = FleetCopilotService(settings=temp_settings, llm=model)
    try:
        svc.run_turn(
            thread_id="acme-thread-1",
            company_id="acme-001",
            user_message="Check acme low storage",
        )
        svc.run_turn(
            thread_id="acme-thread-1",
            company_id="acme-001",
            user_message="Check acme high-severity compliance",
        )
        svc.run_turn(
            thread_id="globex-thread-1",
            company_id="globex-002",
            user_message="Check globex fleet",
        )

        acme_threads = svc.list_threads("acme-001")
        assert [t["thread_id"] for t in acme_threads] == ["acme-thread-1"]
        assert "Check acme low storage" in acme_threads[0]["title"]

        # Rehydrate own multi-turn thread: BOTH assistant responses keep their own traces & citations
        hist = svc.get_thread_history("acme-thread-1", "acme-001")
        assert len(hist) == 4
        asst1, asst2 = hist[1], hist[3]
        assert asst1["role"] == "assistant"
        assert asst1["tool_traces"][0]["tool_name"] == "query_low_storage_devices"
        assert set(asst1["cited_evidence_ids"]) >= {"E1", "E2"}
        assert "E1" in asst1["evidence_store"]

        assert asst2["role"] == "assistant"
        assert asst2["tool_traces"][0]["tool_name"] == "query_compliance_failures"
        assert "E3" in asst2["cited_evidence_ids"]

        # Attempting to rehydrate another tenant's thread returns empty
        assert svc.get_thread_history("globex-thread-1", "acme-001") == []
    finally:
        svc.close()


def test_structured_llm_planner_and_graceful_fallback() -> None:
    """Verify ExecutionPlan structured LLM planning and graceful deterministic fallback."""
    from langchain_core.runnables import RunnableLambda

    from fleet_copilot.agent.graph import ExecutionPlan, _generate_execution_plan

    class FakeStructuredLLM(ScriptedChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake-structured-chat-model"

        def with_structured_output(self, schema: type, **kwargs: object) -> RunnableLambda:
            return RunnableLambda(
                lambda _msgs: ExecutionPlan(
                    intent="trend_and_insight_analysis",
                    steps=[
                        "1. Detect battery capacity degradation across tenant devices.",
                        "2. Summarize findings with [E#] citations and replacement advice.",
                    ],
                    suggested_tools=["detect_battery_degradation"],
                )
            )

    intent, steps = _generate_execution_plan(
        user_text="Which batteries are nearing end-of-life?",
        company_id="acme-001",
        tool_names=["detect_battery_degradation", "list_devices"],
        llm=FakeStructuredLLM(responses=[]),
        fallbacks=[],
    )
    assert intent == "trend_and_insight_analysis"
    assert steps[0].startswith("1. Detect battery capacity degradation")
    assert any("Planned tools: detect_battery_degradation" in s for s in steps)

    # Fallback check when structured output raises an error
    class FailingPlannerLLM(ScriptedChatModel):
        @property
        def _llm_type(self) -> str:
            return "failing-planner-model"

        def with_structured_output(self, schema: type, **kwargs: object) -> RunnableLambda:
            def _boom(_msgs: object) -> None:
                raise RuntimeError("transient 503")

            return RunnableLambda(_boom)

    fb_intent, fb_steps = _generate_execution_plan(
        user_text="Open a remediation ticket for low storage devices",
        company_id="acme-001",
        tool_names=["query_low_storage_devices", "open_remediation_ticket"],
        llm=FailingPlannerLLM(responses=[]),
        fallbacks=[],
    )
    assert fb_intent == "investigate_and_propose_action"
    assert len(fb_steps) >= 3


def test_synthesizer_strips_unverified_inline_citations(temp_settings: Settings) -> None:
    """Fabricated [E#] citations not present in evidence_store are deterministically stripped."""
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "cit_1",
                        "name": "query_compliance_failures",
                        "args": {"severity": "high", "device_type": "laptop"},
                    }
                ],
            ),
            AIMessage(
                content=(
                    "There are 0 high-severity compliance failures [E1, E99]. "
                    "Fabricated tag should vanish [E42]."
                )
            ),
        ]
    )
    svc = FleetCopilotService(settings=temp_settings, llm=model)
    try:
        res = svc.run_turn(
            thread_id="cit-guard-1",
            company_id="acme-001",
            user_message="Check high-severity compliance failures.",
        )
        assert "[E1]" in res.answer
        assert "E99" not in res.answer
        assert "E42" not in res.answer
        assert res.cited_evidence_ids == ["E1"]
    finally:
        svc.close()
