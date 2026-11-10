"""Deterministic tool-calling model used to exercise the pipeline offline."""
from __future__ import annotations

import json
from typing import Any, Sequence
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _json_messages(messages: Sequence[BaseMessage]) -> list[tuple[str, dict[str, Any]]]:
    parsed = []
    for message in messages:
        if isinstance(message, ToolMessage):
            try:
                data = json.loads(message.content)
                if isinstance(data, dict):
                    parsed.append((message.name or "", data))
            except (TypeError, json.JSONDecodeError):
                continue
    return parsed


class ScriptedChatModel(BaseChatModel):
    """A small deterministic policy that calls real tools in pipeline order."""

    bound_tool_names: list[str] = []
    memory: dict[str, Any] = {}

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.memory = {}

    @property
    def _llm_type(self) -> str:
        return "astro-helix-scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        return self.model_copy(update={"bound_tool_names": [getattr(tool, "name", "") for tool in tools], "memory": self.memory})

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        names = set(self.bound_tool_names)
        toolset_key = "|".join(sorted(names))
        calls = self.memory.setdefault("_calls", {}).setdefault(toolset_key, {})
        data = _json_messages(messages)
        data.extend(self.memory.get("_tool_history", []))
        for name, value in data:
            self.memory[name] = value

        def call(name: str, args: dict[str, Any] | None = None):
            calls[name] = calls.get(name, 0) + 1
            return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": str(uuid4()), "type": "tool_call"}])

        def finish(text: str):
            return AIMessage(content=text)

        latest = {**self.memory, **{name: value for name, value in data}}
        scan = latest.get("scan_demo_conjunctions", {})
        events = scan.get("events", []) if isinstance(scan, dict) else []
        cdms = latest.get("get_pending_cdms", [])
        if isinstance(cdms, dict):
            cdms = cdms.get("cdms", cdms.get("messages", []))
        if not events and isinstance(cdms, list):
            events = cdms
        risk_results = [value for name, value in data if name == "assess_risk" and isinstance(value, dict) and not value.get("error")]
        risk_results.sort(key=lambda value: (
            -float(value.get("collision_probability_chan", 0.0) or 0.0),
            float(value.get("miss_distance_m", float("inf"))),
        ))
        ranked_events = sorted(events, key=lambda event: (
            -float(event.get("collision_probability_chan", event.get("chan_probability", 0.0)) or 0.0),
            float(event.get("miss_distance_m", float("inf"))),
        ))
        assessed_ids = {int(value["secondary_id"]) for value in risk_results if value.get("secondary_id") is not None}
        top = risk_results[0] if risk_results else (ranked_events[0] if ranked_events else {})
        top_id = top.get("secondary_id")

        for message in messages:
            if isinstance(message, HumanMessage) and "[Safety review]" in str(message.content):
                self.memory["safety_final"] = str(message.content).split("[Safety review]", 1)[-1].strip()
            if isinstance(message, HumanMessage) and "[Planner recommendations]" in str(message.content):
                self.memory["planner_final"] = str(message.content).split("[Planner recommendations]", 1)[-1].strip()

        # Scout gathers scan and CDM inputs once.
        if "scan_demo_conjunctions" in names and "assess_risk" not in names and calls.get("scan_demo_conjunctions", 0) == 0:
            response = call("scan_demo_conjunctions")
        elif "get_pending_cdms" in names and "assess_risk" not in names and calls.get("get_pending_cdms", 0) == 0:
            response = call("get_pending_cdms")
        # Analyst assesses at most five events in Chan-probability order; no rescans.
        elif "assess_risk" in names and "propose_avoidance_maneuvers" not in names and "record_safety_verdict" not in names and "execute_recommended_maneuver" not in names:
            next_event = next((event for event in ranked_events[:5] if int(event.get("secondary_id", -1)) not in assessed_ids), None)
            if next_event is not None:
                response = call("assess_risk", {"secondary_id": int(next_event["secondary_id"]), "miss_distance_m": float(next_event.get("miss_distance_m", 0.0)), "tca_offset_sec": float(next_event.get("tca_offset_sec", next_event.get("tca_sec", 0.0)))})
            else:
                response = finish("Risk assessment completed for the top available events.")
        # Planner searches only the leading threat, then stores an index rather than a vector.
        elif "propose_avoidance_maneuvers" in names:
            if calls.get("propose_avoidance_maneuvers", 0) == 0:
                if top_id is None:
                    response = finish("No ranked threat is available; no maneuver search was performed.")
                else:
                    response = call("propose_avoidance_maneuvers", {"secondary_id": int(top_id)})
            else:
                search = latest.get("propose_avoidance_maneuvers", {})
                candidates = search.get("candidates", []) if isinstance(search, dict) else []
                sid = int(search.get("secondary_id", top_id or 0)) if isinstance(search, dict) else int(top_id or 0)
                if search.get("no_improving_candidate") or not candidates:
                    response = finish(f"Top threat {sid}: no improving candidate within constraints. No maneuver recommended.")
                elif "record_recommended_maneuver" in names and calls.get("record_recommended_maneuver", 0) == 0:
                    response = call("record_recommended_maneuver", {"secondary_id": sid, "candidate_index": int(candidates[0].get("candidate_index", 0))})
                else:
                    response = finish(f"Top threat {sid} has a stored improving candidate. See the recommended_maneuver record.")
        # Safety checks and verdict operate entirely on the stored record.
        elif "record_safety_verdict" in names:
            if calls.get("get_satellite_status", 0) == 0:
                response = call("get_satellite_status")
            elif calls.get("check_maneuver_feasibility", 0) == 0:
                response = call("check_maneuver_feasibility")
            elif calls.get("check_maneuver_constraints", 0) == 0:
                response = call("check_maneuver_constraints")
            elif calls.get("record_safety_verdict", 0) == 0:
                feasibility = latest.get("check_maneuver_feasibility", {})
                constraints = latest.get("check_maneuver_constraints", {})
                approved = feasibility.get("feasible") is True and constraints.get("overall_pass") is True
                verdict = "APPROVED" if approved else "REJECTED"
                response = call("record_safety_verdict", {"verdict": verdict, "reason": "Both checks passed for the stored recommended maneuver." if approved else "One or more stored-vector safety checks failed."})
            else:
                record_result = latest.get("record_safety_verdict", {})
                record = record_result.get("recommended_maneuver", {})
                verdict = record.get("verdict", "REJECTED")
                response = finish(f"VERDICT: {verdict}. Safety checks were run against the stored recommended maneuver.")
        # Ops calls the argument-free executor only after stored approval, then assesses that target once.
        elif "execute_recommended_maneuver" in names:
            verdict_result = latest.get("record_safety_verdict", {})
            record = verdict_result.get("recommended_maneuver", {})
            verdict = record.get("verdict")
            if verdict == "APPROVED" and calls.get("execute_recommended_maneuver", 0) == 0:
                response = call("execute_recommended_maneuver")
            elif verdict == "APPROVED" and calls.get("assess_risk", 0) == 0:
                response = call("assess_risk", {"secondary_id": int(record["secondary_id"])})
            elif verdict == "APPROVED" and calls.get("get_satellite_status", 0) == 0:
                response = call("get_satellite_status")
            elif verdict == "APPROVED":
                response = finish(f"Operator brief: stored maneuver for secondary {record.get('secondary_id')} executed and risk reassessed.")
            else:
                response = finish("Operator brief: Safety did not approve a stored maneuver; no execution occurred.")
        elif "scan_conjunctions" in names and "scan_demo_conjunctions" not in names and calls.get("scan_conjunctions", 0) == 0:
            response = call("scan_conjunctions")
        else:
            response = finish("Analysis complete using the deterministic demo policy.")

        return ChatResult(generations=[ChatGeneration(message=response)])
