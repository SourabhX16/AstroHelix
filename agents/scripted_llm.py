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
        called = {name for name, _ in data} | {name for name, count in calls.items() if count}
        latest = {name: value for name, value in data}

        def call(name: str, args: dict[str, Any] | None = None):
            calls[name] = calls.get(name, 0) + 1
            return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": str(uuid4()), "type": "tool_call"}])

        def finish(text: str):
            return AIMessage(content=text)

        latest = {**self.memory, **latest}
        threat_data = latest.get("scan_demo_conjunctions", {})
        events = threat_data.get("events", [])
        cdms = latest.get("get_pending_cdms", [])
        if isinstance(cdms, dict):
            cdms = cdms.get("cdms", cdms.get("messages", []))
        if not events and isinstance(cdms, list):
            events = cdms
        for message in messages:
            if isinstance(message, HumanMessage) and "[Safety review]" in str(message.content):
                self.memory["safety_final"] = str(message.content).split("[Safety review]", 1)[-1].strip()
            if isinstance(message, AIMessage) and isinstance(message.content, str) and (message.content.startswith("APPROVED") or message.content.startswith("REJECTED")):
                self.memory["safety_final"] = message.content

        if "scan_demo_conjunctions" in names and "scan_demo_conjunctions" not in called:
            response = call("scan_demo_conjunctions")
        elif "get_pending_cdms" in names and "get_pending_cdms" not in called:
            response = call("get_pending_cdms")
        elif names == {"get_pending_cdms", "scan_conjunctions", "scan_demo_conjunctions"} and "get_satellite_status" in names and "get_satellite_status" not in called:
            response = call("get_satellite_status")
        elif "assess_risk" in names and "execute_maneuver_on_satellite" not in names and "propose_avoidance_maneuvers" not in names and "check_maneuver_constraints" not in names and not any(n == "assess_risk" for n in called):
            ids = [int(x.get("secondary_id")) for x in events[:3] if x.get("secondary_id") is not None]
            if not ids and isinstance(cdms, list):
                ids = [int(x.get("secondary_id")) for x in cdms[:3] if x.get("secondary_id") is not None]
            response = call("assess_risk", {"secondary_id": ids[0]}) if ids else finish("Missing secondary_id in conjunction data; unable to assess risk.")
        elif "assess_risk" in names and "execute_maneuver_on_satellite" not in names and "propose_avoidance_maneuvers" not in names and "check_maneuver_constraints" not in names and calls.get("assess_risk", 0) < 3:
            risk_ids = [x.get("secondary_id") for n, x in data if n == "assess_risk"]
            ids = [int(x.get("secondary_id")) for x in events[:3] if x.get("secondary_id") is not None]
            todo = next((sid for sid in ids if sid not in risk_ids), None)
            if todo:
                response = call("assess_risk", {"secondary_id": todo})
            elif "refine_conjunction_hifi" in names and "refine_conjunction_hifi" not in called:
                sid = risk_ids[0] if risk_ids else (ids[0] if ids else None)
                response = call("refine_conjunction_hifi", {"secondary_id": int(sid)}) if sid is not None else finish("Missing secondary_id; risk assessments completed without refinement.")
            else:
                response = finish("Risk assessments completed for the available top threats.")
        elif "refine_conjunction_hifi" in names and "refine_conjunction_hifi" not in called:
            risk = next((x for n, x in data if n == "assess_risk"), {})
            sid = risk.get("secondary_id") or (events[0].get("secondary_id") if events else None)
            response = call("refine_conjunction_hifi", {"secondary_id": int(sid)}) if sid is not None else finish("Missing secondary_id; high-fidelity refinement was not possible.")
        elif "propose_avoidance_maneuvers" in names and "propose_avoidance_maneuvers" not in called:
            risk = next((x for n, x in data if n == "assess_risk"), {})
            event = events[0] if events else {}
            sid = risk.get("secondary_id", event.get("secondary_id"))
            miss = risk.get("miss_distance_m", event.get("miss_distance_m"))
            tca = event.get("tca_offset_sec", event.get("tca_sec"))
            if sid is None or miss is None or tca is None:
                response = finish("Missing secondary_id, miss_distance_m, or tca_offset_sec; unable to propose a maneuver.")
            else:
                response = call("propose_avoidance_maneuvers", {"secondary_id": int(sid), "tca_offset_sec": float(tca), "miss_distance_m": float(miss)})
        elif "check_maneuver_constraints" in names:
            candidates = latest.get("propose_avoidance_maneuvers", {}).get("candidates", [])
            candidates = [c for c in candidates if c.get("delta_v_magnitude_ms", 999) <= 50.0]
            candidate = max(candidates, key=lambda c: c.get("miss_improvement_m", 0), default={})
            dv = candidate.get("delta_v_vector")
            if dv is None:
                response = finish("Missing candidate delta_v; no maneuver can be reviewed.")
            elif "check_maneuver_constraints" not in called:
                response = call("check_maneuver_constraints", {"delta_v": dv, "burn_time_sec": float(candidate.get("burn_time_sec", 0.0))})
            elif "check_maneuver_feasibility" in names and "check_maneuver_feasibility" not in called:
                response = call("check_maneuver_feasibility", {"delta_v": dv})
            else:
                checks = [x for n, x in data if n in {"check_maneuver_constraints", "check_maneuver_feasibility"}]
                approved = all(x.get("feasible", True) and x.get("overall_pass", True) for x in checks)
                verdict = ("APPROVED" if approved else "REJECTED") + f": candidate delta_v {dv}; checks: {checks}"
                if "execute_maneuver_on_satellite" in names and approved and "execute_maneuver_on_satellite" not in called:
                    self.memory["safety_final"] = verdict
                    response = call("execute_maneuver_on_satellite", {"delta_v": dv})
                elif "execute_maneuver_on_satellite" in names and "execute_maneuver_on_satellite" in called:
                    response = finish("Operator brief: approved simulated maneuver executed. " + verdict)
                else:
                    response = finish(verdict)
        elif "execute_maneuver_on_satellite" in names:
            safety_text = self.memory.get("safety_final", "")
            candidates = latest.get("propose_avoidance_maneuvers", {}).get("candidates", [])
            candidates = [c for c in candidates if c.get("delta_v_magnitude_ms", 999) <= 50.0]
            candidate = max(candidates, key=lambda c: c.get("miss_improvement_m", 0), default={})
            dv = candidate.get("delta_v_vector")
            if safety_text.startswith("APPROVED") and dv is not None and "execute_maneuver_on_satellite" not in called:
                response = call("execute_maneuver_on_satellite", {"delta_v": dv})
            else:
                response = finish("Operator brief: maneuver rejected or not approved; no execution performed.")
        elif "get_satellite_status" in names and "execute_maneuver_on_satellite" in called and "get_satellite_status" not in called:
            response = call("get_satellite_status")
        elif "scan_conjunctions" in names and "scan_demo_conjunctions" not in names and "scan_conjunctions" not in called:
            response = call("scan_conjunctions")
        elif "simulate_maneuver_effect" in names and "simulate_maneuver_effect" not in called and latest.get("propose_avoidance_maneuvers", {}).get("candidates"):
            candidates = latest["propose_avoidance_maneuvers"]["candidates"]
            cand = max((c for c in candidates if c.get("delta_v_magnitude_ms", 999) <= 50.0), key=lambda c: c.get("miss_improvement_m", 0), default={})
            sid = latest["propose_avoidance_maneuvers"].get("secondary_id")
            response = call("simulate_maneuver_effect", {"secondary_id": int(sid), "delta_v": cand["delta_v_vector"], "burn_time_sec": float(cand.get("burn_time_sec", 0.0))}) if sid is not None and cand else finish("Missing maneuver candidate details; simulation skipped.")
        else:
            outputs = {"scout": "Threats ranked from the demo conjunction scan.", "analyst": "Risk assessed for the leading conjunctions.", "planner": "Selected the best available candidate within the maneuver limit.", "safety": "APPROVED: maneuver passed the available constraint checks.", "ops_brief": "Operator brief: approved simulated maneuver executed.", "astro_helix": "Analysis complete using the deterministic demo policy."}
            role = "astro_helix"
            for candidate in ("scout", "analyst", "planner", "safety", "ops_brief"):
                if (candidate == "scout" and "scan_demo_conjunctions" in names) or (candidate == "analyst" and "assess_risk" in names and "propose_avoidance_maneuvers" not in names) or (candidate == "planner" and "propose_avoidance_maneuvers" in names) or (candidate == "safety" and "check_maneuver_constraints" in names) or (candidate == "ops_brief" and "execute_maneuver_on_satellite" in names):
                    role = candidate
                    break
            response = finish(outputs[role])
        return ChatResult(generations=[ChatGeneration(message=response)])
