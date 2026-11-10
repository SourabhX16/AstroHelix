"""
LangGraph multi-agent collision avoidance pipeline.

Five agents form a sequential pipeline, each with specialized tools:
  Scout → Analyst → Planner → Safety → Ops Brief

Uses Nemotron via vLLM's OpenAI-compatible API + LangChain + LangGraph.
"""
from __future__ import annotations

import asyncio
import json
import logging
import operator
import queue as queue_module
import time
import socket
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from agents.config import LLMConfig
from agents.llm import make_llm
from agents.prompts import (
    ANALYST_PROMPT,
    OPS_BRIEF_PROMPT,
    PLANNER_PROMPT,
    SAFETY_PROMPT,
    SCOUT_PROMPT,
)
from agents.tools import (
    ALL_TOOLS,
    ANALYST_TOOLS,
    PLANNER_TOOLS,
    SAFETY_TOOLS,
    SCOUT_TOOLS,
    assess_risk,
    execute_recommended_maneuver,
    get_satellite_status,
    propagate_satellite_orbit,
)
from api.state import get_last_maneuver_search, get_recommended_maneuver, reset_maneuver_pipeline_state

logger = logging.getLogger("astro_helix.agents.graph")


# ─────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    """Shared state that flows through the agent pipeline."""
    # Core conversation messages (appended by each agent)
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Which agent is active
    current_agent: str
    # Structured outputs from each stage (accumulated)
    scout_output: Optional[str]
    analyst_output: Optional[str]
    planner_output: Optional[str]
    safety_output: Optional[str]
    ops_brief: Optional[str]
    # Event stream for real-time UI updates
    events: Annotated[List[Dict[str, Any]], operator.add]
    pipeline_error: Optional[bool]
    recommended_maneuver: Optional[Dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────
# Agent node factory
# ─────────────────────────────────────────────────────────────────────────
def _create_agent_node(
    llm: BaseChatModel,
    config: LLMConfig,
    system_prompt: str,
    tools: list,
    agent_name: str,
    output_key: str,
    next_agent: Optional[str] = None,
    event_queue: Optional[queue_module.Queue] = None,
):
    """
    Create a LangGraph node that:
    1. Calls the LLM with tool-calling enabled
    2. Loops on tool calls until the LLM is done
    3. Stores its final response in state[output_key]
    4. Emits events for real-time streaming via event_queue
    """
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    tools_by_name = {getattr(tool, "name", ""): tool for tool in tools}

    def _emit(event: dict, events: list):
        """Push event to both the local list and the real-time queue."""
        events.append(event)
        if event_queue:
            event_queue.put(event)

    def agent_node(state: AgentState) -> dict:
        """Run one agent to completion (with tool-calling loop)."""
        t0 = time.time()
        events: List[Dict[str, Any]] = []

        if state.get("pipeline_error"):
            return {"messages": [], "current_agent": next_agent or agent_name, "events": []}

        _emit({
            "type": "agent_start",
            "agent": agent_name,
            "llm_provider": config.provider,
            "timestamp": time.time(),
        }, events)

        # Build messages: system prompt + all prior messages + handoff context
        msgs: List[BaseMessage] = [SystemMessage(content=system_prompt)]

        # Add context from previous agents (truncated to stay within token budget)
        _CTX_CAP = 500
        for key, label in [
            ("scout_output", "Scout findings"),
            ("analyst_output", "Analyst assessment"),
            ("planner_output", "Planner recommendations"),
            ("safety_output", "Safety review"),
        ]:
            val = state.get(key)
            if val and key != output_key:
                truncated = val[:_CTX_CAP] + "..." if len(val) > _CTX_CAP else val
                msgs.append(HumanMessage(content=f"[{label}]\n{truncated}"))

        # Add the original user request (first human message)
        for m in state["messages"]:
            if isinstance(m, HumanMessage):
                msgs.append(m)
                break

        # Tool-calling loop
        max_iterations = config.max_iterations
        executed_calls: dict[tuple[str, str], str] = {}
        exhausted_with_tools = False
        for iteration in range(max_iterations):
            _emit({
                "type": "llm_call",
                "agent": agent_name,
                "iteration": iteration + 1,
                "timestamp": time.time(),
            }, events)

            _emit({
                "type": "thinking",
                "agent": agent_name,
                "text": "waiting for model response...",
                "timestamp": time.time(),
            }, events)

            try:
                response = llm_with_tools.invoke(msgs)
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                text = str(exc)
                host = urlparse(config.base_url).hostname or "configured endpoint"
                lowered = text.lower()
                if status in (401, 403):
                    kind, message = "llm_auth", f"Authentication failed for LLM host {host}."
                elif status in (400, 422):
                    kind = "llm_bad_request"
                    safe = text[:300]
                    safe = __import__("re").sub(r"(?i)(api[_ -]?key|authorization|bearer)\s*[:=]?\s*[^\s,;]+", r"\1=[redacted]", safe)
                    if config.api_key and config.api_key != "not-needed":
                        safe = safe.replace(config.api_key, "[redacted]")
                    safe = " ".join(safe.split())
                    message = f"LLM host {host} rejected the request: {safe}"
                elif isinstance(exc, (ConnectionError, TimeoutError, socket.gaierror)) or any(term in lowered for term in ("connect", "timeout", "timed out", "name or service", "dns", "refused")):
                    kind, message = "llm_unreachable", f"Could not reach LLM host {host}."
                else:
                    kind, message = "llm_other", f"LLM request failed at host {host}."
                _emit({"type": "error", "error_kind": kind, "message": message, "timestamp": time.time()}, events)
                return {"messages": [AIMessage(content=message, name=agent_name)], output_key: message,
                        "current_agent": next_agent or agent_name, "events": events, "pipeline_error": True}
            msgs.append(response)

            # If no tool calls, we're done
            if not response.tool_calls:
                break

            # Execute tool calls, replaying prior identical calls within this agent run.
            _emit({
                "type": "tool_calls",
                "agent": agent_name,
                "tools": [tc["name"] for tc in response.tool_calls],
                "timestamp": time.time(),
            }, events)

            if tools:
                # Create a mini-state for the tool node
                _TOOL_CAP = 800
                tool_msgs = []
                for tc in response.tool_calls:
                    tool_name = tc.get("name", "")
                    args = tc.get("args", {})
                    fingerprint = (tool_name, json.dumps(args, sort_keys=True, separators=(",", ":"), default=str))
                    if fingerprint in executed_calls:
                        result_text = "(duplicate call, result unchanged) " + executed_calls[fingerprint]
                    else:
                        try:
                            result = tools_by_name[tool_name].invoke(args)
                            result_text = result if isinstance(result, str) else json.dumps(result, default=str)
                        except Exception as exc:
                            result_text = json.dumps({"error": str(exc)})
                        executed_calls[fingerprint] = result_text
                    tool_msgs.append(ToolMessage(content=result_text, name=tool_name, tool_call_id=tc.get("id", "")))
                # Preserve complete structured outputs for the deterministic offline
                # model before truncating conversation context for hosted models.
                if hasattr(llm_with_tools, "memory"):
                    for raw_tm in tool_msgs:
                        if isinstance(raw_tm, ToolMessage):
                            try:
                                parsed_tool_result = json.loads(raw_tm.content) if isinstance(raw_tm.content, str) else {}
                                llm_with_tools.memory[raw_tm.name] = parsed_tool_result
                                llm_with_tools.memory.setdefault("_tool_history", []).append((raw_tm.name, parsed_tool_result))
                            except (TypeError, json.JSONDecodeError):
                                pass
                # Truncate tool outputs to stay within token budget
                for tm in tool_msgs:
                    if isinstance(tm, ToolMessage) and isinstance(tm.content, str) and len(tm.content) > _TOOL_CAP:
                        tm.content = tm.content[:_TOOL_CAP] + "...[truncated]"
                msgs.extend(tool_msgs)
                exhausted_with_tools = iteration == max_iterations - 1 and bool(response.tool_calls)

                for tm in tool_msgs:
                    if isinstance(tm, ToolMessage):
                        _emit({
                            "type": "tool_result",
                            "agent": agent_name,
                            "tool": tm.name,
                            "timestamp": time.time(),
                            # Truncate large outputs for event stream
                            "summary": tm.content[:1200] if tm.name in {"execute_maneuver_on_satellite", "execute_recommended_maneuver"} and isinstance(tm.content, str) else (tm.content[:200] if isinstance(tm.content, str) else str(tm.content)[:200]),
                        }, events)

                        # When a maneuver is executed, emit post-maneuver state
                        # so the frontend can update the globe visualization.
                        if tm.name in {"execute_maneuver_on_satellite", "execute_recommended_maneuver"}:
                            try:
                                from api.state import get_satellite as _get_sat
                                _sat = _get_sat()
                                # Parse delta_v from the tool result
                                _dv = [0.0, 0.0, 0.0]
                                try:
                                    _res = json.loads(tm.content) if isinstance(tm.content, str) else {}
                                    if "delta_v" in _res:
                                        _dv = _res["delta_v"]
                                except Exception:
                                    pass
                                if _res.get("executed") is True:
                                    _emit({
                                    "type": "maneuver_executed",
                                    "agent": agent_name,
                                    "timestamp": time.time(),
                                    "position": _sat.position.tolist(),
                                    "velocity": _sat.velocity.tolist(),
                                    "delta_v": _dv,
                                    }, events)
                            except Exception:
                                pass

                        if tm.name in {
                            "record_recommended_maneuver", "check_maneuver_feasibility",
                            "check_maneuver_constraints", "record_safety_verdict",
                            "execute_recommended_maneuver", "execute_maneuver_on_satellite",
                        }:
                            _emit({
                                "type": "recommended_maneuver",
                                "agent": agent_name,
                                "stage": tm.name,
                                "recommended_maneuver": get_recommended_maneuver(),
                                "timestamp": time.time(),
                            }, events)
                        if tm.name == "propose_avoidance_maneuvers":
                            _emit({
                                "type": "maneuver_search",
                                "agent": agent_name,
                                "search": get_last_maneuver_search(),
                                "timestamp": time.time(),
                            }, events)

        if exhausted_with_tools:
            _emit({"type": "iteration_limit", "agent": agent_name, "timestamp": time.time()}, events)
            _emit({"type": "llm_call", "agent": agent_name, "iteration": max_iterations + 1, "final": True, "timestamp": time.time()}, events)
            try:
                final_response = llm.invoke(msgs + [SystemMessage(content="Iteration limit reached. Write your final report now, using only the tool results above. No more tool calls.")])
                msgs.append(final_response)
            except Exception as exc:
                logger.warning("[%s] final report call after iteration cap failed: %s", agent_name, exc)

        # Extract final response
        final_content = msgs[-1].content if msgs and isinstance(msgs[-1], AIMessage) else ""
        if isinstance(final_content, (dict, list)):
            final_content = json.dumps(final_content, default=str)
        elapsed = time.time() - t0
        _emit({
            "type": "agent_complete",
            "agent": agent_name,
            "elapsed_sec": round(elapsed, 2),
            "timestamp": time.time(),
        }, events)

        # Emit agent output for real-time display
        if final_content:
            _emit({
                "type": "agent_output",
                "agent": agent_name,
                "content": final_content,
                "timestamp": time.time(),
            }, events)

        logger.info(f"[{agent_name}] completed in {elapsed:.1f}s")

        return {
            "messages": [AIMessage(content=f"[{agent_name}] {final_content}", name=agent_name)],
            output_key: final_content,
            "current_agent": next_agent or agent_name,
            "events": events,
            "recommended_maneuver": get_recommended_maneuver(),
        }

    return agent_node


# ─────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────
def build_avoidance_graph(
    config: Optional[LLMConfig] = None,
    event_queue: Optional[queue_module.Queue] = None,
) -> StateGraph:
    """
    Build the multi-agent LangGraph for collision avoidance.

    Pipeline:
      scout → analyst → planner → safety → ops_brief → END
    """
    if config is None:
        config = LLMConfig.from_env()

    llm = make_llm(config)

    # Create agent nodes
    scout_node = _create_agent_node(
        llm, config, SCOUT_PROMPT, SCOUT_TOOLS, "scout", "scout_output", "analyst",
        event_queue=event_queue,
    )
    analyst_node = _create_agent_node(
        llm, config, ANALYST_PROMPT, ANALYST_TOOLS, "analyst", "analyst_output", "planner",
        event_queue=event_queue,
    )
    planner_node = _create_agent_node(
        llm, config, PLANNER_PROMPT, PLANNER_TOOLS, "planner", "planner_output", "safety",
        event_queue=event_queue,
    )
    safety_node = _create_agent_node(
        llm, config, SAFETY_PROMPT, SAFETY_TOOLS, "safety", "safety_output", "ops_brief",
        event_queue=event_queue,
    )
    OPS_BRIEF_TOOLS = [assess_risk, execute_recommended_maneuver, get_satellite_status, propagate_satellite_orbit]
    ops_brief_node = _create_agent_node(
        llm, config, OPS_BRIEF_PROMPT, OPS_BRIEF_TOOLS, "ops_brief", "ops_brief", None,
        event_queue=event_queue,
    )

    # Build graph
    graph = StateGraph(AgentState)

    graph.add_node("scout", scout_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("planner", planner_node)
    graph.add_node("safety", safety_node)
    graph.add_node("ops_brief", ops_brief_node)

    # Linear pipeline
    graph.set_entry_point("scout")
    graph.add_edge("scout", "analyst")
    graph.add_edge("analyst", "planner")
    graph.add_edge("planner", "safety")
    graph.add_edge("safety", "ops_brief")
    graph.add_edge("ops_brief", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────
# Single-agent mode (simpler, for quick testing)
# ─────────────────────────────────────────────────────────────────────────
def build_single_agent_graph(
    config: Optional[LLMConfig] = None,
    event_queue: Optional[queue_module.Queue] = None,
) -> StateGraph:
    """
    Build a single-agent graph with all tools available.
    Simpler than the multi-agent pipeline, good for testing.
    """
    if config is None:
        config = LLMConfig.from_env()

    llm = make_llm(config)

    SINGLE_AGENT_PROMPT = """You are Astro Helix, an AI collision avoidance copilot for satellites.
You are part of a software simulation of on-board collision-avoidance autonomy.

You have access to physics tools that compute real orbital mechanics. NEVER guess numbers.
Always call tools to get actual data.

When asked to analyze threats or plan avoidance:
1. Scan for conjunctions using scan_conjunctions or scan_demo_conjunctions
2. Assess risk for the most dangerous events
3. If risk is high, propose avoidance maneuvers
4. Check constraints on the best maneuver candidates
5. Present a clear recommendation to the operator

Be concise and actionable. Satellite operators need clear decisions, not essays."""

    agent_node = _create_agent_node(
        llm, config, SINGLE_AGENT_PROMPT, ALL_TOOLS, "astro_helix", "ops_brief", None,
        event_queue=event_queue,
    )

    graph = StateGraph(AgentState)
    graph.add_node("astro_helix", agent_node)
    graph.set_entry_point("astro_helix")
    graph.add_edge("astro_helix", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────
def run_avoidance_pipeline(
    request: str,
    config: Optional[LLMConfig] = None,
    mode: Literal["multi", "single"] = "multi",
) -> Dict[str, Any]:
    """
    Run the collision avoidance agent pipeline.

    Args:
        request: natural language request from the operator
        config: LLM configuration (defaults to env vars)
        mode: "multi" for 5-agent pipeline, "single" for single agent

    Returns:
        dict with ops_brief, events, and intermediate outputs
    """
    reset_maneuver_pipeline_state()
    if mode == "multi":
        graph = build_avoidance_graph(config)
    else:
        graph = build_single_agent_graph(config)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=request)],
        "current_agent": "scout" if mode == "multi" else "astro_helix",
        "scout_output": None,
        "analyst_output": None,
        "planner_output": None,
        "safety_output": None,
        "ops_brief": None,
        "events": [],
        "pipeline_error": False,
        "recommended_maneuver": None,
    }

    result = graph.invoke(initial_state, {"recursion_limit": 50})

    return {
        "ops_brief": result.get("ops_brief", ""),
        "scout_output": result.get("scout_output"),
        "analyst_output": result.get("analyst_output"),
        "planner_output": result.get("planner_output"),
        "safety_output": result.get("safety_output"),
        "events": result.get("events", []),
        "recommended_maneuver": result.get("recommended_maneuver"),
    }


async def stream_avoidance_pipeline(
    request: str,
    config: Optional[LLMConfig] = None,
    mode: Literal["multi", "single"] = "multi",
):
    """
    Async generator that streams agent events as they happen.

    Uses a thread-safe queue so events from synchronous agent nodes
    (which LangGraph runs in a thread pool) are delivered to the
    async SSE generator in real time, instead of batching per-node.
    """
    eq: queue_module.Queue = queue_module.Queue()

    reset_maneuver_pipeline_state()
    if mode == "multi":
        graph = build_avoidance_graph(config, event_queue=eq)
    else:
        graph = build_single_agent_graph(config, event_queue=eq)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=request)],
        "current_agent": "scout" if mode == "multi" else "astro_helix",
        "scout_output": None,
        "analyst_output": None,
        "planner_output": None,
        "safety_output": None,
        "ops_brief": None,
        "events": [],
        "pipeline_error": False,
        "recommended_maneuver": None,
    }

    done = asyncio.Event()

    async def _run_graph():
        try:
            async for _chunk in graph.astream(initial_state, {"recursion_limit": 50}):
                pass  # events are pushed to the queue from inside nodes
        except Exception:
            host = urlparse((config or LLMConfig.from_env()).base_url).hostname or "configured endpoint"
            eq.put({"type": "error", "error_kind": "llm_other", "message": f"LLM pipeline failed at host {host}.", "timestamp": time.time()})
        finally:
            done.set()

    task = asyncio.create_task(_run_graph())

    # Poll the queue, yielding events as they arrive
    while not done.is_set() or not eq.empty():
        try:
            event = eq.get_nowait()
            yield event
        except queue_module.Empty:
            await asyncio.sleep(0.1)

    # Drain any remaining events
    while not eq.empty():
        yield eq.get_nowait()

    yield {"type": "pipeline_complete", "timestamp": time.time()}
