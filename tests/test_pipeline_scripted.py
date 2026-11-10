import asyncio
import json
import time

from fastapi.testclient import TestClient

from agents.config import LLMConfig
from agents.graph import run_avoidance_pipeline, stream_avoidance_pipeline
from api.app import app
from api.demo_data import DEMO_DEBRIS, load_demo_data
import api.demo_data as demo_data_module
from api.state import get_catalog, get_cdm_inbox, get_satellite, reset_state
from agents.tools import scan_demo_conjunctions
from api.state import (
    set_last_maneuver_search, get_recommended_maneuver, get_top_threat,
    record_conjunction_events, record_risk_assessment,
)


PROMPT = "Analyze threats for ISS using the demo dataset"


def setup_function():
    reset_state()
    load_demo_data()


def test_scripted_multi_pipeline_executes_in_order():
    before = get_satellite().velocity.copy()
    result = run_avoidance_pipeline(PROMPT, config=LLMConfig(provider="scripted"), mode="multi")
    completed = [e["agent"] for e in result["events"] if e["type"] == "agent_complete"]
    assert completed == ["scout", "analyst", "planner", "safety", "ops_brief"]
    assert result["recommended_maneuver"]
    assert any(e["type"] == "maneuver_executed" for e in result["events"])
    assert (get_satellite().velocity != before).any()
    assert result["ops_brief"]
    assert result["recommended_maneuver"]["verdict"] == "APPROVED"
    stored_dv = result["recommended_maneuver"]["delta_v_vector"]
    execution = next(e for e in result["events"] if e["type"] == "maneuver_executed")
    executed_dv = execution["delta_v"]
    assert executed_dv == stored_dv
    maneuver_events = [e for e in result["events"] if e["type"] == "recommended_maneuver"]
    assert {e["stage"] for e in maneuver_events} >= {
        "record_recommended_maneuver", "check_maneuver_feasibility",
        "check_maneuver_constraints", "record_safety_verdict", "execute_recommended_maneuver",
    }
    safety_tools = [e["tool"] for e in result["events"] if e["type"] == "tool_result" and e["agent"] == "safety"]
    assert safety_tools.count("check_maneuver_feasibility") == 1
    assert safety_tools.count("check_maneuver_constraints") == 1
    risk_calls = [e for e in result["events"] if e["type"] == "tool_result" and e["tool"] == "assess_risk"]
    assert risk_calls
    assert '3600.0' in risk_calls[0]["summary"]
    ops_risk_calls = [e for e in risk_calls if e["agent"] == "ops_brief"]
    assert len(ops_risk_calls) == 1


def test_scripted_stream_order_and_completion():
    async def collect():
        return [event async for event in stream_avoidance_pipeline(PROMPT, LLMConfig(provider="scripted"), "multi")]

    events = asyncio.run(collect())
    completed = [e["agent"] for e in events if e["type"] == "agent_complete"]
    started = [e["agent"] for e in events if e["type"] == "agent_start"]
    assert completed == ["scout", "analyst", "planner", "safety", "ops_brief"]
    assert started == completed
    assert events[0]["type"] == "agent_start"
    assert events[-1]["type"] == "pipeline_complete"
    assert not any(e["type"] == "error" for e in events)


def test_fastapi_scripted_endpoints(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "scripted")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        status_response = client.get("/status").json()
        assert status_response["llm"]["llm_provider"] == "scripted"
        assert status_response["satellite"]["fuel_kg"] == round(get_satellite().fuel_kg, 2)
        assert status_response["satellite"]["fuel_pct"] == round(get_satellite().fuel_percentage, 2)
        response = client.post("/agent/run", params={"mode": "multi", "prompt": PROMPT})
        assert response.status_code == 200
        assert response.json()["ops_brief"]


def test_unreachable_openai_compatible_endpoint_emits_safe_error():
    config = LLMConfig(base_url="http://127.0.0.1:9/v1", api_key="not-needed", model="test", provider="openai_compat")

    async def collect():
        return [event async for event in stream_avoidance_pipeline(PROMPT, config, "multi")]

    started = time.monotonic()
    events = asyncio.run(asyncio.wait_for(collect(), timeout=15))
    assert time.monotonic() - started < 15
    errors = [e for e in events if e["type"] == "error"]
    assert errors and errors[0]["error_kind"] == "llm_unreachable"
    assert "Traceback" not in json.dumps(events)
    assert events[-1]["type"] == "pipeline_complete"


def test_scripted_single_agent_smoke():
    result = run_avoidance_pipeline(PROMPT, config=LLMConfig(provider="scripted"), mode="single")
    assert result["ops_brief"]
    assert any(e["type"] == "agent_complete" and e["agent"] == "astro_helix" for e in result["events"])


def test_cached_demo_states_match_fresh_generation_and_load_quickly():
    reset_state()
    load_demo_data(use_cache=False)
    fresh = {obj.norad_id: (obj.position.copy(), obj.velocity.copy()) for obj in get_catalog().list_debris()}
    fresh_cdms = [{key: value for key, value in cdm.items() if key != "timestamp"} for cdm in get_cdm_inbox().get_all()]
    reset_state()
    demo_data_module._MEMO.clear()
    started = time.monotonic()
    load_demo_data()
    elapsed = time.monotonic() - started
    cached = {obj.norad_id: (obj.position.copy(), obj.velocity.copy()) for obj in get_catalog().list_debris()}
    assert cached.keys() == fresh.keys() == {spec["norad_id"] for spec in DEMO_DEBRIS}
    for norad_id in fresh:
        for actual, expected in zip(cached[norad_id], fresh[norad_id]):
            assert abs(actual - expected).max() <= 1e-6 * max(1.0, abs(expected).max())
    warm_cdms = [{key: value for key, value in cdm.items() if key != "timestamp"} for cdm in get_cdm_inbox().get_all()]
    assert warm_cdms == fresh_cdms
    assert elapsed < 2


def test_demo_cache_key_change_forces_per_debris_regeneration(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(demo_data_module, "_CACHE_PATH", tmp_path / "demo_cache.json")
    monkeypatch.setattr(demo_data_module, "_MEMO", {})
    monkeypatch.setattr(demo_data_module, "build_conjunction_debris", lambda *args: (calls.append("build") or get_satellite().position.copy(), get_satellite().velocity.copy()))
    monkeypatch.setattr(demo_data_module, "screen_conjunctions", lambda **kwargs: [{"tca_offset_sec": 3600.0, "miss_distance_m": 150.0, "relative_velocity_ms": 1000.0}])
    monkeypatch.setattr(demo_data_module, "assess_conjunction_risk", lambda **kwargs: {"risk_level": "low", "collision_probability_chan": 0.0})
    demo_data_module.load_demo_data()
    first_count = len(calls)
    monkeypatch.setattr(demo_data_module, "SCENARIO_VERSION", demo_data_module.SCENARIO_VERSION + 1)
    reset_state()
    demo_data_module.load_demo_data()
    assert first_count == len(DEMO_DEBRIS)
    assert len(calls) == 2 * len(DEMO_DEBRIS)


def test_demo_scan_is_cached_until_satellite_state_changes(monkeypatch):
    from agents import tools as agent_tools

    calls = []
    monkeypatch.setattr(agent_tools, "screen_conjunctions", lambda **kwargs: (calls.append(1) or [{"secondary_id": 90005, "miss_distance_m": 150.0, "tca_offset_sec": 3600.0}]))
    monkeypatch.setattr(agent_tools, "load_demo_data", lambda: None)
    first = scan_demo_conjunctions.func()
    second = scan_demo_conjunctions.func()
    assert first == second
    assert len(calls) == 1
    get_satellite().position[0] += 0.01
    scan_demo_conjunctions.func()
    assert len(calls) == 2


def test_execute_maneuver_rejects_over_limit_without_applying():
    from agents.tools import execute_maneuver_on_satellite

    sat = get_satellite()
    velocity = sat.velocity.copy()
    fuel = sat.fuel_kg
    result = json.loads(execute_maneuver_on_satellite.func([51.0, 0.0, 0.0]))
    assert result["executed"] is False
    assert result["error"]
    assert not (sat.velocity != velocity).any()
    assert sat.fuel_kg == fuel


def test_status_fuel_precision_is_two_decimals():
    from agents.tools import get_satellite_status

    status = json.loads(get_satellite_status.func())
    assert status["fuel_kg"] == round(get_satellite().fuel_kg, 2)
    assert status["fuel_pct"] == round(get_satellite().fuel_percentage, 2)


def test_status_probe_reports_timeout_as_llm_timeout(monkeypatch):
    import langchain_openai

    class TimeoutChat:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 20

        def invoke(self, prompt):
            raise TimeoutError("request timed out")

    monkeypatch.setattr("api.app._get_config", lambda: LLMConfig(provider="openai_compat", base_url="https://example.invalid/v1", model="test", api_key="test"))
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", TimeoutChat)
    with TestClient(app) as client:
        status = client.get("/status?probe=1").json()
    assert status["llm"]["llm_reachable"] is False
    assert status["llm"]["error_kind"] == "llm_timeout"


def _store_test_recommendation(vector=(0.01, 0.0, 0.0), verdict=None):
    from agents.tools import record_recommended_maneuver, check_maneuver_feasibility, check_maneuver_constraints, record_safety_verdict

    set_last_maneuver_search({"secondary_id": 90005, "candidates": [{
        "candidate_index": 0, "delta_v_vector": list(vector), "type": "along-track",
        "miss_before": 150.0, "miss_after": 1200.0, "burn_time_sec": 0.0,
        "delta_v_magnitude_ms": float(sum(x * x for x in vector) ** 0.5),
    }]})
    stored = json.loads(record_recommended_maneuver.func(90005, 0))["recommended_maneuver"]
    if verdict:
        check_maneuver_feasibility.func()
        check_maneuver_constraints.func()
        stored = json.loads(record_safety_verdict.func(verdict, "test verdict"))["recommended_maneuver"]
    return stored


def test_safety_checks_and_ops_execute_the_stored_vector():
    from agents.tools import check_maneuver_feasibility, check_maneuver_constraints, record_safety_verdict, execute_recommended_maneuver

    stored = _store_test_recommendation()
    feasibility = json.loads(check_maneuver_feasibility.func())
    constraints = json.loads(check_maneuver_constraints.func())
    assert feasibility["checked_delta_v_vector"] == stored["delta_v_vector"]
    assert constraints["checked_delta_v_vector"] == stored["delta_v_vector"]
    approved = json.loads(record_safety_verdict.func("APPROVED", "both checks passed"))["recommended_maneuver"]
    before = get_satellite().velocity.copy()
    result = json.loads(execute_recommended_maneuver.func())
    assert result["executed"] is True
    assert result["delta_v"] == approved["delta_v_vector"]
    assert (get_satellite().velocity - before).tolist() == approved["delta_v_vector"]


def test_recommended_execution_refuses_nonapproved_verdict():
    from agents.tools import execute_recommended_maneuver

    _store_test_recommendation(verdict="REJECTED")
    before = get_satellite().velocity.copy()
    result = json.loads(execute_recommended_maneuver.func())
    assert result["executed"] is False
    assert result["fuel_used_kg"] == 0.0
    assert (get_satellite().velocity == before).all()


def test_deterministic_search_handles_the_150m_threat():
    from agents.tools import assess_risk, propose_avoidance_maneuvers

    scan_demo_conjunctions.func()
    assess_risk.func(90005, miss_distance_m=150.0, tca_offset_sec=3600.0)
    search = json.loads(propose_avoidance_maneuvers.func(90005))
    assert search["secondary_id"] == 90005
    assert search["runtime_sec"] >= 0.0
    if search["candidates"]:
        assert len(search["candidates"]) <= 3
        assert all(candidate["miss_after"] > candidate["miss_before"] for candidate in search["candidates"])
    else:
        assert search["no_improving_candidate"] is True
        assert search["message"] == "no improving candidate within constraints"


def test_top_threat_uses_chan_probability_before_miss_distance():
    record_conjunction_events([
        {"secondary_id": 90005, "miss_distance_m": 150.0, "tca_offset_sec": 3600.0},
        {"secondary_id": 90007, "miss_distance_m": 600.0, "tca_offset_sec": 10800.0},
    ])
    record_risk_assessment(90005, {"collision_probability_chan": 8e-5, "miss_distance_m": 150.0})
    record_risk_assessment(90007, {"collision_probability_chan": 1e-2, "miss_distance_m": 600.0})
    assert get_top_threat()["secondary_id"] == 90007


def test_wrong_vector_in_ops_text_cannot_change_stored_execution():
    from uuid import uuid4
    import numpy as np
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from agents.graph import _create_agent_node
    from agents.tools import execute_recommended_maneuver, execute_maneuver_on_satellite

    class WrongVectorModel(BaseChatModel):
        bound_names: list[str] = []

        @property
        def _llm_type(self):
            return "wrong-vector-text-test"

        def bind_tools(self, tools, **kwargs):
            return self.model_copy(update={"bound_names": [tool.name for tool in tools]})

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if any(isinstance(message, ToolMessage) and message.name == "execute_recommended_maneuver" for message in messages):
                reply = AIMessage(content="I propose [999.0, 999.0, 999.0] m/s.")
            else:
                reply = AIMessage(content="", tool_calls=[{"name": "execute_recommended_maneuver", "args": {}, "id": str(uuid4()), "type": "tool_call"}])
            return ChatResult(generations=[ChatGeneration(message=reply)])

    stored = _store_test_recommendation((0.02, 0.0, 0.0), verdict="APPROVED")
    rejected = json.loads(execute_maneuver_on_satellite.func([999.0, 999.0, 999.0]))
    assert rejected["executed"] is False
    before = get_satellite().velocity.copy()
    node = _create_agent_node(WrongVectorModel(), LLMConfig(provider="scripted", max_iterations=3), "ops", [execute_recommended_maneuver], "ops_brief", "ops_brief")
    node({
        "messages": [HumanMessage(content="Execute approved recommendation")], "current_agent": "ops_brief",
        "scout_output": None, "analyst_output": None, "planner_output": None, "safety_output": "VERDICT: APPROVED",
        "ops_brief": None, "events": [], "pipeline_error": False, "recommended_maneuver": stored,
    })
    assert np.allclose(get_satellite().velocity - before, stored["delta_v_vector"], rtol=0, atol=1e-15)
    assert get_recommended_maneuver()["delta_v_vector"] == stored["delta_v_vector"]
