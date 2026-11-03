import asyncio
import json
import time

from fastapi.testclient import TestClient

from agents.config import LLMConfig
from agents.graph import run_avoidance_pipeline, stream_avoidance_pipeline
from api.app import app
from api.demo_data import DEMO_DEBRIS, load_demo_data
import api.demo_data as demo_data_module
from api.state import get_catalog, get_satellite, reset_state


PROMPT = "Analyze threats for ISS using the demo dataset"


def setup_function():
    reset_state()
    load_demo_data()


def test_scripted_multi_pipeline_executes_in_order():
    before = get_satellite().velocity.copy()
    result = run_avoidance_pipeline(PROMPT, config=LLMConfig(provider="scripted"), mode="multi")
    completed = [e["agent"] for e in result["events"] if e["type"] == "agent_complete"]
    assert completed == ["scout", "analyst", "planner", "safety", "ops_brief"]
    assert any(e["type"] == "maneuver_executed" for e in result["events"])
    assert (get_satellite().velocity != before).any()
    assert result["ops_brief"]


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
        assert client.get("/status").json()["llm"]["llm_provider"] == "scripted"
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
    load_demo_data(use_cache=False)
    fresh = {obj.norad_id: (obj.position.copy(), obj.velocity.copy()) for obj in get_catalog().list_debris()}
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
    assert elapsed < 5
