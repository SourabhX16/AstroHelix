"""
FastAPI backend for the Astro Helix agent system.

Provides SSE streaming of agent pipeline events to the frontend.
The frontend's physics/visualization stays in TypeScript — this API
only exposes the LLM agent pipeline for the terminal drawer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from typing import Optional

if os.getenv("ASTRO_HELIX_SKIP_DOTENV") != "1":
    from dotenv import load_dotenv
    load_dotenv(override=False)

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agents.config import LLMConfig
from agents.graph import run_avoidance_pipeline, stream_avoidance_pipeline
from api.demo_data import load_demo_data
from api.state import get_satellite, reset_state

logger = logging.getLogger("astro_helix.api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load demo data on startup."""
    logger.info("Loading demo data...")
    load_demo_data()
    logger.info("Demo data loaded. Agent API ready.")
    yield


app = FastAPI(
    title="Astro Helix Agent API",
    description="Internal API for the Astro Helix collision avoidance agent system",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_config() -> LLMConfig:
    """Build LLM config from environment."""
    return LLMConfig.from_env()


# ─────────────────────────────────────────────────────────────────────────
# Health / status
# ─────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "astro-helix-agent-api"}


@app.get("/status")
def status(probe: int = Query(default=0)):
    sat = get_satellite()
    config = _get_config()
    llm = {
        "llm_provider": config.provider,
        "llm_host": urlparse(config.base_url).hostname,
        "model": config.model,
        "base_url_source": "env" if "NEMOTRON_BASE_URL" in os.environ else "default",
        "configured": bool(config.provider == "scripted" or (config.base_url and config.model and config.api_key and config.api_key != "not-needed")),
    }
    if probe == 1:
        if config.provider == "scripted":
            llm["llm_reachable"] = True
        else:
            from langchain_openai import ChatOpenAI
            try:
                kwargs = config.to_llm_kwargs()
                kwargs.update(timeout=20, max_tokens=1)
                ChatOpenAI(**kwargs).invoke("Reply with OK")
                llm["llm_reachable"] = True
            except Exception as exc:
                llm["llm_reachable"] = False
                status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                if status_code in (401, 403):
                    llm["error_kind"] = "llm_auth"
                elif status_code in (400, 422):
                    llm["error_kind"] = "llm_bad_request"
                elif isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or any(term in str(exc).lower() for term in ("timeout", "timed out")):
                    llm["error_kind"] = "llm_timeout"
                elif any(term in str(exc).lower() for term in ("connect", "refused", "dns")):
                    llm["error_kind"] = "llm_unreachable"
                else:
                    llm["error_kind"] = "llm_other"
    return {
        "satellite": {
            "norad_id": sat.norad_id,
            "name": sat.name,
            "fuel_kg": round(sat.fuel_kg, 2),
            "fuel_pct": round(sat.fuel_kg / sat.config.fuel_capacity_kg * 100, 2),
        },
        "llm": llm,
    }


# ─────────────────────────────────────────────────────────────────────────
# Agent pipeline — synchronous (returns full result)
# ─────────────────────────────────────────────────────────────────────────

@app.post("/agent/run")
def agent_run(
    prompt: str = "Scan for conjunction threats against the ISS in the next 24 hours. If any are high risk, propose avoidance maneuvers and check constraints. Use the demo dataset.",
    mode: str = "multi",
):
    """
    Run the agent pipeline synchronously and return the full result.
    Use /agent/stream for real-time SSE updates.
    """
    config = _get_config()
    result = run_avoidance_pipeline(prompt, config=config, mode=mode)
    return result


# ─────────────────────────────────────────────────────────────────────────
# Agent pipeline — SSE streaming (real-time events for terminal drawer)
# ─────────────────────────────────────────────────────────────────────────

@app.get("/agent/stream")
async def agent_stream(
    prompt: str = Query(
        default="Scan for conjunction threats against the ISS in the next 24 hours. If any are high risk, propose avoidance maneuvers and check constraints. Use the demo dataset.",
        description="Natural language request for the agent",
    ),
    mode: str = Query(default="multi", description="Agent mode: multi or single"),
):
    """
    Stream agent pipeline events via Server-Sent Events (SSE).
    The frontend terminal drawer connects to this endpoint.
    """
    config = _get_config()

    async def event_generator():
        try:
            async for event in stream_avoidance_pipeline(prompt, config=config, mode=mode):
                data = json.dumps(event, default=str)
                yield f"data: {data}\n\n"
        except Exception:
            host = urlparse(config.base_url).hostname or "configured endpoint"
            yield f"data: {json.dumps({'type': 'error', 'error_kind': 'llm_other', 'message': f'LLM pipeline failed at host {host}.'})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done', 'timestamp': time.time()})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────
# Demo data management
# ─────────────────────────────────────────────────────────────────────────

@app.post("/demo/reload")
def demo_reload():
    """Reset state and reload demo data."""
    reset_state()
    summary = load_demo_data()
    return {"ok": True, "summary": summary}
