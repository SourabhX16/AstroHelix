# Astro Helix — Multi-Agent Satellite Collision Avoidance Simulation

Astro Helix is a software simulation of an autonomous collision-avoidance pipeline for satellites. It combines a Python orbital physics engine, a LangGraph multi-agent workflow, a FastAPI service, and a Next.js dashboard. The agents screen a demo orbital catalog, assess conjunction risk, propose and simulate avoidance maneuvers, check resource constraints, and produce an operator brief. Maneuvers update only the simulated satellite state; no commands are sent to a real spacecraft.

The current project is a software-only simulation designed to run on a development machine. Deployment to on-board hardware such as an ASUS Ascent GX10 / DGX Spark-class system is future scope.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│        On-board compute (target deployment, future scope)        │
│                                                                  │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────┐ │
│  │  SCOUT  │→ │ ANALYST  │→ │ PLANNER  │→ │ SAFETY │→ │ OPS  │ │
│  │ scan &  │  │ risk &   │  │ maneuver │  │ verify │  │BRIEF │ │
│  │ triage  │  │ refine   │  │ design   │  │& exec  │  │      │ │
│  └─────────┘  └──────────┘  └──────────┘  └────────┘  └──────┘ │
│       ↕             ↕             ↕             ↕               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Physics Engine (deterministic)               │   │
│  │  screening · risk · CW dynamics · RK4 · SGP4 · Chan Pc   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│       ↕                                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │          Satellite Model (fuel, power, dynamics)          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│       ↕                                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Nemotron 3 Nano 30B (NVFP4) via vLLM — local inference  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Key Components

| Component | Path | Description |
|-----------|------|-------------|
| **Agent Pipeline** | `agents/` | LangGraph 5-agent pipeline with tool-calling |
| **Physics Engine** | `engine/` | RK4/RK45 propagation, perturbation models, CW dynamics, Chan collision probability |
| **Satellite Model** | `engine/models/active_satellite.py` | Simulated orbital state with resource management (fuel, power, battery) |
| **Tool Wrappers** | `agents/tools.py` | 12 LangChain tools wrapping the physics engine and in-memory state |
| **API** | `api/` | FastAPI service for agent execution, SSE streaming, demo reset, health, and status |
| **Frontend** | `frontend/` | Next.js + React Three Fiber dashboard with a 3D globe and TLE-based orbital data routes |
| **Optional hardware setup** | `scripts/setup_gx10.sh` | Optional future-scope vLLM setup for GX10 / DGX Spark-class hardware |

## Agent Pipeline

| Agent | Role | Tools |
|-------|------|-------|
| **Scout** | Scan catalog for upcoming conjunctions, triage by severity | `get_pending_cdms`, `scan_conjunctions`, `scan_demo_conjunctions` |
| **Analyst** | Deep risk assessment — Chan probability, high-fidelity TCA refinement | `get_pending_cdms`, `scan_conjunctions`, `scan_demo_conjunctions`, `assess_risk`, `refine_conjunction_hifi` |
| **Planner** | Design avoidance maneuvers considering satellite resources | `propose_avoidance_maneuvers`, `simulate_maneuver_effect`, `assess_risk` |
| **Safety** | Validate constraints and check maneuver feasibility | `get_satellite_status`, `check_maneuver_constraints`, `check_maneuver_feasibility`, `propagate_satellite_orbit` |
| **Ops Brief** | Apply the approved simulated burn and produce an operator summary | `assess_risk`, `execute_maneuver_on_satellite`, `get_satellite_status`, `propagate_satellite_orbit` |

## Quick Start

### 1. Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev  # localhost:3000
```

### 3. Agent System (optional future-scope hardware)

```bash
# Optional: start a local model endpoint for future-scope hardware
chmod +x scripts/setup_gx10.sh
./scripts/setup_gx10.sh

# Run agent pipeline
python -m agents.run "Scan for conjunction threats to satellite 25544 in the next 48 hours" --demo
```

### 4. Agent System (development mode, no local GPU)

```bash
# Set an OpenAI-compatible endpoint in .env
NEMOTRON_BASE_URL=https://api.openai.com/v1
NEMOTRON_API_KEY=sk-...
NEMOTRON_MODEL=gpt-4o-mini

python -m agents.run "Scan for threats" --demo
```

## Model

The agent layer uses an OpenAI-compatible chat endpoint. The repository includes configuration for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` and an optional vLLM setup script for a future-scope on-board GX10 / DGX Spark-class deployment. The simulation can also run against any OpenAI-compatible endpoint; no on-board hardware is required.

## Why Simulate This Pipeline?

- **Reproducible scenarios:** the API loads a demo debris catalog, and the tools operate on in-memory orbital state.
- **Inspectable decisions:** the pipeline emits SSE events, intermediate agent outputs, and a final operator brief.
- **Modular physics:** screening, risk assessment, maneuver planning, propagation, and constraint checks live in separate Python modules.
- **Safe experimentation:** maneuvers update only the simulated satellite model; no commands are sent to a real spacecraft.

An on-board deployment is future scope and is not implemented or tested in this repository.

## Team

- <name> — <role>
- <name> — <role>
- <name> — <role>
- <name> — <role>
