"""
LangChain @tool wrappers for the Astro Helix agent system.

Wraps raw physics functions from tools/ and state helpers from api/
into LangChain tools that agents can call via function-calling.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.tools import tool

from api.demo_data import load_demo_data
from api.state import (
    get_catalog, get_cdm_inbox, get_satellite, get_top_threat,
    record_conjunction_events, record_risk_assessment,
    set_last_maneuver_search, get_last_maneuver_search,
    set_recommended_maneuver, get_recommended_maneuver,
    update_recommended_maneuver, record_maneuver_check, get_maneuver_checks,
)
from tools.constraints import check_constraints
from tools.maneuver import simulate_maneuver
from tools.propagate import propagate_orbit
from tools.refine import refine_conjunction
from tools.risk import assess_conjunction_risk
from tools.screening import screen_conjunctions


def _compact_events(events: list[dict]) -> tuple[list[dict], int]:
    closest = {}
    for event in events:
        secondary_id = event.get("secondary_id")
        if secondary_id is None:
            continue
        if (
            secondary_id not in closest
            or event.get("miss_distance_m", float("inf"))
            < closest[secondary_id].get("miss_distance_m", float("inf"))
        ):
            closest[secondary_id] = event
    ordered = sorted(closest.values(), key=lambda event: event.get("miss_distance_m", float("inf")))
    return ordered[:10], len(events)


# ─────────────────────────────────────────────────────────────────────────
# CDM / Catalog tools (Agent 0 — Scout)
# ─────────────────────────────────────────────────────────────────────────

@tool
def get_pending_cdms() -> str:
    """Get all pending (unprocessed) Conjunction Data Messages from the inbox."""
    inbox = get_cdm_inbox()
    pending = inbox.get_pending()
    record_conjunction_events(pending)
    return json.dumps(pending, indent=2, default=str)


@tool
def scan_conjunctions(lookahead_hours: float = 24.0, threshold_km: float = 50.0) -> str:
    """
    Screen the orbital catalog for close approaches to the active satellite.

    Args:
        lookahead_hours: how far ahead to scan (hours)
        threshold_km: only report conjunctions closer than this (km)
    """
    sat = get_satellite()
    catalog = get_catalog()
    debris_list = [obj.to_dict() for obj in catalog.list_debris()]

    if not debris_list:
        return json.dumps({"events": [], "message": "No debris in catalog. Use scan_demo_conjunctions or load data first."})

    events = screen_conjunctions(
        primary_pos=sat.position,
        primary_vel=sat.velocity,
        debris_list=debris_list,
        lookahead_sec=lookahead_hours * 3600,
        threshold_km=threshold_km,
    )
    shown, total = _compact_events(events)
    record_conjunction_events(shown)
    return json.dumps({
        "total_screened": len(debris_list),
        "events_found": len(events),
        "total_encounters_found": total,
        "events_shown": len(shown),
        "events": shown,
    }, indent=2)


@tool
def scan_demo_conjunctions() -> str:
    """Load demo debris data and scan for conjunctions. Use this for testing/demo."""
    sat = get_satellite()
    catalog = get_catalog()
    debris_list = [obj.to_dict() for obj in catalog.list_debris()]
    from api import demo_data
    demo_ids = {int(item["norad_id"]) for item in demo_data.DEMO_DEBRIS}
    if (
        {int(item["norad_id"]) for item in debris_list} != demo_ids
        or demo_data._LOADED_KEY != demo_data._cache_key(sat)
    ):
        load_demo_data()
        debris_list = [obj.to_dict() for obj in get_catalog().list_debris()]
    signature = json.dumps({"sat": [sat.position.tolist(), sat.velocity.tolist()], "debris": debris_list}, sort_keys=True)
    if demo_data._SCAN_CACHE and demo_data._SCAN_CACHE[0] == signature:
        return demo_data._SCAN_CACHE[1]

    events = screen_conjunctions(
        primary_pos=sat.position,
        primary_vel=sat.velocity,
        debris_list=debris_list,
        lookahead_sec=86400,
        threshold_km=50.0,
    )
    shown, total = _compact_events(events)
    record_conjunction_events(shown)
    result = json.dumps({
        "demo_loaded": True,
        "debris_count": len(debris_list),
        "events_found": len(events),
        "total_encounters_found": total,
        "events_shown": len(shown),
        "events": shown,
        "satellite": {"norad_id": sat.norad_id, "name": sat.name},
    }, indent=2)
    demo_data._SCAN_CACHE = (signature, result)
    return result


# ─────────────────────────────────────────────────────────────────────────
# Risk assessment tools (Agent 0 — Analyst)
# ─────────────────────────────────────────────────────────────────────────

@tool
def assess_risk(
    secondary_id: int,
    miss_distance_m: Optional[float] = None,
    tca_offset_sec: Optional[float] = None,
) -> str:
    """
    Compute detailed collision probability and risk for a conjunction event.

    Args:
        secondary_id: NORAD ID of the debris object
        miss_distance_m: known miss distance (optional, computed if not given)
        tca_offset_sec: time to closest approach (optional, read from CDM or screened)
    """
    recommended = get_recommended_maneuver()
    if recommended and recommended.get("executed"):
        secondary_id = int(recommended["secondary_id"])
    sat = get_satellite()
    catalog = get_catalog()
    obj = catalog.get(secondary_id)
    if obj is None:
        return json.dumps({"error": f"Object {secondary_id} not found in catalog"})

    if tca_offset_sec is None:
        cdm = next(
            (message for message in reversed(get_cdm_inbox().get_all())
             if message.get("secondary_id") == secondary_id),
            None,
        )
        if cdm is not None:
            tca_offset_sec = float(cdm["tca_offset_sec"])
        else:
            events = screen_conjunctions(
                primary_pos=sat.position,
                primary_vel=sat.velocity,
                debris_list=[obj.to_dict()],
                lookahead_sec=86400.0,
                threshold_km=50000.0,
            )
            if events:
                tca_offset_sec = float(events[0]["tca_offset_sec"])
            else:
                tca_offset_sec = 0.0

    result = assess_conjunction_risk(
        primary_pos=sat.position,
        primary_vel=sat.velocity,
        secondary_pos=obj.position,
        secondary_vel=obj.velocity,
        miss_distance_m=miss_distance_m,
        tca_offset_sec=tca_offset_sec,
    )
    result["secondary_id"] = secondary_id
    result["secondary_name"] = obj.name
    if recommended and recommended.get("executed"):
        result["miss_distance_before_m"] = recommended.get("miss_before")
        result["miss_distance_after_m"] = result.get("miss_distance_m")
        result["chan_probability_before"] = recommended.get("chan_probability")
        result["chan_probability_after"] = result.get("collision_probability_chan")
        update_recommended_maneuver(
            miss_after_actual_m=result.get("miss_distance_m"),
            chan_probability_after=result.get("collision_probability_chan"),
        )
    record_risk_assessment(secondary_id, result)
    return json.dumps(result, indent=2, default=str)


@tool
def refine_conjunction_hifi(secondary_id: int, window_sec: float = 3600.0) -> str:
    """
    Run Engine2 high-fidelity propagation (RK45 + J2/J3/J4 + drag) for TCA refinement.

    Args:
        secondary_id: NORAD ID of the debris object
        window_sec: propagation window in seconds
    """
    sat = get_satellite()
    catalog = get_catalog()
    obj = catalog.get(secondary_id)
    if obj is None:
        return json.dumps({"error": f"Object {secondary_id} not found in catalog"})

    result = refine_conjunction(
        primary_pos=sat.position,
        primary_vel=sat.velocity,
        secondary_pos=obj.position,
        secondary_vel=obj.velocity,
        window_sec=window_sec,
    )
    result["secondary_id"] = secondary_id
    return json.dumps(result, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────
# Maneuver planning tools (Agent 1 — Planner)
# ─────────────────────────────────────────────────────────────────────────

@tool
def propose_avoidance_maneuvers(secondary_id: int) -> str:
    """Search a deterministic delta-v grid for the highest-ranked threat."""
    started = time.perf_counter()
    sat = get_satellite()
    catalog = get_catalog()
    top = get_top_threat()
    requested_secondary_id = secondary_id
    if top is None:
        return json.dumps({"error": "No ranked threat is available from Scout and Analyst", "candidates": [], "no_improving_candidate": True})
    secondary_id = int(top["secondary_id"])
    tca_offset_sec = float(top.get("tca_offset_sec", top.get("tca_sec", 0.0)))
    miss_distance_m = float(top.get("miss_distance_m", 0.0))
    obj = catalog.get(secondary_id)
    if obj is None:
        return json.dumps({"error": f"Object {secondary_id} not found in catalog"})

    radial = sat.position / np.linalg.norm(sat.position)
    along = sat.velocity - np.dot(sat.velocity, radial) * radial
    along = along / np.linalg.norm(along)
    cross = np.cross(radial, along)
    cross = cross / np.linalg.norm(cross)
    directions = (
        ("along-track", along), ("along-track", -along),
        ("radial", radial), ("radial", -radial),
        ("cross-track", cross), ("cross-track", -cross),
    )
    max_dv = float(sat.config.max_dv_per_burn_ms)
    magnitudes = sorted({float(value) for value in (0.05, 0.1, 0.2, max_dv) if 0 < value <= max_dv})
    # Resolve the TCA neighborhood to keep the grid search focused on the event.
    duration_sec = min(86400.0, max(600.0, float(tca_offset_sec) + 300.0))
    evaluated = []
    for maneuver_type, direction in directions:
        for magnitude in magnitudes:
            dv = (direction * magnitude).tolist()
            constraints = check_constraints(
                delta_v=dv, primary_position=sat.position, primary_velocity=sat.velocity,
                mass_kg=sat.total_mass, isp_s=sat.config.isp_s,
                remaining_fuel_kg=sat.fuel_kg, burn_time_sec=0.0, max_dv_mps=max_dv,
            )
            if not constraints["overall_pass"]:
                continue
            result = simulate_maneuver(
                primary_pos=sat.position, primary_vel=sat.velocity,
                secondary_pos=obj.position, secondary_vel=obj.velocity,
                delta_v=dv, burn_time_sec=0.0, duration_sec=duration_sec,
            )
            before = float(result["before"]["miss_distance_m"])
            after = float(result["after"]["miss_distance_m"])
            improvement = after - before
            if improvement > 0:
                evaluated.append({
                    "type": maneuver_type,
                    "delta_v_vector": dv,
                    "delta_v_magnitude_ms": round(magnitude, 4),
                    "burn_time_sec": 0.0,
                    "miss_before": before,
                    "miss_after": after,
                    "predicted_miss_distance_m": after,
                    "miss_improvement_m": round(improvement, 1),
                    "constraints": constraints,
                })
    evaluated.sort(key=lambda candidate: (-candidate["miss_improvement_m"], candidate["delta_v_magnitude_ms"]))
    candidates = evaluated[:3]
    for index, candidate in enumerate(candidates):
        candidate["candidate_index"] = index
    elapsed = round(time.perf_counter() - started, 3)
    search = {
        "secondary_id": int(secondary_id),
        "tca_offset_sec": float(tca_offset_sec),
        "miss_distance_m": float(miss_distance_m),
        "candidates": candidates,
        "runtime_sec": elapsed,
        "no_improving_candidate": not candidates,
    }
    top_probability = float(top.get("collision_probability_chan", 0.0))
    search["chan_probability"] = top_probability
    set_last_maneuver_search(search)
    if not candidates:
        search["message"] = "no improving candidate within constraints"
    if requested_secondary_id != secondary_id:
        search["requested_secondary_id"] = int(requested_secondary_id)
        search["selection"] = "used top threat by Chan probability, then miss distance"
    return json.dumps(search, indent=2, default=str)


@tool
def record_recommended_maneuver(secondary_id: int, candidate_index: int) -> str:
    """Store a candidate by its zero-based index from the latest Planner search."""
    search = get_last_maneuver_search()
    if not search or int(secondary_id) != int(search["secondary_id"]):
        return json.dumps({"error": "secondary_id does not match the latest maneuver search"})
    candidates = search.get("candidates", [])
    if not isinstance(candidate_index, int) or candidate_index < 0 or candidate_index >= len(candidates):
        return json.dumps({"error": "candidate_index is outside the improving candidate list"})
    candidate = candidates[candidate_index]
    record = {
        "secondary_id": int(secondary_id),
        "delta_v_vector": list(candidate["delta_v_vector"]),
        "type": candidate["type"],
        "miss_before": candidate["miss_before"],
        "miss_after": candidate["miss_after"],
        "chan_probability": float(search.get("chan_probability", 0.0)),
        "verdict": None,
        "reason": None,
        "candidate_index": candidate_index,
        "burn_time_sec": candidate["burn_time_sec"],
        "executed": False,
    }
    set_recommended_maneuver(record)
    return json.dumps({"recommended_maneuver": get_recommended_maneuver()}, indent=2)


@tool
def simulate_maneuver_effect(
    secondary_id: int,
    delta_v: List[float],
    burn_time_sec: float,
) -> str:
    """
    Simulate a specific maneuver and compute before/after miss distance.

    Args:
        secondary_id: NORAD ID of the debris object
        delta_v: delta-v vector [x, y, z] in m/s (ECI)
        burn_time_sec: when to apply the burn (seconds from now)
    """
    sat = get_satellite()
    catalog = get_catalog()
    obj = catalog.get(secondary_id)
    if obj is None:
        return json.dumps({"error": f"Object {secondary_id} not found in catalog"})

    result = simulate_maneuver(
        primary_pos=sat.position,
        primary_vel=sat.velocity,
        secondary_pos=obj.position,
        secondary_vel=obj.velocity,
        delta_v=delta_v,
        burn_time_sec=burn_time_sec,
    )
    return json.dumps(result, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────
# Constraint checking tools (Agent 2 — Safety)
# ─────────────────────────────────────────────────────────────────────────

@tool
def get_satellite_status() -> str:
    """Get current satellite telemetry: fuel, power, position, delta-v budget."""
    sat = get_satellite()
    status = sat.get_status()
    status["fuel_kg"] = round(sat.fuel_kg, 2)
    status["fuel_pct"] = round(sat.fuel_percentage, 2)
    return json.dumps(status, indent=2, default=str)


@tool
def check_maneuver_constraints() -> str:
    """Validate the stored recommended maneuver against operational constraints."""
    recommended = get_recommended_maneuver()
    if not recommended:
        return json.dumps({"overall_pass": False, "error": "No recommended maneuver is stored"})
    sat = get_satellite()
    result = check_constraints(
        delta_v=recommended["delta_v_vector"],
        primary_position=sat.position,
        primary_velocity=sat.velocity,
        mass_kg=sat.total_mass,
        isp_s=sat.config.isp_s,
        remaining_fuel_kg=sat.fuel_kg,
        burn_time_sec=float(recommended.get("burn_time_sec", 0.0)),
        max_dv_mps=sat.config.max_dv_per_burn_ms,
    )
    result["checked_delta_v_vector"] = recommended["delta_v_vector"]
    record_maneuver_check("constraints", result)
    return json.dumps(result, indent=2, default=str)


@tool
def check_maneuver_feasibility() -> str:
    """Check feasibility for the stored recommended maneuver."""
    recommended = get_recommended_maneuver()
    if not recommended:
        return json.dumps({"feasible": False, "error": "No recommended maneuver is stored"})
    sat = get_satellite()
    delta_v = recommended["delta_v_vector"]
    dv_mag = float(np.linalg.norm(delta_v))
    ve = sat.config.isp_s * 9.80665
    fuel_needed = sat.total_mass * (1 - np.exp(-dv_mag / ve))
    feasible = fuel_needed <= max(0.0, sat.fuel_kg - 1.0) and dv_mag <= sat.config.max_dv_per_burn_ms and sat.is_operational
    result = {
        "feasible": feasible,
        "delta_v_ms": round(dv_mag, 4),
        "fuel_required_kg": round(float(fuel_needed), 4),
        "fuel_available_kg": round(sat.fuel_kg, 4),
        "reason": "OK" if feasible else ("Insufficient fuel or satellite not operational" if fuel_needed > max(0.0, sat.fuel_kg - 1.0) or not sat.is_operational else "Exceeds max delta-v"),
        "checked_delta_v_vector": delta_v,
    }
    record_maneuver_check("feasibility", result)
    return json.dumps(result, indent=2)


@tool
def record_safety_verdict(verdict: str, reason: str) -> str:
    """Record Safety's verdict; APPROVED requires both stored-vector checks to pass."""
    recommended = get_recommended_maneuver()
    if not recommended:
        return json.dumps({"error": "No recommended maneuver is stored", "verdict": "REJECTED"})
    requested = verdict.strip().upper()
    if requested not in {"APPROVED", "CONDITIONAL", "REJECTED"}:
        requested = "REJECTED"
        reason = f"Invalid verdict value. {reason}".strip()
    checks = get_maneuver_checks()
    all_checks_pass = (
        checks.get("feasibility", {}).get("feasible") is True
        and checks.get("constraints", {}).get("overall_pass") is True
    )
    if requested == "APPROVED" and not all_checks_pass:
        requested = "REJECTED"
        reason = f"Safety checks missing or failed. {reason}".strip()
    updated = update_recommended_maneuver(verdict=requested, reason=reason)
    return json.dumps({"recommended_maneuver": updated}, indent=2)


# ─────────────────────────────────────────────────────────────────────────
# Execution tools (Agent 3 — Ops)
# ─────────────────────────────────────────────────────────────────────────

@tool
def execute_maneuver_on_satellite(delta_v: List[float]) -> str:
    """
    Execute a maneuver by applying delta-v to the active satellite.

    Args:
        delta_v: delta-v vector [x, y, z] in m/s (ECI)
    """
    sat = get_satellite()
    dv = np.array(delta_v, dtype=float)
    recommended = get_recommended_maneuver()
    if recommended:
        expected = np.asarray(recommended["delta_v_vector"], dtype=float)
        if not np.array_equal(dv, expected):
            return json.dumps({"error": "Delta-v differs from the stored recommended maneuver", "executed": False, "fuel_used_kg": 0.0})
        if recommended.get("verdict") != "APPROVED" or recommended.get("executed"):
            return json.dumps({"error": "Stored maneuver is not approved or was already executed", "executed": False, "fuel_used_kg": 0.0})

    safety = check_constraints(
        delta_v=dv.tolist(), primary_position=sat.position, primary_velocity=sat.velocity,
        mass_kg=sat.total_mass, isp_s=sat.config.isp_s, remaining_fuel_kg=sat.fuel_kg,
        burn_time_sec=0.0, max_dv_mps=sat.config.max_dv_per_burn_ms,
    )
    if not safety["overall_pass"]:
        return json.dumps({"error": "Maneuver rejected by execution safety gate", "executed": False,
                           "fuel_used_kg": 0.0, "checks": safety}, indent=2, default=str)

    # Apply maneuver (handles fuel, power, velocity update)
    result = sat.apply_maneuver(dv)
    result.setdefault("fuel_used_kg", 0.0)
    if result.get("executed"):
        update_recommended_maneuver(executed=True, fuel_used_kg=result["fuel_used_kg"])
    return json.dumps(result, indent=2, default=str)


@tool
def execute_recommended_maneuver() -> str:
    """Execute the stored recommendation only after Safety has APPROVED it."""
    recommended = get_recommended_maneuver()
    if not recommended:
        return json.dumps({"error": "No recommended maneuver is stored", "executed": False, "fuel_used_kg": 0.0})
    if recommended.get("verdict") != "APPROVED":
        return json.dumps({"error": "Recommended maneuver has no APPROVED Safety verdict", "executed": False, "fuel_used_kg": 0.0, "recommended_maneuver": recommended})
    if recommended.get("executed"):
        return json.dumps({"error": "Recommended maneuver was already executed", "executed": False, "fuel_used_kg": 0.0, "recommended_maneuver": recommended})
    sat = get_satellite()
    dv = np.asarray(recommended["delta_v_vector"], dtype=float)
    feasibility = {
        "feasible": sat.can_perform_maneuver(float(np.linalg.norm(dv))),
        "delta_v_vector": dv.tolist(),
    }
    if not feasibility["feasible"]:
        return json.dumps({"error": "Maneuver rejected by execution feasibility gate", "executed": False,
                           "fuel_used_kg": 0.0, "feasibility": feasibility})
    checks = check_constraints(
        delta_v=dv.tolist(), primary_position=sat.position, primary_velocity=sat.velocity,
        mass_kg=sat.total_mass, isp_s=sat.config.isp_s, remaining_fuel_kg=sat.fuel_kg,
        burn_time_sec=float(recommended.get("burn_time_sec", 0.0)), max_dv_mps=sat.config.max_dv_per_burn_ms,
    )
    if not checks.get("overall_pass"):
        return json.dumps({"error": "Maneuver rejected by execution safety gate", "executed": False,
                           "fuel_used_kg": 0.0, "feasibility": feasibility, "checks": checks})
    result = sat.apply_maneuver(dv)
    result.setdefault("fuel_used_kg", 0.0)
    if result.get("executed"):
        update_recommended_maneuver(executed=True, fuel_used_kg=result["fuel_used_kg"], execution=result)
    return json.dumps(result, indent=2, default=str)


@tool
def propagate_satellite_orbit(duration_hours: float = 1.5) -> str:
    """
    Propagate the satellite orbit forward to verify trajectory.

    Args:
        duration_hours: how far ahead to propagate (hours)
    """
    sat = get_satellite()
    result = propagate_orbit(
        position=sat.position,
        velocity=sat.velocity,
        duration_sec=duration_hours * 3600,
        dt=60.0,
    )
    # Return summary, not full trajectory
    return json.dumps({
        "duration_hours": duration_hours,
        "total_points": result["total_points"],
        "start_altitude_km": result["start_altitude_km"],
        "end_altitude_km": result["end_altitude_km"],
        "altitude_range_km": [min(result["altitudes_km"]), max(result["altitudes_km"])],
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────────
# Tool groups for each agent
# ─────────────────────────────────────────────────────────────────────────

SCOUT_TOOLS = [
    get_pending_cdms,
    scan_conjunctions,
    scan_demo_conjunctions,
]

ANALYST_TOOLS = [
    get_pending_cdms,
    scan_conjunctions,
    scan_demo_conjunctions,
    assess_risk,
    refine_conjunction_hifi,
]

PLANNER_TOOLS = [propose_avoidance_maneuvers, record_recommended_maneuver]

SAFETY_TOOLS = [
    get_satellite_status,
    check_maneuver_constraints,
    check_maneuver_feasibility,
    record_safety_verdict,
    propagate_satellite_orbit,
]

ALL_TOOLS = list({id(t): t for group in [SCOUT_TOOLS, ANALYST_TOOLS, PLANNER_TOOLS, SAFETY_TOOLS,
                                   [execute_maneuver_on_satellite, execute_recommended_maneuver, propagate_satellite_orbit]] for t in group}.values())
