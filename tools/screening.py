"""Conjunction screening using RK4 propagation and TCA refinement."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np

from engine.physics.state import State
from engine.physics.forces import NewtonianGravity, J2Perturbation, CompositeForce
from engine.physics.solver import RK4Solver

_FORCE = CompositeForce(NewtonianGravity(), J2Perturbation())


def propagate_state(position: np.ndarray, velocity: np.ndarray, duration_sec: float, dt: float = 1.0) -> State:
    """Propagate a state with the screening force model, including partial final steps."""
    position_key = tuple(float(value) for value in position)
    velocity_key = tuple(float(value) for value in velocity)
    propagated_position, propagated_velocity = _propagate_state_cached(
        position_key,
        velocity_key,
        float(duration_sec),
        float(dt),
    )
    return State(np.array(propagated_position), np.array(propagated_velocity))


@lru_cache(maxsize=64)
def _propagate_state_cached(
    position: Tuple[float, float, float],
    velocity: Tuple[float, float, float],
    duration_sec: float,
    dt: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    state = State(np.array(position), np.array(velocity))
    solver = RK4Solver(_FORCE)
    direction = 1.0 if duration_sec >= 0 else -1.0
    remaining = abs(float(duration_sec))
    while remaining > 1e-9:
        step = direction * min(dt, remaining)
        state = solver.step(state, step)
        remaining -= abs(step)
    return tuple(state.r), tuple(state.v)


def _propagate_pair(
    primary_pos: np.ndarray,
    primary_vel: np.ndarray,
    secondary_pos: np.ndarray,
    secondary_vel: np.ndarray,
    duration_sec: float,
    dt: float,
) -> Tuple[State, State]:
    return (
        propagate_state(primary_pos, primary_vel, duration_sec, dt),
        propagate_state(secondary_pos, secondary_vel, duration_sec, dt),
    )


def _range_rate(primary: State, secondary: State) -> float:
    return float(np.dot(primary.r - secondary.r, primary.v - secondary.v))


def _refine_tca(
    primary_start: State,
    secondary_start: State,
    left_sec: float,
    right_sec: float,
) -> Tuple[float, State, State]:
    bracket_width = right_sec - left_sec
    right_state = _propagate_pair(primary_start.r, primary_start.v, secondary_start.r, secondary_start.v, bracket_width, 0.5)
    left_rate = _range_rate(primary_start, secondary_start)
    right_rate = _range_rate(*right_state)

    if left_rate > 0 or right_rate < 0:
        states = _propagate_pair(primary_start.r, primary_start.v, secondary_start.r, secondary_start.v, bracket_width / 2.0, 0.5)
        return left_sec + bracket_width / 2.0, *states

    left_offset = 0.0
    right_offset = bracket_width
    while right_offset - left_offset > 0.001:
        middle_offset = (left_offset + right_offset) / 2.0
        states = _propagate_pair(
            primary_start.r, primary_start.v,
            secondary_start.r, secondary_start.v,
            middle_offset, 0.5,
        )
        if _range_rate(*states) < 0:
            left_offset = middle_offset
        else:
            right_offset = middle_offset
            right_state = states

    tca_offset = (left_offset + right_offset) / 2.0
    states = _propagate_pair(
        primary_start.r, primary_start.v,
        secondary_start.r, secondary_start.v,
        tca_offset, 0.5,
    )
    return left_sec + tca_offset, *states


def screen_conjunctions(
    primary_pos: np.ndarray,
    primary_vel: np.ndarray,
    debris_list: List[Dict[str, Any]],
    lookahead_sec: float = 86400.0,
    threshold_km: float = 50.0,
    dt: float = 10.0,
    refine: bool = True,
) -> List[Dict[str, Any]]:
    """Screen for close approaches and optionally refine each candidate TCA."""
    threshold_m = threshold_km * 1000.0
    steps = int(np.ceil(lookahead_sec / dt))
    times = [min(i * dt, lookahead_sec) for i in range(steps + 1)]
    primary_states = [State(np.asarray(primary_pos, dtype=float), np.asarray(primary_vel, dtype=float))]
    primary_solver = RK4Solver(_FORCE)
    for index in range(steps):
        step = times[index + 1] - times[index]
        primary_states.append(primary_solver.step(primary_states[-1], step))

    events: List[Dict[str, Any]] = []
    for debris in debris_list:
        secondary_states = [State(np.asarray(debris["position"], dtype=float), np.asarray(debris["velocity"], dtype=float))]
        secondary_solver = RK4Solver(_FORCE)
        for index in range(steps):
            step = times[index + 1] - times[index]
            secondary_states.append(secondary_solver.step(secondary_states[-1], step))

        relative_positions = [p.r - s.r for p, s in zip(primary_states, secondary_states)]
        distances = [float(np.linalg.norm(relative)) for relative in relative_positions]
        range_rates = [_range_rate(p, s) for p, s in zip(primary_states, secondary_states)]
        speeds = [float(np.linalg.norm(p.v - s.v)) for p, s in zip(primary_states, secondary_states)]
        min_index = int(np.argmin(distances))

        candidate_indices = set()
        for index in range(steps):
            if range_rates[index] <= 0.0 < range_rates[index + 1]:
                candidate_indices.add(index)
        if not any(abs(index - min_index) <= 1 for index in candidate_indices):
            candidate_indices.add(max(0, min(min_index - 1, steps - 1)))

        for index in sorted(candidate_indices):
            allowance = 0.5 * max(speeds[index], speeds[index + 1]) * (times[index + 1] - times[index])
            if min(distances[index], distances[index + 1]) > threshold_m + allowance and index != min_index:
                continue

            if refine:
                tca_sec, primary_tca, secondary_tca = _refine_tca(
                    primary_states[index],
                    secondary_states[index],
                    times[index],
                    times[index + 1],
                )
                miss_distance = float(np.linalg.norm(primary_tca.r - secondary_tca.r))
                relative_speed = float(np.linalg.norm(primary_tca.v - secondary_tca.v))
            else:
                sample_index = min_index if index == min_index else index
                tca_sec = times[sample_index]
                miss_distance = distances[sample_index]
                relative_speed = speeds[sample_index]

            if miss_distance > threshold_m:
                continue
            events.append({
                "secondary_id": debris.get("norad_id", 0),
                "secondary_name": debris.get("name", "UNKNOWN"),
                "miss_distance_m": round(miss_distance, 1),
                "relative_velocity_ms": round(relative_speed, 1),
                "tca_offset_sec": round(tca_sec, 1),
                "risk_level": _classify_risk(miss_distance, relative_speed),
                "probability_estimate": _quick_probability(miss_distance),
            })

    events.sort(key=lambda event: event["miss_distance_m"])
    return events


def _classify_risk(miss_m: float, rel_vel_ms: float) -> str:
    if miss_m < 500:
        return "critical"
    if miss_m < 2000:
        return "high"
    if miss_m < 5000:
        return "medium"
    return "low"


def _quick_probability(miss_m: float) -> float:
    return float(np.exp(-0.5 * (miss_m / 200.0) ** 2))
