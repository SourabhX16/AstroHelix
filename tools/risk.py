"""
risk.py — Detailed risk assessment using Chan collision probability.

Computes composite probability of collision (PoC), covariance integration,
and risk scoring for conjunction events.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from engine.config.settings import GM, RE, COLLISION_RADIUS
from engine.physics.chan_probability import chan_collision_probability
from engine.physics.entity import Entity
from engine.physics.state import State
from tools.screening import propagate_state, screen_conjunctions

logger = logging.getLogger("astro_helix.tools.risk")


def assess_conjunction_risk(
    primary_pos: np.ndarray,
    primary_vel: np.ndarray,
    secondary_pos: np.ndarray,
    secondary_vel: np.ndarray,
    covariance_diag: Optional[list] = None,
    miss_distance_m: Optional[float] = None,
    collision_radius: float = COLLISION_RADIUS,
    tca_offset_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute detailed risk assessment for a conjunction event.

    Uses Chan B-plane collision probability with covariance integration.

    Args:
        primary_pos/vel: satellite state (ECI, meters, m/s)
        secondary_pos/vel: debris state (ECI, meters, m/s)
        covariance_diag: [σx², σy², σz²] position covariance diagonal (m²)
        miss_distance_m: known miss distance (if pre-computed)
        collision_radius: combined object radii (m)

    Returns:
        Risk assessment dict with PoC, risk level, recommendations
    """
    p_pos = np.array(primary_pos, dtype=float)
    p_vel = np.array(primary_vel, dtype=float)
    s_pos = np.array(secondary_pos, dtype=float)
    s_vel = np.array(secondary_vel, dtype=float)

    if tca_offset_sec is None:
        events = screen_conjunctions(
            primary_pos=p_pos,
            primary_vel=p_vel,
            debris_list=[{"position": s_pos, "velocity": s_vel}],
            lookahead_sec=86400.0,
            threshold_km=50000.0,
        )
        tca_sec = float(events[0]["tca_offset_sec"]) if events else 0.0
    else:
        tca_sec = float(tca_offset_sec)
    primary_tca = propagate_state(p_pos, p_vel, tca_sec, dt=1.0)
    secondary_tca = propagate_state(s_pos, s_vel, tca_sec, dt=1.0)
    rel_pos = secondary_tca.r - primary_tca.r
    rel_vel = secondary_tca.v - primary_tca.v
    dist = float(np.linalg.norm(rel_pos))
    rel_speed = float(np.linalg.norm(rel_vel))

    if miss_distance_m is not None and dist > 0 and abs(float(miss_distance_m) - dist) / dist > 0.1:
        logger.warning(
            "Supplied miss distance %.1f m differs from propagated TCA miss %.1f m; using propagated value",
            miss_distance_m,
            dist,
        )

    # Build covariance matrix
    if covariance_diag is not None:
        cov = np.diag(np.array(covariance_diag, dtype=float))
    else:
        # Default covariance based on object type
        sigma = max(50.0, dist * 0.1)  # 10% of miss distance, min 50m
        cov = np.diag([sigma**2, sigma**2, sigma**2])

    # Chan collision probability
    chan_status = "ok"
    try:
        poc = chan_collision_probability(
            rel_pos=rel_pos,
            rel_vel=rel_vel,
            cov_rel=cov,
            collision_radius=collision_radius,
        )
    except Exception as e:
        logger.warning("Chan PoC failed, using Gaussian fallback: %s", e)
        sigma_eff = np.sqrt(np.trace(cov) / 3.0)
        poc = float(np.exp(-0.5 * (dist / max(sigma_eff, 1.0)) ** 2))
        chan_status = "fallback"

    # Crude distance likelihood for display; it is not a collision probability.
    sigma_eff = np.sqrt(np.trace(cov) / 3.0)
    gaussian_prob = float(np.exp(-0.5 * (dist / max(sigma_eff, 1.0)) ** 2))

    # Composite risk score (0-1)
    # Weighted combination of PoC, miss distance, and relative velocity
    dist_score = max(0, 1.0 - dist / 10000.0)  # 0 at 10km, 1 at 0
    vel_score = min(1.0, rel_speed / 15000.0)   # normalized to ~15 km/s max
    poc_score = min(1.0, poc * 1000.0)           # scale up small probabilities

    risk_score = 0.5 * poc_score + 0.3 * dist_score + 0.2 * vel_score
    risk_score = max(0.0, min(1.0, risk_score))

    # Risk classification is based on the Chan collision probability.
    if poc > 1e-4:
        risk_level = "critical"
        recommendation = "IMMEDIATE maneuver required"
    elif poc > 1e-5:
        risk_level = "high"
        recommendation = "Plan avoidance maneuver"
    elif poc > 1e-6:
        risk_level = "medium"
        recommendation = "Monitor closely, prepare contingency"
    else:
        risk_level = "low"
        recommendation = "Continue monitoring"

    return {
        "miss_distance_m": round(dist, 1),
        "relative_velocity_ms": round(rel_speed, 1),
        "collision_probability_chan": float(f"{poc:.2e}"),
        "collision_probability_gaussian": float(f"{gaussian_prob:.2e}"),
        "chan_status": chan_status,
        "tca_offset_sec": round(tca_sec, 1),
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "covariance_trace_m2": round(float(np.trace(cov)), 1),
        "collision_radius_m": collision_radius,
        "analysis": {
            "distance_score": round(dist_score, 3),
            "velocity_score": round(vel_score, 3),
            "poc_score": round(poc_score, 3),
        },
    }
