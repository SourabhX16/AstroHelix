"""Utilities for constructing deterministic conjunction scenarios."""
from __future__ import annotations

import numpy as np

from tools.screening import propagate_state


def build_conjunction_debris(
    sat_pos: np.ndarray,
    sat_vel: np.ndarray,
    tca_sec: float,
    miss_m: float,
    cross_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an epoch debris state whose propagated TCA miss matches the design."""
    satellite_tca = propagate_state(sat_pos, sat_vel, tca_sec, dt=2.0)
    radial_unit = satellite_tca.r / np.linalg.norm(satellite_tca.r)
    angle = np.radians(cross_deg)
    debris_velocity_tca = (
        satellite_tca.v * np.cos(angle)
        + np.cross(radial_unit, satellite_tca.v) * np.sin(angle)
        + radial_unit * np.dot(radial_unit, satellite_tca.v) * (1.0 - np.cos(angle))
    )
    miss_direction = np.cross(debris_velocity_tca - satellite_tca.v, radial_unit)
    miss_direction /= np.linalg.norm(miss_direction)
    debris_position_tca = satellite_tca.r + miss_m * miss_direction

    debris_epoch = propagate_state(debris_position_tca, debris_velocity_tca, -tca_sec, dt=2.0)
    return debris_epoch.r, debris_epoch.v
