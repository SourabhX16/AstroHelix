import numpy as np

from api.demo_data import DEMO_DEBRIS, load_demo_data
from api.state import get_catalog, get_satellite, reset_state
from engine.physics.chan_probability import chan_collision_probability
from tools.risk import assess_conjunction_risk
from tools.scenario import build_conjunction_debris
from tools.screening import screen_conjunctions


def _debris_state(tca_sec, miss_m=150.0, cross_deg=70.0):
    satellite = get_satellite()
    return build_conjunction_debris(
        satellite.position,
        satellite.velocity,
        tca_sec,
        miss_m,
        cross_deg,
    )


def test_chan_probability_works_and_risk_reports_chan_status():
    probability = chan_collision_probability(
        np.array([150.0, 0.0, 0.0]),
        np.array([0.0, 1000.0, 0.0]),
        np.diag([10000.0, 10000.0, 10000.0]),
        10.0,
    )
    assert probability >= 0.0

    reset_state()
    satellite = get_satellite()
    secondary_pos, secondary_vel = _debris_state(3600.0)
    result = assess_conjunction_risk(
        satellite.position,
        satellite.velocity,
        secondary_pos,
        secondary_vel,
        covariance_diag=[10000.0, 10000.0, 10000.0],
        tca_offset_sec=3600.0,
    )
    assert result["chan_status"] == "ok"


def test_screening_refines_designed_tca_and_miss():
    reset_state()
    satellite = get_satellite()
    for tca_sec in (3600.0, 3605.0, 3607.0):
        secondary_pos, secondary_vel = _debris_state(tca_sec)
        events = screen_conjunctions(
            satellite.position,
            satellite.velocity,
            [{"norad_id": 123, "position": secondary_pos, "velocity": secondary_vel}],
            lookahead_sec=tca_sec + 20.0,
            threshold_km=1.0,
        )
        assert events
        event = min(events, key=lambda value: value["miss_distance_m"])
        assert abs(event["miss_distance_m"] - 150.0) <= max(7.5, 10.0)
        assert abs(event["tca_offset_sec"] - tca_sec) <= 1.0


def test_assess_risk_uses_chan_at_tca():
    reset_state()
    satellite = get_satellite()
    secondary_pos, secondary_vel = _debris_state(3600.0)
    result = assess_conjunction_risk(
        satellite.position,
        satellite.velocity,
        secondary_pos,
        secondary_vel,
        covariance_diag=[10000.0, 10000.0, 10000.0],
        miss_distance_m=150.0,
        tca_offset_sec=3600.0,
    )
    assert result["collision_probability_chan"] > 0.0
    assert result["collision_probability_chan"] != result["collision_probability_gaussian"]


def test_demo_cdms_are_measured_conjunctions():
    reset_state()
    load_demo_data()
    satellite = get_satellite()
    debris = [item.to_dict() for item in get_catalog().list_debris()]
    events = screen_conjunctions(
        satellite.position,
        satellite.velocity,
        debris,
        lookahead_sec=max(item["tca_offset_sec"] for item in DEMO_DEBRIS) + 10.0,
        threshold_km=50.0,
        refine=True,
    )
    for design in DEMO_DEBRIS:
        matching_events = [event for event in events if event["secondary_id"] == design["norad_id"]]
        assert matching_events
        measured = min(
            matching_events,
            key=lambda event: abs(event["tca_offset_sec"] - design["tca_offset_sec"]),
        )
        assert abs(measured["miss_distance_m"] - design["miss_m"]) <= max(
            design["miss_m"] * 0.05, 10.0
        )
