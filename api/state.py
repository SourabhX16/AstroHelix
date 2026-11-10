"""
Global state management for the Astro Helix system.

Holds the orbital catalog, active satellite, and CDM inbox.
Thread-safe singleton pattern for use by both API routes and agent tools.
"""
from __future__ import annotations

import threading
import copy
from typing import Any, Dict, List, Optional

import numpy as np

from engine.models.active_satellite import Satellite, SatelliteConfig, create_default_satellite
from engine.physics.entity import Entity
from engine.config.settings import RE, GM


class OrbitalObject:
    """
    A tracked orbital object (satellite or debris) in the catalog.
    Wraps position/velocity state with metadata.
    """

    def __init__(
        self,
        norad_id: int,
        name: str,
        position: np.ndarray,
        velocity: np.ndarray,
        object_type: str = "debris",
        rcs_m2: float = 1.0,
        mass_kg: float = 10.0,
    ):
        self.norad_id = norad_id
        self.name = name
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.object_type = object_type
        self.rcs_m2 = rcs_m2
        self.mass_kg = mass_kg

    def to_entity(self) -> Entity:
        """Convert to physics Entity for engine computations."""
        return Entity(position=self.position.copy(), velocity=self.velocity.copy())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "norad_id": self.norad_id,
            "name": self.name,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "object_type": self.object_type,
            "rcs_m2": self.rcs_m2,
            "mass_kg": self.mass_kg,
        }


class OrbitalCatalog:
    """
    In-memory catalog of tracked orbital objects.
    In production this would be backed by Space-Track CDMs or TLEs.
    """

    def __init__(self):
        self._objects: Dict[int, OrbitalObject] = {}
        self._lock = threading.Lock()

    def add(self, obj: OrbitalObject) -> None:
        with self._lock:
            self._objects[obj.norad_id] = obj

    def get(self, norad_id: int) -> Optional[OrbitalObject]:
        with self._lock:
            return self._objects.get(norad_id)

    def remove(self, norad_id: int) -> None:
        with self._lock:
            self._objects.pop(norad_id, None)

    def list_all(self) -> List[OrbitalObject]:
        with self._lock:
            return list(self._objects.values())

    def list_debris(self) -> List[OrbitalObject]:
        with self._lock:
            return [o for o in self._objects.values() if o.object_type == "debris"]

    def count(self) -> int:
        with self._lock:
            return len(self._objects)


class CDMInbox:
    """
    Conjunction Data Message inbox.
    Stores incoming CDMs for processing by Agent 0.
    """

    def __init__(self):
        self._messages: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, cdm: Dict[str, Any]) -> None:
        with self._lock:
            self._messages.append(cdm)

    def get_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._messages)

    def get_pending(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [m for m in self._messages if not m.get("processed", False)]

    def mark_processed(self, index: int) -> None:
        with self._lock:
            if 0 <= index < len(self._messages):
                self._messages[index]["processed"] = True

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()


# ── Singleton state ──────────────────────────────────────────────────────
_catalog: Optional[OrbitalCatalog] = None
_satellite: Optional[Satellite] = None
_cdm_inbox: Optional[CDMInbox] = None
_init_lock = threading.Lock()
_maneuver_lock = threading.RLock()
_maneuver_pipeline: Dict[str, Any] = {
    "events": {}, "risk_assessments": {}, "last_search": None,
    "recommended_maneuver": None, "checks": {},
    "constraints": {"max_dv_ms": 0.35, "max_burns": 1, "axis": "along-track", "horizon_h": 24.0},
    "scan_events": None,
}


def reset_maneuver_pipeline_state() -> None:
    """Clear handoff and threat records at the start of an agent run."""
    with _maneuver_lock:
        _maneuver_pipeline.update({
            "events": {}, "risk_assessments": {}, "last_search": None,
            "recommended_maneuver": None, "checks": {},
            "scan_events": None,
        })


def record_conjunction_events(events: List[Dict[str, Any]]) -> None:
    with _maneuver_lock:
        deduped = {}
        for event in events:
            sid = event.get("secondary_id")
            if sid is not None:
                current = deduped.get(int(sid))
                if current is None or float(event.get("miss_distance_m", float("inf"))) < float(current.get("miss_distance_m", float("inf"))):
                    deduped[int(sid)] = copy.deepcopy(event)
        _maneuver_pipeline["scan_events"] = deduped
        _maneuver_pipeline["events"] = copy.deepcopy(deduped)


def get_scan_event(secondary_id: int) -> Optional[Dict[str, Any]]:
    with _maneuver_lock:
        events = _maneuver_pipeline.get("scan_events")
        event = events.get(int(secondary_id)) if events is not None else None
        return copy.deepcopy(event)


def record_risk_assessment(secondary_id: int, assessment: Dict[str, Any]) -> None:
    with _maneuver_lock:
        _maneuver_pipeline["risk_assessments"][int(secondary_id)] = copy.deepcopy(assessment)


def get_top_threat() -> Optional[Dict[str, Any]]:
    with _maneuver_lock:
        ranked = []
        for sid, event in _maneuver_pipeline["events"].items():
            risk = _maneuver_pipeline["risk_assessments"].get(sid, {})
            probability = risk.get("collision_probability_chan", event.get("collision_probability_chan", event.get("chan_probability", 0.0)))
            ranked.append((float(probability or 0.0), float(event.get("miss_distance_m", float("inf"))), sid, event, risk))
        if not ranked:
            return None
        probability, miss, sid, event, risk = sorted(ranked, key=lambda row: (-row[0], row[1]))[0]
        result = copy.deepcopy(event)
        result.update({"secondary_id": sid, "miss_distance_m": miss, "collision_probability_chan": probability})
        if risk:
            result["risk_assessment"] = copy.deepcopy(risk)
        return result


def set_last_maneuver_search(search: Dict[str, Any]) -> None:
    with _maneuver_lock:
        _maneuver_pipeline["last_search"] = copy.deepcopy(search)


def get_last_maneuver_search() -> Optional[Dict[str, Any]]:
    with _maneuver_lock:
        return copy.deepcopy(_maneuver_pipeline["last_search"])


def set_recommended_maneuver(record: Optional[Dict[str, Any]]) -> None:
    with _maneuver_lock:
        _maneuver_pipeline["recommended_maneuver"] = copy.deepcopy(record)
        _maneuver_pipeline["checks"] = {}


def get_recommended_maneuver() -> Optional[Dict[str, Any]]:
    with _maneuver_lock:
        return copy.deepcopy(_maneuver_pipeline["recommended_maneuver"])


def update_recommended_maneuver(**updates: Any) -> Optional[Dict[str, Any]]:
    with _maneuver_lock:
        record = _maneuver_pipeline["recommended_maneuver"]
        if record is None:
            return None
        record.update(copy.deepcopy(updates))
        return copy.deepcopy(record)


def record_maneuver_check(name: str, result: Dict[str, Any]) -> None:
    with _maneuver_lock:
        _maneuver_pipeline["checks"][name] = copy.deepcopy(result)


def get_maneuver_checks() -> Dict[str, Any]:
    with _maneuver_lock:
        return copy.deepcopy(_maneuver_pipeline["checks"])


def set_pipeline_constraints(constraints: Dict[str, Any]) -> None:
    with _maneuver_lock:
        _maneuver_pipeline["constraints"] = copy.deepcopy(constraints)


def get_pipeline_constraints() -> Dict[str, Any]:
    with _maneuver_lock:
        return copy.deepcopy(_maneuver_pipeline["constraints"])


def get_catalog() -> OrbitalCatalog:
    """Get or create the global orbital catalog."""
    global _catalog
    with _init_lock:
        if _catalog is None:
            _catalog = OrbitalCatalog()
    return _catalog


def get_satellite() -> Satellite:
    """Get or create the active satellite."""
    global _satellite
    with _init_lock:
        if _satellite is None:
            _satellite = create_default_satellite()
    return _satellite


def set_satellite(sat: Satellite) -> None:
    """Replace the active satellite."""
    global _satellite
    with _init_lock:
        _satellite = sat


def get_cdm_inbox() -> CDMInbox:
    """Get or create the CDM inbox."""
    global _cdm_inbox
    with _init_lock:
        if _cdm_inbox is None:
            _cdm_inbox = CDMInbox()
    return _cdm_inbox


def reset_state() -> None:
    """Reset all global state (for testing)."""
    global _catalog, _satellite, _cdm_inbox
    with _init_lock:
        _catalog = None
        _satellite = None
        _cdm_inbox = None
    reset_maneuver_pipeline_state()
