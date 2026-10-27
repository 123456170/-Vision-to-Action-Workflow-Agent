import io
import json
import math
import os
import random
import tempfile
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

st.set_page_config(
    page_title="Vision-to-Action Workflow Agent",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

VIDEO_W = 960
VIDEO_H = 540
FPS = 15
DURATION_S = 8.0
NUM_FRAMES = int(FPS * DURATION_S)
FALLBACK_SIZE = (768, 432)

DEFAULT_OBSTACLES = [
    (0.30, 0.16, 0.14, 0.16),
    (0.62, 0.56, 0.18, 0.14),
    (0.16, 0.70, 0.12, 0.12),
]

CLUTTER_OBSTACLES = DEFAULT_OBSTACLES + [
    (0.48, 0.34, 0.10, 0.10),
    (0.74, 0.22, 0.09, 0.16),
    (0.34, 0.72, 0.16, 0.08),
]

TOOL_REGISTRY = [
    {
        "tool": "vision.detect_color",
        "scope": "camera.frame",
        "risk": "low",
        "writes": False,
        "description": "Detects colored objects in the synthetic camera frame.",
    },
    {
        "tool": "memory.update_belief",
        "scope": "agent.state",
        "risk": "low",
        "writes": True,
        "description": "Updates target belief, velocity estimate, and confidence.",
    },
    {
        "tool": "planner.create_mark_action",
        "scope": "task.plan",
        "risk": "low",
        "writes": True,
        "description": "Creates a proposed map-marking action from current belief.",
    },
    {
        "tool": "policy.evaluate_risk",
        "scope": "policy.gate",
        "risk": "low",
        "writes": False,
        "description": "Scores action risk using confidence, visibility, uncertainty, and collision.",
    },
    {
        "tool": "map.check_collision",
        "scope": "mock.map.read",
        "risk": "low",
        "writes": False,
        "description": "Checks whether a proposed marker would collide with a mapped obstacle.",
    },
    {
        "tool": "map.mark_location",
        "scope": "mock.map.write",
        "risk": "medium",
        "writes": True,
        "description": "Writes a location marker onto the mock map.",
    },
    {
        "tool": "verifier.check_marker",
        "scope": "mock.map.read + ground_truth",
        "risk": "low",
        "writes": False,
        "description": "Verifies whether the marker is close enough to the true target.",
    },
    {
        "tool": "recovery.clear_marker",
        "scope": "mock.map.write",
        "risk": "medium",
        "writes": True,
        "description": "Clears an invalid marker and returns the agent to recovery mode.",
    },
]

SCENARIOS = [
    {"id": "S01", "name": "Nominal red object", "overrides": {}},
    {"id": "S02", "name": "Fast mover", "overrides": {"speed_multiplier": 1.9}},
    {"id": "S03", "name": "Occluded target", "overrides": {"occlusion_multiplier": 2.0}},
    {"id": "S04", "name": "Noisy sensor", "overrides": {"noise_multiplier": 2.0}},
    {"id": "S05", "name": "Red distractor", "overrides": {"red_distractor": True}},
    {"id": "S06", "name": "Late appearance", "overrides": {"appear_after": 2.8}},
    {"id": "S07", "name": "Strict policy", "overrides": {"policy_threshold_delta": -0.14}},
    {"id": "S08", "name": "Loose policy", "overrides": {"policy_threshold_delta": 0.22}},
    {"id": "S09", "name": "Tight verification", "overrides": {"tolerance_delta": -0.035}},
    {"id": "S10", "name": "Wide verification", "overrides": {"tolerance_delta": 0.07}},
    {"id": "S11", "name": "Cluttered map", "overrides": {"clutter": True}},
    {
        "id": "S12",
        "name": "Recovery stress",
        "overrides": {
            "recovery_stress": True,
            "speed_multiplier": 1.25,
            "occlusion_multiplier": 1.4,
        },
    },
]

CODEC_CANDIDATES = [
    (".mp4", "avc1", "video/mp4"),
    (".mp4", "H264", "video/mp4"),
    (".mp4", "X264", "video/mp4"),
    (".webm", "VP80", "video/webm"),
    (".webm", "VP90", "video/webm"),
]


# ---------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------

class Detection(BaseModel):
    label: str
    color: str
    x: float
    y: float
    w: float
    h: float
    confidence: float


class VisionEvent(BaseModel):
    frame: int
    t: float
    visible_truth: bool
    detections: List[Detection] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def pyd_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def norm_to_panel(x: float, y: float, x0: int, y0: int, x1: int, y1: int):
    px = int(x0 + clamp(x, 0.0, 1.0) * (x1 - x0))
    py = int(y0 + clamp(y, 0.0, 1.0) * (y1 - y0))
    return px, py


def in_obstacle(x: float, y: float, obstacles: List[Any], margin: float = 0.0) -> bool:
    for ox, oy, ow, oh in obstacles:
        if ox - margin <= x <= ox + ow + margin and oy - margin <= y <= oy + oh + margin:
            return True
    return False


# ---------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------

def make_occlusion_windows(cfg: Dict[str, Any]) -> List[Any]:
    dur = float(cfg.get("occlusion_duration", 0.0))
    windows = []

    if dur > 0.05:
        start = max(float(cfg.get("appear_after", 1.0)) + 1.4, 2.4)
        end = min(start + dur, DURATION_S - 1.0)
        if end > start:
            windows.append((start, end))

    if cfg.get("recovery_stress", False):
        windows.append((5.0, 5.8))

    return windows


def build_base_config(
    speed: float,
    noise: float,
    occlusion_duration: float,
    policy_threshold: float,
    tolerance: float,
    min_detection_conf: float,
) -> Dict[str, Any]:
    cfg = {
        "trial_id": "CUSTOM",
        "scenario_name": "Current controls",
        "speed": float(speed),
        "noise": float(noise),
        "occlusion_duration": float(occlusion_duration),
        "policy_threshold": float(policy_threshold),
        "tolerance": float(tolerance),
        "min_detection_conf": float(min_detection_conf),
        "policy_conf_threshold": max(float(min_detection_conf), 0.60),
        "stable_frames_required": 8,
        "verify_delay": 0.75,
        "max_recoveries": 2,
        "smoothing": 0.58,
        "seed": 42,
        "appear_after": 1.0,
        "red_distractor": False,
        "clutter": False,
        "recovery_stress": False,
        "low_contrast": False,
        "obstacles": DEFAULT_OBSTACLES,
    }
    cfg["occlusion_windows"] = make_occlusion_windows(cfg)
    return cfg


def build_scenario_config(base: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    cfg = base.copy()
    ov = scenario.get("overrides", {})

    cfg["trial_id"] = scenario["id"]
    cfg["scenario_name"] = scenario["name"]

    cfg["speed"] = clamp(base.get("speed", 0.06) * ov.get("speed_multiplier", 1.0), 0.005, 0.35)
    cfg["noise"] = clamp(base.get("noise", 0.25) * ov.get("noise_multiplier", 1.0), 0.0, 1.5)
    cfg["occlusion_duration"] = clamp(
        base.get("occlusion_duration", 1.0) * ov.get("occlusion_multiplier", 1.0),
        0.0,
        3.5,
    )
    cfg["policy_threshold"] = clamp(
        base.get("policy_threshold", 0.35) + ov.get("policy_threshold_delta", 0.0),
        0.02,
        0.95,
    )
    cfg["tolerance"] = clamp(
        base.get("tolerance", 0.10) + ov.get("tolerance_delta", 0.0),
        0.02,
        0.35,
    )
    cfg["min_detection_conf"] = clamp(
        base.get("min_detection_conf", 0.55) + ov.get("min_conf_delta", 0.0),
        0.20,
        0.95,
    )
    cfg["policy_conf_threshold"] = max(cfg["min_detection_conf"], 0.60)

    cfg["red_distractor"] = bool(ov.get("red_distractor", base.get("red_distractor", False)))
    cfg["clutter"] = bool(ov.get("clutter", base.get("clutter", False)))
    cfg["recovery_stress"] = bool(ov.get("recovery_stress", base.get("recovery_stress", False)))
    cfg["low_contrast"] = bool(ov.get("low_contrast", base.get("low_contrast", False)))
    cfg["appear_after"] = float(ov.get("appear_after", base.get("appear_after", 1.0)))

    cfg["obstacles"] = CLUTTER_OBSTACLES if cfg["clutter"] else DEFAULT_OBSTACLES
    cfg["seed"] = int(base.get("seed", 42)) + sum(ord(ch) for ch in scenario["id"])
    cfg["occlusion_windows"] = make_occlusion_windows(cfg)

    return cfg


# ---------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------

def truth_position(t: float, cfg: Dict[str, Any]):
    s = float(cfg.get("seed", 42)) * 0.017
    x = 0.18 + float(cfg.get("speed", 0.06)) * t + 0.07 * math.sin(2 * math.pi * 0.33 * t + s)
    y = (
        0.56
        + 0.20 * math.sin(2 * math.pi * 0.21 * t + s * 1.3)
        + 0.04 * math.cos(2 * math.pi * 0.55 * t + s * 0.7)
    )
    return clamp(x, 0.04, 0.96), clamp(y, 0.08, 0.92)


def visible_at(t: float, cfg: Dict[str, Any]) -> bool:
    if t < float(cfg.get("appear_after", 1.0)):
        return False
    for start, end in cfg.get("occlusion_windows", []):
        if start <= t <= end:
            return False
    return True


def make_detections(
    t: float,
    cfg: Dict[str, Any],
    rng: random.Random,
    tx: float,
    ty: float,
    visible: bool,
) -> List[Detection]:
    dets: List[Detection] = []
    noise = float(cfg.get("noise", 0.25))

    if visible:
        jx = rng.gauss(0.0, 0.010 + noise * 0.040)
        jy = rng.gauss(0.0, 0.010 + noise * 0.040)

        conf = 0.94 - noise * 0.24 + rng.gauss(0.0, 0.025)
        if cfg.get("low_contrast", False):
            conf -= 0.12
        conf = clamp(conf, 0.18, 0.99)

        w = clamp(0.085 + rng.uniform(-0.012, 0.012), 0.04, 0.16)
        h = clamp(0.085 + rng.uniform(-0.012, 0.012), 0.04, 0.16)

        dets.append(
            Detection(
                label="red_candidate",
                color="red",
                x=float(clamp(tx + jx, 0.02, 0.98)),
                y=float(clamp(ty + jy, 0.04, 0.96)),
                w=float(w),
                h=float(h),
                confidence=float(conf),
            )
        )

    bx = clamp(0.78 + 0.08 * math.sin(2 * math.pi * 0.17 * t + float(cfg.get("seed", 42)) * 0.013), 0.02, 0.98)
    by = clamp(0.22 + 0.07 * math.cos(2 * math.pi * 0.13 * t + float(cfg.get("seed", 42)) * 0.019), 0.05, 0.95)
    dets.append(
        Detection(
            label="blue_distractor",
            color="blue",
            x=float(bx),
            y=float(by),
            w=0.07,
            h=0.07,
            confidence=0.82,
        )
    )

    gx = clamp(0.24 + 0.09 * math.cos(2 * math.pi * 0.15 * t + float(cfg.get("seed", 42)) * 0.023), 0.02, 0.98)
    gy = clamp(0.34 + 0.08 * math.sin(2 * math.pi * 0.11 * t + float(cfg.get("seed", 42)) * 0.029), 0.05, 0.95)
    dets.append(
        Detection(
            label="green_distractor",
            color="green",
            x=float(gx),
            y=float(gy),
            w=0.065,
            h=0.065,
            confidence=0.78,
        )
    )

    if cfg.get("red_distractor", False):
        rx = clamp(0.22 + 0.10 * math.sin(2 * math.pi * 0.23 * t + float(cfg.get("seed", 42)) * 0.031), 0.02, 0.98)
        ry = clamp(0.76 + 0.08 * math.cos(2 * math.pi * 0.19 * t + float(cfg.get("seed", 42)) * 0.037), 0.05, 0.95)
        conf = 0.60 + rng.gauss(0.0, 0.06) - noise * 0.08
        if not visible:
            conf += 0.04
        dets.append(
            Detection(
                label="red_distractor",
                color="red",
                x=float(rx),
                y=float(ry),
                w=0.065,
                h=0.065,
                confidence=float(clamp(conf, 0.12, 0.94)),
            )
        )

    return dets


# ---------------------------------------------------------------------
# Agent state machine
# ---------------------------------------------------------------------

def init_state() -> Dict[str, Any]:
    return {
        "frame": 0,
        "phase": "OBSERVE",
        "belief_x": 0.5,
        "belief_y": 0.5,
        "vel_x": 0.0,
        "vel_y": 0.0,
        "confidence": 0.0,
        "uncertainty": 0.5,
        "stable_frames": 0,
        "last_detect_t": -10.0,
        "recent_visible": False,
        "marker": None,
        "verify_due_t": None,
        "verification_status": "none",
        "completed": False,
        "recovery_success": False,
        "escalated": False,
        "actions": 0,
        "invalid_actions": 0,
        "verification_failures": 0,
        "recoveries": 0,
        "tool_calls": 0,
        "history": [],
        "confidence_history": [],
        "risk_history": [],
        "trace": [],
        "final_dist": None,
    }


def add_trace(
    state: Dict[str, Any],
    t: float,
    phase: str,
    event: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    state["trace"].append(
        {
            "frame": state["frame"],
            "t": round(float(t), 2),
            "phase": phase,
            "event": event,
            "message": message,
            "data": data or {},
        }
    )


def compute_policy_risk(state: Dict[str, Any], cfg: Dict[str, Any], t: float) -> float:
    confidence = float(state["confidence"])
    base_risk = (1.0 - confidence) * 0.65
    visibility_risk = 0.0 if state["recent_visible"] else 0.18
    uncertainty_risk = min(float(state["uncertainty"]), 0.6) * 0.35
    collision_risk = (
        0.22
        if in_obstacle(state["belief_x"], state["belief_y"], cfg.get("obstacles", []), margin=0.035)
        else 0.0
    )
    stale_risk = 0.15 if (t - float(state["last_detect_t"])) > 1.0 else 0.0

    return clamp(base_risk + visibility_risk + uncertainty_risk + collision_risk + stale_risk, 0.0, 1.0)


def step_agent(state: Dict[str, Any], ve: VisionEvent, cfg: Dict[str, Any], dt: float) -> None:
    state["frame"] += 1
    state["tool_calls"] += 1

    red_dets = [
        d
        for d in ve.detections
        if d.color == "red" and d.confidence >= float(cfg.get("min_detection_conf", 0.55))
    ]
    best = max(red_dets, key=lambda d: d.confidence) if red_dets else None

    state["recent_visible"] = bool(
        ve.visible_truth or (best is not None and best.confidence >= 0.70)
    )

    pred_x = clamp(state["belief_x"] + state["vel_x"] * dt, 0.0, 1.0)
    pred_y = clamp(state["belief_y"] + state["vel_y"] * dt, 0.0, 1.0)

    if best is not None:
        alpha = float(cfg.get("smoothing", 0.58))
        new_x = clamp(alpha * best.x + (1.0 - alpha) * pred_x, 0.0, 1.0)
        new_y = clamp(alpha * best.y + (1.0 - alpha) * pred_y, 0.0, 1.0)

        if state["last_detect_t"] >= 0.0 and dt > 0:
            vx = (new_x - state["belief_x"]) / dt
            vy = (new_y - state["belief_y"]) / dt
            state["vel_x"] = clamp(vx, -0.6, 0.6)
            state["vel_y"] = clamp(vy, -0.6, 0.6)

        state["belief_x"] = new_x
        state["belief_y"] = new_y
        state["confidence"] = clamp(max(float(best.confidence), state["confidence"] * 0.75), 0.0, 1.0)
        state["stable_frames"] += 1
        state["uncertainty"] = clamp(
            (1.0 - float(best.confidence)) + float(cfg.get("noise", 0.25)) * 0.15,
            0.02,
            0.9,
        )
        state["last_detect_t"] = float(ve.t)
        state["phase"] = "UPDATE_BELIEF"

        add_trace(
            state,
            ve.t,
            state["phase"],
            "detection_accepted",
            f"Accepted red detection conf={best.confidence:.2f}",
            {"detection": pyd_dump(best)},
        )
    else:
        state["belief_x"] = pred_x
        state["belief_y"] = pred_y
        state["confidence"] = clamp(state["confidence"] * 0.86, 0.0, 1.0)
        state["stable_frames"] = 0
        state["uncertainty"] = clamp(state["uncertainty"] + 0.04, 0.0, 1.0)
        state["phase"] = "PREDICT" if state["confidence"] > 0.25 else "OBSERVE"

        add_trace(
            state,
            ve.t,
            state["phase"],
            "no_reliable_detection",
            "No reliable red detection; propagating belief.",
            {"confidence": round(state["confidence"], 3)},
        )

    if state["confidence"] > 0.18:
        state["history"].append(
            {
                "t": round(float(ve.t), 2),
                "x": round(float(state["belief_x"]), 4),
                "y": round(float(state["belief_y"]), 4),
                "conf": round(float(state["confidence"]), 3),
            }
        )
        if len(state["history"]) > 140:
            state["history"].pop(0)

    state["confidence_history"].append(
        {
            "frame": state["frame"],
            "t": round(float(ve.t), 2),
            "value": round(float(state["confidence"]), 3),
        }
    )

    # Verify pending marker.
    if (
        state["marker"] is not None
        and state["verify_due_t"] is not None
        and ve.t >= state["verify_due_t"]
    ):
        tx, ty = truth_position(ve.t, cfg)
        dist = math.hypot(state["marker"]["x"] - tx, state["marker"]["y"] - ty)
        collision = in_obstacle(state["marker"]["x"], state["marker"]["y"], cfg.get("obstacles", []), margin=0.0)

        state["tool_calls"] += 1
        state["verify_due_t"] = None
        state["final_dist"] = round(dist, 4)

        if dist <= float(cfg.get("tolerance", 0.10)) and not collision:
            state["completed"] = True
            state["verification_status"] = "success"
            state["phase"] = "VERIFIED"
            if state["recoveries"] > 0:
                state["recovery_success"] = True

            add_trace(
                state,
                ve.t,
                state["phase"],
                "verification_passed",
                f"Marker verified. dist={dist:.3f}",
                {"distance": round(dist, 4), "collision": collision},
            )
        else:
            state["verification_failures"] += 1
            state["invalid_actions"] += 1
            state["verification_status"] = "failed"

            add_trace(
                state,
                ve.t,
                "VERIFY",
                "verification_failed",
                f"Marker invalid. dist={dist:.3f}, collision={collision}",
                {"distance": round(dist, 4), "collision": collision},
            )

            if state["recoveries"] < int(cfg.get("max_recoveries", 2)):
                state["recoveries"] += 1
                state["marker"] = None
                state["verify_due_t"] = None
                state["confidence"] = clamp(state["confidence"] * 0.55, 0.0, 1.0)
                state["stable_frames"] = 0
                state["phase"] = "RECOVERY"

                add_trace(
                    state,
                    ve.t,
                    state["phase"],
                    "recovery_started",
                    "Clearing invalid marker and re-entering observation loop.",
                    {"recoveries": state["recoveries"]},
                )
            else:
                state["escalated"] = True
                state["marker"] = None
                state["verify_due_t"] = None
                state["phase"] = "ESCALATED"

                add_trace(
                    state,
                    ve.t,
                    state["phase"],
                    "recovery_exhausted",
                    "Maximum recoveries exceeded; escalating.",
                    {},
                )

    if (
        state["completed"]
        and state["marker"] is not None
        and ve.t - float(state["marker"]["t"]) > 0.8
    ):
        state["phase"] = "DONE"

    # Plan and execute only when no marker is pending and task is not complete.
    if (not state["completed"]) and state["marker"] is None and not state["escalated"]:
        if (
            state["stable_frames"] >= int(cfg.get("stable_frames_required", 8))
            and state["confidence"] >= float(cfg.get("policy_conf_threshold", 0.60))
        ):
            risk = compute_policy_risk(state, cfg, ve.t)
            state["risk_history"].append(
                {
                    "frame": state["frame"],
                    "t": round(float(ve.t), 2),
                    "value": round(float(risk), 3),
                }
            )
            state["tool_calls"] += 1

            if risk <= float(cfg.get("policy_threshold", 0.35)):
                marker = {
                    "x": round(float(state["belief_x"]), 4),
                    "y": round(float(state["belief_y"]), 4),
                    "t": round(float(ve.t), 2),
                    "risk": round(float(risk), 3),
                }
                state["marker"] = marker
                state["verify_due_t"] = ve.t + float(cfg.get("verify_delay", 0.75))
                state["actions"] += 1
                state["tool_calls"] += 1
                state["phase"] = "EXECUTE"
                state["verification_status"] = "pending"

                add_trace(
                    state,
                    ve.t,
                    state["phase"],
                    "action_executed",
                    f"Approved and executed map mark. risk={risk:.2f}",
                    {"marker": marker},
                )
            else:
                state["phase"] = "POLICY_HOLD"
                add_trace(
                    state,
                    ve.t,
                    state["phase"],
                    "policy_hold",
                    f"Action held by policy gate. risk={risk:.2f}",
                    {"risk": round(float(risk), 3)},
                )


def summarize_state(state: Dict[str, Any], cfg: Dict[str, Any], latency_ms: float) -> Dict[str, Any]:
    return {
        "trial_id": cfg.get("trial_id", "CUSTOM"),
        "scenario": cfg.get("scenario_name", "Custom"),
        "completed": bool(state["completed"]),
        "actions": int(state["actions"]),
        "invalid_actions": int(state["invalid_actions"]),
        "verification_failures": int(state["verification_failures"]),
        "recoveries": int(state["recoveries"]),
        "recovery_success": bool(state["recovery_success"]),
        "successful_recovery_attempts": int(state["recoveries"]) if state["recovery_success"] else 0,
        "latency_ms": round(float(latency_ms), 2),
        "tool_calls": int(state["tool_calls"]),
        "final_confidence": round(float(state["confidence"]), 3),
        "final_dist": state.get("final_dist"),
        "escalated": bool(state["escalated"]),
    }


def simulate_trial(cfg: Dict[str, Any]) -> Dict[str, Any]:
    rng = random.Random(int(cfg.get("seed", 42)))
    state = init_state()
    dt = 1.0 / FPS

    start = time.perf_counter()
    for i in range(NUM_FRAMES):
        t = i * dt
        tx, ty = truth_position(t, cfg)
        vis = visible_at(t, cfg)
        dets = make_detections(t, cfg, rng, tx, ty, vis)
        ve = VisionEvent(frame=i, t=t, visible_truth=vis, detections=dets)
        step_agent(state, ve, cfg, dt)

    latency_ms = (time.perf_counter() - start) * 1000.0
    return summarize_state(state, cfg, latency_ms)


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------

def draw_camera_panel(
    frame: np.ndarray,
    cfg: Dict[str, Any],
    state: Dict[str, Any],
    ve: VisionEvent,
    tx: float,
    ty: float,
    visible: bool,
    t: float,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> None:
    cv2.rectangle(frame, (x0, y0), (x1, y1), (56, 63, 72), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (125, 132, 142), 1)
    cv2.putText(
        frame,
        "CAMERA OBSERVATION",
        (x0 + 8, y0 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Simple floor grid.
    for i in range(1, 6):
        yy = y0 + int(i * (y1 - y0) / 6)
        cv2.line(frame, (x0, yy), (x1, yy), (72, 80, 90), 1)
    for i in range(1, 8):
        xx = x0 + int(i * (x1 - x0) / 8)
        cv2.line(frame, (xx, y0), (xx, y1), (72, 80, 90), 1)

    # Obstacles.
    for ox, oy, ow, oh in cfg.get("obstacles", []):
        p1 = norm_to_panel(ox, oy, x0, y0, x1, y1)
        p2 = norm_to_panel(ox + ow, oy + oh, x0, y0, x1, y1)
        cv2.rectangle(frame, p1, p2, (88, 88, 98), -1)
        cv2.rectangle(frame, p1, p2, (132, 132, 148), 1)

    if visible:
        cx, cy = norm_to_panel(tx, ty, x0, y0, x1, y1)
        radius = max(8, int(0.045 * (x1 - x0)))

        cv2.ellipse(frame, (cx, cy + radius // 2), (radius, radius // 3), 0, 0, 360, (28, 28, 28), -1)
        cv2.circle(frame, (cx, cy), radius, (0, 0, 225), -1)
        cv2.circle(frame, (cx, cy), radius, (0, 0, 255), 2)
        cv2.putText(
            frame,
            "RED TARGET",
            (max(x0 + 4, cx - 42), max(y0 + 16, cy - radius - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 + 25, y0 + 55), (x1 - 25, y1 - 55), (96, 96, 108), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        label = "TARGET NOT YET IN VIEW" if t < float(cfg.get("appear_after", 1.0)) else "OCCLUDER ACTIVE"
        cv2.putText(
            frame,
            label,
            (x0 + 42, (y0 + y1) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    # Detections.
    for d in ve.detections:
        accepted = d.color == "red" and d.confidence >= float(cfg.get("min_detection_conf", 0.55))
        color = (0, 255, 0) if accepted else (170, 170, 170)
        thickness = 2 if accepted else 1

        cx, cy = norm_to_panel(d.x, d.y, x0, y0, x1, y1)
        bw = max(8, int(d.w * (x1 - x0)))
        bh = max(8, int(d.h * (y1 - y0)))

        p1 = (cx - bw // 2, cy - bh // 2)
        p2 = (cx + bw // 2, cy + bh // 2)
        cv2.rectangle(frame, p1, p2, color, thickness)
        cv2.putText(
            frame,
            f"{d.color} {d.confidence:.2f}",
            (max(x0 + 4, p1[0]), max(y0 + 14, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )

    # Belief crosshair.
    bx, by = norm_to_panel(state["belief_x"], state["belief_y"], x0, y0, x1, y1)
    cv2.line(frame, (bx - 10, by), (bx + 10, by), (0, 255, 255), 1)
    cv2.line(frame, (bx, by - 10), (bx, by + 10), (0, 255, 255), 1)


def draw_map_panel(
    frame: np.ndarray,
    cfg: Dict[str, Any],
    state: Dict[str, Any],
    tx: float,
    ty: float,
    visible: bool,
    true_history: List[Any],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> None:
    cv2.rectangle(frame, (x0, y0), (x1, y1), (32, 42, 56), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (105, 120, 138), 1)
    cv2.putText(
        frame,
        "MOCK MAP / ACTION SPACE",
        (x0 + 8, y0 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Grid.
    for i in range(1, 10):
        xx = x0 + int(i * (x1 - x0) / 10)
        cv2.line(frame, (xx, y0), (xx, y1), (52, 66, 82), 1)
    for j in range(1, 7):
        yy = y0 + int(j * (y1 - y0) / 7)
        cv2.line(frame, (x0, yy), (x1, yy), (52, 66, 82), 1)

    # Obstacles.
    for ox, oy, ow, oh in cfg.get("obstacles", []):
        p1 = norm_to_panel(ox, oy, x0, y0, x1, y1)
        p2 = norm_to_panel(ox + ow, oy + oh, x0, y0, x1, y1)
        cv2.rectangle(frame, p1, p2, (70, 70, 82), -1)
        cv2.rectangle(frame, p1, p2, (120, 120, 135), 1)

    # Ground-truth trail.
    for pt in true_history[-100:]:
        px, py = norm_to_panel(pt[0], pt[1], x0, y0, x1, y1)
        cv2.circle(frame, (px, py), 2, (175, 175, 175), -1)

    # Current truth ghost.
    if visible or state["completed"]:
        gx, gy = norm_to_panel(tx, ty, x0, y0, x1, y1)
        cv2.circle(frame, (gx, gy), 9, (245, 245, 245), 1)
        cv2.putText(
            frame,
            "TRUTH",
            (gx + 12, gy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    # Belief trail.
    for h in state["history"][-100:]:
        px, py = norm_to_panel(h["x"], h["y"], x0, y0, x1, y1)
        conf = float(h["conf"])
        color = (0, int(180 + 60 * conf), 255)
        cv2.circle(frame, (px, py), 2, color, -1)

    # Belief crosshair.
    bx, by = norm_to_panel(state["belief_x"], state["belief_y"], x0, y0, x1, y1)
    cv2.drawMarker(frame, (bx, by), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(
        frame,
        f"BELIEF {state['confidence']:.2f}",
        (bx + 12, by + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Marker.
    if state["marker"] is not None:
        mx, my = norm_to_panel(state["marker"]["x"], state["marker"]["y"], x0, y0, x1, y1)

        if state["verification_status"] == "success":
            vcolor = (0, 255, 0)
        elif state["verification_status"] == "failed":
            vcolor = (0, 0, 255)
        else:
            vcolor = (0, 165, 255)

        cv2.circle(frame, (mx, my), 14, vcolor, 2)
        cv2.line(frame, (mx - 10, my - 10), (mx + 10, my + 10), vcolor, 2)
        cv2.line(frame, (mx - 10, my + 10), (mx + 10, my - 10), vcolor, 2)

        tol_px = max(8, int(float(cfg.get("tolerance", 0.10)) * (x1 - x0)))
        cv2.circle(frame, (mx, my), tol_px, (235, 235, 235), 1)

        cv2.putText(
            frame,
            "MARK",
            (mx + 16, my - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            vcolor,
            1,
            cv2.LINE_AA,
        )

    # Risk box.
    risk = state["risk_history"][-1]["value"] if state["risk_history"] else 0.0
    cv2.rectangle(frame, (x0 + 8, y1 - 62), (x0 + 218, y1 - 10), (18, 18, 24), -1)
    cv2.rectangle(frame, (x0 + 8, y1 - 62), (x0 + 218, y1 - 10), (90, 90, 105), 1)
    cv2.putText(
        frame,
        f"RISK {risk:.2f}",
        (x0 + 18, y1 - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"THRESH {cfg.get('policy_threshold', 0.35):.2f}",
        (x0 + 18, y1 - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )


def draw_frame(
    frame_idx: int,
    t: float,
    cfg: Dict[str, Any],
    state: Dict[str, Any],
    ve: VisionEvent,
    tx: float,
    ty: float,
    visible: bool,
    true_history: List[Any],
) -> np.ndarray:
    frame = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
    frame[:] = (38, 38, 42)

    # Header.
    cv2.rectangle(frame, (0, 0), (VIDEO_W, 40), (22, 22, 26), -1)
    cv2.putText(
        frame,
        "VISION-TO-ACTION WORKFLOW AGENT",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    status = f"FRAME {frame_idx:03d} | T={t:0.2f}s | PHASE={state['phase']}"
    cv2.putText(
        frame,
        status,
        (VIDEO_W - 365, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    cam_x0, cam_y0, cam_x1, cam_y1 = 10, 50, VIDEO_W // 2 - 12, VIDEO_H - 38
    map_x0, map_y0, map_x1, map_y1 = VIDEO_W // 2 + 12, 50, VIDEO_W - 10, VIDEO_H - 38

    draw_camera_panel(frame, cfg, state, ve, tx, ty, visible, t, cam_x0, cam_y0, cam_x1, cam_y1)
    draw_map_panel(frame, cfg, state, tx, ty, visible, true_history, map_x0, map_y0, map_x1, map_y1)

    # Bottom bar.
    cv2.rectangle(frame, (0, VIDEO_H - 34), (VIDEO_W, VIDEO_H), (22, 22, 26), -1)
    risk = state["risk_history"][-1]["value"] if state["risk_history"] else 0.0
    marker_txt = "YES" if state["marker"] is not None else "NO"
    bottom = (
        f"CONF {state['confidence']:.2f} | RISK {risk:.2f} | MARKER {marker_txt} | "
        f"VERIFY {state['verification_status']} | ACTIONS {state['actions']} | "
        f"INVALID {state['invalid_actions']} | RECOVERIES {state['recoveries']}"
    )
    cv2.putText(
        frame,
        bottom,
        (10, VIDEO_H - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return frame


def select_video_codec():
    dummy = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)

    for suffix, cc, mime in CODEC_CANDIDATES:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        try:
            fourcc = cv2.VideoWriter_fourcc(*cc)
            writer = cv2.VideoWriter(path, fourcc, FPS, (VIDEO_W, VIDEO_H))
            if writer.isOpened():
                writer.write(dummy)
                writer.release()
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    return suffix, cc, mime
        except Exception:
            pass
        finally:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass

    return None


def try_imageio_mp4(frames: List[np.ndarray]) -> Optional[bytes]:
    if not frames:
        return None

    try:
        import imageio.v2 as imageio
    except Exception:
        return None

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        with imageio.get_writer(
            path,
            fps=FPS,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,
        ) as writer:
            for fr in frames:
                writer.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                data = f.read()
            os.unlink(path)
            return data

        if os.path.exists(path):
            os.unlink(path)
    except Exception:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass

    return None


def frames_to_gif(frames: List[np.ndarray]) -> Optional[bytes]:
    if not frames:
        return None

    try:
        from PIL import Image

        step = 2 if len(frames) > 60 else 1
        selected = frames[::step]
        duration = int(1000.0 * step / FPS)

        rgb_frames = [Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) for fr in selected]
        buf = io.BytesIO()
        rgb_frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=rgb_frames[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )
        return buf.getvalue()
    except Exception:
        return None


def render_video(cfg: Dict[str, Any]):
    rng = random.Random(int(cfg.get("seed", 42)))
    state = init_state()
    dt = 1.0 / FPS

    codec = select_video_codec()
    buffer_frames = codec is None
    frames = [] if buffer_frames else None

    writer = None
    path = None
    mime = "image/gif"

    if not buffer_frames:
        suffix, cc, mime = codec
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        try:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), FPS, (VIDEO_W, VIDEO_H))
        except Exception:
            writer = None

        if writer is None or not writer.isOpened():
            buffer_frames = True
            frames = []
            mime = "image/gif"

            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass

            path = None

    true_history: List[Any] = []
    last_ve: Optional[VisionEvent] = None

    start = time.perf_counter()

    for i in range(NUM_FRAMES):
        t = i * dt
        tx, ty = truth_position(t, cfg)
        vis = visible_at(t, cfg)
        dets = make_detections(t, cfg, rng, tx, ty, vis)
        ve = VisionEvent(frame=i, t=t, visible_truth=vis, detections=dets)

        step_agent(state, ve, cfg, dt)
        last_ve = ve

        if vis or t >= float(cfg.get("appear_after", 1.0)):
            true_history.append((tx, ty))

        frame = draw_frame(i, t, cfg, state, ve, tx, ty, vis, true_history)

        if buffer_frames:
            frames.append(cv2.resize(frame, FALLBACK_SIZE, interpolation=cv2.INTER_AREA))
        else:
            writer.write(frame)

    latency_ms = (time.perf_counter() - start) * 1000.0

    if buffer_frames:
        video_bytes = try_imageio_mp4(frames)
        if video_bytes is not None:
            mime = "video/mp4"
            filename = "vision_to_action_demo.mp4"
        else:
            video_bytes = frames_to_gif(frames)
            mime = "image/gif"
            filename = "vision_to_action_demo.gif"

        if video_bytes is None:
            video_bytes = b""
            mime = "application/octet-stream"
            filename = "vision_to_action_demo.bin"
    else:
        writer.release()
        with open(path, "rb") as f:
            video_bytes = f.read()
        try:
            os.unlink(path)
        except Exception:
            pass

        filename = "vision_to_action_demo.mp4" if mime == "video/mp4" else "vision_to_action_demo.webm"

    metric = summarize_state(state, cfg, latency_ms)
    return video_bytes, mime, filename, state, last_ve, metric


# ---------------------------------------------------------------------
# Full demo runner
# ---------------------------------------------------------------------

def run_demo(base_cfg: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    selected_cfg = build_scenario_config(base_cfg, scenario)
    suite_cfgs = [build_scenario_config(base_cfg, s) for s in SCENARIOS]

    metric_rows = []
    for c in suite_cfgs:
        metric_rows.append(simulate_trial(c))

    metrics_df = pd.DataFrame(metric_rows)

    video_bytes, mime, filename, state, last_ve, selected_metric = render_video(selected_cfg)

    trace = state["trace"]
    histories = {
        "confidence": state["confidence_history"],
        "risk": state["risk_history"],
    }

    last_detections = []
    if last_ve is not None:
        last_detections = [pyd_dump(d) for d in last_ve.detections]

    state_summary = {
        "phase": state["phase"],
        "completed": state["completed"],
        "actions": state["actions"],
        "invalid_actions": state["invalid_actions"],
        "recoveries": state["recoveries"],
        "verification_status": state["verification_status"],
        "final_confidence": round(float(state["confidence"]), 3),
        "final_dist": state.get("final_dist"),
        "marker": state.get("marker"),
        "escalated": state.get("escalated"),
    }

    return {
        "video_bytes": video_bytes,
        "mime": mime,
        "filename": filename,
        "metrics_df": metrics_df,
        "trace": trace,
        "histories": histories,
        "last_detections": last_detections,
        "state_summary": state_summary,
        "cfg": selected_cfg,
        "selected_metric": selected_metric,
    }


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

def main() -> None:
    st.title("🛰️ Vision-to-Action Workflow Agent")
    st.caption(
        "Local, no-API-key demo. The agent observes a synthetic camera stream, maintains state, "
        "plans a tool action, passes a policy gate, executes on a mock map, verifies the result, "
        "and recovers when needed."
    )

    with st.sidebar:
        st.header("Mission Controls")

        scenario_idx = st.selectbox(
            "Scenario",
            list(range(len(SCENARIOS))),
            format_func=lambda i: f"{SCENARIOS[i]['id']} — {SCENARIOS[i]['name']}",
        )

        speed = st.slider("Object speed", 0.01, 0.18, 0.06, 0.01)
        noise = st.slider("Sensor noise", 0.0, 1.0, 0.25, 0.05)
        occlusion = st.slider("Occlusion duration (s)", 0.0, 2.5, 1.0, 0.1)
        policy_threshold = st.slider("Policy risk threshold", 0.05, 0.80, 0.35, 0.05)
        tolerance = st.slider("Verification tolerance", 0.03, 0.25, 0.10, 0.01)
        min_conf = st.slider("Min detection confidence", 0.30, 0.90, 0.55, 0.05)

        run_btn = st.button("▶️ Run agent & render video", type="primary", use_container_width=True)

        st.markdown("---")
        st.subheader("No API key")
        st.write(
            "All planning, tool use, state tracking, policy gating, simulation, and verification "
            "run locally in Python."
        )

    if "result" not in st.session_state:
        st.session_state.result = None

    if st.session_state.result is None or run_btn:
        base_cfg = build_base_config(
            speed=speed,
            noise=noise,
            occlusion_duration=occlusion,
            policy_threshold=policy_threshold,
            tolerance=tolerance,
            min_detection_conf=min_conf,
        )
        scenario = SCENARIOS[scenario_idx]

        try:
            with st.spinner(
                "Loading realistic demo data, running planner/policy/executor/verifier, and rendering video..."
            ):
                st.session_state.result = run_demo(base_cfg, scenario)
        except Exception as exc:
            st.error("Demo generation failed. See exception below.")
            st.exception(exc)
            st.stop()

    result = st.session_state.result
    selected = result["selected_metric"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Selected scenario", result["cfg"]["scenario_name"])
    c2.metric("Completed", "✅ Yes" if selected["completed"] else "❌ No")
    c3.metric("Actions", selected["actions"])
    c4.metric("Invalid actions", selected["invalid_actions"])
    c5.metric("Recoveries", selected["recoveries"])

    tabs = st.tabs(
        [
            "🎥 Live Demo",
            "🧠 Agent Trace",
            "🧰 Tools & Policy",
            "📊 Evaluation",
            "🏗️ Architecture",
        ]
    )

    with tabs[0]:
        st.subheader("Generated demo video")

        if not result["video_bytes"]:
            st.warning("Video bytes are empty. The encoder fallback did not produce a file.")
        elif result["mime"] in ("video/mp4", "video/webm"):
            st.video(
                result["video_bytes"],
                format=result["mime"],
                loop=True,
                autoplay=True,
                muted=True,
            )
        else:
            st.image(result["video_bytes"], use_container_width=True)

        st.download_button(
            "⬇️ Download demo file",
            data=result["video_bytes"],
            file_name=result["filename"],
            mime=result["mime"],
            use_container_width=True,
        )

        st.caption(
            "The video shows synthetic camera observations, vision events, belief updates, policy gating, "
            "map marking, verification, and recovery."
        )

        colA, colB = st.columns(2)

        with colA:
            st.markdown("#### Latest vision event")
            if result["last_detections"]:
                st.dataframe(
                    pd.DataFrame(result["last_detections"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("No detections in final frame.")

        with colB:
            st.markdown("#### Agent state")
            st.json(result["state_summary"])

        st.markdown("#### Confidence and policy risk over time")

        conf_df = pd.DataFrame(result["histories"]["confidence"])
        risk_df = pd.DataFrame(result["histories"]["risk"])

        cc1, cc2 = st.columns(2)

        with cc1:
            if not conf_df.empty:
                st.line_chart(
                    conf_df.set_index("t")[["value"]].rename(columns={"value": "Confidence"})
                )
            else:
                st.info("No confidence history yet.")

        with cc2:
            if not risk_df.empty:
                st.line_chart(
                    risk_df.set_index("t")[["value"]].rename(columns={"value": "Policy risk"})
                )
            else:
                st.info("No policy evaluation yet in this run.")

    with tabs[1]:
        st.subheader("Structured trace")

        trace_df = pd.DataFrame(result["trace"])
        if not trace_df.empty and "data" in trace_df.columns:
            trace_df["data"] = trace_df["data"].apply(
                lambda x: json.dumps(x, default=str) if isinstance(x, dict) else str(x)
            )

        st.dataframe(trace_df, use_container_width=True, hide_index=True)

        with st.expander("Raw trace JSON — last 80 events"):
            st.json(result["trace"][-80:])

    with tabs[2]:
        st.subheader("Tool registry")
        st.dataframe(pd.DataFrame(TOOL_REGISTRY), use_container_width=True, hide_index=True)

        st.subheader("Policy gate")
        st.markdown(
            """
            The policy gate estimates risk before any write action:

            - Low detection confidence increases risk.
            - Lost visibility increases risk.
            - High belief uncertainty increases risk.
            - Proposed marker collision with obstacles increases risk.
            - Stale detections increase risk.

            An action is executed only when estimated risk is below the configured threshold.
            """
        )

        cfg = result["cfg"]
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Risk threshold", f"{cfg['policy_threshold']:.2f}")
        pc2.metric("Verification tolerance", f"{cfg['tolerance']:.2f}")
        pc3.metric("Min detection confidence", f"{cfg['min_detection_conf']:.2f}")

    with tabs[3]:
        st.subheader("Task-suite evaluation")

        df = result["metrics_df"]

        completion_rate = float(df["completed"].mean()) if not df.empty else 0.0
        total_actions = int(df["actions"].sum()) if not df.empty else 0
        total_invalid = int(df["invalid_actions"].sum()) if not df.empty else 0
        invalid_rate = total_invalid / total_actions if total_actions else 0.0

        total_failures = int(df["verification_failures"].sum()) if not df.empty else 0
        successful_recoveries = (
            int(df["successful_recovery_attempts"].sum()) if not df.empty else 0
        )
        recovery_rate = successful_recoveries / total_failures if total_failures else 1.0

        avg_latency = float(df["latency_ms"].mean()) if not df.empty else 0.0
        total_tools = int(df["tool_calls"].sum()) if not df.empty else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Completion rate", f"{completion_rate * 100:.1f}%")
        m2.metric("Invalid action rate", f"{invalid_rate * 100:.1f}%")
        m3.metric("Recovery rate", f"{recovery_rate * 100:.1f}%")
        m4.metric("Avg latency", f"{avg_latency:.1f} ms")
        m5.metric("Tool calls", total_tools)

        st.dataframe(df, use_container_width=True, hide_index=True)

        if not df.empty:
            chart_df = df[["scenario", "completed", "invalid_actions", "recoveries"]].copy()
            chart_df["completed"] = chart_df["completed"].astype(int)
            st.bar_chart(chart_df.set_index("scenario"))

    with tabs[4]:
        st.subheader("Architecture")
        st.markdown(
            """
            ### Implemented modules

            - **Vision events**: synthetic camera frames produce structured detection objects.
            - **Task planner**: converts belief state into a proposed `map.mark_location` action.
            - **Tool registry**: explicit tool names, scopes, risk levels, and write permissions.
            - **Policy gate**: blocks risky actions based on confidence, visibility, uncertainty, and collision.
            - **Execution simulator**: applies approved actions to a mock map.
            - **Verifier**: checks marker correctness against ground truth and triggers recovery.
            - **State machine**: OBSERVE → UPDATE_BELIEF → PLAN → POLICY_GATE → EXECUTE → VERIFY → RECOVERY/DONE.
            - **Evaluation harness**: runs a synthetic task suite and reports completion, invalid-action rate, recovery rate, latency, and tool-call cost.

            ### Why this matters

            This is not a text-only agent. It keeps state, chooses tools, respects policy, executes in a simulator,
            verifies outcomes, and recovers from mistakes.
            """
        )


if __name__ == "__main__":
    main()