import io
import json
import math
import os
import tempfile
import hashlib
import importlib
import time
import ctypes
import concurrent.futures as cf
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from body_attitude import build_body_reference_from_json, compute_body_attitude_summary
from center_map_tool_v5 import SuspensionGeometryExact
from dynamic_optimization import (
    CandidateEvaluation,
    OptimizationConstraint,
    OptimizationProblem,
    OptimizationResult,
    OptimizationVariable,
    candidates_to_csv,
    json_safe,
    run_optimization,
)
from upright_solver import solve_upright_for_zw

try:
    from calibrator import calibrate as calibrate_geometry, write_calibrated_json
    _CALIBRATOR_AVAILABLE = True
except Exception:
    calibrate_geometry = None  # type: ignore[assignment]
    write_calibrated_json = None  # type: ignore[assignment]
    _CALIBRATOR_AVAILABLE = False

try:
    from suspension_model import VEHICLE_REGISTRY
except Exception:
    VEHICLE_REGISTRY = {}

_PIPELINE_SOURCE = "local_workspace"
_CALIBRATION_MODEL_FALLBACK = ["F2_2026", "F4_T421", "D324_EF", "F3_2026", "default"]


app = Flask(__name__)

_state: Dict[str, object] = {
    "json_data": None,
    "json_path": None,
    "last_csv": "",
    "last_opt_global_csv": "",
    "last_opt_top_csv": "",
    "gg_calibration_cache": {},
    "geometry_solver_cache": {},
    "body_attitude_ref_cache": {},
    "center_state_cache": {},
    "json_signature_cache": {},
    "windows_perf_tuning": {},
}


def _json_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value
    try:
        num = float(value)
    except Exception:
        return None
    return num if np.isfinite(num) else None


def _configure_windows_performance_mode() -> Dict[str, Any]:
    """
    Best-effort Windows tuning:
    - Disable process execution-speed throttling (efficiency mode style throttling)
    - Raise process priority class to HIGH
    This only affects the current Python process.
    """
    status = {
        "platform": os.name,
        "attempted": False,
        "throttling_disabled": False,
        "priority_high": False,
        "message": "",
    }
    if os.name != "nt":
        status["message"] = "Not running on Windows; tuning skipped."
        return status
    if str(os.environ.get("DYNAMIC_SIM_DISABLE_WIN_TUNING", "0")).strip() in {"1", "true", "TRUE"}:
        status["message"] = "Windows tuning disabled by DYNAMIC_SIM_DISABLE_WIN_TUNING."
        return status

    status["attempted"] = True
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process = kernel32.GetCurrentProcess()

        PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
        ProcessPowerThrottling = 4
        HIGH_PRIORITY_CLASS = 0x00000080

        class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
            _fields_ = [
                ("Version", ctypes.c_ulong),
                ("ControlMask", ctypes.c_ulong),
                ("StateMask", ctypes.c_ulong),
            ]

        state = PROCESS_POWER_THROTTLING_STATE(
            Version=1,
            ControlMask=PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask=0,  # 0 => disable execution-speed throttling
        )
        set_info_ok = kernel32.SetProcessInformation(
            process,
            ProcessPowerThrottling,
            ctypes.byref(state),
            ctypes.sizeof(state),
        )
        status["throttling_disabled"] = bool(set_info_ok)

        set_prio_ok = kernel32.SetPriorityClass(process, HIGH_PRIORITY_CLASS)
        status["priority_high"] = bool(set_prio_ok)

        if status["throttling_disabled"] and status["priority_high"]:
            status["message"] = "Windows process tuning applied (throttling disabled, priority high)."
        elif status["throttling_disabled"] or status["priority_high"]:
            status["message"] = "Windows process tuning partially applied."
        else:
            err = ctypes.get_last_error()
            status["message"] = f"Windows process tuning call completed without success (GetLastError={err})."
    except Exception as exc:
        status["message"] = f"Windows process tuning failed: {exc}"
    return status


def _json_clean(value):
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, (int, float, np.integer, np.floating, np.bool_)):
        return _json_scalar(value)
    return value


def _decode_json_bytes(raw: bytes) -> dict:
    for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            continue
    raise ValueError("Could not decode uploaded file as JSON.")


def _ensure_json_path(json_data: dict) -> str:
    json_path = _state.get("json_path")
    if isinstance(json_path, str) and os.path.isfile(json_path):
        return json_path
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
    json.dump(json_data, tmp, indent=2)
    tmp.close()
    _state["json_path"] = tmp.name
    return tmp.name


def _dynamic_range(start: float, end: float, step: float) -> List[float]:
    if abs(float(step)) < 1e-12:
        raise ValueError("Step must be non-zero.")
    direction = 1.0 if float(end) >= float(start) else -1.0
    step = abs(float(step)) * direction
    values: List[float] = []
    x = float(start)
    n = 0
    while (x <= float(end) + 1e-12) if direction > 0 else (x >= float(end) - 1e-12):
        values.append(float(x))
        x += step
        n += 1
        if n > 50000:
            raise ValueError("Range is too large.")
    return values


def _eval_poly_terms(poly_terms: List[dict], variables: Dict[str, float]) -> float:
    total = 0.0
    for term in poly_terms or []:
        expr = str(term.get("expression", "Const")).strip() or "Const"
        coeff = float(term.get("coefficient", 0.0))
        basis = 1.0
        for token in expr.split("*"):
            tk = token.strip()
            if not tk or tk == "Const":
                continue
            basis *= float(variables.get(tk, 0.0))
        total += coeff * basis
    return float(total)


def _mf_tyre_combined_from_json(
    tyre_cfg: dict,
    fz_n: float,
    slip_angle_deg: float,
    slip_ratio: float,
    camber_deg: float,
    grip_scale: float = 1.0,
) -> Dict[str, float]:
    lat = tyre_cfg.get("LATERAL_COEFFICIENTS", {}) if isinstance(tyre_cfg.get("LATERAL_COEFFICIENTS"), dict) else {}
    lon = tyre_cfg.get("LONGITUDINAL_COEFFICIENTS", {}) if isinstance(tyre_cfg.get("LONGITUDINAL_COEFFICIENTS"), dict) else {}
    sc = tyre_cfg.get("SCALING_COEFFICIENTS", {}) if isinstance(tyre_cfg.get("SCALING_COEFFICIENTS"), dict) else {}
    vert = tyre_cfg.get("VERTICAL", {}) if isinstance(tyre_cfg.get("VERTICAL"), dict) else {}

    def _g(d: dict, key: str, default: float = 0.0) -> float:
        try:
            return float(d.get(key, default))
        except Exception:
            return float(default)

    fz = max(0.0, float(fz_n))
    fz0 = max(1.0, _g(vert, "FNOMIN", 4000.0) * _g(sc, "LFZ0", 1.0))
    dfz = (fz - fz0) / fz0
    gamma = math.radians(float(camber_deg))
    alpha = math.radians(float(slip_angle_deg))
    kappa = float(slip_ratio)
    eps = 1e-9

    lcy = _g(sc, "LCY", 1.0)
    lmuy = _g(sc, "LMUY", 1.0) * float(grip_scale)
    ley = _g(sc, "LEY", 1.0)
    lky = _g(sc, "LKY", 1.0)
    lhy = _g(sc, "LHY", 1.0)
    lvy = _g(sc, "LVY", 1.0)
    lyka = _g(sc, "LYKA", 1.0)

    # Lateral base force Fy0
    cy = _g(lat, "PCY1", 1.3) * lcy
    mu_y = (_g(lat, "PDY1", 1.0) + _g(lat, "PDY2", 0.0) * dfz) * (1.0 - _g(lat, "PDY3", 0.0) * gamma * gamma) * lmuy
    dy = mu_y * fz
    shy = (_g(lat, "PHY1", 0.0) + _g(lat, "PHY2", 0.0) * dfz) * lhy + _g(lat, "PHY3", 0.0) * gamma
    alpha_y = alpha + shy
    ky = _g(lat, "PKY1", 20.0) * fz0 * math.sin(
        2.0 * math.atan(fz / max(eps, _g(lat, "PKY2", 1.5) * fz0))
    ) * (1.0 - _g(lat, "PKY3", 0.0) * abs(gamma)) * lky
    by = ky / max(eps, cy * dy)
    ey = (_g(lat, "PEY1", -1.5) + _g(lat, "PEY2", 0.0) * dfz) * (
        1.0 - (_g(lat, "PEY3", 0.0) + _g(lat, "PEY4", 0.0) * gamma) * math.copysign(1.0, alpha_y if abs(alpha_y) > eps else 1.0)
    ) * ley
    sv = ((_g(lat, "PVY1", 0.0) + _g(lat, "PVY2", 0.0) * dfz) * lvy
          + (_g(lat, "PVY3", 0.0) + _g(lat, "PVY4", 0.0) * dfz) * gamma * lvy) * fz
    fy0 = dy * math.sin(cy * math.atan(by * alpha_y - ey * (by * alpha_y - math.atan(by * alpha_y)))) + sv

    # Longitudinal base force Fx0
    lcx = _g(sc, "LCX", 1.0)
    lmux = _g(sc, "LMUX", 1.0) * float(grip_scale)
    lex = _g(sc, "LEX", 1.0)
    lkx = _g(sc, "LKX", 1.0)
    lhx = _g(sc, "LHX", 1.0)
    lvx = _g(sc, "LVX", 1.0)
    lxal = _g(sc, "LXAL", 1.0)

    cx = _g(lon, "PCX1", 1.6) * lcx
    mu_x = (_g(lon, "PDX1", 1.0) + _g(lon, "PDX2", 0.0) * dfz) * (1.0 - _g(lon, "PDX3", 0.0) * gamma * gamma) * lmux
    dx = mu_x * fz
    shx = (_g(lon, "PHX1", 0.0) + _g(lon, "PHX2", 0.0) * dfz) * lhx
    kappa_x = kappa + shx
    kx = fz * (_g(lon, "PKX1", 20.0) + _g(lon, "PKX2", 0.0) * dfz) * math.exp(_g(lon, "PKX3", 0.0) * dfz) * lkx
    bx = kx / max(eps, cx * dx)
    ex = (_g(lon, "PEX1", 0.0) + _g(lon, "PEX2", 0.0) * dfz + _g(lon, "PEX3", 0.0) * dfz * dfz) * (
        1.0 - _g(lon, "PEX4", 0.0) * math.copysign(1.0, kappa_x if abs(kappa_x) > eps else 1.0)
    ) * lex
    svx = fz * (_g(lon, "PVX1", 0.0) + _g(lon, "PVX2", 0.0) * dfz) * lvx
    fx0 = dx * math.sin(cx * math.atan(bx * kappa_x - ex * (bx * kappa_x - math.atan(bx * kappa_x)))) + svx

    cyk = _g(lat, "RCY1", 1.0)
    byk = _g(lat, "RBY1", 1.0) * math.cos(math.atan(_g(lat, "RBY2", 0.0) * (alpha - _g(lat, "RBY3", 0.0)))) * lyka
    eyk = _g(lat, "REY1", 0.0) + _g(lat, "REY2", 0.0) * dfz
    shyk = _g(lat, "RHY1", 0.0) + _g(lat, "RHY2", 0.0) * dfz

    def _gfun(k: float) -> float:
        return math.cos(cyk * math.atan(byk * k - eyk * (byk * k - math.atan(byk * k))))

    gyk = _gfun(kappa + shyk) / max(eps, _gfun(shyk))
    fy = fy0 * gyk

    # Combined-slip scaling for Fx with slip angle dependency
    cxa = _g(lon, "RCX1", 1.0)
    bxa = _g(lon, "RBX1", 1.0) * math.cos(math.atan(_g(lon, "RBX2", 0.0) * kappa)) * lxal
    exa = _g(lon, "REX1", 0.0) + _g(lon, "REX2", 0.0) * dfz
    shxa = _g(lon, "RHX1", 0.0)
    alpha_s = alpha + shxa

    def _gxa(a: float) -> float:
        return math.cos(cxa * math.atan(bxa * a - exa * (bxa * a - math.atan(bxa * a))))

    gxa = _gxa(alpha_s) / max(eps, _gxa(shxa))
    fx = fx0 * gxa

    return {"fy_n": float(fy), "fx_n": float(fx)}


def _get_cached_suspension_geometry(json_data: dict) -> SuspensionGeometryExact:
    json_path = _ensure_json_path(json_data)
    signature = _json_signature_cached(json_data)
    key = f"{json_path}:{signature}"
    cache = _state.get("geometry_solver_cache")
    if not isinstance(cache, dict):
        cache = {}
        _state["geometry_solver_cache"] = cache
    susp = cache.get(key)
    if not isinstance(susp, SuspensionGeometryExact):
        susp = SuspensionGeometryExact(json_path=json_path)
        # Keep cache bounded: this app normally works on one setup at a time.
        if len(cache) > 3:
            cache.clear()
        cache[key] = susp
    return susp


def _get_cached_body_attitude_ref(json_data: dict):
    signature = _json_signature_cached(json_data)
    cache = _state.get("body_attitude_ref_cache")
    if not isinstance(cache, dict):
        cache = {}
        _state["body_attitude_ref_cache"] = cache
    ref = cache.get(signature)
    if ref is None:
        try:
            ref = build_body_reference_from_json(json_data)
        except Exception:
            ref = None
        if len(cache) > 6:
            cache.clear()
        cache[signature] = ref
    return ref


def _estimate_cg_from_state_fast(
    body_ref: Any,
    hf_mm: float,
    hr_mm: float,
    rf_mm_equiv: float,
    rr_mm_equiv: float,
    front_track_mm: float,
    rear_track_mm: float,
) -> Tuple[float, float, float, float]:
    if body_ref is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    try:
        wheelbase_mm = float(getattr(body_ref, "wheelbase_mm", 0.0))
        cg_body = np.asarray(getattr(body_ref, "cg_body_mm", np.zeros(3)), dtype=float).reshape(3)
        if wheelbase_mm <= 1e-9:
            return float("nan"), float("nan"), float("nan"), float("nan")
        roll_front = math.atan(float(rf_mm_equiv) / max(1e-9, float(front_track_mm)))
        roll_rear = math.atan(float(rr_mm_equiv) / max(1e-9, float(rear_track_mm)))
        roll_body = 0.5 * (roll_front + roll_rear)
        pitch_body = math.atan((float(hr_mm) - float(hf_mm)) / max(1e-9, wheelbase_mm))
        heave_body = 0.5 * (float(hf_mm) + float(hr_mm))

        cr = math.cos(roll_body)
        sr = math.sin(roll_body)
        cp = math.cos(pitch_body)
        sp = math.sin(pitch_body)
        r_pitch_roll = np.array(
            [
                [cp, sp * sr, sp * cr],
                [0.0, cr, -sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=float,
        )
        translation = np.array([0.0, 0.0, -heave_body], dtype=float)
        cg_global = translation + r_pitch_roll @ cg_body
        return float(cg_global[0]), float(cg_global[1]), float(cg_global[2]), float(cg_global[2])
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")


def compute_center_antis_for_state(
    json_data: dict,
    hf: float,
    rf: float,
    hr: float,
    rr: float,
    state_cache: Optional[Dict[Tuple[float, float, float, float], Dict[str, float]]] = None,
) -> Dict[str, float]:
    state_key = (round(float(hf), 6), round(float(rf), 6), round(float(hr), 6), round(float(rr), 6))
    if isinstance(state_cache, dict) and state_key in state_cache:
        return dict(state_cache[state_key])

    global_cache = _state.get("center_state_cache")
    if not isinstance(global_cache, dict):
        global_cache = {}
        _state["center_state_cache"] = global_cache
    signature = _json_signature_cached(json_data)
    sig_cache = global_cache.get(signature)
    if not isinstance(sig_cache, dict):
        sig_cache = {}
        global_cache[signature] = sig_cache
    if state_key in sig_cache:
        cached = dict(sig_cache[state_key])
        if isinstance(state_cache, dict):
            state_cache[state_key] = cached
        return cached

    susp = _get_cached_suspension_geometry(json_data)
    vehicle = susp._vehicle_params

    df = susp.generate_4wheel_map(
        hf_values=[float(hf)],
        hr_values=[float(hr)],
        rf_values=[float(rf)],
        rr_values=[float(rr)],
        x_cg=vehicle["x_cg_mm"],
        z_cg=vehicle["z_cg_mm"],
        contact_patch_mode="static_offset",
        roll_mode="chassis_rotation",
        verbose=False,
    )
    if df.empty:
        raise RuntimeError("Could not solve requested state.")

    row = df.iloc[0]
    zfl = float(row.get("zfl", 0.0))
    zfr = float(row.get("zfr", 0.0))
    zrl = float(row.get("zrl", 0.0))
    zrr = float(row.get("zrr", 0.0))

    def _num(key: str, default: float = np.nan) -> float:
        value = row.get(key, default)
        return float(default) if pd.isna(value) else float(value)

    front_rh = float(vehicle.get("ride_height_f_mm", 0.0)) - float(hf)
    rear_rh = float(vehicle.get("ride_height_r_mm", 0.0)) - float(hr)

    cg_x_mm = float("nan")
    cg_y_mm = float("nan")
    cg_z_mm = float("nan")
    h_cg_mm = float("nan")
    ba_ref = _get_cached_body_attitude_ref(json_data)
    cg_x_mm, cg_y_mm, cg_z_mm, h_cg_mm = _estimate_cg_from_state_fast(
        body_ref=ba_ref,
        hf_mm=float(hf),
        hr_mm=float(hr),
        rf_mm_equiv=_num("rf_mm_equiv", 0.0),
        rr_mm_equiv=_num("rr_mm_equiv", 0.0),
        front_track_mm=float(getattr(susp, "front_track_mm", np.nan)),
        rear_track_mm=float(getattr(susp, "rear_track_mm", np.nan)),
    )
    if not np.isfinite(cg_x_mm):
        try:
            ba_summary = compute_body_attitude_summary(
                json_data=json_data,
                state_4w={"hf": float(hf), "rf": _num("rf_mm_equiv", 0.0), "hr": float(hr), "rr": _num("rr_mm_equiv", 0.0)},
                ref=ba_ref,
                roll_strategy="average_axle_roll",
            )
            cg_global = ba_summary.get("cg_global_mm")
            if isinstance(cg_global, (list, tuple)) and len(cg_global) == 3:
                cg_x_mm = float(cg_global[0])
                cg_y_mm = float(cg_global[1])
                cg_z_mm = float(cg_global[2])
            h_cg_mm = float(ba_summary.get("h_cg_mm", cg_z_mm))
        except Exception:
            pass

    result = {
        "hf": float(hf),
        "rf": float(rf),
        "hr": float(hr),
        "rr": float(rr),
        "rf_mm_equiv": _num("rf_mm_equiv", np.nan),
        "rr_mm_equiv": _num("rr_mm_equiv", np.nan),
        "front_ride_height": front_rh,
        "rear_ride_height": rear_rh,
        "zfl": zfl,
        "zfr": zfr,
        "zrl": zrl,
        "zrr": zrr,
        "front_track_mm": float(getattr(susp, "front_track_mm", np.nan)),
        "rear_track_mm": float(getattr(susp, "rear_track_mm", np.nan)),
        "cg_x_mm": cg_x_mm,
        "cg_y_mm": cg_y_mm,
        "cg_z_mm": cg_z_mm,
        "h_cg_mm": h_cg_mm,
        "front_rc_z_mm": _num("front_rc_z", np.nan),
        "rear_rc_z_mm": _num("rear_rc_z", np.nan),
        "front_rc_height_mm": _num("front_rc_height", np.nan),
        "rear_rc_height_mm": _num("rear_rc_height", np.nan),
        "pitch_x_mm": _num("pitch_x", np.nan),
        "pitch_z_mm": _num("pitch_z", np.nan),
        "pitch_ic_front_x_mm": _num("pitch_ic_front_x", np.nan),
        "pitch_ic_front_z_mm": _num("pitch_ic_front_z", np.nan),
        "pitch_ic_rear_x_mm": _num("pitch_ic_rear_x", np.nan),
        "pitch_ic_rear_z_mm": _num("pitch_ic_rear_z", np.nan),
        "anti_dive_pct": _num("anti_dive_pct", np.nan),
        "anti_squat_pct": _num("anti_squat_pct", np.nan),
    }
    if isinstance(state_cache, dict):
        state_cache[state_key] = dict(result)
    # keep bounded
    if len(sig_cache) > 4000:
        sig_cache.clear()
    sig_cache[state_key] = dict(result)
    return result


def _build_dynamic_aero_rows(json_data: dict, body: dict) -> Dict[str, object]:
    cfg = (json_data or {}).get("config", {})
    aero = cfg.get("aero", {}) if isinstance(cfg.get("aero"), dict) else {}
    if not aero:
        raise ValueError("Missing config.aero in loaded JSON.")

    speed_mode = str(body.get("speed_mode", "single")).lower()
    speeds_kph = _dynamic_range(float(body.get("speed_min_kph", 120.0)), float(body.get("speed_max_kph", 220.0)), float(body.get("speed_step_kph", 20.0))) if speed_mode == "range" else [float(body.get("speed_kph", 140.0))]
    sweep_var = str(body.get("sweep_variable", "front_heave"))

    if sweep_var in {"roll_custom", "heave_custom"}:
        front_vals = _dynamic_range(float(body.get("front_min", -2.0)), float(body.get("front_max", 2.0)), float(body.get("front_step", 0.2)))
        rear_vals = _dynamic_range(float(body.get("rear_min", -2.0)), float(body.get("rear_max", 2.0)), float(body.get("rear_step", 0.2)))
        sweep_grid = [(float(fv), float(rv)) for fv in front_vals for rv in rear_vals]
    else:
        vals = _dynamic_range(float(body.get("sweep_min", -10.0)), float(body.get("sweep_max", 10.0)), float(body.get("sweep_step", 1.0)))
        sweep_grid = [(float(v), None) for v in vals]

    flap = aero.get("flapAngles", {}) if isinstance(aero.get("flapAngles"), dict) else {}
    offsets = aero.get("coefficientOffsets", {}) if isinstance(aero.get("coefficientOffsets"), dict) else {}
    drs_block = aero.get("DRS", {}) if isinstance(aero.get("DRS"), dict) else {}

    cl_f_poly = aero.get("PolynomialCLiftBodyFDefinition", [])
    cl_r_poly = aero.get("PolynomialCLiftBodyRDefinition", [])
    cd_poly = aero.get("PolynomialCDragBodyDefinition", [])
    drs_cl_f_poly = drs_block.get("CLiftBodyFDRSPolynomial", [])
    drs_cl_r_poly = drs_block.get("CLiftBodyRDRSPolynomial", [])
    drs_cd_poly = drs_block.get("CDragBodyDRSPolynomial", [])

    a_ref = float(aero.get("ARef", 1.0))
    rho = float(body.get("air_density", 1.225))
    drs_effective = bool(body.get("drs_on", False) and drs_block.get("bDRSEnabled", aero.get("bDRSEnabled", False)))

    cl_front_factor = float(aero.get("rCLiftBodyFFactor", 1.0))
    cl_rear_factor = float(aero.get("rCLiftBodyRFactor", 1.0))
    cd_factor = float(aero.get("rCDragBodyFactor", 1.0))
    cl_front_offset = float(offsets.get("CLiftBodyFUserOffset", 0.0))
    cl_rear_offset = float(offsets.get("CLiftBodyRUserOffset", 0.0))
    cl_global_offset = float(offsets.get("CLiftBodyUserOffset", 0.0))
    cd_offset = float(offsets.get("CDragBodyUserOffset", 0.0))
    aero_bal_offset = float(offsets.get("rAeroBalanceUserOffset", 0.0))
    front_global_share = np.clip(0.5 + aero_bal_offset, 0.0, 1.0)
    rear_global_share = 1.0 - front_global_share

    chassis = cfg.get("chassis", {}) if isinstance(cfg.get("chassis"), dict) else {}
    car_mass_block = chassis.get("carRunningMass", {}) if isinstance(chassis.get("carRunningMass"), dict) else {}
    total_mass_kg = float(car_mass_block.get("mCar", 0.0))
    r_weight_bal_f = float(car_mass_block.get("rWeightBalF", 0.5))
    m_hub_f = float(chassis.get("mHubF", 0.0))
    m_hub_r = float(chassis.get("mHubR", 0.0))
    mass_unsprung_total = 2.0 * (m_hub_f + m_hub_r)
    mass_sprung_total = max(0.0, total_mass_kg - mass_unsprung_total)
    g = 9.80665

    tyres_root = cfg.get("tyres", {}) if isinstance(cfg.get("tyres"), dict) else {}
    tyre_front_cfg = tyres_root.get("front", {}) if isinstance(tyres_root.get("front"), dict) else {}
    tyre_rear_cfg = tyres_root.get("rear", {}) if isinstance(tyres_root.get("rear"), dict) else {}
    tyre_global_grip = float(tyres_root.get("rGripFactor", 1.0))
    tyre_front_grip = float(tyre_front_cfg.get("rGripFactor", 1.0))
    tyre_rear_grip = float(tyre_rear_cfg.get("rGripFactor", 1.0))

    slip_angle_fl_deg = float(body.get("slip_angle_fl_deg", 0.0))
    slip_angle_fr_deg = float(body.get("slip_angle_fr_deg", 0.0))
    slip_angle_rl_deg = float(body.get("slip_angle_rl_deg", 0.0))
    slip_angle_rr_deg = float(body.get("slip_angle_rr_deg", 0.0))
    slip_ratio_fl = float(body.get("slip_ratio_fl", 0.0))
    slip_ratio_fr = float(body.get("slip_ratio_fr", 0.0))
    slip_ratio_rl = float(body.get("slip_ratio_rl", 0.0))
    slip_ratio_rr = float(body.get("slip_ratio_rr", 0.0))
    camber_fl_deg = float(body.get("camber_fl_deg", 0.0))
    camber_fr_deg = float(body.get("camber_fr_deg", 0.0))
    camber_rl_deg = float(body.get("camber_rl_deg", 0.0))
    camber_rr_deg = float(body.get("camber_rr_deg", 0.0))

    acc_units = str(body.get("acc_units", "g")).lower()
    ax_input = float(body.get("ax", 0.0))
    ay_input = float(body.get("ay", 0.0))
    ax_mps2 = ax_input * g if acc_units in {"g", "gee"} else ax_input
    ay_mps2 = ay_input * g if acc_units in {"g", "gee"} else ay_input

    sus = cfg.get("suspension", {}) if isinstance(cfg.get("suspension"), dict) else {}
    front_pick = (((sus.get("front") or {}).get("external") or {}).get("pickUpPts") or {})
    rear_pick = (((sus.get("rear") or {}).get("external") or {}).get("pickUpPts") or {})
    front_axle_c = front_pick.get("rAxleC", [0.0, 0.0, 0.0])
    rear_axle_c = rear_pick.get("rAxleC", [-1.0, 0.0, 0.0])
    try:
        front_axle_x_mm = float(front_axle_c[0]) * 1000.0
    except Exception:
        front_axle_x_mm = 0.0
    try:
        rear_axle_x_mm = float(rear_axle_c[0]) * 1000.0
    except Exception:
        rear_axle_x_mm = -1000.0
    wheelbase_mm = max(1.0, abs(front_axle_x_mm - rear_axle_x_mm))

    rows: List[Dict[str, object]] = []
    static_state = compute_center_antis_for_state(json_data, hf=0.0, rf=0.0, hr=0.0, rr=0.0)
    base_front_rh_mm = float(static_state.get("front_ride_height", 0.0))
    base_rear_rh_mm = float(static_state.get("rear_ride_height", 0.0))
    base_h_cg_mm = float(static_state.get("h_cg_mm", 0.0))
    base_cg_x_mm = float(static_state.get("cg_x_mm", np.nan))
    base_cg_y_mm = float(static_state.get("cg_y_mm", np.nan))
    base_long_ratio = np.clip(1.0 - r_weight_bal_f, 0.0, 1.0)

    for speed_kph in speeds_kph:
        v_mps = float(speed_kph) / 3.6
        qA = 0.5 * rho * (v_mps ** 2) * a_ref

        for main_val, rear_val in sweep_grid:
            hf = hr = rf = rr = 0.0
            if sweep_var == "front_roll":
                rf = float(main_val)
            elif sweep_var == "rear_roll":
                rr = float(main_val)
            elif sweep_var == "global_roll":
                rf = rr = float(main_val)
            elif sweep_var == "front_heave":
                hf = float(main_val)
            elif sweep_var == "rear_heave":
                hr = float(main_val)
            elif sweep_var == "global_heave":
                hf = hr = float(main_val)
            elif sweep_var == "roll_custom":
                rf = float(main_val)
                rr = float(rear_val if rear_val is not None else 0.0)
            elif sweep_var == "heave_custom":
                hf = float(main_val)
                hr = float(rear_val if rear_val is not None else 0.0)

            analysis = compute_center_antis_for_state(json_data, hf=hf, rf=rf, hr=hr, rr=rr)
            zfl = float(analysis.get("zfl", 0.0))
            zfr = float(analysis.get("zfr", 0.0))
            zrl = float(analysis.get("zrl", 0.0))
            zrr = float(analysis.get("zrr", 0.0))
            rf_eq = float(analysis.get("rf_mm_equiv", 0.0))
            rr_eq = float(analysis.get("rr_mm_equiv", 0.0))
            if abs(zfl) < 1e-9 and abs(zfr) < 1e-9 and abs(rf_eq) > 1e-9:
                zfl, zfr = hf + 0.5 * rf_eq, hf - 0.5 * rf_eq
            if abs(zrl) < 1e-9 and abs(zrr) < 1e-9 and abs(rr_eq) > 1e-9:
                zrl, zrr = hr + 0.5 * rr_eq, hr - 0.5 * rr_eq

            ride_fl_mm = base_front_rh_mm + zfl
            ride_fr_mm = base_front_rh_mm + zfr
            ride_rl_mm = base_rear_rh_mm + zrl
            ride_rr_mm = base_rear_rh_mm + zrr
            hRideF_mm = 0.5 * (ride_fl_mm + ride_fr_mm)
            hRideR_mm = 0.5 * (ride_rl_mm + ride_rr_mm)
            h_cg_now_mm = float(analysis.get("h_cg_mm", base_h_cg_mm))
            body_heave_correction_mm = h_cg_now_mm - base_h_cg_mm
            hRideF_m = (hRideF_mm + body_heave_correction_mm) / 1000.0
            hRideR_m = (hRideR_mm + body_heave_correction_mm) / 1000.0

            variables = {"hRideF": hRideF_m, "hRideR": hRideR_m, "aFlapF": float(flap.get("aFlapF", 0.0)), "aFlapR": float(flap.get("aFlapR", 0.0))}
            raw_cl_front = _eval_poly_terms(cl_f_poly, variables)
            raw_cl_rear = _eval_poly_terms(cl_r_poly, variables)
            raw_cd = _eval_poly_terms(cd_poly, variables)
            drs_delta_cl_front = _eval_poly_terms(drs_cl_f_poly, variables) if drs_effective else 0.0
            drs_delta_cl_rear = _eval_poly_terms(drs_cl_r_poly, variables) if drs_effective else 0.0
            drs_delta_cd = _eval_poly_terms(drs_cd_poly, variables) if drs_effective else 0.0
            effective_cl_front = ((raw_cl_front + cl_front_offset + cl_global_offset * front_global_share) * cl_front_factor) + drs_delta_cl_front
            effective_cl_rear = ((raw_cl_rear + cl_rear_offset + cl_global_offset * rear_global_share) * cl_rear_factor) + drs_delta_cl_rear
            effective_cd = ((raw_cd + cd_offset) * cd_factor) + drs_delta_cd

            front_aero_load_n = qA * effective_cl_front
            rear_aero_load_n = qA * effective_cl_rear
            total_aero_load_n = front_aero_load_n + rear_aero_load_n
            aero_balance_front_pct = (
                100.0 * front_aero_load_n / total_aero_load_n
                if abs(total_aero_load_n) > 1e-12
                else np.nan
            )
            cg_x_mm = float(analysis.get("cg_x_mm", np.nan))
            cg_y_mm = float(analysis.get("cg_y_mm", np.nan))
            cg_z_mm = float(analysis.get("cg_z_mm", np.nan))

            longitudinal_ratio = base_long_ratio
            axle_dx_mm = (rear_axle_x_mm - front_axle_x_mm)
            if np.isfinite(cg_x_mm) and np.isfinite(base_cg_x_mm) and abs(axle_dx_mm) > 1e-9:
                raw_base_ratio = (base_cg_x_mm - front_axle_x_mm) / axle_dx_mm
                raw_now_ratio = (cg_x_mm - front_axle_x_mm) / axle_dx_mm
                delta_ratio = raw_now_ratio - raw_base_ratio
                longitudinal_ratio = np.clip(base_long_ratio + delta_ratio, 0.0, 1.0)

            x_cg_from_front_mm = wheelbase_mm * longitudinal_ratio
            x_cg_from_rear_mm = wheelbase_mm * (1.0 - longitudinal_ratio)
            if not np.isfinite(cg_x_mm):
                cg_x_mm = rear_axle_x_mm + x_cg_from_rear_mm
            if not np.isfinite(cg_y_mm):
                cg_y_mm = 0.0
            if not np.isfinite(cg_z_mm):
                cg_z_mm = h_cg_now_mm if np.isfinite(h_cg_now_mm) else 0.0

            sprung_weight_n = mass_sprung_total * g
            unsprung_fl_n = m_hub_f * g
            unsprung_fr_n = m_hub_f * g
            unsprung_rl_n = m_hub_r * g
            unsprung_rr_n = m_hub_r * g

            front_sprung_axle_n = sprung_weight_n * (1.0 - longitudinal_ratio)
            rear_sprung_axle_n = sprung_weight_n * longitudinal_ratio

            front_track_mm = float(analysis.get("front_track_mm", np.nan))
            rear_track_mm = float(analysis.get("rear_track_mm", np.nan))
            if not np.isfinite(front_track_mm) or front_track_mm <= 1e-9:
                front_track_mm = 2.0 * abs(float(front_axle_c[1]) * 1000.0 if len(front_axle_c) > 1 else 800.0)
            if not np.isfinite(rear_track_mm) or rear_track_mm <= 1e-9:
                rear_track_mm = 2.0 * abs(float(rear_axle_c[1]) * 1000.0 if len(rear_axle_c) > 1 else 800.0)

            delta_cg_y_mm = (cg_y_mm - base_cg_y_mm) if np.isfinite(base_cg_y_mm) else cg_y_mm
            ratio_left_front = np.clip(0.5 + delta_cg_y_mm / max(front_track_mm, 1e-9), 0.0, 1.0)
            ratio_left_rear = np.clip(0.5 + delta_cg_y_mm / max(rear_track_mm, 1e-9), 0.0, 1.0)
            front_left_sprung_n = front_sprung_axle_n * (1.0 - ratio_left_front)
            front_right_sprung_n = front_sprung_axle_n - front_left_sprung_n
            rear_left_sprung_n = rear_sprung_axle_n * (1.0 - ratio_left_rear)
            rear_right_sprung_n = rear_sprung_axle_n - rear_left_sprung_n

            front_left_static_load_n = front_left_sprung_n + unsprung_fl_n
            front_right_static_load_n = front_right_sprung_n + unsprung_fr_n
            rear_left_static_load_n = rear_left_sprung_n + unsprung_rl_n
            rear_right_static_load_n = rear_right_sprung_n + unsprung_rr_n
            front_static_load_n = front_left_static_load_n + front_right_static_load_n
            rear_static_load_n = rear_left_static_load_n + rear_right_static_load_n

            h_cg_m = (h_cg_now_mm / 1000.0) if np.isfinite(h_cg_now_mm) else abs(float(chassis.get("zCoG", -0.25)))
            wheelbase_m = wheelbase_mm / 1000.0
            front_track_m = front_track_mm / 1000.0
            rear_track_m = rear_track_mm / 1000.0
            sprung_mass_front_kg = front_sprung_axle_n / g
            sprung_mass_rear_kg = rear_sprung_axle_n / g

            front_rc_h_mm = float(analysis.get("front_rc_height_mm", np.nan))
            rear_rc_h_mm = float(analysis.get("rear_rc_height_mm", np.nan))
            if not np.isfinite(front_rc_h_mm):
                front_rc_h_mm = float(analysis.get("front_rc_z_mm", np.nan))
            if not np.isfinite(rear_rc_h_mm):
                rear_rc_h_mm = float(analysis.get("rear_rc_z_mm", np.nan))

            lam_cg = float(np.clip(longitudinal_ratio, 0.0, 1.0))
            h_ra_cg_mm = (
                front_rc_h_mm + lam_cg * (rear_rc_h_mm - front_rc_h_mm)
                if (np.isfinite(front_rc_h_mm) and np.isfinite(rear_rc_h_mm))
                else np.nan
            )
            h_ra_cg_m = (h_ra_cg_mm / 1000.0) if np.isfinite(h_ra_cg_mm) else 0.0

            pitch_ic_f_z_mm = float(analysis.get("pitch_ic_front_z_mm", np.nan))
            pitch_ic_r_z_mm = float(analysis.get("pitch_ic_rear_z_mm", np.nan))
            if np.isfinite(pitch_ic_f_z_mm) and np.isfinite(pitch_ic_r_z_mm):
                h_pc_mm = pitch_ic_f_z_mm + lam_cg * (pitch_ic_r_z_mm - pitch_ic_f_z_mm)
            else:
                h_pc_mm = float(analysis.get("pitch_z_mm", np.nan))
            h_pc_m = (h_pc_mm / 1000.0) if np.isfinite(h_pc_mm) else 0.0

            pitch_arm_m = h_cg_m - h_pc_m
            roll_arm_m = h_cg_m - h_ra_cg_m

            longitudinal_transfer_total_n = mass_sprung_total * ax_mps2 * pitch_arm_m / max(wheelbase_m, 1e-9)
            lateral_transfer_front_total_n = sprung_mass_front_kg * ay_mps2 * roll_arm_m / max(front_track_m, 1e-9)
            lateral_transfer_rear_total_n = sprung_mass_rear_kg * ay_mps2 * roll_arm_m / max(rear_track_m, 1e-9)

            delta_long_front_each_n = -0.5 * longitudinal_transfer_total_n
            delta_long_rear_each_n = +0.5 * longitudinal_transfer_total_n
            delta_lat_fl_n = -0.5 * lateral_transfer_front_total_n
            delta_lat_fr_n = +0.5 * lateral_transfer_front_total_n
            delta_lat_rl_n = -0.5 * lateral_transfer_rear_total_n
            delta_lat_rr_n = +0.5 * lateral_transfer_rear_total_n

            delta_load_fl_mech_n = delta_long_front_each_n + delta_lat_fl_n
            delta_load_fr_mech_n = delta_long_front_each_n + delta_lat_fr_n
            delta_load_rl_mech_n = delta_long_rear_each_n + delta_lat_rl_n
            delta_load_rr_mech_n = delta_long_rear_each_n + delta_lat_rr_n

            fz_mechanical_fl_n = front_left_static_load_n + delta_load_fl_mech_n
            fz_mechanical_fr_n = front_right_static_load_n + delta_load_fr_mech_n
            fz_mechanical_rl_n = rear_left_static_load_n + delta_load_rl_mech_n
            fz_mechanical_rr_n = rear_right_static_load_n + delta_load_rr_mech_n

            fz_total_fl_n = fz_mechanical_fl_n + 0.5 * front_aero_load_n
            fz_total_fr_n = fz_mechanical_fr_n + 0.5 * front_aero_load_n
            fz_total_rl_n = fz_mechanical_rl_n + 0.5 * rear_aero_load_n
            fz_total_rr_n = fz_mechanical_rr_n + 0.5 * rear_aero_load_n

            tyre_fl = _mf_tyre_combined_from_json(
                tyre_front_cfg, fz_total_fl_n, slip_angle_fl_deg, slip_ratio_fl, camber_fl_deg,
                grip_scale=tyre_global_grip * tyre_front_grip,
            )
            tyre_fr = _mf_tyre_combined_from_json(
                tyre_front_cfg, fz_total_fr_n, slip_angle_fr_deg, slip_ratio_fr, camber_fr_deg,
                grip_scale=tyre_global_grip * tyre_front_grip,
            )
            tyre_rl = _mf_tyre_combined_from_json(
                tyre_rear_cfg, fz_total_rl_n, slip_angle_rl_deg, slip_ratio_rl, camber_rl_deg,
                grip_scale=tyre_global_grip * tyre_rear_grip,
            )
            tyre_rr = _mf_tyre_combined_from_json(
                tyre_rear_cfg, fz_total_rr_n, slip_angle_rr_deg, slip_ratio_rr, camber_rr_deg,
                grip_scale=tyre_global_grip * tyre_rear_grip,
            )
            fy_fl_n = abs(float(tyre_fl["fy_n"]))
            fy_fr_n = abs(float(tyre_fr["fy_n"]))
            fy_rl_n = abs(float(tyre_rl["fy_n"]))
            fy_rr_n = abs(float(tyre_rr["fy_n"]))
            fx_fl_n = abs(float(tyre_fl["fx_n"]))
            fx_fr_n = abs(float(tyre_fr["fx_n"]))
            fx_rl_n = abs(float(tyre_rl["fx_n"]))
            fx_rr_n = abs(float(tyre_rr["fx_n"]))

            row = {
                "speed_kph": float(speed_kph),
                "sweep_variable": sweep_var,
                "sweep_value_front": float(main_val),
                "sweep_value_rear": _json_scalar(rear_val),
                "hf_mm": float(hf),
                "rf_deg": float(rf),
                "hr_mm": float(hr),
                "rr_deg": float(rr),
                "front_aero_load_n": float(front_aero_load_n),
                "rear_aero_load_n": float(rear_aero_load_n),
                "total_aero_load_n": float(total_aero_load_n),
                "aero_balance_front_pct": float(aero_balance_front_pct) if np.isfinite(aero_balance_front_pct) else None,
                "drag_force_n": float(qA * effective_cd),
                "drs_enabled_effective": bool(drs_effective),
                "hRideF": float(hRideF_m),
                "hRideR": float(hRideR_m),
                "effective_cl_front": float(effective_cl_front),
                "effective_cl_rear": float(effective_cl_rear),
                "effective_cd": float(effective_cd),
                "fz_mechanical_fl_n": float(fz_mechanical_fl_n),
                "fz_mechanical_fr_n": float(fz_mechanical_fr_n),
                "fz_mechanical_rl_n": float(fz_mechanical_rl_n),
                "fz_mechanical_rr_n": float(fz_mechanical_rr_n),
                "front_left_static_load_n": float(front_left_static_load_n),
                "front_right_static_load_n": float(front_right_static_load_n),
                "rear_left_static_load_n": float(rear_left_static_load_n),
                "rear_right_static_load_n": float(rear_right_static_load_n),
                "front_static_load_n": float(front_static_load_n),
                "rear_static_load_n": float(rear_static_load_n),
                "longitudinal_ratio": float(longitudinal_ratio),
                "cg_x_mm": float(cg_x_mm),
                "cg_y_mm": float(cg_y_mm),
                "cg_z_mm": float(cg_z_mm),
                "front_track_mm": float(front_track_mm),
                "rear_track_mm": float(rear_track_mm),
                "load_transfer_longitudinal_n": float(longitudinal_transfer_total_n),
                "lateral_transfer_front_total_n": float(lateral_transfer_front_total_n),
                "lateral_transfer_rear_total_n": float(lateral_transfer_rear_total_n),
                "h_pc_mm": float(h_pc_mm) if np.isfinite(h_pc_mm) else None,
                "h_ra_cg_mm": float(h_ra_cg_mm) if np.isfinite(h_ra_cg_mm) else None,
                "pitch_arm_m": float(pitch_arm_m),
                "roll_arm_m": float(roll_arm_m),
                "delta_load_fl_mech_n": float(delta_load_fl_mech_n),
                "delta_load_fr_mech_n": float(delta_load_fr_mech_n),
                "delta_load_rl_mech_n": float(delta_load_rl_mech_n),
                "delta_load_rr_mech_n": float(delta_load_rr_mech_n),
                "fz_total_fl_n": float(fz_total_fl_n),
                "fz_total_fr_n": float(fz_total_fr_n),
                "fz_total_rl_n": float(fz_total_rl_n),
                "fz_total_rr_n": float(fz_total_rr_n),
                "fy_fl_n": float(fy_fl_n),
                "fy_fr_n": float(fy_fr_n),
                "fy_rl_n": float(fy_rl_n),
                "fy_rr_n": float(fy_rr_n),
                "fy_total_n": float(fy_fl_n + fy_fr_n + fy_rl_n + fy_rr_n),
                "fx_fl_n": float(fx_fl_n),
                "fx_fr_n": float(fx_fr_n),
                "fx_rl_n": float(fx_rl_n),
                "fx_rr_n": float(fx_rr_n),
                "fx_total_n": float(fx_fl_n + fx_fr_n + fx_rl_n + fx_rr_n),
            }
            rows.append(_json_clean(row))

    csv_keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in csv_keys:
                csv_keys.append(key)
    csv_io = io.StringIO()
    if csv_keys:
        csv_io.write(",".join(csv_keys) + "\n")
        for row in rows:
            values = []
            for key in csv_keys:
                text = "" if row.get(key) is None else str(row.get(key))
                if ("," in text) or ('"' in text) or ("\n" in text):
                    text = '"' + text.replace('"', '""') + '"'
                values.append(text)
            csv_io.write(",".join(values) + "\n")

    return {
        "rows": rows,
        "csv": csv_io.getvalue(),
        "speed_values_kph": [float(v) for v in speeds_kph],
        "sweep_variable": sweep_var,
        "speed_mode": speed_mode,
    }


def _compute_operating_point(json_data: dict, params: dict, _static_ref: dict = None) -> Dict[str, float]:
    """Compute key vehicle outputs for a single operating point."""
    cfg = (json_data or {}).get("config", {})
    aero = cfg.get("aero", {}) if isinstance(cfg.get("aero"), dict) else {}
    if not aero:
        raise ValueError("Missing config.aero in loaded JSON.")

    flap = aero.get("flapAngles", {}) if isinstance(aero.get("flapAngles"), dict) else {}
    offsets = aero.get("coefficientOffsets", {}) if isinstance(aero.get("coefficientOffsets"), dict) else {}
    drs_block = aero.get("DRS", {}) if isinstance(aero.get("DRS"), dict) else {}

    cl_f_poly = aero.get("PolynomialCLiftBodyFDefinition", [])
    cl_r_poly = aero.get("PolynomialCLiftBodyRDefinition", [])
    cd_poly = aero.get("PolynomialCDragBodyDefinition", [])
    drs_cl_f_poly = drs_block.get("CLiftBodyFDRSPolynomial", [])
    drs_cl_r_poly = drs_block.get("CLiftBodyRDRSPolynomial", [])
    drs_cd_poly = drs_block.get("CDragBodyDRSPolynomial", [])

    a_ref = float(aero.get("ARef", 1.0))
    rho = float(params.get("air_density", 1.225))
    drs_effective = bool(params.get("drs_on", False)) and bool(drs_block.get("bDRSEnabled", aero.get("bDRSEnabled", False)))

    cl_front_factor = float(aero.get("rCLiftBodyFFactor", 1.0))
    cl_rear_factor = float(aero.get("rCLiftBodyRFactor", 1.0))
    cd_factor = float(aero.get("rCDragBodyFactor", 1.0))
    cl_front_offset = float(offsets.get("CLiftBodyFUserOffset", 0.0))
    cl_rear_offset = float(offsets.get("CLiftBodyRUserOffset", 0.0))
    cl_global_offset = float(offsets.get("CLiftBodyUserOffset", 0.0))
    cd_offset = float(offsets.get("CDragBodyUserOffset", 0.0))
    aero_bal_offset = float(offsets.get("rAeroBalanceUserOffset", 0.0))
    front_global_share = float(np.clip(0.5 + aero_bal_offset, 0.0, 1.0))
    rear_global_share = 1.0 - front_global_share

    chassis = cfg.get("chassis", {}) if isinstance(cfg.get("chassis"), dict) else {}
    car_mass_block = chassis.get("carRunningMass", {}) if isinstance(chassis.get("carRunningMass"), dict) else {}
    total_mass_kg = float(car_mass_block.get("mCar", 0.0))
    r_weight_bal_f = float(car_mass_block.get("rWeightBalF", 0.5))
    m_hub_f = float(chassis.get("mHubF", 0.0))
    m_hub_r = float(chassis.get("mHubR", 0.0))
    mass_sprung_total = max(0.0, total_mass_kg - 2.0 * (m_hub_f + m_hub_r))
    g = 9.80665

    tyres_root = cfg.get("tyres", {}) if isinstance(cfg.get("tyres"), dict) else {}
    tyre_front_cfg = tyres_root.get("front", {}) if isinstance(tyres_root.get("front"), dict) else {}
    tyre_rear_cfg = tyres_root.get("rear", {}) if isinstance(tyres_root.get("rear"), dict) else {}
    tyre_global_grip = float(tyres_root.get("rGripFactor", 1.0))
    tyre_front_grip = float(tyre_front_cfg.get("rGripFactor", 1.0))
    tyre_rear_grip = float(tyre_rear_cfg.get("rGripFactor", 1.0))

    sus = cfg.get("suspension", {}) if isinstance(cfg.get("suspension"), dict) else {}
    front_pick = (((sus.get("front") or {}).get("external") or {}).get("pickUpPts") or {})
    rear_pick = (((sus.get("rear") or {}).get("external") or {}).get("pickUpPts") or {})
    front_axle_c = front_pick.get("rAxleC", [0.0, 0.0, 0.0])
    rear_axle_c = rear_pick.get("rAxleC", [-1.0, 0.0, 0.0])
    try:
        front_axle_x_mm = float(front_axle_c[0]) * 1000.0
    except Exception:
        front_axle_x_mm = 0.0
    try:
        rear_axle_x_mm = float(rear_axle_c[0]) * 1000.0
    except Exception:
        rear_axle_x_mm = -1000.0
    wheelbase_mm = max(1.0, abs(front_axle_x_mm - rear_axle_x_mm))

    static_state = _static_ref if isinstance(_static_ref, dict) else compute_center_antis_for_state(json_data, hf=0.0, rf=0.0, hr=0.0, rr=0.0)
    base_front_rh_mm = float(static_state.get("front_ride_height", 0.0))
    base_rear_rh_mm = float(static_state.get("rear_ride_height", 0.0))
    base_h_cg_mm = float(static_state.get("h_cg_mm", 0.0))
    base_cg_x_mm = float(static_state.get("cg_x_mm", np.nan))
    base_cg_y_mm = float(static_state.get("cg_y_mm", np.nan))
    base_long_ratio = float(np.clip(1.0 - r_weight_bal_f, 0.0, 1.0))

    hf = float(params.get("hf", 0.0))
    rf = float(params.get("rf", 0.0))
    hr = float(params.get("hr", 0.0))
    rr = float(params.get("rr", 0.0))
    speed_kph = float(params.get("speed_kph", 140.0))
    ax_mps2 = float(params.get("ax_g", 0.0)) * g
    ay_mps2 = float(params.get("ay_g", 0.0)) * g

    qA = 0.5 * rho * (speed_kph / 3.6) ** 2 * a_ref

    analysis = compute_center_antis_for_state(json_data, hf=hf, rf=rf, hr=hr, rr=rr)
    zfl = float(analysis.get("zfl", 0.0))
    zfr = float(analysis.get("zfr", 0.0))
    zrl = float(analysis.get("zrl", 0.0))
    zrr = float(analysis.get("zrr", 0.0))
    rf_eq = float(analysis.get("rf_mm_equiv", 0.0))
    rr_eq = float(analysis.get("rr_mm_equiv", 0.0))
    if abs(zfl) < 1e-9 and abs(zfr) < 1e-9 and abs(rf_eq) > 1e-9:
        zfl, zfr = hf + 0.5 * rf_eq, hf - 0.5 * rf_eq
    if abs(zrl) < 1e-9 and abs(zrr) < 1e-9 and abs(rr_eq) > 1e-9:
        zrl, zrr = hr + 0.5 * rr_eq, hr - 0.5 * rr_eq

    hRideF_mm = float(analysis.get("front_ride_height", np.nan))
    hRideR_mm = float(analysis.get("rear_ride_height", np.nan))
    if not np.isfinite(hRideF_mm) or not np.isfinite(hRideR_mm):
        hRideF_mm = 0.5 * (base_front_rh_mm + zfl + base_front_rh_mm + zfr)
        hRideR_mm = 0.5 * (base_rear_rh_mm + zrl + base_rear_rh_mm + zrr)
    h_cg_now_mm = float(analysis.get("h_cg_mm", base_h_cg_mm))
    hRideF_m = hRideF_mm / 1000.0
    hRideR_m = hRideR_mm / 1000.0

    variables = {
        "hRideF": hRideF_m, "hRideR": hRideR_m,
        "aFlapF": float(flap.get("aFlapF", 0.0)),
        "aFlapR": float(flap.get("aFlapR", 0.0)),
    }
    raw_cl_front = _eval_poly_terms(cl_f_poly, variables)
    raw_cl_rear = _eval_poly_terms(cl_r_poly, variables)
    raw_cd = _eval_poly_terms(cd_poly, variables)
    dcl_f = _eval_poly_terms(drs_cl_f_poly, variables) if drs_effective else 0.0
    dcl_r = _eval_poly_terms(drs_cl_r_poly, variables) if drs_effective else 0.0
    dcd = _eval_poly_terms(drs_cd_poly, variables) if drs_effective else 0.0
    effective_cl_front = ((raw_cl_front + cl_front_offset + cl_global_offset * front_global_share) * cl_front_factor) + dcl_f
    effective_cl_rear = ((raw_cl_rear + cl_rear_offset + cl_global_offset * rear_global_share) * cl_rear_factor) + dcl_r
    effective_cd = ((raw_cd + cd_offset) * cd_factor) + dcd

    front_aero_load_n = qA * effective_cl_front
    rear_aero_load_n = qA * effective_cl_rear
    total_aero_load_n = front_aero_load_n + rear_aero_load_n
    aero_balance_front_pct = (100.0 * front_aero_load_n / total_aero_load_n if abs(total_aero_load_n) > 1e-12 else np.nan)
    drag_force_n = qA * effective_cd

    cg_x_mm = float(analysis.get("cg_x_mm", np.nan))
    cg_y_mm = float(analysis.get("cg_y_mm", np.nan))

    longitudinal_ratio = base_long_ratio
    axle_dx_mm = rear_axle_x_mm - front_axle_x_mm
    if np.isfinite(cg_x_mm) and np.isfinite(base_cg_x_mm) and abs(axle_dx_mm) > 1e-9:
        raw_base_ratio = (base_cg_x_mm - front_axle_x_mm) / axle_dx_mm
        raw_now_ratio = (cg_x_mm - front_axle_x_mm) / axle_dx_mm
        longitudinal_ratio = float(np.clip(base_long_ratio + (raw_now_ratio - raw_base_ratio), 0.0, 1.0))

    if not np.isfinite(cg_x_mm):
        cg_x_mm = rear_axle_x_mm + wheelbase_mm * (1.0 - longitudinal_ratio)
    if not np.isfinite(cg_y_mm):
        cg_y_mm = 0.0

    front_track_mm = float(analysis.get("front_track_mm", np.nan))
    rear_track_mm = float(analysis.get("rear_track_mm", np.nan))
    if not np.isfinite(front_track_mm) or front_track_mm <= 1e-9:
        front_track_mm = 2.0 * abs(float(front_axle_c[1]) * 1000.0 if len(front_axle_c) > 1 else 800.0)
    if not np.isfinite(rear_track_mm) or rear_track_mm <= 1e-9:
        rear_track_mm = 2.0 * abs(float(rear_axle_c[1]) * 1000.0 if len(rear_axle_c) > 1 else 800.0)

    sprung_weight_n = mass_sprung_total * g
    front_sprung_axle_n = sprung_weight_n * (1.0 - longitudinal_ratio)
    rear_sprung_axle_n = sprung_weight_n * longitudinal_ratio

    delta_cg_y_mm = (cg_y_mm - base_cg_y_mm) if np.isfinite(base_cg_y_mm) else cg_y_mm
    ratio_lf = float(np.clip(0.5 + delta_cg_y_mm / max(front_track_mm, 1e-9), 0.0, 1.0))
    ratio_lr = float(np.clip(0.5 + delta_cg_y_mm / max(rear_track_mm, 1e-9), 0.0, 1.0))
    fl_static_n = front_sprung_axle_n * (1.0 - ratio_lf) + m_hub_f * g
    fr_static_n = front_sprung_axle_n * ratio_lf + m_hub_f * g
    rl_static_n = rear_sprung_axle_n * (1.0 - ratio_lr) + m_hub_r * g
    rr_static_n = rear_sprung_axle_n * ratio_lr + m_hub_r * g

    h_cg_m = (h_cg_now_mm / 1000.0) if np.isfinite(h_cg_now_mm) else abs(float(chassis.get("zCoG", -0.25)))
    wb_m = wheelbase_mm / 1000.0
    ft_m = front_track_mm / 1000.0
    rt_m = rear_track_mm / 1000.0

    front_rc_h_mm = float(analysis.get("front_rc_height_mm", np.nan))
    rear_rc_h_mm = float(analysis.get("rear_rc_height_mm", np.nan))
    if not np.isfinite(front_rc_h_mm):
        front_rc_h_mm = float(analysis.get("front_rc_z_mm", np.nan))
    if not np.isfinite(rear_rc_h_mm):
        rear_rc_h_mm = float(analysis.get("rear_rc_z_mm", np.nan))

    lam_cg = float(np.clip(longitudinal_ratio, 0.0, 1.0))
    h_ra_cg_mm = (
        front_rc_h_mm + lam_cg * (rear_rc_h_mm - front_rc_h_mm)
        if (np.isfinite(front_rc_h_mm) and np.isfinite(rear_rc_h_mm))
        else np.nan
    )
    h_ra_cg_m = (h_ra_cg_mm / 1000.0) if np.isfinite(h_ra_cg_mm) else 0.0

    pitch_ic_f_z_mm = float(analysis.get("pitch_ic_front_z_mm", np.nan))
    pitch_ic_r_z_mm = float(analysis.get("pitch_ic_rear_z_mm", np.nan))
    if np.isfinite(pitch_ic_f_z_mm) and np.isfinite(pitch_ic_r_z_mm):
        h_pc_mm = pitch_ic_f_z_mm + lam_cg * (pitch_ic_r_z_mm - pitch_ic_f_z_mm)
    else:
        h_pc_mm = float(analysis.get("pitch_z_mm", np.nan))
    h_pc_m = (h_pc_mm / 1000.0) if np.isfinite(h_pc_mm) else 0.0

    pitch_arm_m = h_cg_m - h_pc_m
    roll_arm_m = h_cg_m - h_ra_cg_m

    long_tf_n = mass_sprung_total * ax_mps2 * pitch_arm_m / max(wb_m, 1e-9)
    lat_tf_f_n = (front_sprung_axle_n / g) * ay_mps2 * roll_arm_m / max(ft_m, 1e-9)
    lat_tf_r_n = (rear_sprung_axle_n / g) * ay_mps2 * roll_arm_m / max(rt_m, 1e-9)

    fz_fl = fl_static_n + (-0.5 * long_tf_n) + (-0.5 * lat_tf_f_n) + 0.5 * front_aero_load_n
    fz_fr = fr_static_n + (-0.5 * long_tf_n) + (+0.5 * lat_tf_f_n) + 0.5 * front_aero_load_n
    fz_rl = rl_static_n + (+0.5 * long_tf_n) + (-0.5 * lat_tf_r_n) + 0.5 * rear_aero_load_n
    fz_rr = rr_static_n + (+0.5 * long_tf_n) + (+0.5 * lat_tf_r_n) + 0.5 * rear_aero_load_n

    gs_f = tyre_global_grip * tyre_front_grip
    gs_r = tyre_global_grip * tyre_rear_grip
    tyre_fl = _mf_tyre_combined_from_json(
        tyre_front_cfg,
        fz_fl,
        float(params.get("slip_angle_fl_deg", 0.0)),
        float(params.get("slip_ratio_fl", 0.0)),
        float(params.get("camber_fl_deg", 0.0)),
        grip_scale=gs_f,
    )
    tyre_fr = _mf_tyre_combined_from_json(
        tyre_front_cfg,
        fz_fr,
        float(params.get("slip_angle_fr_deg", 0.0)),
        float(params.get("slip_ratio_fr", 0.0)),
        float(params.get("camber_fr_deg", 0.0)),
        grip_scale=gs_f,
    )
    tyre_rl = _mf_tyre_combined_from_json(
        tyre_rear_cfg,
        fz_rl,
        float(params.get("slip_angle_rl_deg", 0.0)),
        float(params.get("slip_ratio_rl", 0.0)),
        float(params.get("camber_rl_deg", 0.0)),
        grip_scale=gs_r,
    )
    tyre_rr = _mf_tyre_combined_from_json(
        tyre_rear_cfg,
        fz_rr,
        float(params.get("slip_angle_rr_deg", 0.0)),
        float(params.get("slip_ratio_rr", 0.0)),
        float(params.get("camber_rr_deg", 0.0)),
        grip_scale=gs_r,
    )
    fy_fl = abs(float(tyre_fl["fy_n"]))
    fy_fr = abs(float(tyre_fr["fy_n"]))
    fy_rl = abs(float(tyre_rl["fy_n"]))
    fy_rr = abs(float(tyre_rr["fy_n"]))
    fx_fl = abs(float(tyre_fl["fx_n"]))
    fx_fr = abs(float(tyre_fr["fx_n"]))
    fx_rl = abs(float(tyre_rl["fx_n"]))
    fx_rr = abs(float(tyre_rr["fx_n"]))
    front_aero_each_n = 0.5 * front_aero_load_n
    rear_aero_each_n = 0.5 * rear_aero_load_n

    front_total_load_n = float(fz_fl + fz_fr)
    rear_total_load_n = float(fz_rl + fz_rr)
    fy_front_n = float(fy_fl + fy_fr)
    fy_rear_n = float(fy_rl + fy_rr)
    fx_front_n = float(fx_fl + fx_fr)
    fx_rear_n = float(fx_rl + fx_rr)

    return {
        "state_4w": {"hf": float(hf), "rf": float(rf), "hr": float(hr), "rr": float(rr)},
        "cg_global_mm": [float(cg_x_mm), float(cg_y_mm), float(h_cg_now_mm)],
        "h_cg_mm": float(h_cg_now_mm),
        "hRideF": float(hRideF_m),
        "hRideR": float(hRideR_m),
        "hRideF_mm": float(hRideF_mm),
        "hRideR_mm": float(hRideR_mm),
        "front_aero_load_n": float(front_aero_load_n),
        "rear_aero_load_n": float(rear_aero_load_n),
        "total_aero_load_n": float(total_aero_load_n),
        "aero_balance_front_pct": float(aero_balance_front_pct) if np.isfinite(aero_balance_front_pct) else None,
        "drag_force_n": float(drag_force_n),
        "front_left_static_load_n": float(fl_static_n),
        "front_right_static_load_n": float(fr_static_n),
        "rear_left_static_load_n": float(rl_static_n),
        "rear_right_static_load_n": float(rr_static_n),
        "front_static_load_n": float(fl_static_n + fr_static_n),
        "rear_static_load_n": float(rl_static_n + rr_static_n),
        "load_transfer_longitudinal_n": float(long_tf_n),
        "lateral_transfer_front_total_n": float(lat_tf_f_n),
        "lateral_transfer_rear_total_n": float(lat_tf_r_n),
        "h_pc_mm": float(h_pc_mm) if np.isfinite(h_pc_mm) else None,
        "h_ra_cg_mm": float(h_ra_cg_mm) if np.isfinite(h_ra_cg_mm) else None,
        "pitch_arm_m": float(pitch_arm_m),
        "roll_arm_m": float(roll_arm_m),
        "fz_mechanical_fl_n": float(fl_static_n + (-0.5 * long_tf_n) + (-0.5 * lat_tf_f_n)),
        "fz_mechanical_fr_n": float(fr_static_n + (-0.5 * long_tf_n) + (+0.5 * lat_tf_f_n)),
        "fz_mechanical_rl_n": float(rl_static_n + (+0.5 * long_tf_n) + (-0.5 * lat_tf_r_n)),
        "fz_mechanical_rr_n": float(rr_static_n + (+0.5 * long_tf_n) + (+0.5 * lat_tf_r_n)),
        "fz_aero_fl_n": float(front_aero_each_n),
        "fz_aero_fr_n": float(front_aero_each_n),
        "fz_aero_rl_n": float(rear_aero_each_n),
        "fz_aero_rr_n": float(rear_aero_each_n),
        "total_load_fl_n": float(fz_fl),
        "total_load_fr_n": float(fz_fr),
        "total_load_rl_n": float(fz_rl),
        "total_load_rr_n": float(fz_rr),
        "fz_total_fl_n": float(fz_fl),
        "fz_total_fr_n": float(fz_fr),
        "fz_total_rl_n": float(fz_rl),
        "fz_total_rr_n": float(fz_rr),
        "front_total_load_n": front_total_load_n,
        "rear_total_load_n": rear_total_load_n,
        "fy_fl_n": float(fy_fl),
        "fy_fr_n": float(fy_fr),
        "fy_rl_n": float(fy_rl),
        "fy_rr_n": float(fy_rr),
        "front_axle_lateral_force_n": fy_front_n,
        "rear_axle_lateral_force_n": fy_rear_n,
        "total_lateral_force_n": float(fy_front_n + fy_rear_n),
        "fy_front_n": fy_front_n,
        "fy_rear_n": fy_rear_n,
        "fx_fl_n": float(fx_fl),
        "fx_fr_n": float(fx_fr),
        "fx_rl_n": float(fx_rl),
        "fx_rr_n": float(fx_rr),
        "front_axle_longitudinal_force_n": float(fx_front_n),
        "rear_axle_longitudinal_force_n": float(fx_rear_n),
        "total_longitudinal_force_n": float(fx_front_n + fx_rear_n),
        "fz_front_n": front_total_load_n,
        "fz_rear_n": rear_total_load_n,
        "cg_x_mm": float(cg_x_mm),
        "cg_y_mm": float(cg_y_mm),
        "front_track_mm": float(front_track_mm),
        "rear_track_mm": float(rear_track_mm),
    }


def _build_sensitivity_data(json_data: dict, body: dict) -> Dict[str, object]:
    """Perturb each input one at a time and compute output sensitivities."""
    g = 9.80665
    acc_units = str(body.get("acc_units", "g")).lower()

    def _to_g(v: object) -> float:
        return float(v) if acc_units in {"g", "gee"} else float(v) / g

    base_params: Dict[str, object] = {
        "speed_kph": float(body.get("speed_kph", 140.0)),
        "hf": float(body.get("hf", 0.0)),
        "rf": float(body.get("rf", 0.0)),
        "hr": float(body.get("hr", 0.0)),
        "rr": float(body.get("rr", 0.0)),
        "ax_g": _to_g(body.get("ax", 0.0)),
        "ay_g": _to_g(body.get("ay", 0.0)),
        "air_density": float(body.get("air_density", 1.225)),
        "drs_on": bool(body.get("drs_on", False)),
        "slip_angle_fl_deg": float(body.get("slip_angle_fl_deg", 4.0)),
        "slip_angle_fr_deg": float(body.get("slip_angle_fr_deg", 4.0)),
        "slip_angle_rl_deg": float(body.get("slip_angle_rl_deg", 4.0)),
        "slip_angle_rr_deg": float(body.get("slip_angle_rr_deg", 4.0)),
        "slip_ratio_fl": float(body.get("slip_ratio_fl", 0.0)),
        "slip_ratio_fr": float(body.get("slip_ratio_fr", 0.0)),
        "slip_ratio_rl": float(body.get("slip_ratio_rl", 0.0)),
        "slip_ratio_rr": float(body.get("slip_ratio_rr", 0.0)),
        "camber_fl_deg": 0.0, "camber_fr_deg": 0.0, "camber_rl_deg": 0.0, "camber_rr_deg": 0.0,
    }

    perturbations: List[Dict[str, object]] = [
        {"label": "Front Heave (compression +)", "unit": "mm",   "param": "hf",        "delta": float(body.get("delta_front_heave", 1.0))},
        {"label": "Rear Heave (compression +)",  "unit": "mm",   "param": "hr",        "delta": float(body.get("delta_rear_heave", 1.0))},
        {"label": "Front Roll",  "unit": "°",    "param": "rf",        "delta": float(body.get("delta_front_roll", 0.1))},
        {"label": "Rear Roll",   "unit": "°",    "param": "rr",        "delta": float(body.get("delta_rear_roll", 0.1))},
        {"label": "Speed",       "unit": "km/h", "param": "speed_kph", "delta": float(body.get("delta_speed", 1.0))},
        {"label": "Ay",          "unit": "g",    "param": "ay_g",      "delta": float(body.get("delta_ay", 0.1))},
        {"label": "Ax",          "unit": "g",    "param": "ax_g",      "delta": float(body.get("delta_ax", 0.1))},
    ]

    static_ref = compute_center_antis_for_state(json_data, hf=0.0, rf=0.0, hr=0.0, rr=0.0)
    baseline = _compute_operating_point(json_data, base_params, _static_ref=static_ref)

    output_keys = [
        "fy_fl_n",
        "fy_fr_n",
        "fy_rl_n",
        "fy_rr_n",
        "fz_mechanical_fl_n",
        "fz_mechanical_fr_n",
        "fz_mechanical_rl_n",
        "fz_mechanical_rr_n",
        "fz_aero_fl_n",
        "fz_aero_fr_n",
        "fz_aero_rl_n",
        "fz_aero_rr_n",
        "fz_total_fl_n",
        "fz_total_fr_n",
        "fz_total_rl_n",
        "fz_total_rr_n",
        "total_aero_load_n",
        "aero_balance_front_pct",
        "drag_force_n",
    ]
    output_labels = {
        "fy_fl_n": "Fy FL (N)",
        "fy_fr_n": "Fy FR (N)",
        "fy_rl_n": "Fy RL (N)",
        "fy_rr_n": "Fy RR (N)",
        "fz_mechanical_fl_n": "Fz Mechanical FL (N)",
        "fz_mechanical_fr_n": "Fz Mechanical FR (N)",
        "fz_mechanical_rl_n": "Fz Mechanical RL (N)",
        "fz_mechanical_rr_n": "Fz Mechanical RR (N)",
        "fz_aero_fl_n": "Fz Aero FL (N)",
        "fz_aero_fr_n": "Fz Aero FR (N)",
        "fz_aero_rl_n": "Fz Aero RL (N)",
        "fz_aero_rr_n": "Fz Aero RR (N)",
        "fz_total_fl_n": "Fz Total FL (N)",
        "fz_total_fr_n": "Fz Total FR (N)",
        "fz_total_rl_n": "Fz Total RL (N)",
        "fz_total_rr_n": "Fz Total RR (N)",
        "total_aero_load_n": "Total Aero Load (N)",
        "aero_balance_front_pct": "Aero Balance F (%)",
        "drag_force_n": "Drag (N)",
    }

    rows: List[Dict[str, object]] = []
    for pert in perturbations:
        delta = float(pert["delta"])
        if abs(delta) < 1e-12:
            continue
        p2 = dict(base_params)
        p2[pert["param"]] = float(base_params[pert["param"]]) + delta  # type: ignore[arg-type]
        try:
            perturbed = _compute_operating_point(json_data, p2, _static_ref=static_ref)
        except Exception:
            perturbed = {}
        row: Dict[str, object] = {
            "parameter": str(pert["label"]),
            "unit": str(pert["unit"]),
            "delta": delta,
            "label": f"{pert['label']} +{delta}{pert['unit']}",
        }
        for k in output_keys:
            bv = baseline.get(k)
            pv = perturbed.get(k)
            row[f"sens_{k}"] = float((pv - bv) / delta) if (bv is not None and pv is not None) else None  # type: ignore[operator]
        rows.append(_json_clean(row))

    all_cols = ["parameter", "unit", "delta"] + [f"sens_{k}" for k in output_keys]
    csv_io = io.StringIO()
    csv_io.write(",".join(all_cols) + "\n")
    for row in rows:
        vals = []
        for k in all_cols:
            v = row.get(k, "")
            text = "" if v is None else str(v)
            if "," in text or '"' in text or "\n" in text:
                text = '"' + text.replace('"', '""') + '"'
            vals.append(text)
        csv_io.write(",".join(vals) + "\n")

    return {
        "baseline": _json_clean(baseline),
        "rows": rows,
        "output_keys": output_keys,
        "output_labels": output_labels,
        "csv": csv_io.getvalue(),
    }


def _json_signature(json_data: Dict[str, Any]) -> str:
    raw = json.dumps(json_data, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _json_signature_cached(json_data: Dict[str, Any]) -> str:
    cache = _state.get("json_signature_cache")
    if not isinstance(cache, dict):
        cache = {}
        _state["json_signature_cache"] = cache
    key = id(json_data)
    sig = cache.get(key)
    if isinstance(sig, str) and sig:
        return sig
    sig = _json_signature(json_data)
    if len(cache) > 32:
        cache.clear()
    cache[key] = sig
    return sig


def _infer_model_id(json_data: Dict[str, Any], requested_model_id: Optional[str] = None) -> str:
    if not VEHICLE_REGISTRY:
        if requested_model_id:
            return str(requested_model_id).strip()
        cfg = json_data.get("config", {}) if isinstance(json_data.get("config"), dict) else {}
        return str(cfg.get("model_id", json_data.get("model_id", "default"))).strip() or "default"

    if requested_model_id:
        model_id = str(requested_model_id).strip()
        if model_id not in VEHICLE_REGISTRY:
            raise ValueError(
                f"Unknown model_id '{model_id}'. Available: {', '.join(sorted(VEHICLE_REGISTRY.keys()))}"
            )
        return model_id

    cfg = json_data.get("config", {}) if isinstance(json_data.get("config"), dict) else {}
    candidates = [
        cfg.get("model_id"),
        cfg.get("vehicleModelId"),
        cfg.get("vehicle_model_id"),
        json_data.get("model_id"),
        "F2_2026",
    ]
    for value in candidates:
        if value is None:
            continue
        model_id = str(value).strip()
        if model_id in VEHICLE_REGISTRY:
            return model_id
    if VEHICLE_REGISTRY:
        return sorted(VEHICLE_REGISTRY.keys())[0]
    raise ValueError("No registered vehicle models are available in suspension_model.VEHICLE_REGISTRY.")


def _build_calibrated_json_base(
    json_data: Dict[str, Any],
    model_id: Optional[str] = None,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    if not isinstance(json_data, dict):
        raise ValueError("Invalid JSON payload for calibration.")

    resolved_model_id = _infer_model_id(json_data, model_id)
    signature = _json_signature_cached(json_data)
    cache_key = f"{resolved_model_id}:{signature}"
    cache = _state.get("gg_calibration_cache")
    if not isinstance(cache, dict):
        cache = {}
        _state["gg_calibration_cache"] = cache

    if (not force_rebuild) and (cache_key in cache):
        cached = cache.get(cache_key) or {}
        return {
            "calibrated_json_data": cached.get("calibrated_json_data"),
            "calibration_result": cached.get("calibration_result"),
            "metadata": dict(cached.get("metadata") or {}),
        }

    global _CALIBRATOR_AVAILABLE, calibrate_geometry, write_calibrated_json
    if not _CALIBRATOR_AVAILABLE or calibrate_geometry is None or write_calibrated_json is None:
        try:
            cal_mod = importlib.import_module("calibrator")
            calibrate_geometry = getattr(cal_mod, "calibrate", None)
            write_calibrated_json = getattr(cal_mod, "write_calibrated_json", None)
            _CALIBRATOR_AVAILABLE = callable(calibrate_geometry) and callable(write_calibrated_json)
        except Exception:
            _CALIBRATOR_AVAILABLE = False

    if not _CALIBRATOR_AVAILABLE or calibrate_geometry is None or write_calibrated_json is None:
        metadata = {
            "calibration_applied": False,
            "calibration_model_id": resolved_model_id,
            "calibration_model_desc": "",
            "json_signature": signature,
            "warning": "calibrator.py dependencies are unavailable; using raw JSON geometry.",
        }
        cache[cache_key] = {
            "calibrated_json_data": json_data,
            "calibration_result": {},
            "metadata": metadata,
        }
        return {
            "calibrated_json_data": json_data,
            "calibration_result": {},
            "metadata": metadata,
        }

    calibration_result = calibrate_geometry(json_data, model_id=resolved_model_id, verbose=False)
    calibrated_json_data = write_calibrated_json(json_data, calibration_result)
    metadata = {
        "calibration_applied": True,
        "calibration_model_id": resolved_model_id,
        "calibration_model_desc": calibration_result.get("model_desc", ""),
        "json_signature": signature,
    }

    cache[cache_key] = {
        "calibrated_json_data": calibrated_json_data,
        "calibration_result": calibration_result,
        "metadata": metadata,
    }
    return {
        "calibrated_json_data": calibrated_json_data,
        "calibration_result": calibration_result,
        "metadata": metadata,
    }


def _get_or_build_gg_base_geometry(json_data: Dict[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
    model_id = body.get("model_id")
    force_rebuild = bool(body.get("force_rebuild_calibration", False))
    return _build_calibrated_json_base(json_data=json_data, model_id=model_id, force_rebuild=force_rebuild)


def _safe_axis_values(start: float, end: float, step: float, max_points: int = 41) -> Tuple[List[float], bool]:
    values = _dynamic_range(start, end, step)
    if len(values) <= max_points:
        return values, False
    idx = np.linspace(0, len(values) - 1, max_points).round().astype(int)
    sampled = [float(values[int(i)]) for i in idx]
    return sampled, True


def _extract_camber_defaults_deg(json_data: Dict[str, Any]) -> Dict[str, float]:
    cfg = json_data.get("config", {}) if isinstance(json_data.get("config"), dict) else {}
    sus = cfg.get("suspension", {}) if isinstance(cfg.get("suspension"), dict) else {}
    front = (((sus.get("front") or {}).get("external") or {}).get("aCamberSetupAlignment") or {})
    rear = (((sus.get("rear") or {}).get("external") or {}).get("aCamberSetupAlignment") or {})

    def _deg(block: Dict[str, Any]) -> float:
        try:
            return float(math.degrees(float(block.get("aCamberSetup", 0.0))))
        except Exception:
            return 0.0

    f_deg = _deg(front)
    r_deg = _deg(rear)
    return {
        "camber_fl_deg": f_deg,
        "camber_fr_deg": f_deg,
        "camber_rl_deg": r_deg,
        "camber_rr_deg": r_deg,
    }


def _reduce_envelope_points(points: List[Dict[str, float]], max_points: int = 250) -> List[Dict[str, float]]:
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).round().astype(int)
    return [points[int(i)] for i in idx]


def _build_tyre_fx_fy_envelope(
    tyre_cfg: Dict[str, Any],
    fz_n: float,
    camber_deg: float,
    grip_scale: float,
    alpha_min_deg: float,
    alpha_max_deg: float,
    alpha_steps: int,
    kappa_min: float,
    kappa_max: float,
    kappa_steps: int,
    envelope_cache: Optional[Dict[Tuple[Any, ...], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    cache_key: Optional[Tuple[Any, ...]] = None
    if envelope_cache is not None:
        # Quantize Fz to reuse nearby operating points with minimal loss.
        fz_q = round(float(fz_n) / 25.0) * 25.0
        cache_key = (
            int(id(tyre_cfg)),
            float(fz_q),
            round(float(camber_deg), 3),
            round(float(grip_scale), 4),
            round(float(alpha_min_deg), 4),
            round(float(alpha_max_deg), 4),
            int(alpha_steps),
            round(float(kappa_min), 5),
            round(float(kappa_max), 5),
            int(kappa_steps),
        )
        cached = envelope_cache.get(cache_key)
        if isinstance(cached, dict):
            return cached

    alpha_steps_i = int(max(3, alpha_steps))
    kappa_steps_i = int(max(3, kappa_steps))
    alpha_vals = np.linspace(float(alpha_min_deg), float(alpha_max_deg), alpha_steps_i)
    kappa_vals = np.linspace(float(kappa_min), float(kappa_max), kappa_steps_i)

    samples: List[Dict[str, float]] = []
    max_fx_abs = -1.0
    max_fy_abs = -1.0
    sample_fxmax: Optional[Dict[str, float]] = None
    sample_fymax: Optional[Dict[str, float]] = None

    for alpha in alpha_vals:
        for kappa in kappa_vals:
            out = _mf_tyre_combined_from_json(
                tyre_cfg=tyre_cfg,
                fz_n=float(fz_n),
                slip_angle_deg=float(alpha),
                slip_ratio=float(kappa),
                camber_deg=float(camber_deg),
                grip_scale=float(grip_scale),
            )
            fx_n = float(out.get("fx_n", 0.0))
            fy_n = float(out.get("fy_n", 0.0))
            row = {
                "slip_angle_deg": float(alpha),
                "slip_ratio": float(kappa),
                "fx_n": fx_n,
                "fy_n": fy_n,
            }
            samples.append(row)
            if abs(fx_n) > max_fx_abs:
                max_fx_abs = abs(fx_n)
                sample_fxmax = row
            if abs(fy_n) > max_fy_abs:
                max_fy_abs = abs(fy_n)
                sample_fymax = row

    if sample_fxmax is None:
        sample_fxmax = {"fx_n": 0.0, "fy_n": 0.0, "slip_angle_deg": 0.0, "slip_ratio": 0.0}
    if sample_fymax is None:
        sample_fymax = {"fx_n": 0.0, "fy_n": 0.0, "slip_angle_deg": 0.0, "slip_ratio": 0.0}

    result = {
        "samples": _reduce_envelope_points(samples, max_points=320),
        "envelope_points": _reduce_envelope_points(samples, max_points=220),
        "fx_max_n": float(max_fx_abs if max_fx_abs > 0.0 else 0.0),
        "fy_max_n": float(max_fy_abs if max_fy_abs > 0.0 else 0.0),
        "fx_at_fymax_n": float(sample_fymax.get("fx_n", 0.0)),
        "fy_at_fxmax_n": float(sample_fxmax.get("fy_n", 0.0)),
        "slip_angle_at_fxmax_deg": float(sample_fxmax.get("slip_angle_deg", 0.0)),
        "slip_ratio_at_fxmax": float(sample_fxmax.get("slip_ratio", 0.0)),
        "slip_angle_at_fymax_deg": float(sample_fymax.get("slip_angle_deg", 0.0)),
        "slip_ratio_at_fymax": float(sample_fymax.get("slip_ratio", 0.0)),
    }
    if envelope_cache is not None and cache_key is not None:
        envelope_cache[cache_key] = result
    return result


def _select_envelope_point_by_direction(
    envelope: Dict[str, Any],
    ax_g_target: float,
    ay_g_target: float,
    fx_pos_scale: float = 1.0,
    fx_neg_scale: float = 1.0,
) -> Dict[str, float]:
    samples = envelope.get("samples", [])
    if not isinstance(samples, list) or not samples:
        return {"fx_n": 0.0, "fy_n": 0.0, "slip_angle_deg": 0.0, "slip_ratio": 0.0}
    dx = float(ax_g_target)
    dy = float(ay_g_target)
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        return {"fx_n": 0.0, "fy_n": 0.0, "slip_angle_deg": 0.0, "slip_ratio": 0.0}
    ux = dx / norm
    uy = dy / norm

    best_fx = 0.0
    best_fy = 0.0
    best_sa = 0.0
    best_sr = 0.0
    best_proj = -1e18
    for s in samples:
        if not isinstance(s, dict):
            continue
        fx_raw = float(s.get("fx_n", 0.0))
        fx = fx_raw * (float(fx_pos_scale) if fx_raw >= 0.0 else float(fx_neg_scale))
        fy = float(s.get("fy_n", 0.0))
        proj = fx * ux + fy * uy
        if proj > best_proj:
            best_proj = proj
            best_fx = fx
            best_fy = fy
            best_sa = float(s.get("slip_angle_deg", 0.0))
            best_sr = float(s.get("slip_ratio", 0.0))
    return {"fx_n": best_fx, "fy_n": best_fy, "slip_angle_deg": best_sa, "slip_ratio": best_sr}


def _gg_longitudinal_caps(body: Dict[str, Any]) -> Dict[str, float]:
    drive_layout = str(body.get("drive_layout", "rwd")).strip().lower()
    if drive_layout == "fwd":
        default_front_trac, default_rear_trac = 1.0, 0.0
    elif drive_layout == "awd":
        default_front_trac, default_rear_trac = 1.0, 1.0
    else:
        default_front_trac, default_rear_trac = 0.0, 1.0

    traction_front = float(body.get("traction_front_scale", default_front_trac))
    traction_rear = float(body.get("traction_rear_scale", default_rear_trac))
    bb_front_pct = float(body.get("brake_balance_front_pct", 60.0))
    bb_front = float(np.clip(bb_front_pct / 100.0, 0.0, 1.0))
    bb_rear = 1.0 - bb_front
    # Backward compatibility: explicit per-axle scales still override balance if sent.
    brake_front = float(body.get("brake_front_scale", bb_front))
    brake_rear = float(body.get("brake_rear_scale", bb_rear))

    return {
        "fl_pos": max(0.0, traction_front),
        "fr_pos": max(0.0, traction_front),
        "rl_pos": max(0.0, traction_rear),
        "rr_pos": max(0.0, traction_rear),
        "fl_neg": max(0.0, brake_front),
        "fr_neg": max(0.0, brake_front),
        "rl_neg": max(0.0, brake_rear),
        "rr_neg": max(0.0, brake_rear),
    }


def _gg_fixed_inputs(calibrated_json_data: Dict[str, Any], body: Dict[str, Any], base_state: Dict[str, float]) -> Dict[str, Any]:
    g = 9.80665
    acc_units = str(body.get("acc_units", "g")).lower()
    ax_raw = float(body.get("ax_g_target", body.get("ax_g", body.get("ax", 0.0))))
    ay_raw = float(body.get("ay_g_target", body.get("ay_g", body.get("ay", 0.0))))
    ax_g = ax_raw if acc_units in {"g", "gee"} else ax_raw / g
    ay_g = ay_raw if acc_units in {"g", "gee"} else ay_raw / g

    camber_defaults = _extract_camber_defaults_deg(calibrated_json_data)
    payload = {
        "speed_kph": float(body.get("speed_kph", 140.0)),
        "air_density": float(body.get("air_density", 1.225)),
        "drs_on": bool(body.get("drs_on", False)),
        "ax_g": float(ax_g),
        "ay_g": float(ay_g),
        # GG philosophy:
        # - Slip is not a user input here; envelope sweep already scans alpha/kappa space.
        # - Camber comes from setup/calibrated base geometry.
        "slip_angle_fl_deg": 0.0,
        "slip_angle_fr_deg": 0.0,
        "slip_angle_rl_deg": 0.0,
        "slip_angle_rr_deg": 0.0,
        "slip_ratio_fl": 0.0,
        "slip_ratio_fr": 0.0,
        "slip_ratio_rl": 0.0,
        "slip_ratio_rr": 0.0,
        "camber_fl_deg": float(camber_defaults["camber_fl_deg"]),
        "camber_fr_deg": float(camber_defaults["camber_fr_deg"]),
        "camber_rl_deg": float(camber_defaults["camber_rl_deg"]),
        "camber_rr_deg": float(camber_defaults["camber_rr_deg"]),
        "base_state": {
            "hf": float(base_state.get("hf", 0.0)),
            "hr": float(base_state.get("hr", 0.0)),
            "rf": float(base_state.get("rf", 0.0)),
            "rr": float(base_state.get("rr", 0.0)),
        },
    }
    return payload


def _build_gg_row_from_outputs(
    point_outputs: Dict[str, Any],
    env_fl: Dict[str, Any],
    env_fr: Dict[str, Any],
    env_rl: Dict[str, Any],
    env_rr: Dict[str, Any],
    ax_g_target: float,
    ay_g_target: float,
    longitudinal_caps: Optional[Dict[str, float]] = None,
    include_envelopes: bool = False,
) -> Dict[str, Any]:
    longitudinal_caps = longitudinal_caps or {}
    # Use real combined-slip cloud and pick the point aligned with requested
    # acceleration direction. This avoids the "star/rays" artifact.
    pt_fl = _select_envelope_point_by_direction(
        env_fl, ax_g_target, ay_g_target,
        fx_pos_scale=float(longitudinal_caps.get("fl_pos", 1.0)),
        fx_neg_scale=float(longitudinal_caps.get("fl_neg", 1.0)),
    )
    pt_fr = _select_envelope_point_by_direction(
        env_fr, ax_g_target, ay_g_target,
        fx_pos_scale=float(longitudinal_caps.get("fr_pos", 1.0)),
        fx_neg_scale=float(longitudinal_caps.get("fr_neg", 1.0)),
    )
    pt_rl = _select_envelope_point_by_direction(
        env_rl, ax_g_target, ay_g_target,
        fx_pos_scale=float(longitudinal_caps.get("rl_pos", 1.0)),
        fx_neg_scale=float(longitudinal_caps.get("rl_neg", 1.0)),
    )
    pt_rr = _select_envelope_point_by_direction(
        env_rr, ax_g_target, ay_g_target,
        fx_pos_scale=float(longitudinal_caps.get("rr_pos", 1.0)),
        fx_neg_scale=float(longitudinal_caps.get("rr_neg", 1.0)),
    )
    fx_fl, fy_fl = float(pt_fl["fx_n"]), float(pt_fl["fy_n"])
    fx_fr, fy_fr = float(pt_fr["fx_n"]), float(pt_fr["fy_n"])
    fx_rl, fy_rl = float(pt_rl["fx_n"]), float(pt_rl["fy_n"])
    fx_rr, fy_rr = float(pt_rr["fx_n"]), float(pt_rr["fy_n"])

    state = point_outputs.get("state_4w", {}) if isinstance(point_outputs.get("state_4w"), dict) else {}
    row = {
        "ax_g_target": float(ax_g_target),
        "ay_g_target": float(ay_g_target),
        "hf": float(state.get("hf", 0.0)),
        "hr": float(state.get("hr", 0.0)),
        "rf": float(state.get("rf", 0.0)),
        "rr": float(state.get("rr", 0.0)),
        "hRideF": float(point_outputs.get("hRideF", 0.0)),
        "hRideR": float(point_outputs.get("hRideR", 0.0)),
        "hRideF_mm": float(point_outputs.get("hRideF", 0.0)) * 1000.0,
        "hRideR_mm": float(point_outputs.get("hRideR", 0.0)) * 1000.0,
        "pitch_deg": float(point_outputs.get("pitch_deg", 0.0)),
        "roll_deg": float(point_outputs.get("roll_deg", 0.0)),
        "fz_total_fl_n": float(point_outputs.get("fz_total_fl_n", point_outputs.get("total_load_fl_n", 0.0))),
        "fz_total_fr_n": float(point_outputs.get("fz_total_fr_n", point_outputs.get("total_load_fr_n", 0.0))),
        "fz_total_rl_n": float(point_outputs.get("fz_total_rl_n", point_outputs.get("total_load_rl_n", 0.0))),
        "fz_total_rr_n": float(point_outputs.get("fz_total_rr_n", point_outputs.get("total_load_rr_n", 0.0)),
        ),
        "fx_fl_n": fx_fl,
        "fx_fr_n": fx_fr,
        "fx_rl_n": fx_rl,
        "fx_rr_n": fx_rr,
        "fy_fl_n": fy_fl,
        "fy_fr_n": fy_fr,
        "fy_rl_n": fy_rl,
        "fy_rr_n": fy_rr,
        "fx_front_axle_n": float(fx_fl + fx_fr),
        "fy_front_axle_n": float(fy_fl + fy_fr),
        "fx_rear_axle_n": float(fx_rl + fx_rr),
        "fy_rear_axle_n": float(fy_rl + fy_rr),
        "fx_total_n": float(fx_fl + fx_fr + fx_rl + fx_rr),
        "fy_total_n": float(fy_fl + fy_fr + fy_rl + fy_rr),
        "slip_angle_fl_deg_used": float(pt_fl["slip_angle_deg"]),
        "slip_angle_fr_deg_used": float(pt_fr["slip_angle_deg"]),
        "slip_angle_rl_deg_used": float(pt_rl["slip_angle_deg"]),
        "slip_angle_rr_deg_used": float(pt_rr["slip_angle_deg"]),
        "slip_ratio_fl_used": float(pt_fl["slip_ratio"]),
        "slip_ratio_fr_used": float(pt_fr["slip_ratio"]),
        "slip_ratio_rl_used": float(pt_rl["slip_ratio"]),
        "slip_ratio_rr_used": float(pt_rr["slip_ratio"]),
        "aero_balance_front_pct": _json_scalar(point_outputs.get("aero_balance_front_pct")),
        "drag_force_n": _json_scalar(point_outputs.get("drag_force_n")),
        "front_aero_load_n": _json_scalar(point_outputs.get("front_aero_load_n")),
        "rear_aero_load_n": _json_scalar(point_outputs.get("rear_aero_load_n")),
        "total_aero_load_n": _json_scalar(point_outputs.get("total_aero_load_n")),
    }
    if include_envelopes:
        row["tyre_envelope_fl"] = env_fl
        row["tyre_envelope_fr"] = env_fr
        row["tyre_envelope_rl"] = env_rl
        row["tyre_envelope_rr"] = env_rr
    return _json_clean(row)


def _evaluate_gg_point_frozen(
    calibrated_json_data: Dict[str, Any],
    speed_kph: float,
    ax_g_target: float,
    ay_g_target: float,
    base_state: Dict[str, float],
    envelope_settings: Dict[str, Any],
    static_ref: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    envelope_cache: Optional[Dict[Tuple[Any, ...], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    body = body or {}
    fixed = _gg_fixed_inputs(calibrated_json_data, {**body, "speed_kph": speed_kph, "ax_g_target": ax_g_target, "ay_g_target": ay_g_target}, base_state)
    params = {
        **fixed,
        "hf": float(base_state.get("hf", 0.0)),
        "hr": float(base_state.get("hr", 0.0)),
        "rf": float(base_state.get("rf", 0.0)),
        "rr": float(base_state.get("rr", 0.0)),
    }
    outputs = _compute_operating_point(calibrated_json_data, params, _static_ref=static_ref)

    cfg = calibrated_json_data.get("config", {}) if isinstance(calibrated_json_data.get("config"), dict) else {}
    tyres = cfg.get("tyres", {}) if isinstance(cfg.get("tyres"), dict) else {}
    front_tyre = tyres.get("front", {}) if isinstance(tyres.get("front"), dict) else {}
    rear_tyre = tyres.get("rear", {}) if isinstance(tyres.get("rear"), dict) else {}
    grip_global = float(tyres.get("rGripFactor", 1.0))
    grip_front = grip_global * float(front_tyre.get("rGripFactor", 1.0))
    grip_rear = grip_global * float(rear_tyre.get("rGripFactor", 1.0))

    env_fl = _build_tyre_fx_fy_envelope(front_tyre, float(outputs.get("total_load_fl_n", 0.0)), float(fixed["camber_fl_deg"]), grip_front, **envelope_settings, envelope_cache=envelope_cache)
    env_fr = _build_tyre_fx_fy_envelope(front_tyre, float(outputs.get("total_load_fr_n", 0.0)), float(fixed["camber_fr_deg"]), grip_front, **envelope_settings, envelope_cache=envelope_cache)
    env_rl = _build_tyre_fx_fy_envelope(rear_tyre, float(outputs.get("total_load_rl_n", 0.0)), float(fixed["camber_rl_deg"]), grip_rear, **envelope_settings, envelope_cache=envelope_cache)
    env_rr = _build_tyre_fx_fy_envelope(rear_tyre, float(outputs.get("total_load_rr_n", 0.0)), float(fixed["camber_rr_deg"]), grip_rear, **envelope_settings, envelope_cache=envelope_cache)
    caps = _gg_longitudinal_caps(body)
    return _build_gg_row_from_outputs(outputs, env_fl, env_fr, env_rl, env_rr, ax_g_target, ay_g_target, longitudinal_caps=caps)


def _evaluate_gg_point_relaxed(
    calibrated_json_data: Dict[str, Any],
    speed_kph: float,
    ax_g_target: float,
    ay_g_target: float,
    base_state: Dict[str, float],
    state_bounds: Dict[str, Any],
    ride_height_limits: Dict[str, Any],
    envelope_settings: Dict[str, Any],
    optimization_settings: Dict[str, Any],
    static_ref: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    envelope_cache: Optional[Dict[Tuple[Any, ...], Dict[str, Any]]] = None,
    prev_state: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    body = body or {}
    seed_state = dict(base_state or {})
    if isinstance(prev_state, dict):
        for k in ("hf", "hr", "rf", "rr"):
            if k in prev_state:
                seed_state[k] = float(prev_state[k])
    fixed = _gg_fixed_inputs(
        calibrated_json_data,
        {**body, "speed_kph": speed_kph, "ax_g_target": ax_g_target, "ay_g_target": ay_g_target},
        seed_state,
    )

    variables = [
        OptimizationVariable("front_heave", float(state_bounds.get("front_heave_min", -8.0)), float(state_bounds.get("front_heave_max", 8.0)), float(seed_state.get("hf", 0.0)), True),
        OptimizationVariable("rear_heave", float(state_bounds.get("rear_heave_min", -8.0)), float(state_bounds.get("rear_heave_max", 8.0)), float(seed_state.get("hr", 0.0)), True),
        OptimizationVariable("front_roll", float(state_bounds.get("front_roll_min", -12.0)), float(state_bounds.get("front_roll_max", 12.0)), float(seed_state.get("rf", 0.0)), True),
        OptimizationVariable("rear_roll", float(state_bounds.get("rear_roll_min", -12.0)), float(state_bounds.get("rear_roll_max", 12.0)), float(seed_state.get("rr", 0.0)), True),
    ]

    constraints = [
        OptimizationConstraint("hRideF", "ge", float(ride_height_limits.get("hRideF_min_m", 0.020)), 180.0),
        OptimizationConstraint("hRideR", "ge", float(ride_height_limits.get("hRideR_min_m", 0.030)), 180.0),
        OptimizationConstraint("total_load_fl_n", "ge", 1.0, 25.0),
        OptimizationConstraint("total_load_fr_n", "ge", 1.0, 25.0),
        OptimizationConstraint("total_load_rl_n", "ge", 1.0, 25.0),
        OptimizationConstraint("total_load_rr_n", "ge", 1.0, 25.0),
    ]

    max_global_points = int(optimization_settings.get("max_global_points", 1))
    n_best_candidates = int(optimization_settings.get("n_best_candidates", 1))
    local_maxiter = int(optimization_settings.get("local_maxiter", 48))
    local_xtol = float(optimization_settings.get("local_xtol", 1e-3))
    local_ftol = float(optimization_settings.get("local_ftol", 1e-3))
    max_global_points = int(np.clip(max_global_points, 1, 12))
    n_best_candidates = int(np.clip(n_best_candidates, 1, 2))
    local_maxiter = int(np.clip(local_maxiter, 10, 120))

    problem = OptimizationProblem(
        objective_mode="minimize",
        objective_name="drag_force_n",
        objective_target=None,
        variables=variables,
        constraints=constraints,
        fixed_inputs=fixed,
        search_method=str(optimization_settings.get("search_method", optimization_settings.get("method", "auto"))),
        max_global_points=max_global_points,
        n_best_candidates=n_best_candidates,
        local_maxiter=local_maxiter,
        local_xtol=local_xtol,
        local_ftol=local_ftol,
    )

    def _evaluator(sim_inputs: Dict[str, Any]) -> Dict[str, Any]:
        return _compute_operating_point(calibrated_json_data, sim_inputs, _static_ref=static_ref)

    result = run_optimization(problem, _evaluator)
    best = result.best_candidate
    outputs = dict(best.outputs or {})
    outputs["state_4w"] = dict(best.outputs.get("state_4w") or {})

    cfg = calibrated_json_data.get("config", {}) if isinstance(calibrated_json_data.get("config"), dict) else {}
    tyres = cfg.get("tyres", {}) if isinstance(cfg.get("tyres"), dict) else {}
    front_tyre = tyres.get("front", {}) if isinstance(tyres.get("front"), dict) else {}
    rear_tyre = tyres.get("rear", {}) if isinstance(tyres.get("rear"), dict) else {}
    grip_global = float(tyres.get("rGripFactor", 1.0))
    grip_front = grip_global * float(front_tyre.get("rGripFactor", 1.0))
    grip_rear = grip_global * float(rear_tyre.get("rGripFactor", 1.0))

    env_fl = _build_tyre_fx_fy_envelope(front_tyre, float(outputs.get("total_load_fl_n", 0.0)), float(fixed["camber_fl_deg"]), grip_front, **envelope_settings, envelope_cache=envelope_cache)
    env_fr = _build_tyre_fx_fy_envelope(front_tyre, float(outputs.get("total_load_fr_n", 0.0)), float(fixed["camber_fr_deg"]), grip_front, **envelope_settings, envelope_cache=envelope_cache)
    env_rl = _build_tyre_fx_fy_envelope(rear_tyre, float(outputs.get("total_load_rl_n", 0.0)), float(fixed["camber_rl_deg"]), grip_rear, **envelope_settings, envelope_cache=envelope_cache)
    env_rr = _build_tyre_fx_fy_envelope(rear_tyre, float(outputs.get("total_load_rr_n", 0.0)), float(fixed["camber_rr_deg"]), grip_rear, **envelope_settings, envelope_cache=envelope_cache)

    caps = _gg_longitudinal_caps(body)
    row = _build_gg_row_from_outputs(outputs, env_fl, env_fr, env_rl, env_rr, ax_g_target, ay_g_target, longitudinal_caps=caps)
    row["optimizer_total_cost"] = float(best.total_cost)
    row["optimizer_penalty_value"] = float(best.penalty_value)
    row["optimizer_objective_raw"] = float(best.objective_value_raw)
    return row


def _split_gg_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    rows_total: List[Dict[str, Any]] = []
    rows_front: List[Dict[str, Any]] = []
    rows_rear: List[Dict[str, Any]] = []
    rows_fl: List[Dict[str, Any]] = []
    rows_fr: List[Dict[str, Any]] = []
    rows_rl: List[Dict[str, Any]] = []
    rows_rr: List[Dict[str, Any]] = []

    for r in rows:
        ax = float(r.get("ax_g_target", 0.0))
        ay = float(r.get("ay_g_target", 0.0))
        rows_total.append({"ax": ax, "ay": ay, "fx": float(r.get("fx_total_n", 0.0)), "fy": float(r.get("fy_total_n", 0.0)), "metadata": r})
        rows_front.append({"ax": ax, "ay": ay, "fx": float(r.get("fx_front_axle_n", 0.0)), "fy": float(r.get("fy_front_axle_n", 0.0)), "metadata": r})
        rows_rear.append({"ax": ax, "ay": ay, "fx": float(r.get("fx_rear_axle_n", 0.0)), "fy": float(r.get("fy_rear_axle_n", 0.0)), "metadata": r})
        rows_fl.append({"ax": ax, "ay": ay, "fx": float(r.get("fx_fl_n", 0.0)), "fy": float(r.get("fy_fl_n", 0.0)), "metadata": r})
        rows_fr.append({"ax": ax, "ay": ay, "fx": float(r.get("fx_fr_n", 0.0)), "fy": float(r.get("fy_fr_n", 0.0)), "metadata": r})
        rows_rl.append({"ax": ax, "ay": ay, "fx": float(r.get("fx_rl_n", 0.0)), "fy": float(r.get("fy_rl_n", 0.0)), "metadata": r})
        rows_rr.append({"ax": ax, "ay": ay, "fx": float(r.get("fx_rr_n", 0.0)), "fy": float(r.get("fy_rr_n", 0.0)), "metadata": r})

    return {
        "rows_total": _json_clean(rows_total),
        "rows_front_axle": _json_clean(rows_front),
        "rows_rear_axle": _json_clean(rows_rear),
        "rows_fl": _json_clean(rows_fl),
        "rows_fr": _json_clean(rows_fr),
        "rows_rl": _json_clean(rows_rl),
        "rows_rr": _json_clean(rows_rr),
    }


def _compute_gg_static_map(
    calibrated_json_data: Dict[str, Any],
    body: Dict[str, Any],
    mode: str = "frozen",
) -> Dict[str, Any]:
    envelope_settings = dict(body.get("envelope_settings") or {})
    envelope_settings = {
        "alpha_min_deg": float(envelope_settings.get("alpha_min_deg", -18.0)),
        "alpha_max_deg": float(envelope_settings.get("alpha_max_deg", 18.0)),
        "alpha_steps": int(envelope_settings.get("alpha_steps", 41)),
        "kappa_min": float(envelope_settings.get("kappa_min", -0.20)),
        "kappa_max": float(envelope_settings.get("kappa_max", 0.20)),
        "kappa_steps": int(envelope_settings.get("kappa_steps", 41)),
    }
    base_state_in = body.get("base_state") if isinstance(body.get("base_state"), dict) else {}
    base_state = {
        "hf": float(base_state_in.get("hf", 0.0)),
        "hr": float(base_state_in.get("hr", 0.0)),
        "rf": float(base_state_in.get("rf", 0.0)),
        "rr": float(base_state_in.get("rr", 0.0)),
    }
    state_bounds = dict(body.get("state_bounds") or {})
    ride_height_limits = dict(body.get("ride_height_limits") or {})
    optimization_settings = dict(body.get("optimization") or {})

    axis_cap = 41 if mode == "frozen" else 13
    ax_values, ax_clipped = _safe_axis_values(
        float(body.get("ax_min_g", -3.0)),
        float(body.get("ax_max_g", 2.0)),
        float(body.get("ax_step_g", 0.25)),
        max_points=axis_cap,
    )
    ay_values, ay_clipped = _safe_axis_values(
        float(body.get("ay_min_g", -4.0)),
        float(body.get("ay_max_g", 4.0)),
        float(body.get("ay_step_g", 0.25)),
        max_points=axis_cap,
    )

    warnings: List[str] = []
    if ax_clipped:
        warnings.append(f"Ax grid was reduced to {axis_cap} points for runtime control.")
    if ay_clipped:
        warnings.append(f"Ay grid was reduced to {axis_cap} points for runtime control.")
    if mode == "relaxed":
        warnings.append("Relaxed mode uses warm-start continuation + fast local solve.")

    # Runtime control for tyre envelope sweep.
    grid_points = max(1, len(ax_values) * len(ay_values))
    if mode != "frozen":
        if grid_points > 160:
            envelope_settings["alpha_steps"] = int(min(int(envelope_settings.get("alpha_steps", 41)), 17))
            envelope_settings["kappa_steps"] = int(min(int(envelope_settings.get("kappa_steps", 41)), 17))
            warnings.append("Slip envelope resolution auto-reduced for relaxed/families runtime control.")
        if grid_points > 240:
            envelope_settings["alpha_steps"] = int(min(int(envelope_settings.get("alpha_steps", 41)), 13))
            envelope_settings["kappa_steps"] = int(min(int(envelope_settings.get("kappa_steps", 41)), 13))
            warnings.append("Slip envelope resolution was strongly reduced due dense GG grid.")

    static_ref = compute_center_antis_for_state(calibrated_json_data, hf=0.0, rf=0.0, hr=0.0, rr=0.0)
    envelope_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    prev_state: Optional[Dict[str, float]] = None
    for idx_ax, ax_g in enumerate(ax_values):
        ay_iter = ay_values if (idx_ax % 2 == 0 or mode == "frozen") else list(reversed(ay_values))
        for ay_g in ay_iter:
            if mode == "relaxed":
                row = _evaluate_gg_point_relaxed(
                    calibrated_json_data=calibrated_json_data,
                    speed_kph=float(body.get("speed_kph", 140.0)),
                    ax_g_target=float(ax_g),
                    ay_g_target=float(ay_g),
                    base_state=base_state,
                    state_bounds=state_bounds,
                    ride_height_limits=ride_height_limits,
                    envelope_settings=envelope_settings,
                    optimization_settings=optimization_settings,
                    static_ref=static_ref,
                    body=body,
                    envelope_cache=envelope_cache,
                    prev_state=prev_state,
                )
            else:
                row = _evaluate_gg_point_frozen(
                    calibrated_json_data=calibrated_json_data,
                    speed_kph=float(body.get("speed_kph", 140.0)),
                    ax_g_target=float(ax_g),
                    ay_g_target=float(ay_g),
                    base_state=base_state,
                    envelope_settings=envelope_settings,
                    static_ref=static_ref,
                    body=body,
                    envelope_cache=envelope_cache,
                )
            rows.append(row)
            prev_state = {
                "hf": float(row.get("hf", 0.0)),
                "hr": float(row.get("hr", 0.0)),
                "rf": float(row.get("rf", 0.0)),
                "rr": float(row.get("rr", 0.0)),
            }

    split = _split_gg_rows(rows)
    return {
        "mode": mode,
        "rows": _json_clean(rows),
        **split,
        "meta": {
            "speed_kph": float(body.get("speed_kph", 140.0)),
            "grid_shape": [len(ax_values), len(ay_values)],
            "warnings": warnings,
        },
    }


def _compute_gg_state_families(calibrated_json_data: Dict[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
    family_mode = str(body.get("family_mode", "relaxed")).strip().lower()
    families = body.get("families") if isinstance(body.get("families"), list) else []
    if not families:
        raise ValueError("State Families requires at least one family definition.")

    runs: List[Dict[str, Any]] = []
    for idx, family in enumerate(families):
        if not isinstance(family, dict):
            continue
        name = str(family.get("name", f"Family {idx + 1}")).strip() or f"Family {idx + 1}"
        fam_body = dict(body)
        fam_body["base_state"] = family.get("base_state", {})
        result = _compute_gg_static_map(calibrated_json_data, fam_body, mode="relaxed" if family_mode == "relaxed" else "frozen")
        runs.append({"name": name, "base_state": fam_body["base_state"], "result": result})

    if not runs:
        raise ValueError("No valid family entries were provided.")
    return {"family_mode": family_mode, "families": runs}


def _cfg_get(root: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    node: Any = root
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    return default if node is None else node


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return float(v) if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _interpolate_1d(x: float, x_values: List[float], y_values: List[float]) -> Tuple[float, float]:
    if not x_values or not y_values or len(x_values) != len(y_values):
        return 0.0, 0.0
    if len(x_values) == 1:
        y = float(y_values[0])
        k = y / max(float(x_values[0]), 1e-9) if abs(float(x_values[0])) > 1e-9 else 0.0
        return y, k
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    xq = float(np.clip(x, float(np.min(x_arr)), float(np.max(x_arr))))
    yq = float(np.interp(xq, x_arr, y_arr))
    idx = int(np.searchsorted(x_arr, xq, side="right"))
    i0 = int(np.clip(idx - 1, 0, len(x_arr) - 2))
    i1 = i0 + 1
    dx = float(x_arr[i1] - x_arr[i0])
    slope = float((y_arr[i1] - y_arr[i0]) / dx) if abs(dx) > 1e-12 else 0.0
    return yq, slope


def _extract_platform_solver_parameters_from_json(calibrated_json_data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = calibrated_json_data.get("config", {}) if isinstance(calibrated_json_data.get("config"), dict) else {}
    chassis = cfg.get("chassis", {}) if isinstance(cfg.get("chassis"), dict) else {}
    mass_block = chassis.get("carRunningMass", {}) if isinstance(chassis.get("carRunningMass"), dict) else {}
    suspension = cfg.get("suspension", {}) if isinstance(cfg.get("suspension"), dict) else {}
    aero = cfg.get("aero", {}) if isinstance(cfg.get("aero"), dict) else {}
    drs_block = aero.get("DRS", {}) if isinstance(aero.get("DRS"), dict) else {}
    front_pick = _cfg_get(cfg, ["suspension", "front", "external", "pickUpPts"], {})
    rear_pick = _cfg_get(cfg, ["suspension", "rear", "external", "pickUpPts"], {})
    front_axle = front_pick.get("rAxleC", [0.0, 0.0, 0.0]) if isinstance(front_pick, dict) else [0.0, 0.0, 0.0]
    rear_axle = rear_pick.get("rAxleC", [-1.0, 0.0, 0.0]) if isinstance(rear_pick, dict) else [-1.0, 0.0, 0.0]
    front_axle_x_mm = _safe_float(front_axle[0] if len(front_axle) > 0 else 0.0, 0.0) * 1000.0
    rear_axle_x_mm = _safe_float(rear_axle[0] if len(rear_axle) > 0 else -1.0, -1.0) * 1000.0
    wheelbase_mm = max(1.0, abs(front_axle_x_mm - rear_axle_x_mm))

    m_car = _safe_float(mass_block.get("mCar", chassis.get("mCar", 0.0)), 0.0)
    m_hub_f = _safe_float(chassis.get("mHubF", 0.0), 0.0)
    m_hub_r = _safe_float(chassis.get("mHubR", 0.0), 0.0)
    mass_unsprung_total = 2.0 * (m_hub_f + m_hub_r)
    mass_sprung_base = max(0.0, m_car - mass_unsprung_total)
    r_weight_bal_f = float(np.clip(_safe_float(mass_block.get("rWeightBalF", 0.5), 0.5), 0.0, 1.0))

    def _spring_block(axle: str) -> Dict[str, Any]:
        return _cfg_get(suspension, [axle, "internal", "spring"], {})

    def _bump_block(axle: str) -> Dict[str, Any]:
        return _cfg_get(suspension, [axle, "internal", "bumpStop"], {})

    def _arb_block(axle: str) -> Dict[str, Any]:
        return _cfg_get(suspension, [axle, "internal", "antiRollBar"], {})

    spring_f = _spring_block("front")
    spring_r = _spring_block("rear")
    bump_f = _bump_block("front")
    bump_r = _bump_block("rear")
    arb_f = _arb_block("front")
    arb_r = _arb_block("rear")
    axle_f = suspension.get("front", {}) if isinstance(suspension.get("front"), dict) else {}
    axle_r = suspension.get("rear", {}) if isinstance(suspension.get("rear"), dict) else {}

    def _spring_wheel_rate(spring: Dict[str, Any]) -> float:
        k_spring = _safe_float(spring.get("kSpring", 0.0), 0.0)
        mr = _safe_float(spring.get("MR_WD", 1.0), 1.0)
        if abs(mr) < 1e-9:
            mr = 1.0
        return max(1.0, k_spring / (mr * mr))

    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and (not isinstance(value, bool)) and np.isfinite(float(value))

    def _find_first_numeric_by_keys(obj: Any, keys_lower: set) -> Optional[float]:
        stack = [obj]
        visited = set()
        while stack:
            cur = stack.pop()
            oid = id(cur)
            if oid in visited:
                continue
            visited.add(oid)
            if isinstance(cur, dict):
                for k, v in cur.items():
                    if isinstance(k, str) and k.strip().lower() in keys_lower and _is_number(v):
                        return float(v)
                    if isinstance(v, (dict, list, tuple)):
                        stack.append(v)
            elif isinstance(cur, (list, tuple)):
                for v in cur:
                    if isinstance(v, (dict, list, tuple)):
                        stack.append(v)
        return None

    def _extract_damper_static_stroke_m(axle_data: Dict[str, Any]) -> Optional[float]:
        keys_mm = {
            "damper_static_mm",
            "damperstatic_mm",
            "damper_static_stroke_mm",
            "damperstaticstrokemm",
        }
        keys_m = {
            "damper_static_m",
            "damperstatic_m",
            "damper_static_stroke_m",
            "damperstaticstrokem",
        }
        keys_auto = {
            "damper_static",
            "damperstatic",
            "damper_static_stroke",
            "damperstaticstroke",
            "s_damper_static",
            "sdamperstatic",
            "x_damper_static",
            "xdamperstatic",
        }
        mm_val = _find_first_numeric_by_keys(axle_data, keys_mm)
        if mm_val is not None:
            return float(mm_val) * 1e-3
        m_val = _find_first_numeric_by_keys(axle_data, keys_m)
        if m_val is not None:
            return float(m_val)
        auto_val = _find_first_numeric_by_keys(axle_data, keys_auto)
        if auto_val is None:
            return None
        return float(auto_val) * 1e-3 if abs(float(auto_val)) > 2.0 else float(auto_val)

    def _extract_stroke_limits_m(axle_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        keys_min_mm = {
            "damper_min_mm",
            "damperstroke_min_mm",
            "s_damper_min_mm",
            "x_damper_min_mm",
        }
        keys_max_mm = {
            "damper_max_mm",
            "damperstroke_max_mm",
            "s_damper_max_mm",
            "x_damper_max_mm",
        }
        keys_min_m = {
            "damper_min_m",
            "damperstroke_min_m",
            "s_damper_min_m",
            "x_damper_min_m",
        }
        keys_max_m = {
            "damper_max_m",
            "damperstroke_max_m",
            "s_damper_max_m",
            "x_damper_max_m",
        }
        keys_min_auto = {"damper_min", "damperstroke_min", "s_damper_min", "x_damper_min"}
        keys_max_auto = {"damper_max", "damperstroke_max", "s_damper_max", "x_damper_max"}
        vmin_mm = _find_first_numeric_by_keys(axle_data, keys_min_mm)
        vmax_mm = _find_first_numeric_by_keys(axle_data, keys_max_mm)
        if (vmin_mm is not None) and (vmax_mm is not None):
            return float(vmin_mm) * 1e-3, float(vmax_mm) * 1e-3
        vmin_m = _find_first_numeric_by_keys(axle_data, keys_min_m)
        vmax_m = _find_first_numeric_by_keys(axle_data, keys_max_m)
        if (vmin_m is not None) and (vmax_m is not None):
            return float(vmin_m), float(vmax_m)
        vmin_auto = _find_first_numeric_by_keys(axle_data, keys_min_auto)
        vmax_auto = _find_first_numeric_by_keys(axle_data, keys_max_auto)
        if (vmin_auto is None) or (vmax_auto is None):
            return None, None
        min_out = float(vmin_auto) * 1e-3 if abs(float(vmin_auto)) > 2.0 else float(vmin_auto)
        max_out = float(vmax_auto) * 1e-3 if abs(float(vmax_auto)) > 2.0 else float(vmax_auto)
        return min_out, max_out

    def _travel_from_static_and_stroke_limits(axle_data: Dict[str, Any], mr_wd: float) -> Tuple[Optional[float], Optional[float]]:
        s_static = _extract_damper_static_stroke_m(axle_data)
        s_min, s_max = _extract_stroke_limits_m(axle_data)
        if (s_static is None) or (s_min is None) or (s_max is None):
            return None, None
        deltas = [float(s_min - s_static), float(s_max - s_static)]
        stroke_bump_m = max(0.0, max(deltas))
        stroke_droop_m = max(0.0, max(-d for d in deltas))
        mr_scale = abs(float(mr_wd)) if np.isfinite(mr_wd) and abs(float(mr_wd)) > 1e-9 else 1.0
        # In this app MR_WD is treated as linear wheel-domain ratio; convert stroke -> wheel travel.
        return float(stroke_bump_m * mr_scale), float(stroke_droop_m * mr_scale)

    def _arb_wheel_rate(arb: Dict[str, Any]) -> Tuple[float, str]:
        k_arb = _safe_float(arb.get("kAntiRollBar", 0.0), 0.0)
        mr_da = _safe_float(arb.get("MR_DA_Linear", 1.0), 1.0)
        if abs(mr_da) < 1e-9:
            mr_da = 1.0
        # V1 hypothesis: ARB rate is transformed to wheel domain with linear MR_DA.
        return (max(0.0, k_arb / (mr_da * mr_da)), "arb_linear_mr_da_hypothesis")

    def _compliance_to_m_per_n(value: Any) -> float:
        raw = abs(_safe_float(value, 0.0))
        if raw <= 1e-12:
            return 0.0
        # If value is very small, treat as compliance [m/N]. Otherwise treat as stiffness [N/m].
        return raw if raw < 1e-2 else (1.0 / raw)

    def _bump_scale(x_data: List[float]) -> float:
        x_max = max(abs(float(v)) for v in x_data) if x_data else 0.0
        return 1.0 if x_max < 0.2 else 0.001

    front_bump_x_data = [float(_safe_float(x, 0.0)) for x in (bump_f.get("xData") or [])]
    rear_bump_x_data = [float(_safe_float(x, 0.0)) for x in (bump_r.get("xData") or [])]
    front_scale = _bump_scale(front_bump_x_data)
    rear_scale = _bump_scale(rear_bump_x_data)
    front_free_gap_m = float(_safe_float(bump_f.get("xFreeGap", 0.0), 0.0)) * front_scale
    rear_free_gap_m = float(_safe_float(bump_r.get("xFreeGap", 0.0), 0.0)) * rear_scale
    front_bump_range_m = max(0.0, max(front_bump_x_data) * front_scale) if front_bump_x_data else 0.0
    rear_bump_range_m = max(0.0, max(rear_bump_x_data) * rear_scale) if rear_bump_x_data else 0.0
    front_max_comp_m = float(np.clip((front_free_gap_m + front_bump_range_m) if front_bump_range_m > 0.0 else 0.060, 0.015, 0.120))
    rear_max_comp_m = float(np.clip((rear_free_gap_m + rear_bump_range_m) if rear_bump_range_m > 0.0 else 0.060, 0.015, 0.120))
    front_max_droop_m = 0.040
    rear_max_droop_m = 0.040
    travel_limit_source = "fallback_bumpstop_plus_default_droop"
    mr_front_wd = _safe_float(spring_f.get("MR_WD", 1.0), 1.0)
    mr_rear_wd = _safe_float(spring_r.get("MR_WD", 1.0), 1.0)
    front_bump_from_stroke_m, front_droop_from_stroke_m = _travel_from_static_and_stroke_limits(axle_f, mr_front_wd)
    rear_bump_from_stroke_m, rear_droop_from_stroke_m = _travel_from_static_and_stroke_limits(axle_r, mr_rear_wd)
    if (front_bump_from_stroke_m is not None) and (front_droop_from_stroke_m is not None):
        front_max_comp_m = float(np.clip(front_bump_from_stroke_m, 0.005, 0.150))
        front_max_droop_m = float(np.clip(front_droop_from_stroke_m, 0.005, 0.150))
        travel_limit_source = "damper_static_plus_stroke_limits_front"
    if (rear_bump_from_stroke_m is not None) and (rear_droop_from_stroke_m is not None):
        rear_max_comp_m = float(np.clip(rear_bump_from_stroke_m, 0.005, 0.150))
        rear_max_droop_m = float(np.clip(rear_droop_from_stroke_m, 0.005, 0.150))
        if travel_limit_source == "damper_static_plus_stroke_limits_front":
            travel_limit_source = "damper_static_plus_stroke_limits_front_rear"
        else:
            travel_limit_source = "damper_static_plus_stroke_limits_rear"

    z_cg_raw = _safe_float(chassis.get("zCoG", 0.0), 0.0)
    z_cg_ref_mm = abs(z_cg_raw) * 1000.0 if abs(z_cg_raw) < 10.0 else abs(z_cg_raw)

    return {
        "m_car_kg": float(m_car),
        "m_hub_f_kg": float(m_hub_f),
        "m_hub_r_kg": float(m_hub_r),
        "mass_sprung_base_kg": float(mass_sprung_base),
        "r_weight_bal_f": float(r_weight_bal_f),
        "wheelbase_mm": float(wheelbase_mm),
        "front_axle_x_mm": float(front_axle_x_mm),
        "rear_axle_x_mm": float(rear_axle_x_mm),
        "z_cg_ref_mm": float(z_cg_ref_mm),
        "front_spring_wheel_rate_npm": float(_spring_wheel_rate(spring_f)),
        "rear_spring_wheel_rate_npm": float(_spring_wheel_rate(spring_r)),
        "front_preload_n": float(_safe_float(spring_f.get("FSpringPreload", 0.0), 0.0)),
        "rear_preload_n": float(_safe_float(spring_r.get("FSpringPreload", 0.0), 0.0)),
        "front_bump_x_data": front_bump_x_data,
        "front_bump_f_data": [float(_safe_float(x, 0.0)) for x in (bump_f.get("FData") or [])],
        "front_bump_free_gap": float(_safe_float(bump_f.get("xFreeGap", 0.0), 0.0)),
        "rear_bump_x_data": rear_bump_x_data,
        "rear_bump_f_data": [float(_safe_float(x, 0.0)) for x in (bump_r.get("FData") or [])],
        "rear_bump_free_gap": float(_safe_float(bump_r.get("xFreeGap", 0.0), 0.0)),
        "front_max_compression_m": float(front_max_comp_m),
        "rear_max_compression_m": float(rear_max_comp_m),
        "front_max_droop_m": float(front_max_droop_m),
        "rear_max_droop_m": float(rear_max_droop_m),
        "travel_limit_source": str(travel_limit_source),
        "front_compliance_mpn": float(_compliance_to_m_per_n(chassis.get("kVerticalSuspensionComplianceF", 0.0))),
        "rear_compliance_mpn": float(_compliance_to_m_per_n(chassis.get("kVerticalSuspensionComplianceR", 0.0))),
        "front_arb_wheel_rate_npm": float(_arb_wheel_rate(arb_f)[0]),
        "rear_arb_wheel_rate_npm": float(_arb_wheel_rate(arb_r)[0]),
        "arb_model_note": _arb_wheel_rate(arb_f)[1],
        "aero": {
            "ARef": float(_safe_float(aero.get("ARef", 1.0), 1.0)),
            "flap_angles": dict(aero.get("flapAngles", {})) if isinstance(aero.get("flapAngles"), dict) else {},
            "offsets": dict(aero.get("coefficientOffsets", {})) if isinstance(aero.get("coefficientOffsets"), dict) else {},
            "drs_enabled": bool(drs_block.get("bDRSEnabled", aero.get("bDRSEnabled", False))),
            "cl_front_poly": list(aero.get("PolynomialCLiftBodyFDefinition", []) or []),
            "cl_rear_poly": list(aero.get("PolynomialCLiftBodyRDefinition", []) or []),
            "cd_poly": list(aero.get("PolynomialCDragBodyDefinition", []) or []),
            "drs_cl_front_poly": list(drs_block.get("CLiftBodyFDRSPolynomial", []) or []),
            "drs_cl_rear_poly": list(drs_block.get("CLiftBodyRDRSPolynomial", []) or []),
            "drs_cd_poly": list(drs_block.get("CDragBodyDRSPolynomial", []) or []),
            "cl_front_factor": float(_safe_float(aero.get("rCLiftBodyFFactor", 1.0), 1.0)),
            "cl_rear_factor": float(_safe_float(aero.get("rCLiftBodyRFactor", 1.0), 1.0)),
            "cd_factor": float(_safe_float(aero.get("rCDragBodyFactor", 1.0), 1.0)),
        },
    }


def _platform_eval_aero_loads(
    params: Dict[str, Any],
    h_ride_f_m: float,
    h_ride_r_m: float,
    speed_kph: float,
    drs_on: bool,
    air_density: float = 1.225,
) -> Dict[str, float]:
    aero = params.get("aero", {})
    flap = aero.get("flap_angles", {}) if isinstance(aero.get("flap_angles"), dict) else {}
    offsets = aero.get("offsets", {}) if isinstance(aero.get("offsets"), dict) else {}
    vars_map = {
        "hRideF": float(h_ride_f_m),
        "hRideR": float(h_ride_r_m),
        "aFlapF": float(_safe_float(flap.get("aFlapF", 0.0), 0.0)),
        "aFlapR": float(_safe_float(flap.get("aFlapR", 0.0), 0.0)),
    }
    raw_clf = _eval_poly_terms(list(aero.get("cl_front_poly", []) or []), vars_map)
    raw_clr = _eval_poly_terms(list(aero.get("cl_rear_poly", []) or []), vars_map)
    raw_cd = _eval_poly_terms(list(aero.get("cd_poly", []) or []), vars_map)
    drs_effective = bool(drs_on and bool(aero.get("drs_enabled", False)))
    dclf = _eval_poly_terms(list(aero.get("drs_cl_front_poly", []) or []), vars_map) if drs_effective else 0.0
    dclr = _eval_poly_terms(list(aero.get("drs_cl_rear_poly", []) or []), vars_map) if drs_effective else 0.0
    dcd = _eval_poly_terms(list(aero.get("drs_cd_poly", []) or []), vars_map) if drs_effective else 0.0

    cl_front_offset = _safe_float(offsets.get("CLiftBodyFUserOffset", 0.0), 0.0)
    cl_rear_offset = _safe_float(offsets.get("CLiftBodyRUserOffset", 0.0), 0.0)
    cl_global_offset = _safe_float(offsets.get("CLiftBodyUserOffset", 0.0), 0.0)
    cd_offset = _safe_float(offsets.get("CDragBodyUserOffset", 0.0), 0.0)
    aero_bal_offset = _safe_float(offsets.get("rAeroBalanceUserOffset", 0.0), 0.0)
    front_share = float(np.clip(0.5 + aero_bal_offset, 0.0, 1.0))
    rear_share = 1.0 - front_share
    cl_front = ((raw_clf + cl_front_offset + cl_global_offset * front_share) * _safe_float(aero.get("cl_front_factor", 1.0), 1.0)) + dclf
    cl_rear = ((raw_clr + cl_rear_offset + cl_global_offset * rear_share) * _safe_float(aero.get("cl_rear_factor", 1.0), 1.0)) + dclr
    cd_total = ((raw_cd + cd_offset) * _safe_float(aero.get("cd_factor", 1.0), 1.0)) + dcd

    v_mps = float(speed_kph) / 3.6
    qA = 0.5 * float(air_density) * (v_mps * v_mps) * max(1e-9, _safe_float(aero.get("ARef", 1.0), 1.0))
    front_load = qA * cl_front
    rear_load = qA * cl_rear
    total_load = front_load + rear_load
    aero_balance_front_pct = 100.0 * front_load / total_load if abs(total_load) > 1e-9 else float("nan")
    return {
        "front_aero_load_n": float(front_load),
        "rear_aero_load_n": float(rear_load),
        "total_aero_load_n": float(total_load),
        "aero_balance_front_pct": float(aero_balance_front_pct) if np.isfinite(aero_balance_front_pct) else float("nan"),
        "drag_force_n": float(qA * cd_total),
        "drs_effective": bool(drs_effective),
    }


def _platform_bump_force_and_rate(
    wheel_travel_m: float,
    x_data: List[float],
    f_data: List[float],
    free_gap: float,
) -> Tuple[float, float]:
    if not x_data or not f_data or len(x_data) != len(f_data):
        return 0.0, 0.0
    x_max = max(abs(float(v)) for v in x_data) if x_data else 0.0
    scale = 1.0 if x_max < 0.2 else 0.001
    gap_m = float(max(0.0, free_gap)) * scale
    travel_m = max(0.0, float(wheel_travel_m))
    bump_comp = travel_m - gap_m
    if bump_comp <= 0.0:
        return 0.0, 0.0
    xx = [float(v) * scale for v in x_data]
    yy = [float(v) for v in f_data]
    force_n, k_npm = _interpolate_1d(bump_comp, xx, yy)
    return max(0.0, force_n), max(0.0, k_npm)


def _platform_state_from_dz(
    dz_fl_m: float,
    dz_fr_m: float,
    dz_rl_m: float,
    dz_rr_m: float,
    front_track_mm: float,
    rear_track_mm: float,
) -> Dict[str, float]:
    hf_mm = 1000.0 * 0.5 * (dz_fl_m + dz_fr_m)
    hr_mm = 1000.0 * 0.5 * (dz_rl_m + dz_rr_m)
    rf_mm_eq = 1000.0 * (dz_fl_m - dz_fr_m)
    rr_mm_eq = 1000.0 * (dz_rl_m - dz_rr_m)
    ft_mm = max(1e-6, float(front_track_mm))
    rt_mm = max(1e-6, float(rear_track_mm))
    rf_deg = math.degrees(math.atan(rf_mm_eq / ft_mm))
    rr_deg = math.degrees(math.atan(rr_mm_eq / rt_mm))
    return {
        "hf_mm": float(hf_mm),
        "hr_mm": float(hr_mm),
        "rf_deg": float(rf_deg),
        "rr_deg": float(rr_deg),
        "rf_mm_equiv": float(rf_mm_eq),
        "rr_mm_equiv": float(rr_mm_eq),
    }


def _platform_equivalent_stiffness(
    params: Dict[str, Any],
    dz: np.ndarray,
    front_track_mm: float,
    rear_track_mm: float,
) -> Dict[str, float]:
    kf_spring = _safe_float(params.get("front_spring_wheel_rate_npm", 0.0), 0.0)
    kr_spring = _safe_float(params.get("rear_spring_wheel_rate_npm", 0.0), 0.0)
    cf = _safe_float(params.get("front_compliance_mpn", 0.0), 0.0)
    cr = _safe_float(params.get("rear_compliance_mpn", 0.0), 0.0)
    arb_f_rate = _safe_float(params.get("front_arb_wheel_rate_npm", 0.0), 0.0)
    arb_r_rate = _safe_float(params.get("rear_arb_wheel_rate_npm", 0.0), 0.0)

    bf_force_fl, bf_k_fl = _platform_bump_force_and_rate(float(dz[0]), params.get("front_bump_x_data", []), params.get("front_bump_f_data", []), _safe_float(params.get("front_bump_free_gap", 0.0), 0.0))
    bf_force_fr, bf_k_fr = _platform_bump_force_and_rate(float(dz[1]), params.get("front_bump_x_data", []), params.get("front_bump_f_data", []), _safe_float(params.get("front_bump_free_gap", 0.0), 0.0))
    br_force_rl, br_k_rl = _platform_bump_force_and_rate(float(dz[2]), params.get("rear_bump_x_data", []), params.get("rear_bump_f_data", []), _safe_float(params.get("rear_bump_free_gap", 0.0), 0.0))
    br_force_rr, br_k_rr = _platform_bump_force_and_rate(float(dz[3]), params.get("rear_bump_x_data", []), params.get("rear_bump_f_data", []), _safe_float(params.get("rear_bump_free_gap", 0.0), 0.0))

    def _k_eff(k_wheel: float, compliance_mpn: float) -> float:
        k = max(1.0, float(k_wheel))
        c = max(0.0, float(compliance_mpn))
        if c <= 1e-15:
            return k
        return 1.0 / max(1e-12, (1.0 / k) + c)

    k_fl = _k_eff(kf_spring + bf_k_fl, cf)
    k_fr = _k_eff(kf_spring + bf_k_fr, cf)
    k_rl = _k_eff(kr_spring + br_k_rl, cr)
    k_rr = _k_eff(kr_spring + br_k_rr, cr)

    ft_m = max(1e-9, float(front_track_mm) / 1000.0)
    rt_m = max(1e-9, float(rear_track_mm) / 1000.0)
    kz_front = k_fl + k_fr
    kz_rear = k_rl + k_rr
    kphi_front = ((k_fl + k_fr) * (ft_m ** 2) * 0.25) + (arb_f_rate * (ft_m ** 2) * 0.5)
    kphi_rear = ((k_rl + k_rr) * (rt_m ** 2) * 0.25) + (arb_r_rate * (rt_m ** 2) * 0.5)

    return {
        "k_fl_npm": float(k_fl),
        "k_fr_npm": float(k_fr),
        "k_rl_npm": float(k_rl),
        "k_rr_npm": float(k_rr),
        "kz_front_npm": float(max(1.0, kz_front)),
        "kz_rear_npm": float(max(1.0, kz_rear)),
        "kphi_front_npm": float(max(1e-6, kphi_front)),
        "kphi_rear_npm": float(max(1e-6, kphi_rear)),
        "front_bump_force_n": float(bf_force_fl + bf_force_fr),
        "rear_bump_force_n": float(br_force_rl + br_force_rr),
    }


def _iterate_platform_state_solver(
    calibrated_json_data: Dict[str, Any],
    params: Dict[str, Any],
    point_inputs: Dict[str, Any],
    initial_guess: Optional[Dict[str, float]] = None,
    geometry_state_cache: Optional[Dict[Tuple[float, float, float, float], Dict[str, float]]] = None,
    max_iter: int = 40,
    tol_m: float = 1e-5,
) -> Dict[str, Any]:
    g = 9.80665
    ay = _safe_float(point_inputs.get("ay_g", 0.0), 0.0) * g
    ax = _safe_float(point_inputs.get("ax_g", 0.0), 0.0) * g
    speed_kph = _safe_float(point_inputs.get("speed_kph", 0.0), 0.0)
    fuel_mass_kg = max(0.0, _safe_float(point_inputs.get("fuel_mass_kg", 0.0), 0.0))
    drs_on = bool(point_inputs.get("drs_on", False))
    rho = _safe_float(point_inputs.get("air_density", 1.225), 1.225)

    base_front_track_mm = _safe_float(params.get("base_front_track_mm", 1200.0), 1200.0)
    base_rear_track_mm = _safe_float(params.get("base_rear_track_mm", 1200.0), 1200.0)
    solver_mode = str(point_inputs.get("solver_mode", "fast")).strip().lower()
    geometry_refresh_stride = 1
    if solver_mode == "accurate":
        max_iter = int(max(max_iter, 40))
        tol_m = min(float(tol_m), 1e-5)
        relax = 0.45
    elif solver_mode == "turbo":
        max_iter = min(int(max_iter), 8)
        tol_m = max(float(tol_m), 8e-5)
        relax = 0.62
        geometry_refresh_stride = 2
    else:
        # Fast mode: fewer iterations and slightly looser tolerance.
        max_iter = min(int(max_iter), 24)
        tol_m = max(float(tol_m), 2e-5)
        relax = 0.52
    # Optional per-point overrides (used by auto policies for specific sweep types).
    if "solver_max_iter" in point_inputs:
        max_iter = int(max(4, _safe_float(point_inputs.get("solver_max_iter", max_iter), max_iter)))
    if "solver_relax" in point_inputs:
        relax = float(np.clip(_safe_float(point_inputs.get("solver_relax", relax), relax), 0.35, 0.85))
    if "geometry_refresh_stride" in point_inputs:
        geometry_refresh_stride = int(max(1, _safe_float(point_inputs.get("geometry_refresh_stride", geometry_refresh_stride), geometry_refresh_stride)))

    dz0 = np.zeros(4, dtype=float)
    if isinstance(initial_guess, dict):
        dz0 = np.array(
            [
                _safe_float(initial_guess.get("dz_fl_m", 0.0), 0.0),
                _safe_float(initial_guess.get("dz_fr_m", 0.0), 0.0),
                _safe_float(initial_guess.get("dz_rl_m", 0.0), 0.0),
                _safe_float(initial_guess.get("dz_rr_m", 0.0), 0.0),
            ],
            dtype=float,
        )
    dz = np.copy(dz0)
    has_initial_guess = isinstance(initial_guess, dict) and bool(initial_guess)
    local_branch_lock = bool(point_inputs.get("local_branch_lock", False)) and has_initial_guess
    max_delta_point_m = max(0.001, _safe_float(point_inputs.get("max_delta_per_point_mm", 10.0), 10.0) / 1000.0)
    converged = False
    solver_message = "max iterations reached"
    last_geom: Dict[str, Any] = dict(params.get("base_geometry", {}))
    last_aux: Dict[str, Any] = {}
    front_max_comp = float(max(0.001, _safe_float(params.get("front_max_compression_m", 0.060), 0.060)))
    rear_max_comp = float(max(0.001, _safe_float(params.get("rear_max_compression_m", 0.060), 0.060)))
    front_max_droop = float(max(0.001, _safe_float(params.get("front_max_droop_m", 0.040), 0.040)))
    rear_max_droop = float(max(0.001, _safe_float(params.get("rear_max_droop_m", 0.040), 0.040)))
    # Travel limits are precomputed once at run start; solver only applies direct clips here.
    saturation_count = 0

    for it in range(1, int(max_iter) + 1):
        st = _platform_state_from_dz(
            dz_fl_m=float(dz[0]),
            dz_fr_m=float(dz[1]),
            dz_rl_m=float(dz[2]),
            dz_rr_m=float(dz[3]),
            front_track_mm=_safe_float(last_geom.get("front_track_mm", base_front_track_mm), base_front_track_mm),
            rear_track_mm=_safe_float(last_geom.get("rear_track_mm", base_rear_track_mm), base_rear_track_mm),
        )
        must_refresh_geom = (it == 1) or (it % int(max(1, geometry_refresh_stride)) == 0)
        if must_refresh_geom:
            try:
                geom = compute_center_antis_for_state(
                    calibrated_json_data,
                    hf=float(st["hf_mm"]),
                    rf=float(st["rf_deg"]),
                    hr=float(st["hr_mm"]),
                    rr=float(st["rr_deg"]),
                    state_cache=geometry_state_cache,
                )
                last_geom = geom
            except Exception as exc:
                solver_message = f"geometry solve failed: {exc}"
                break
        geom = last_geom

        front_track_mm = _safe_float(geom.get("front_track_mm", base_front_track_mm), base_front_track_mm)
        rear_track_mm = _safe_float(geom.get("rear_track_mm", base_rear_track_mm), base_rear_track_mm)
        h_ride_f_m = _safe_float(geom.get("front_ride_height", 0.0), 0.0) / 1000.0
        h_ride_r_m = _safe_float(geom.get("rear_ride_height", 0.0), 0.0) / 1000.0
        aero = _platform_eval_aero_loads(params, h_ride_f_m, h_ride_r_m, speed_kph, drs_on=drs_on, air_density=rho)
        stiffness = _platform_equivalent_stiffness(params, dz, front_track_mm, rear_track_mm)

        m_s_eff = _safe_float(params.get("mass_sprung_base_kg", 0.0), 0.0) + fuel_mass_kg
        fuel_weight_n = fuel_mass_kg * g
        wbf = _safe_float(params.get("r_weight_bal_f", 0.5), 0.5)
        wb = _safe_float(params.get("wheelbase_mm", 3000.0), 3000.0) / 1000.0
        cg_x_mm = _safe_float(geom.get("cg_x_mm", np.nan), np.nan)
        if np.isfinite(cg_x_mm):
            x_front = abs(cg_x_mm - _safe_float(params.get("front_axle_x_mm", 0.0), 0.0)) / 1000.0
            x_rear = abs(_safe_float(params.get("rear_axle_x_mm", -3000.0), -3000.0) - cg_x_mm) / 1000.0
            if (x_front + x_rear) > 1e-9:
                wb = x_front + x_rear
            else:
                x_front = (1.0 - wbf) * wb
                x_rear = wbf * wb
        else:
            x_front = (1.0 - wbf) * wb
            x_rear = wbf * wb

        rc_f = _safe_float(geom.get("front_rc_height_mm", geom.get("front_rc_z_mm", np.nan)), np.nan)
        rc_r = _safe_float(geom.get("rear_rc_height_mm", geom.get("rear_rc_z_mm", np.nan)), np.nan)
        lam = float(np.clip(x_front / max(wb, 1e-9), 0.0, 1.0))
        roll_axis_h_mm = (rc_f + lam * (rc_r - rc_f)) if (np.isfinite(rc_f) and np.isfinite(rc_r)) else 0.0
        pic_f = _safe_float(geom.get("pitch_ic_front_z_mm", np.nan), np.nan)
        pic_r = _safe_float(geom.get("pitch_ic_rear_z_mm", np.nan), np.nan)
        pitch_axis_h_mm = (pic_f + lam * (pic_r - pic_f)) if (np.isfinite(pic_f) and np.isfinite(pic_r)) else _safe_float(geom.get("pitch_z_mm", 0.0), 0.0)
        z_cg_mm = _safe_float(geom.get("h_cg_mm", geom.get("cg_z_mm", params.get("z_cg_ref_mm", 250.0))), _safe_float(params.get("z_cg_ref_mm", 250.0), 250.0))
        roll_arm_m = (z_cg_mm - roll_axis_h_mm) / 1000.0
        pitch_arm_m = (z_cg_mm - pitch_axis_h_mm) / 1000.0

        delta_f_front_n = fuel_weight_n * wbf + _safe_float(aero.get("front_aero_load_n", 0.0), 0.0)
        delta_f_rear_n = fuel_weight_n * (1.0 - wbf) + _safe_float(aero.get("rear_aero_load_n", 0.0), 0.0)
        front_heave_m = delta_f_front_n / max(1.0, _safe_float(stiffness.get("kz_front_npm", 1.0), 1.0))
        rear_heave_m = delta_f_rear_n / max(1.0, _safe_float(stiffness.get("kz_rear_npm", 1.0), 1.0))

        m_pitch_el = m_s_eff * ax * pitch_arm_m
        k_theta = _safe_float(stiffness.get("kz_front_npm", 0.0), 0.0) * (x_front ** 2) + _safe_float(stiffness.get("kz_rear_npm", 0.0), 0.0) * (x_rear ** 2)
        theta_pitch = m_pitch_el / max(1e-6, k_theta)
        dz_front_pitch_m = -x_front * theta_pitch
        dz_rear_pitch_m = +x_rear * theta_pitch

        kphi_f = max(1e-6, _safe_float(stiffness.get("kphi_front_npm", 1e-6), 1e-6))
        kphi_r = max(1e-6, _safe_float(stiffness.get("kphi_rear_npm", 1e-6), 1e-6))
        kphi_sum = max(1e-6, kphi_f + kphi_r)
        lambda_f = kphi_f / kphi_sum
        lambda_r = kphi_r / kphi_sum
        m_roll_el = m_s_eff * ay * roll_arm_m
        phi_front = (m_roll_el * lambda_f) / kphi_f
        phi_rear = (m_roll_el * lambda_r) / kphi_r
        roll_dz_f_m = 0.5 * (front_track_mm / 1000.0) * phi_front
        roll_dz_r_m = 0.5 * (rear_track_mm / 1000.0) * phi_rear

        new_dz = np.array(
            [
                front_heave_m + dz_front_pitch_m + roll_dz_f_m,
                front_heave_m + dz_front_pitch_m - roll_dz_f_m,
                rear_heave_m + dz_rear_pitch_m + roll_dz_r_m,
                rear_heave_m + dz_rear_pitch_m - roll_dz_r_m,
            ],
            dtype=float,
        )
        if local_branch_lock:
            # Keep the solver on the same local branch between neighbouring sweep points.
            lo = dz0 - max_delta_point_m
            hi = dz0 + max_delta_point_m
            new_dz = np.clip(new_dz, lo, hi)
        unclipped_dz = np.copy(new_dz)
        new_dz[0] = float(np.clip(new_dz[0], -front_max_droop, front_max_comp))
        new_dz[1] = float(np.clip(new_dz[1], -front_max_droop, front_max_comp))
        new_dz[2] = float(np.clip(new_dz[2], -rear_max_droop, rear_max_comp))
        new_dz[3] = float(np.clip(new_dz[3], -rear_max_droop, rear_max_comp))
        if np.max(np.abs(unclipped_dz - new_dz)) > 1e-12:
            saturation_count += 1
        dz_next = (1.0 - relax) * dz + relax * new_dz
        max_err = float(np.max(np.abs(dz_next - dz)))
        dz = dz_next
        last_aux = {
            "m_roll_el": float(m_roll_el),
            "m_pitch_el": float(m_pitch_el),
            "roll_arm_m": float(roll_arm_m),
            "pitch_arm_m": float(pitch_arm_m),
            "lambda_f": float(lambda_f),
            "lambda_r": float(lambda_r),
            "phi_front_rad": float(phi_front),
            "phi_rear_rad": float(phi_rear),
            "front_heave_m": float(front_heave_m + dz_front_pitch_m),
            "rear_heave_m": float(rear_heave_m + dz_rear_pitch_m),
            "theta_pitch_rad": float(theta_pitch),
            "aero": aero,
            "stiffness": stiffness,
            "front_track_mm": float(front_track_mm),
            "rear_track_mm": float(rear_track_mm),
            "roll_axis_h_mm": float(roll_axis_h_mm),
            "pitch_axis_h_mm": float(pitch_axis_h_mm),
            "z_cg_mm": float(z_cg_mm),
            "iterations": it,
            "travel_saturation_count": int(saturation_count),
        }
        if max_err < float(tol_m):
            converged = True
            solver_message = "converged"
            break

    if not converged and solver_message == "max iterations reached":
        solver_message = "not converged"
    return {
        "converged": bool(converged),
        "iterations": int(last_aux.get("iterations", int(max_iter))),
        "solver_message": solver_message,
        "dz": dz,
        "geometry": last_geom,
        "aux": last_aux,
    }


def _compute_platform_state_point(
    calibrated_json_data: Dict[str, Any],
    calibration_meta: Dict[str, Any],
    platform_params: Dict[str, Any],
    point_inputs: Dict[str, Any],
    initial_guess: Optional[Dict[str, float]] = None,
    geometry_state_cache: Optional[Dict[Tuple[float, float, float, float], Dict[str, float]]] = None,
) -> Dict[str, Any]:
    solve = _iterate_platform_state_solver(
        calibrated_json_data=calibrated_json_data,
        params=platform_params,
        point_inputs=point_inputs,
        initial_guess=initial_guess,
        geometry_state_cache=geometry_state_cache,
    )
    dz = np.asarray(solve.get("dz", np.zeros(4)), dtype=float)
    geom = dict(solve.get("geometry", {}))
    aux = dict(solve.get("aux", {}))
    aero = dict(aux.get("aero", {}))
    stiffness = dict(aux.get("stiffness", {}))
    g = 9.80665
    ax = _safe_float(point_inputs.get("ax_g", 0.0), 0.0) * g
    ay = _safe_float(point_inputs.get("ay_g", 0.0), 0.0) * g
    fuel_mass_kg = max(0.0, _safe_float(point_inputs.get("fuel_mass_kg", 0.0), 0.0))
    m_s_eff = _safe_float(platform_params.get("mass_sprung_base_kg", 0.0), 0.0) + fuel_mass_kg
    wbf = _safe_float(platform_params.get("r_weight_bal_f", 0.5), 0.5)
    wb_m = _safe_float(platform_params.get("wheelbase_mm", 3000.0), 3000.0) / 1000.0
    front_track_m = _safe_float(aux.get("front_track_mm", platform_params.get("base_front_track_mm", 1200.0)), 1200.0) / 1000.0
    rear_track_m = _safe_float(aux.get("rear_track_mm", platform_params.get("base_rear_track_mm", 1200.0)), 1200.0) / 1000.0
    m_roll_el = _safe_float(aux.get("m_roll_el", 0.0), 0.0)
    lambda_f = _safe_float(aux.get("lambda_f", 0.5), 0.5)
    lambda_r = _safe_float(aux.get("lambda_r", 0.5), 0.5)
    roll_axis_h_mm = _safe_float(aux.get("roll_axis_h_mm", np.nan), np.nan)
    pitch_axis_h_mm = _safe_float(aux.get("pitch_axis_h_mm", np.nan), np.nan)
    z_cg_mm = _safe_float(aux.get("z_cg_mm", platform_params.get("z_cg_ref_mm", 250.0)), _safe_float(platform_params.get("z_cg_ref_mm", 250.0), 250.0))
    roll_arm_m = _safe_float(aux.get("roll_arm_m", (z_cg_mm - roll_axis_h_mm) / 1000.0 if np.isfinite(roll_axis_h_mm) else 0.0), 0.0)
    pitch_arm_m = _safe_float(aux.get("pitch_arm_m", (z_cg_mm - pitch_axis_h_mm) / 1000.0 if np.isfinite(pitch_axis_h_mm) else 0.0), 0.0)

    fuel_weight_n = fuel_mass_kg * g
    front_static_axle_n = (m_s_eff * g * wbf) + (2.0 * _safe_float(platform_params.get("m_hub_f_kg", 0.0), 0.0) * g)
    rear_static_axle_n = (m_s_eff * g * (1.0 - wbf)) + (2.0 * _safe_float(platform_params.get("m_hub_r_kg", 0.0), 0.0) * g)
    front_left_static_n = front_static_axle_n * 0.5
    front_right_static_n = front_static_axle_n * 0.5
    rear_left_static_n = rear_static_axle_n * 0.5
    rear_right_static_n = rear_static_axle_n * 0.5

    long_transfer_n = m_s_eff * ax * pitch_arm_m / max(1e-9, wb_m)
    front_lat_transfer_n = (2.0 * m_roll_el * lambda_f) / max(1e-9, front_track_m)
    rear_lat_transfer_n = (2.0 * m_roll_el * lambda_r) / max(1e-9, rear_track_m)

    fz_fl_mech = front_left_static_n - 0.5 * long_transfer_n - 0.5 * front_lat_transfer_n
    fz_fr_mech = front_right_static_n - 0.5 * long_transfer_n + 0.5 * front_lat_transfer_n
    fz_rl_mech = rear_left_static_n + 0.5 * long_transfer_n - 0.5 * rear_lat_transfer_n
    fz_rr_mech = rear_right_static_n + 0.5 * long_transfer_n + 0.5 * rear_lat_transfer_n
    fz_fl = fz_fl_mech + 0.5 * _safe_float(aero.get("front_aero_load_n", 0.0), 0.0)
    fz_fr = fz_fr_mech + 0.5 * _safe_float(aero.get("front_aero_load_n", 0.0), 0.0)
    fz_rl = fz_rl_mech + 0.5 * _safe_float(aero.get("rear_aero_load_n", 0.0), 0.0)
    fz_rr = fz_rr_mech + 0.5 * _safe_float(aero.get("rear_aero_load_n", 0.0), 0.0)

    front_axle_load = fz_fl + fz_fr
    rear_axle_load = fz_rl + fz_rr
    lltd_total_front = (
        100.0 * abs(front_lat_transfer_n) / (abs(front_lat_transfer_n) + abs(rear_lat_transfer_n))
        if (abs(front_lat_transfer_n) + abs(rear_lat_transfer_n)) > 1e-9
        else float("nan")
    )

    hridef_m = _safe_float(geom.get("front_ride_height", 0.0), 0.0) / 1000.0
    hrider_m = _safe_float(geom.get("rear_ride_height", 0.0), 0.0) / 1000.0
    base_hridef_m = _safe_float(platform_params.get("base_hRideF_m", hridef_m), hridef_m)
    base_hrider_m = _safe_float(platform_params.get("base_hRideR_m", hrider_m), hrider_m)
    front_heave_m = _safe_float(aux.get("front_heave_m", 0.0), 0.0)
    rear_heave_m = _safe_float(aux.get("rear_heave_m", 0.0), 0.0)
    # body_heave_mm follows ride-height direction (upwards positive, downwards negative).
    body_heave_m = 0.5 * ((hridef_m - base_hridef_m) + (hrider_m - base_hrider_m))
    body_heave_compression_m = -body_heave_m
    front_platform_travel_m = (base_hridef_m - hridef_m)
    rear_platform_travel_m = (base_hrider_m - hrider_m)
    base_rake_m = base_hrider_m - base_hridef_m
    body_pitch_deg = math.degrees(math.atan2((hrider_m - hridef_m) - base_rake_m, max(1e-9, wb_m)))
    body_roll_deg = math.degrees(0.5 * (_safe_float(aux.get("phi_front_rad", 0.0), 0.0) + _safe_float(aux.get("phi_rear_rad", 0.0), 0.0)))

    rc_f_mm = _safe_float(geom.get("front_rc_height_mm", geom.get("front_rc_z_mm", np.nan)), np.nan)
    rc_r_mm = _safe_float(geom.get("rear_rc_height_mm", geom.get("rear_rc_z_mm", np.nan)), np.nan)
    pitch_axis_out_mm = _safe_float(aux.get("pitch_axis_h_mm", geom.get("pitch_z_mm", np.nan)), np.nan)

    row = {
        "ax_g": float(_safe_float(point_inputs.get("ax_g", 0.0), 0.0)),
        "ay_g": float(_safe_float(point_inputs.get("ay_g", 0.0), 0.0)),
        "speed_kph": float(_safe_float(point_inputs.get("speed_kph", 0.0), 0.0)),
        "drs_on": bool(point_inputs.get("drs_on", False)),
        "fuel_mass_kg": float(fuel_mass_kg),
        "dz_fl_mm": float(dz[0] * 1000.0),
        "dz_fr_mm": float(dz[1] * 1000.0),
        "dz_rl_mm": float(dz[2] * 1000.0),
        "dz_rr_mm": float(dz[3] * 1000.0),
        "front_platform_travel_mm": float(front_platform_travel_m * 1000.0),
        "rear_platform_travel_mm": float(rear_platform_travel_m * 1000.0),
        "body_heave_mm": float(body_heave_compression_m * 1000.0),
        "body_heave_compression_mm": float(body_heave_compression_m * 1000.0),
        "body_heave_chassis_up_mm": float(body_heave_m * 1000.0),
        "body_pitch_deg": float(body_pitch_deg),
        "body_roll_deg": float(body_roll_deg),
        "hRideF_m": float(hridef_m),
        "hRideR_m": float(hrider_m),
        "hRideF_mm": float(hridef_m * 1000.0),
        "hRideR_mm": float(hrider_m * 1000.0),
        "rake_mm": float((hrider_m - hridef_m) * 1000.0),
        "cg_x_mm": _json_scalar(_safe_float(geom.get("cg_x_mm", np.nan), np.nan)),
        "cg_y_mm": _json_scalar(_safe_float(geom.get("cg_y_mm", np.nan), np.nan)),
        "cg_z_mm": _json_scalar(_safe_float(geom.get("cg_z_mm", np.nan), np.nan)),
        "h_cg_mm": _json_scalar(_safe_float(geom.get("h_cg_mm", np.nan), np.nan)),
        "roll_center_front_mm": _json_scalar(rc_f_mm),
        "roll_center_rear_mm": _json_scalar(rc_r_mm),
        "roll_axis_height_at_cg_mm": _json_scalar(roll_axis_h_mm),
        "pitch_axis_height_at_cg_mm": _json_scalar(pitch_axis_out_mm),
        "fz_fl_n": float(fz_fl),
        "fz_fr_n": float(fz_fr),
        "fz_rl_n": float(fz_rl),
        "fz_rr_n": float(fz_rr),
        "front_axle_load_n": float(front_axle_load),
        "rear_axle_load_n": float(rear_axle_load),
        "front_aero_load_n": float(_safe_float(aero.get("front_aero_load_n", 0.0), 0.0)),
        "rear_aero_load_n": float(_safe_float(aero.get("rear_aero_load_n", 0.0), 0.0)),
        "total_aero_load_n": float(_safe_float(aero.get("total_aero_load_n", 0.0), 0.0)),
        "aero_balance_front_pct": _json_scalar(_safe_float(aero.get("aero_balance_front_pct", np.nan), np.nan)),
        "drag_force_n": float(_safe_float(aero.get("drag_force_n", 0.0), 0.0)),
        "elastic_roll_distribution_front_pct": float(lambda_f * 100.0),
        "lltd_total_front_pct": _json_scalar(lltd_total_front),
        "front_lateral_transfer_n": float(front_lat_transfer_n),
        "rear_lateral_transfer_n": float(rear_lat_transfer_n),
        "longitudinal_load_transfer_n": float(long_transfer_n),
        "calibration_applied": bool(calibration_meta.get("calibration_applied", False)),
        "calibration_model_id": str(calibration_meta.get("calibration_model_id", "")),
        "converged": bool(solve.get("converged", False)),
        "iterations": int(solve.get("iterations", 0)),
        "solver_message": str(solve.get("solver_message", "")),
        "arb_model_note": str(platform_params.get("arb_model_note", "")),
        "mr_mode": "linear_constant_json_v1",
        "damper_used": False,
        "fuel_cg_shift_enabled": False,
        "lltd_used_as_solver_input": False,
        "kz_front_eq_npm": float(_safe_float(stiffness.get("kz_front_npm", np.nan), np.nan)),
        "kz_rear_eq_npm": float(_safe_float(stiffness.get("kz_rear_npm", np.nan), np.nan)),
        "kphi_front_eq_npm": float(_safe_float(stiffness.get("kphi_front_npm", np.nan), np.nan)),
        "kphi_rear_eq_npm": float(_safe_float(stiffness.get("kphi_rear_npm", np.nan), np.nan)),
        "travel_limit_front_compression_mm": float(_safe_float(platform_params.get("front_max_compression_m", np.nan), np.nan) * 1000.0),
        "travel_limit_rear_compression_mm": float(_safe_float(platform_params.get("rear_max_compression_m", np.nan), np.nan) * 1000.0),
        "travel_limit_front_droop_mm": float(_safe_float(platform_params.get("front_max_droop_m", np.nan), np.nan) * 1000.0),
        "travel_limit_rear_droop_mm": float(_safe_float(platform_params.get("rear_max_droop_m", np.nan), np.nan) * 1000.0),
        "hRideF_min_mm": float(_safe_float(platform_params.get("min_hRideF_m", np.nan), np.nan) * 1000.0),
        "hRideF_max_mm": float(_safe_float(platform_params.get("max_hRideF_m", np.nan), np.nan) * 1000.0),
        "hRideR_min_mm": float(_safe_float(platform_params.get("min_hRideR_m", np.nan), np.nan) * 1000.0),
        "hRideR_max_mm": float(_safe_float(platform_params.get("max_hRideR_m", np.nan), np.nan) * 1000.0),
        "travel_limit_source": str(platform_params.get("travel_limit_source", "")),
        "travel_saturation_count": int(_safe_float(aux.get("travel_saturation_count", 0), 0)),
    }
    return {
        "row": _json_clean(row),
        "state_guess": {
            "dz_fl_m": float(dz[0]),
            "dz_fr_m": float(dz[1]),
            "dz_rl_m": float(dz[2]),
            "dz_rr_m": float(dz[3]),
        },
    }


def _platform_axis_values(
    variable: Optional[str],
    sweep_block: Dict[str, Any],
    fixed_inputs: Dict[str, Any],
    prefix: str,
) -> Tuple[List[float], bool]:
    if not variable:
        return [float(_safe_float(fixed_inputs.get(prefix, 0.0), 0.0))], False
    variable = str(variable).strip()
    if variable == "drs_on":
        return [0.0, 1.0], True
    vmin = _safe_float(sweep_block.get(f"{prefix}_min", fixed_inputs.get(variable, 0.0)), _safe_float(fixed_inputs.get(variable, 0.0), 0.0))
    vmax = _safe_float(sweep_block.get(f"{prefix}_max", fixed_inputs.get(variable, 0.0)), _safe_float(fixed_inputs.get(variable, 0.0), 0.0))
    vstep = _safe_float(sweep_block.get(f"{prefix}_step", 1.0), 1.0)
    return [float(v) for v in _dynamic_range(vmin, vmax, vstep)], False


def _platform_continuation_refine(
    calibrated_json_data: Dict[str, Any],
    calibration_meta: Dict[str, Any],
    platform_params: Dict[str, Any],
    prev_inputs: Dict[str, Any],
    target_inputs: Dict[str, Any],
    initial_guess: Optional[Dict[str, float]],
    geometry_state_cache: Optional[Dict[Tuple[float, float, float, float], Dict[str, float]]],
    n_substeps: int = 4,
) -> Dict[str, Any]:
    lock_mm = float(max(1.5, _safe_float(target_inputs.get("max_delta_per_point_mm", 2.5), 2.5)))
    if bool(prev_inputs.get("drs_on", False)) != bool(target_inputs.get("drs_on", False)):
        return _compute_platform_state_point(
            calibrated_json_data=calibrated_json_data,
            calibration_meta=calibration_meta,
            platform_params=platform_params,
            point_inputs={**target_inputs, "solver_mode": "fast", "local_branch_lock": True, "max_delta_per_point_mm": lock_mm},
            initial_guess=initial_guess,
            geometry_state_cache=geometry_state_cache,
        )
    numeric_keys = ["ax_g", "ay_g", "speed_kph", "fuel_mass_kg", "air_density"]
    state_guess = dict(initial_guess or {})
    last_point: Dict[str, Any] = {}
    for i in range(1, int(max(1, n_substeps)) + 1):
        t = float(i) / float(max(1, n_substeps))
        step_inputs = dict(target_inputs)
        for key in numeric_keys:
            a = _safe_float(prev_inputs.get(key, step_inputs.get(key, 0.0)), _safe_float(step_inputs.get(key, 0.0), 0.0))
            b = _safe_float(target_inputs.get(key, step_inputs.get(key, 0.0)), _safe_float(step_inputs.get(key, 0.0), 0.0))
            step_inputs[key] = float(a + t * (b - a))
        step_inputs["solver_mode"] = "fast"
        step_inputs["local_branch_lock"] = True
        step_inputs["max_delta_per_point_mm"] = lock_mm
        last_point = _compute_platform_state_point(
            calibrated_json_data=calibrated_json_data,
            calibration_meta=calibration_meta,
            platform_params=platform_params,
            point_inputs=step_inputs,
            initial_guess=state_guess if state_guess else None,
            geometry_state_cache=geometry_state_cache,
        )
        state_guess = dict(last_point.get("state_guess", {}))
    return last_point


_PLATFORM_WORKER_CTX: Dict[str, Any] = {}


def _platform_worker_init(
    calibrated_json_data: Dict[str, Any],
    calibration_meta: Dict[str, Any],
    platform_params: Dict[str, Any],
    fixed_inputs: Dict[str, Any],
    x_variable: str,
    y_variable: str,
    x_values: List[float],
    jump_threshold_mm: float,
) -> None:
    global _PLATFORM_WORKER_CTX
    _PLATFORM_WORKER_CTX = {
        "calibrated_json_data": calibrated_json_data,
        "calibration_meta": calibration_meta,
        "platform_params": platform_params,
        "fixed_inputs": fixed_inputs,
        "x_variable": x_variable,
        "y_variable": y_variable,
        "x_values": x_values,
        "jump_threshold_mm": jump_threshold_mm,
    }


def _platform_compute_row_sequence(
    calibrated_json_data: Dict[str, Any],
    calibration_meta: Dict[str, Any],
    platform_params: Dict[str, Any],
    fixed: Dict[str, Any],
    x_variable: str,
    y_variable: Optional[str],
    x_values: List[float],
    y_value: Optional[float],
    initial_guess: Optional[Dict[str, float]] = None,
    geometry_state_cache: Optional[Dict[Tuple[float, float, float, float], Dict[str, float]]] = None,
    jump_threshold_mm: float = 8.0,
) -> Dict[str, Any]:
    row_points: List[Dict[str, Any]] = []
    guess = dict(initial_guess or {})
    prev_row: Optional[Dict[str, Any]] = None
    prev_prev_row: Optional[Dict[str, Any]] = None
    prev_emitted_guess: Optional[Dict[str, float]] = dict(initial_guess or {}) if isinstance(initial_guess, dict) else None
    prev_prev_emitted_guess: Optional[Dict[str, float]] = None
    prev_point_inputs: Optional[Dict[str, Any]] = None
    converged_count = 0
    jump_refined = 0
    retry_nonconverged = bool(fixed.get("retry_on_nonconverged", False))
    speed_internal_step_kph = float(max(1.0, _safe_float(fixed.get("speed_internal_step_kph", 2.0), 2.0)))
    is_speed_1d = (y_variable is None) and (x_variable == "speed_kph")

    def _speed_continuity_score(
        row_now: Dict[str, Any],
        row_prev: Dict[str, Any],
        row_prev2: Optional[Dict[str, Any]],
    ) -> float:
        h_now = float(_safe_float(row_now.get("hRideF_mm", np.nan), np.nan))
        hr_now = float(_safe_float(row_now.get("hRideR_mm", np.nan), np.nan))
        h_prev = float(_safe_float(row_prev.get("hRideF_mm", np.nan), np.nan))
        hr_prev = float(_safe_float(row_prev.get("hRideR_mm", np.nan), np.nan))
        score = 0.0
        if np.isfinite(h_now) and np.isfinite(h_prev):
            dh = h_now - h_prev
            if dh > 0.10:
                score += 80.0 + 60.0 * dh
            else:
                score += abs(dh) * 0.6
        if np.isfinite(hr_now) and np.isfinite(hr_prev):
            dhr = hr_now - hr_prev
            if dhr > 0.10:
                score += 60.0 + 40.0 * dhr
            else:
                score += abs(dhr) * 0.4
        if isinstance(row_prev2, dict):
            h_prev2 = float(_safe_float(row_prev2.get("hRideF_mm", np.nan), np.nan))
            if np.isfinite(h_now) and np.isfinite(h_prev) and np.isfinite(h_prev2):
                slope_prev = h_prev - h_prev2
                slope_now = h_now - h_prev
                if (slope_prev < -0.05) and (slope_now > 0.05):
                    score += 60.0 + 40.0 * abs(slope_now - slope_prev)
                else:
                    score += 0.8 * abs(slope_now - slope_prev)
        if not bool(row_now.get("converged", False)):
            score += 40.0
        score += 0.05 * float(_safe_float(row_now.get("iterations", 0), 0.0))
        return float(score)

    schedule: List[Tuple[float, bool]] = []
    if (y_variable is None) and (x_variable == "speed_kph") and (len(x_values) > 1):
        x_sorted = [float(v) for v in x_values]
        schedule.append((x_sorted[0], True))
        for i in range(1, len(x_sorted)):
            xa = float(x_sorted[i - 1])
            xb = float(x_sorted[i])
            delta = xb - xa
            n_sub = int(max(1, math.ceil(abs(delta) / max(1e-9, speed_internal_step_kph))))
            for j in range(1, n_sub + 1):
                xv_sub = xa + (delta * float(j) / float(n_sub))
                emit = (j == n_sub)
                schedule.append((float(xv_sub), bool(emit)))
    else:
        schedule = [(float(v), True) for v in x_values]

    for xv, emit_row in schedule:
        guess_before_point = dict(guess or {})
        point_inputs = dict(fixed)
        if x_variable == "drs_on":
            point_inputs["drs_on"] = bool(float(xv) >= 0.5)
        else:
            point_inputs[x_variable] = float(xv)
        if y_variable:
            if y_variable == "drs_on":
                point_inputs["drs_on"] = bool(float(y_value or 0.0) >= 0.5)
            else:
                point_inputs[y_variable] = float(y_value or 0.0)
        point_inputs["local_branch_lock"] = bool(fixed.get("local_branch_lock", True))
        point_inputs["max_delta_per_point_mm"] = float(fixed.get("max_delta_per_point_mm", 10.0))

        point = _compute_platform_state_point(
            calibrated_json_data=calibrated_json_data,
            calibration_meta=calibration_meta,
            platform_params=platform_params,
            point_inputs=point_inputs,
            initial_guess=guess if guess else None,
            geometry_state_cache=geometry_state_cache,
        )
        row = dict(point.get("row", {}))
        should_refine = False
        if emit_row and (prev_row is not None):
            d_rear = abs(_safe_float(row.get("rear_platform_travel_mm", 0.0), 0.0) - _safe_float(prev_row.get("rear_platform_travel_mm", 0.0), 0.0))
            d_front = abs(_safe_float(row.get("front_platform_travel_mm", 0.0), 0.0) - _safe_float(prev_row.get("front_platform_travel_mm", 0.0), 0.0))
            d_body = abs(_safe_float(row.get("body_heave_mm", 0.0), 0.0) - _safe_float(prev_row.get("body_heave_mm", 0.0), 0.0))
            d_hf = abs(_safe_float(row.get("hRideF_mm", np.nan), np.nan) - _safe_float(prev_row.get("hRideF_mm", np.nan), np.nan))
            d_hr = abs(_safe_float(row.get("hRideR_mm", np.nan), np.nan) - _safe_float(prev_row.get("hRideR_mm", np.nan), np.nan))
            ride_jump_thr = max(0.8, 0.5 * float(jump_threshold_mm))
            if (
                (d_rear > jump_threshold_mm)
                or (d_front > jump_threshold_mm)
                or (d_body > jump_threshold_mm)
                or (np.isfinite(d_hf) and d_hf > ride_jump_thr)
                or (np.isfinite(d_hr) and d_hr > ride_jump_thr)
            ):
                should_refine = True

        enable_jump_refine = bool(fixed.get("enable_jump_refine", False))
        need_retry = (retry_nonconverged and (not bool(row.get("converged", False)))) or (enable_jump_refine and should_refine and emit_row)
        if need_retry:
            if isinstance(prev_point_inputs, dict):
                substeps = 3 if (enable_jump_refine and should_refine) else 1
                refined_point = _platform_continuation_refine(
                    calibrated_json_data=calibrated_json_data,
                    calibration_meta=calibration_meta,
                    platform_params=platform_params,
                    prev_inputs=prev_point_inputs,
                    target_inputs=point_inputs,
                    initial_guess=guess if guess else None,
                    geometry_state_cache=geometry_state_cache,
                    n_substeps=substeps,
                )
                point = refined_point
                row = dict(point.get("row", {}))
                if enable_jump_refine and should_refine:
                    jump_refined += 1

        # Targeted low-cost rescue for residual 1D speed dents:
        # only re-solve anomalous emitted points with a predictive seed.
        if emit_row and is_speed_1d and isinstance(prev_row, dict):
            hf_now = float(_safe_float(row.get("hRideF_mm", np.nan), np.nan))
            hf_prev = float(_safe_float(prev_row.get("hRideF_mm", np.nan), np.nan))
            hr_now = float(_safe_float(row.get("hRideR_mm", np.nan), np.nan))
            hr_prev = float(_safe_float(prev_row.get("hRideR_mm", np.nan), np.nan))
            anomaly = (
                (np.isfinite(hf_now) and np.isfinite(hf_prev) and ((hf_now - hf_prev) > 0.10))
                or (np.isfinite(hr_now) and np.isfinite(hr_prev) and ((hr_now - hr_prev) > 0.10))
            )
            if anomaly:
                strict_inputs = dict(point_inputs)
                strict_inputs["solver_mode"] = "fast"
                strict_inputs["solver_max_iter"] = int(max(12, _safe_float(strict_inputs.get("solver_max_iter", 10), 10)))
                strict_inputs["solver_relax"] = 0.48
                strict_inputs["geometry_refresh_stride"] = 1
                strict_inputs["local_branch_lock"] = True
                strict_inputs["max_delta_per_point_mm"] = float(min(_safe_float(strict_inputs.get("max_delta_per_point_mm", 2.5), 2.5), 0.6))

                predictive_guess: Dict[str, float] = dict(guess_before_point or {})
                if isinstance(prev_emitted_guess, dict) and prev_emitted_guess:
                    predictive_guess = dict(prev_emitted_guess)
                if isinstance(prev_emitted_guess, dict) and isinstance(prev_prev_emitted_guess, dict):
                    alpha = 0.60
                    for k in ("dz_fl_m", "dz_fr_m", "dz_rl_m", "dz_rr_m"):
                        a = _safe_float(prev_prev_emitted_guess.get(k, 0.0), 0.0)
                        b = _safe_float(prev_emitted_guess.get(k, 0.0), 0.0)
                        p = b + alpha * (b - a)
                        band = 0.0015  # +/-1.5 mm around previous emitted solution
                        predictive_guess[k] = float(np.clip(p, b - band, b + band))

                rescue_point = _compute_platform_state_point(
                    calibrated_json_data=calibrated_json_data,
                    calibration_meta=calibration_meta,
                    platform_params=platform_params,
                    point_inputs=strict_inputs,
                    initial_guess=predictive_guess if predictive_guess else None,
                    geometry_state_cache=geometry_state_cache,
                )
                rescue_row = dict(rescue_point.get("row", {}))
                base_score = _speed_continuity_score(row, prev_row, prev_prev_row)
                rescue_score = _speed_continuity_score(rescue_row, prev_row, prev_prev_row)
                if rescue_score + 1e-6 < base_score:
                    point = rescue_point
                    row = rescue_row
                    jump_refined += 1

        guess = dict(point.get("state_guess", {}))
        if emit_row:
            row["x_value"] = float(xv)
            row["y_value"] = float(y_value) if y_variable is not None else None
            row_points.append(row)
            prev_prev_row = dict(prev_row) if isinstance(prev_row, dict) else None
            prev_row = dict(row)
            prev_prev_emitted_guess = dict(prev_emitted_guess) if isinstance(prev_emitted_guess, dict) else None
            prev_emitted_guess = dict(guess)
            prev_point_inputs = dict(point_inputs)
            if bool(row.get("converged", False)):
                converged_count += 1

    return {
        "rows": row_points,
        "last_guess": guess,
        "converged_count": int(converged_count),
        "jump_refined": int(jump_refined),
    }


def _platform_worker_run_row(task: Tuple[int, float]) -> Dict[str, Any]:
    row_index, y_value = int(task[0]), float(task[1])
    ctx = _PLATFORM_WORKER_CTX
    geometry_state_cache: Dict[Tuple[float, float, float, float], Dict[str, float]] = {}
    out = _platform_compute_row_sequence(
        calibrated_json_data=ctx["calibrated_json_data"],
        calibration_meta=ctx["calibration_meta"],
        platform_params=ctx["platform_params"],
        fixed=ctx["fixed_inputs"],
        x_variable=ctx["x_variable"],
        y_variable=ctx["y_variable"],
        x_values=ctx["x_values"],
        y_value=y_value,
        initial_guess=None,
        geometry_state_cache=geometry_state_cache,
        jump_threshold_mm=float(ctx.get("jump_threshold_mm", 8.0)),
    )
    return {
        "row_index": row_index,
        "rows": out["rows"],
        "converged_count": out["converged_count"],
        "jump_refined": out["jump_refined"],
    }


def _apply_speed_physics_guard(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    if not rows:
        return rows, 0
    ordered = sorted(rows, key=lambda r: float(_safe_float(r.get("speed_kph", 0.0), 0.0)))
    correction_count = 0

    def _guard_metric(key: str, increasing: bool, tol: float, min_step: float) -> None:
        nonlocal correction_count
        if len(ordered) < 2:
            return
        prev = float(_safe_float(ordered[0].get(key, 0.0), 0.0))
        for i in range(1, len(ordered)):
            curr = float(_safe_float(ordered[i].get(key, prev), prev))
            if increasing:
                if curr < (prev - tol):
                    curr = prev + min_step
                    ordered[i][key] = curr
                    correction_count += 1
            else:
                if curr > (prev + tol):
                    curr = prev - min_step
                    ordered[i][key] = curr
                    correction_count += 1
            prev = curr

    # Speed sweep expectations (steady-state): ride heights usually decrease with speed,
    # platform travel/heave compression usually increase with speed.
    for r in ordered:
        # Guard always works in mm so the plotted fields stay consistent.
        if "hRideF_mm" not in r:
            r["hRideF_mm"] = float(_safe_float(r.get("hRideF_m", 0.0), 0.0) * 1000.0)
        if "hRideR_mm" not in r:
            r["hRideR_mm"] = float(_safe_float(r.get("hRideR_m", 0.0), 0.0) * 1000.0)

    _guard_metric("hRideF_mm", increasing=False, tol=0.01, min_step=0.0)
    _guard_metric("hRideR_mm", increasing=False, tol=0.01, min_step=0.0)

    # Rebuild dependent platform metrics from smoothed ride heights to keep physical consistency.
    base_front_candidates = [
        float(_safe_float(r.get("hRideF_mm", np.nan), np.nan) + _safe_float(r.get("front_platform_travel_mm", np.nan), np.nan))
        for r in ordered
    ]
    base_rear_candidates = [
        float(_safe_float(r.get("hRideR_mm", np.nan), np.nan) + _safe_float(r.get("rear_platform_travel_mm", np.nan), np.nan))
        for r in ordered
    ]
    base_front_med = float(np.nanmedian(base_front_candidates)) if base_front_candidates else float("nan")
    base_rear_med = float(np.nanmedian(base_rear_candidates)) if base_rear_candidates else float("nan")
    base_front_mm = base_front_med if np.isfinite(base_front_med) else float(_safe_float(ordered[0].get("hRideF_mm", 0.0), 0.0))
    base_rear_mm = base_rear_med if np.isfinite(base_rear_med) else float(_safe_float(ordered[0].get("hRideR_mm", 0.0), 0.0))

    for r in ordered:
        hf_mm = float(_safe_float(r.get("hRideF_mm", 0.0), 0.0))
        hr_mm = float(_safe_float(r.get("hRideR_mm", 0.0), 0.0))
        front_travel_mm = float(base_front_mm - hf_mm)
        rear_travel_mm = float(base_rear_mm - hr_mm)
        body_heave_comp_mm = 0.5 * (front_travel_mm + rear_travel_mm)

        r["hRideF_mm"] = hf_mm
        r["hRideR_mm"] = hr_mm
        r["hRideF_m"] = hf_mm / 1000.0
        r["hRideR_m"] = hr_mm / 1000.0
        r["front_platform_travel_mm"] = front_travel_mm
        r["rear_platform_travel_mm"] = rear_travel_mm
        r["body_heave_compression_mm"] = body_heave_comp_mm
        # Keep legacy field consistent.
        r["body_heave_mm"] = body_heave_comp_mm
        r["body_heave_chassis_up_mm"] = -body_heave_comp_mm
        r["rake_mm"] = hr_mm - hf_mm

    _guard_metric("front_platform_travel_mm", increasing=True, tol=0.01, min_step=0.0)
    _guard_metric("rear_platform_travel_mm", increasing=True, tol=0.01, min_step=0.0)
    _guard_metric("body_heave_compression_mm", increasing=True, tol=0.01, min_step=0.0)

    index_map = {
        (float(_safe_float(r.get("x_value", 0.0), 0.0)), float(_safe_float(r.get("y_value", 0.0), 0.0) if r.get("y_value") is not None else 0.0)): r
        for r in ordered
    }
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = (float(_safe_float(r.get("x_value", 0.0), 0.0)), float(_safe_float(r.get("y_value", 0.0), 0.0) if r.get("y_value") is not None else 0.0))
        out.append(dict(index_map.get(key, r)))
    return out, int(correction_count)


def _platform_refine_speed_local_windows(
    calibrated_json_data: Dict[str, Any],
    calibration_meta: Dict[str, Any],
    platform_params: Dict[str, Any],
    fixed: Dict[str, Any],
    rows: List[Dict[str, Any]],
    geometry_state_cache: Optional[Dict[Tuple[float, float, float, float], Dict[str, float]]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    if len(rows) < 5:
        return rows, 0
    ordered = sorted(rows, key=lambda r: float(_safe_float(r.get("speed_kph", r.get("x_value", 0.0)), 0.0)))
    hf = np.asarray([_safe_float(r.get("hRideF_mm", np.nan), np.nan) for r in ordered], dtype=float)
    hr = np.asarray([_safe_float(r.get("hRideR_mm", np.nan), np.nan) for r in ordered], dtype=float)
    if np.any(~np.isfinite(hf)):
        return rows, 0

    candidates: List[Tuple[float, int]] = []
    for i in range(1, len(ordered) - 1):
        if not np.isfinite(hf[i - 1]) or not np.isfinite(hf[i]) or not np.isfinite(hf[i + 1]):
            continue
        s1f = hf[i] - hf[i - 1]
        s2f = hf[i + 1] - hf[i]
        sev_f = max(0.0, s1f) + max(0.0, s1f - s2f)
        # Also penalize abrupt downward cliffs and curvature kinks.
        sev_f += max(0.0, abs(s2f - s1f) - 0.35)
        sev = sev_f
        if np.isfinite(hr[i - 1]) and np.isfinite(hr[i]) and np.isfinite(hr[i + 1]):
            s1r = hr[i] - hr[i - 1]
            s2r = hr[i + 1] - hr[i]
            sev += 0.8 * (max(0.0, s1r) + max(0.0, s1r - s2r))
            sev += 0.8 * max(0.0, abs(s2r - s1r) - 0.45)
        is_bump_f = (s1f > 0.05 and s2f < -0.03)
        is_cliff_f = (s1f < -0.8 and s2f > -0.3)
        is_cliff_r = False
        if np.isfinite(hr[i - 1]) and np.isfinite(hr[i]) and np.isfinite(hr[i + 1]):
            s1r = hr[i] - hr[i - 1]
            s2r = hr[i + 1] - hr[i]
            is_cliff_r = (s1r < -1.5 and s2r > -0.6)
        if (is_bump_f or is_cliff_f or is_cliff_r) and (sev > 0.12):
            candidates.append((float(sev), i))
    if not candidates:
        return rows, 0
    candidates.sort(reverse=True, key=lambda x: x[0])
    picked = [idx for _, idx in candidates[:3]]

    windows: List[Tuple[int, int]] = []
    for idx in sorted(picked):
        a = max(0, idx - 1)
        b = min(len(ordered) - 1, idx + 1)
        if windows and a <= windows[-1][1] + 1:
            windows[-1] = (windows[-1][0], max(windows[-1][1], b))
        else:
            windows.append((a, b))

    def _row_guess(r: Dict[str, Any]) -> Dict[str, float]:
        return {
            "dz_fl_m": float(_safe_float(r.get("dz_fl_mm", 0.0), 0.0) / 1000.0),
            "dz_fr_m": float(_safe_float(r.get("dz_fr_mm", 0.0), 0.0) / 1000.0),
            "dz_rl_m": float(_safe_float(r.get("dz_rl_mm", 0.0), 0.0) / 1000.0),
            "dz_rr_m": float(_safe_float(r.get("dz_rr_mm", 0.0), 0.0) / 1000.0),
        }

    def _window_score(rr: List[Dict[str, Any]], ia: int, ib: int) -> float:
        lo = max(0, ia - 1)
        hi = min(len(rr) - 1, ib + 1)
        vals = np.asarray([_safe_float(rr[k].get("hRideF_mm", np.nan), np.nan) for k in range(lo, hi + 1)], dtype=float)
        if np.any(~np.isfinite(vals)):
            return 1e12
        score = 0.0
        for j in range(1, len(vals)):
            dv = vals[j] - vals[j - 1]
            if dv > 0.0:
                score += 80.0 * dv
            score += 0.15 * abs(dv)
        if len(vals) >= 3:
            for j in range(1, len(vals) - 1):
                dd = vals[j + 1] - (2.0 * vals[j]) + vals[j - 1]
                score += 0.9 * abs(dd)
        for k in range(lo, hi + 1):
            if not bool(rr[k].get("converged", False)):
                score += 25.0
        return float(score)

    refined_count = 0
    for a, b in windows:
        baseline = [dict(r) for r in ordered]
        best_score = _window_score(baseline, a, b)
        seed = _row_guess(ordered[a - 1]) if a > 0 else _row_guess(ordered[a])
        strict_common = dict(fixed)
        strict_common["solver_mode"] = "fast"
        strict_common["solver_max_iter"] = int(max(12, _safe_float(strict_common.get("solver_max_iter", 10), 10)))
        strict_common["solver_relax"] = 0.46
        strict_common["geometry_refresh_stride"] = 1
        strict_common["local_branch_lock"] = True
        strict_common["max_delta_per_point_mm"] = float(min(_safe_float(strict_common.get("max_delta_per_point_mm", 1.0), 1.0), 0.5))

        candidate = [dict(r) for r in ordered]
        local_guess = dict(seed)
        for k in range(a, b + 1):
            inp = dict(strict_common)
            inp["speed_kph"] = float(_safe_float(ordered[k].get("speed_kph", ordered[k].get("x_value", 0.0)), 0.0))
            point = _compute_platform_state_point(
                calibrated_json_data=calibrated_json_data,
                calibration_meta=calibration_meta,
                platform_params=platform_params,
                point_inputs=inp,
                initial_guess=local_guess if local_guess else None,
                geometry_state_cache=geometry_state_cache,
            )
            rr = dict(point.get("row", {}))
            rr["x_value"] = float(_safe_float(ordered[k].get("x_value", inp["speed_kph"]), inp["speed_kph"]))
            rr["y_value"] = ordered[k].get("y_value", None)
            candidate[k] = rr
            local_guess = dict(point.get("state_guess", {}))

        cand_score = _window_score(candidate, a, b)
        if cand_score + 1e-6 < best_score:
            for k in range(a, b + 1):
                ordered[k] = candidate[k]
            refined_count += int(b - a + 1)

    key_map = {
        (float(_safe_float(r.get("x_value", 0.0), 0.0)), float(_safe_float(r.get("y_value", 0.0), 0.0) if r.get("y_value") is not None else 0.0)): r
        for r in ordered
    }
    out = []
    for r in rows:
        key = (float(_safe_float(r.get("x_value", 0.0), 0.0)), float(_safe_float(r.get("y_value", 0.0), 0.0) if r.get("y_value") is not None else 0.0))
        out.append(dict(key_map.get(key, r)))
    return out, int(refined_count)


def _platform_soft_physics_limiter_speed(
    rows: List[Dict[str, Any]],
    up_tolerance_mm: float = 0.03,
    max_point_correction_mm: float = 0.35,
) -> Tuple[List[Dict[str, Any]], int]:
    if len(rows) < 3:
        return rows, 0
    ordered = sorted(rows, key=lambda r: float(_safe_float(r.get("speed_kph", r.get("x_value", 0.0)), 0.0)))
    corrections = 0

    def _limit_key(key: str) -> None:
        nonlocal corrections
        speeds = np.asarray([_safe_float(r.get("speed_kph", r.get("x_value", np.nan)), np.nan) for r in ordered], dtype=float)
        vals = np.asarray([_safe_float(r.get(key, np.nan), np.nan) for r in ordered], dtype=float)
        if np.any(~np.isfinite(vals)) or np.any(~np.isfinite(speeds)):
            return
        diffs = np.diff(vals)
        dx = np.diff(speeds)
        finite = np.isfinite(diffs) & np.isfinite(dx) & (dx > 1e-9)
        if not np.any(finite):
            return
        dx_med = float(np.median(dx[finite]))
        abs_diffs = np.abs(diffs[finite])
        typical_step = float(np.median(abs_diffs)) if abs_diffs.size > 0 else 0.4
        if not np.isfinite(typical_step) or typical_step <= 1e-9:
            typical_step = 0.4
        max_drop_base = max(1.2, 2.3 * typical_step)

        prev_env = float(_safe_float(ordered[0].get(key, np.nan), np.nan))
        prev_val = prev_env
        if not np.isfinite(prev_env):
            return
        for i in range(1, len(ordered)):
            v = float(_safe_float(ordered[i].get(key, prev_env), prev_env))
            if not np.isfinite(v):
                continue
            v_allowed = prev_env + float(up_tolerance_mm)
            if v > v_allowed:
                v_new = max(v - float(max_point_correction_mm), v_allowed)
                ordered[i][key] = float(v_new)
                v = float(v_new)
                corrections += 1
            dx_i = float(_safe_float(ordered[i].get("speed_kph", ordered[i].get("x_value", dx_med)), dx_med) - _safe_float(ordered[i - 1].get("speed_kph", ordered[i - 1].get("x_value", 0.0)), 0.0))
            dx_scale = float(np.clip(dx_i / max(1e-9, dx_med), 0.5, 2.5))
            max_drop = max_drop_base * dx_scale
            if v < (prev_val - max_drop):
                v_new = prev_val - max_drop
                ordered[i][key] = float(v_new)
                v = float(v_new)
                corrections += 1
            prev_val = v
            prev_env = min(prev_env, v)

    _limit_key("hRideF_mm")
    _limit_key("hRideR_mm")

    base_front_candidates = [
        float(_safe_float(r.get("hRideF_mm", np.nan), np.nan) + _safe_float(r.get("front_platform_travel_mm", np.nan), np.nan))
        for r in ordered
    ]
    base_rear_candidates = [
        float(_safe_float(r.get("hRideR_mm", np.nan), np.nan) + _safe_float(r.get("rear_platform_travel_mm", np.nan), np.nan))
        for r in ordered
    ]
    bf = float(np.nanmedian(base_front_candidates)) if base_front_candidates else float("nan")
    br = float(np.nanmedian(base_rear_candidates)) if base_rear_candidates else float("nan")
    if not np.isfinite(bf):
        bf = float(_safe_float(ordered[0].get("hRideF_mm", 0.0), 0.0))
    if not np.isfinite(br):
        br = float(_safe_float(ordered[0].get("hRideR_mm", 0.0), 0.0))

    for r in ordered:
        hf_mm = float(_safe_float(r.get("hRideF_mm", 0.0), 0.0))
        hr_mm = float(_safe_float(r.get("hRideR_mm", 0.0), 0.0))
        front_travel_mm = float(bf - hf_mm)
        rear_travel_mm = float(br - hr_mm)
        body_heave_comp_mm = 0.5 * (front_travel_mm + rear_travel_mm)
        r["hRideF_m"] = hf_mm / 1000.0
        r["hRideR_m"] = hr_mm / 1000.0
        r["front_platform_travel_mm"] = front_travel_mm
        r["rear_platform_travel_mm"] = rear_travel_mm
        r["body_heave_compression_mm"] = body_heave_comp_mm
        r["body_heave_mm"] = body_heave_comp_mm
        r["body_heave_chassis_up_mm"] = -body_heave_comp_mm
        r["rake_mm"] = hr_mm - hf_mm

    key_map = {
        (float(_safe_float(r.get("x_value", 0.0), 0.0)), float(_safe_float(r.get("y_value", 0.0), 0.0) if r.get("y_value") is not None else 0.0)): r
        for r in ordered
    }
    out = []
    for r in rows:
        key = (float(_safe_float(r.get("x_value", 0.0), 0.0)), float(_safe_float(r.get("y_value", 0.0), 0.0) if r.get("y_value") is not None else 0.0))
        out.append(dict(key_map.get(key, r)))
    return out, int(corrections)


def _platform_apply_hard_ride_height_bounds(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    if not rows:
        return rows, 0
    out: List[Dict[str, Any]] = []
    corrected = 0
    for r in rows:
        rr = dict(r)
        hf_mm = float(_safe_float(rr.get("hRideF_mm", _safe_float(rr.get("hRideF_m", 0.0), 0.0) * 1000.0), 0.0))
        hr_mm = float(_safe_float(rr.get("hRideR_mm", _safe_float(rr.get("hRideR_m", 0.0), 0.0) * 1000.0), 0.0))
        hf_min = float(max(0.0, _safe_float(rr.get("hRideF_min_mm", 0.0), 0.0)))
        hr_min = float(max(0.0, _safe_float(rr.get("hRideR_min_mm", 0.0), 0.0)))
        hf_max_raw = _safe_float(rr.get("hRideF_max_mm", np.nan), np.nan)
        hr_max_raw = _safe_float(rr.get("hRideR_max_mm", np.nan), np.nan)
        hf_max = float(hf_max_raw) if np.isfinite(hf_max_raw) else float(max(hf_mm, hf_min))
        hr_max = float(hr_max_raw) if np.isfinite(hr_max_raw) else float(max(hr_mm, hr_min))

        hf_clip = float(np.clip(hf_mm, hf_min, hf_max))
        hr_clip = float(np.clip(hr_mm, hr_min, hr_max))
        if (abs(hf_clip - hf_mm) > 1e-9) or (abs(hr_clip - hr_mm) > 1e-9):
            corrected += 1
        rr["hRideF_mm"] = hf_clip
        rr["hRideR_mm"] = hr_clip
        rr["hRideF_m"] = hf_clip / 1000.0
        rr["hRideR_m"] = hr_clip / 1000.0
        base_front = _safe_float(rr.get("hRideF_mm", np.nan), np.nan) + _safe_float(rr.get("front_platform_travel_mm", np.nan), np.nan)
        base_rear = _safe_float(rr.get("hRideR_mm", np.nan), np.nan) + _safe_float(rr.get("rear_platform_travel_mm", np.nan), np.nan)
        if np.isfinite(base_front) and np.isfinite(base_rear):
            front_travel = float(base_front - hf_clip)
            rear_travel = float(base_rear - hr_clip)
            body_heave_comp = 0.5 * (front_travel + rear_travel)
            rr["front_platform_travel_mm"] = front_travel
            rr["rear_platform_travel_mm"] = rear_travel
            rr["body_heave_compression_mm"] = body_heave_comp
            rr["body_heave_mm"] = body_heave_comp
            rr["body_heave_chassis_up_mm"] = -body_heave_comp
            rr["rake_mm"] = hr_clip - hf_clip
        out.append(rr)
    return out, int(corrected)


def _compute_platform_explorer_grid(
    calibrated_json_data: Dict[str, Any],
    calibration_meta: Dict[str, Any],
    platform_params: Dict[str, Any],
    body: Dict[str, Any],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    geometry_state_cache: Dict[Tuple[float, float, float, float], Dict[str, float]] = {}
    fixed_inputs = dict(body.get("fixed_inputs") or {})
    sweep = dict(body.get("sweep") or {})
    fixed = {
        "ax_g": float(_safe_float(fixed_inputs.get("ax_g", 0.0), 0.0)),
        "ay_g": float(_safe_float(fixed_inputs.get("ay_g", 0.0), 0.0)),
        "speed_kph": float(_safe_float(fixed_inputs.get("speed_kph", 180.0), 180.0)),
        "fuel_mass_kg": float(max(0.0, _safe_float(fixed_inputs.get("fuel_mass_kg", 0.0), 0.0))),
        "drs_on": bool(body.get("drs_on", fixed_inputs.get("drs_on", False))),
        "air_density": float(_safe_float(fixed_inputs.get("air_density", body.get("air_density", 1.225)), 1.225)),
        # Production defaults (always on): prioritize speed while forcing continuity safeguards.
        "solver_mode": "turbo",
        "local_branch_lock": True,
        "max_delta_per_point_mm": float(max(1.5, _safe_float(body.get("max_delta_per_point_mm", 2.5), 2.5))),
        "enable_jump_refine": True,
        "retry_on_nonconverged": False,
    }
    x_variable = str(sweep.get("x_variable", "")).strip() or "speed_kph"
    y_variable_raw = sweep.get("y_variable")
    y_variable = str(y_variable_raw).strip() if y_variable_raw not in (None, "", "none", "null") else None
    if x_variable not in {"ax_g", "ay_g", "speed_kph", "fuel_mass_kg", "drs_on"}:
        raise ValueError("x_variable must be one of: ax_g, ay_g, speed_kph, fuel_mass_kg, drs_on")
    if y_variable and y_variable not in {"ax_g", "ay_g", "speed_kph", "fuel_mass_kg", "drs_on"}:
        raise ValueError("y_variable must be one of: ax_g, ay_g, speed_kph, fuel_mass_kg, drs_on")
    x_values, _ = _platform_axis_values(x_variable, sweep, fixed, "x")
    y_values, _ = _platform_axis_values(y_variable, sweep, fixed, "y") if y_variable else ([0.0], False)
    is_speed_1d = (not y_variable) and (x_variable == "speed_kph")
    if is_speed_1d:
        # Adaptive balance: keep runtime low while preserving continuity for speed sweeps.
        fixed["solver_mode"] = "turbo"
        fixed["retry_on_nonconverged"] = False
        fixed["solver_max_iter"] = 10
        fixed["solver_relax"] = 0.55
        fixed["geometry_refresh_stride"] = 1
        if len(x_values) > 1:
            x_step_nom = float(np.median(np.abs(np.diff(np.asarray(x_values, dtype=float)))))
        else:
            x_step_nom = 5.0
        fixed["speed_internal_step_kph"] = float(np.clip(0.45 * x_step_nom, 1.5, 4.0))
        # Strong continuity lock for speed sweeps to avoid non-physical branch hopping.
        branch_mm = float(np.clip(0.08 * x_step_nom, 0.4, 0.8))
        fixed["max_delta_per_point_mm"] = float(min(_safe_float(fixed.get("max_delta_per_point_mm", 2.5), 2.5), branch_mm))

    rows: List[Dict[str, Any]] = []
    point_count = len(x_values) * len(y_values)
    if point_count > 1200:
        raise ValueError("Sweep grid too large. Reduce ranges/step (max 1200 points).")

    cache_key = f"{calibration_meta.get('calibration_model_id', 'default')}:{calibration_meta.get('json_signature', '')}"
    last_state_cache = _state.get("platform_last_state")
    if not isinstance(last_state_cache, dict):
        last_state_cache = {}
        _state["platform_last_state"] = last_state_cache
    guess = dict(last_state_cache.get(cache_key) or {})
    converged_count = 0
    jump_threshold_mm = float(max(1.0, _safe_float(body.get("jump_threshold_mm", 8.0), 8.0)))
    if is_speed_1d:
        jump_threshold_mm = float(min(jump_threshold_mm, 2.0))
    jump_refined_points = 0
    workers_req = int(max(0, _safe_float(body.get("workers", 0), 0)))
    cpu_count = max(1, int(os.cpu_count() or 1))
    workers_auto = max(1, min(cpu_count, 16))
    workers = workers_req if workers_req > 0 else workers_auto
    # Always parallelize 2D sweeps when possible.
    enable_parallel = bool(y_variable) and (len(y_values) > 1) and (workers > 1) and (point_count >= 40)

    if enable_parallel:
        jobs = [(idx, float(yv)) for idx, yv in enumerate(y_values)]
        results_by_idx: Dict[int, Dict[str, Any]] = {}
        with cf.ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_platform_worker_init,
            initargs=(
                calibrated_json_data,
                calibration_meta,
                platform_params,
                fixed,
                x_variable,
                str(y_variable),
                [float(v) for v in x_values],
                float(jump_threshold_mm),
            ),
        ) as ex:
            futures = [ex.submit(_platform_worker_run_row, job) for job in jobs]
            for fut in cf.as_completed(futures):
                got = fut.result()
                results_by_idx[int(got["row_index"])] = got
        for idx in range(len(y_values)):
            got = results_by_idx.get(idx, {})
            rows.extend(list(got.get("rows", [])))
            converged_count += int(got.get("converged_count", 0))
            jump_refined_points += int(got.get("jump_refined", 0))
    else:
        for yi, yv in enumerate(y_values):
            x_iter = x_values if (yi % 2 == 0) else list(reversed(x_values))
            seq = _platform_compute_row_sequence(
                calibrated_json_data=calibrated_json_data,
                calibration_meta=calibration_meta,
                platform_params=platform_params,
                fixed=fixed,
                x_variable=x_variable,
                y_variable=y_variable,
                x_values=[float(v) for v in x_iter],
                y_value=float(yv),
                initial_guess=guess if isinstance(guess, dict) and guess else None,
                geometry_state_cache=geometry_state_cache,
                jump_threshold_mm=jump_threshold_mm,
            )
            rows.extend(list(seq.get("rows", [])))
            guess = dict(seq.get("last_guess", {}))
            converged_count += int(seq.get("converged_count", 0))
            jump_refined_points += int(seq.get("jump_refined", 0))
    if guess:
        last_state_cache[cache_key] = guess

    speed_guard_corrections = 0
    local_window_refined_points = 0
    soft_limiter_corrections = 0
    if is_speed_1d and rows and (not enable_parallel):
        rows, local_window_refined_points = _platform_refine_speed_local_windows(
            calibrated_json_data=calibrated_json_data,
            calibration_meta=calibration_meta,
            platform_params=platform_params,
            fixed=fixed,
            rows=rows,
            geometry_state_cache=geometry_state_cache,
        )
        rows, soft_limiter_corrections = _platform_soft_physics_limiter_speed(rows)
    hard_bound_corrections = 0
    rows, hard_bound_corrections = _platform_apply_hard_ride_height_bounds(rows)

    metric_candidates = [
        "front_platform_travel_mm",
        "rear_platform_travel_mm",
        "body_heave_mm",
        "body_heave_compression_mm",
        "body_pitch_deg",
        "body_roll_deg",
        "hRideF_mm",
        "hRideR_mm",
        "aero_balance_front_pct",
        "drag_force_n",
        "front_aero_load_n",
        "rear_aero_load_n",
        "cg_z_mm",
        "roll_axis_height_at_cg_mm",
        "elastic_roll_distribution_front_pct",
        "lltd_total_front_pct",
    ]
    return {
        "rows": _json_clean(rows),
        "sweep": {
            "x_variable": x_variable,
            "x_values": [float(v) for v in x_values],
            "y_variable": y_variable,
            "y_values": [float(v) for v in y_values] if y_variable else [],
            "is_2d": bool(y_variable),
            "point_count": int(len(rows)),
        },
        "available_visual_fields": metric_candidates,
        "default_visual_field": "body_heave_mm",
        "summary": {
            "converged_points": int(converged_count),
            "total_points": int(len(rows)),
            "jump_refined_points": int(jump_refined_points),
            "elapsed_s": float(max(0.0, time.perf_counter() - t0)),
            "solver_mode": str(fixed.get("solver_mode", "fast")),
            "parallel_enabled": bool(enable_parallel),
            "workers_used": int(workers if enable_parallel else 1),
            "speed_guard_corrections": int(speed_guard_corrections),
            "local_window_refined_points": int(local_window_refined_points),
            "soft_limiter_corrections": int(soft_limiter_corrections),
            "hard_bound_corrections": int(hard_bound_corrections),
        },
        "calibration": _json_clean(calibration_meta),
    }


def _norm_constraint_kind(kind: str) -> str:
    k = str(kind or "").strip().lower()
    if k in {"eq", "equals", "="}:
        return "eq"
    if k in {"ge", ">=", "gte"}:
        return "ge"
    if k in {"le", "<=", "lte"}:
        return "le"
    raise ValueError(f"Unsupported constraint kind '{kind}'.")


def _build_opt_fixed_inputs(body: Dict[str, Any]) -> Dict[str, Any]:
    fixed = dict(body.get("fixed_inputs") or {})
    fixed.setdefault("speed_kph", float(body.get("speed_kph", fixed.get("speed_kph", 180.0))))
    fixed.setdefault("air_density", float(body.get("air_density", fixed.get("air_density", 1.225))))
    fixed.setdefault("drs_on", bool(body.get("drs_on", fixed.get("drs_on", False))))
    acc_units = str(body.get("acc_units", fixed.get("acc_units", "g"))).lower()
    ax = float(body.get("ax", fixed.get("ax", 0.0)))
    ay = float(body.get("ay", fixed.get("ay", 0.0)))
    g = 9.80665
    if "ax_g" not in fixed:
        fixed["ax_g"] = ax if acc_units in {"g", "gee"} else ax / g
    if "ay_g" not in fixed:
        fixed["ay_g"] = ay if acc_units in {"g", "gee"} else ay / g
    fixed["acc_units"] = "g"
    for key, default in (
        ("slip_angle_fl_deg", 4.0),
        ("slip_angle_fr_deg", 4.0),
        ("slip_angle_rl_deg", 4.0),
        ("slip_angle_rr_deg", 4.0),
        ("slip_ratio_fl", 0.0),
        ("slip_ratio_fr", 0.0),
        ("slip_ratio_rl", 0.0),
        ("slip_ratio_rr", 0.0),
        ("camber_fl_deg", 0.0),
        ("camber_fr_deg", 0.0),
        ("camber_rl_deg", 0.0),
        ("camber_rr_deg", 0.0),
    ):
        fixed.setdefault(key, float(body.get(key, fixed.get(key, default))))
    base_state = fixed.get("base_state") if isinstance(fixed.get("base_state"), dict) else {}
    fixed["base_state"] = {
        "hf": float(base_state.get("hf", 0.0)),
        "hr": float(base_state.get("hr", 0.0)),
        "rf": float(base_state.get("rf", 0.0)),
        "rr": float(base_state.get("rr", 0.0)),
    }
    return fixed


def _parse_optimization_problem(body: Dict[str, Any]) -> OptimizationProblem:
    objective = dict(body.get("objective") or {})
    variables_payload = body.get("variables") or []
    constraints_payload = body.get("constraints") or []
    search = dict(body.get("search_settings") or {})

    if not isinstance(variables_payload, list) or not variables_payload:
        raise ValueError("Optimization variables are required.")

    variables: List[OptimizationVariable] = []
    for item in variables_payload:
        if not isinstance(item, dict):
            continue
        variables.append(
            OptimizationVariable(
                name=str(item.get("name", "")).strip(),
                min_value=float(item.get("min_value", item.get("min", 0.0))),
                max_value=float(item.get("max_value", item.get("max", 0.0))),
                initial_guess=float(item.get("initial_guess", item.get("initial", 0.0))),
                enabled=bool(item.get("enabled", True)),
            )
        )
    if not variables:
        raise ValueError("No valid optimization variables found.")

    constraints: List[OptimizationConstraint] = []
    for item in constraints_payload:
        if not isinstance(item, dict):
            continue
        constraints.append(
            OptimizationConstraint(
                name=str(item.get("name", "")).strip(),
                kind=_norm_constraint_kind(str(item.get("kind", "eq"))),
                target=float(item.get("target", 0.0)),
                weight=float(item.get("weight", 1.0)),
            )
        )

    fixed_inputs = _build_opt_fixed_inputs(body)
    return OptimizationProblem(
        objective_mode=str(objective.get("mode", "maximize")).strip().lower(),
        objective_name=str(objective.get("name", "total_aero_load_n")).strip(),
        objective_target=(
            float(objective.get("target"))
            if objective.get("target") is not None and str(objective.get("target")).strip() != ""
            else None
        ),
        variables=variables,
        constraints=constraints,
        fixed_inputs=fixed_inputs,
        search_method=str(search.get("method", body.get("search_method", "auto"))).strip().lower(),
        max_global_points=(
            int(search.get("max_global_points"))
            if search.get("max_global_points") not in (None, "", "null")
            else None
        ),
        n_best_candidates=int(search.get("n_best_candidates", body.get("n_best_candidates", 5))),
    )


def _summarize_value(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            v = float(value)
            return f"{v:.6g}" if np.isfinite(v) else "nan"
        except Exception:
            return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return f"{{...}} ({len(value)} keys)"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in value):
            try:
                nums = [float(v) for v in value]
                if len(nums) > 5:
                    return f"[{nums[0]:.4g}, ..., {nums[-1]:.4g}] (n={len(nums)})"
                return "[" + ", ".join(f"{v:.4g}" for v in nums) + "]"
            except Exception:
                pass
        if len(value) > 5:
            return f"[... ] (n={len(value)})"
        return json.dumps(value)
    return str(value)


def _flatten_for_view(value, prefix: str, out: List[Dict[str, str]], limit: int = 280) -> None:
    if len(out) >= limit:
        return
    if isinstance(value, dict):
        for k in sorted(value.keys(), key=str):
            p = f"{prefix}.{k}" if prefix else str(k)
            _flatten_for_view(value[k], p, out, limit=limit)
            if len(out) >= limit:
                return
        return
    if isinstance(value, (list, tuple)):
        out.append({"key": prefix, "value": _summarize_value(value)})
        return
    out.append({"key": prefix, "value": _summarize_value(value)})


def _build_tyre_preview(tyre_cfg: dict, grip_scale: float) -> Dict[str, object]:
    vert = tyre_cfg.get("VERTICAL", {}) if isinstance(tyre_cfg.get("VERTICAL"), dict) else {}
    sc = tyre_cfg.get("SCALING_COEFFICIENTS", {}) if isinstance(tyre_cfg.get("SCALING_COEFFICIENTS"), dict) else {}
    fz_nom_n = max(800.0, float(vert.get("FNOMIN", 4000.0)) * float(sc.get("LFZ0", 1.0)))

    sa_vals = [float(v) for v in np.linspace(-12.0, 12.0, 25)]
    sr_vals = [float(v) for v in np.linspace(-0.2, 0.2, 21)]

    fy_vs_sa = []
    for sa in sa_vals:
        out = _mf_tyre_combined_from_json(
            tyre_cfg=tyre_cfg,
            fz_n=fz_nom_n,
            slip_angle_deg=sa,
            slip_ratio=0.0,
            camber_deg=0.0,
            grip_scale=grip_scale,
        )
        fy_vs_sa.append({"slip_angle_deg": sa, "fy_n": float(out.get("fy_n", 0.0))})

    fx_vs_sr = []
    for sr in sr_vals:
        out = _mf_tyre_combined_from_json(
            tyre_cfg=tyre_cfg,
            fz_n=fz_nom_n,
            slip_angle_deg=0.0,
            slip_ratio=sr,
            camber_deg=0.0,
            grip_scale=grip_scale,
        )
        fx_vs_sr.append({"slip_ratio": sr, "fx_n": float(out.get("fx_n", 0.0))})

    return {
        "fz_nominal_n": float(fz_nom_n),
        "fy_vs_slip_angle": fy_vs_sa,
        "fx_vs_slip_ratio": fx_vs_sr,
    }


def _build_setup_overview(json_data: dict) -> Dict[str, object]:
    cfg = json_data.get("config", {}) if isinstance(json_data.get("config"), dict) else {}
    aero = cfg.get("aero", {}) if isinstance(cfg.get("aero"), dict) else {}
    chassis = cfg.get("chassis", {}) if isinstance(cfg.get("chassis"), dict) else {}
    mass = chassis.get("carRunningMass", {}) if isinstance(chassis.get("carRunningMass"), dict) else {}
    tyres = cfg.get("tyres", {}) if isinstance(cfg.get("tyres"), dict) else {}
    tyre_front = tyres.get("front", {}) if isinstance(tyres.get("front"), dict) else {}
    tyre_rear = tyres.get("rear", {}) if isinstance(tyres.get("rear"), dict) else {}
    tyre_global_grip = float(tyres.get("rGripFactor", 1.0))
    tyre_front_grip = float(tyre_front.get("rGripFactor", 1.0))
    tyre_rear_grip = float(tyre_rear.get("rGripFactor", 1.0))

    cl_f_poly = aero.get("PolynomialCLiftBodyFDefinition", [])
    cl_r_poly = aero.get("PolynomialCLiftBodyRDefinition", [])
    cd_poly = aero.get("PolynomialCDragBodyDefinition", [])
    flap = aero.get("flapAngles", {}) if isinstance(aero.get("flapAngles"), dict) else {}
    offsets = aero.get("coefficientOffsets", {}) if isinstance(aero.get("coefficientOffsets"), dict) else {}
    drs_block = aero.get("DRS", {}) if isinstance(aero.get("DRS"), dict) else {}
    drs_enabled = bool(drs_block.get("bDRSEnabled", aero.get("bDRSEnabled", False)))

    cl_front_factor = float(aero.get("rCLiftBodyFFactor", 1.0))
    cl_rear_factor = float(aero.get("rCLiftBodyRFactor", 1.0))
    cd_factor = float(aero.get("rCDragBodyFactor", 1.0))
    cl_front_offset = float(offsets.get("CLiftBodyFUserOffset", 0.0))
    cl_rear_offset = float(offsets.get("CLiftBodyRUserOffset", 0.0))
    cl_global_offset = float(offsets.get("CLiftBodyUserOffset", 0.0))
    cd_offset = float(offsets.get("CDragBodyUserOffset", 0.0))
    aero_bal_offset = float(offsets.get("rAeroBalanceUserOffset", 0.0))
    front_global_share = float(np.clip(0.5 + aero_bal_offset, 0.0, 1.0))
    rear_global_share = 1.0 - front_global_share

    drs_cl_f_poly = drs_block.get("CLiftBodyFDRSPolynomial", [])
    drs_cl_r_poly = drs_block.get("CLiftBodyRDRSPolynomial", [])
    drs_cd_poly = drs_block.get("CDragBodyDRSPolynomial", [])

    static_state = {}
    try:
        static_state = compute_center_antis_for_state(json_data, hf=0.0, rf=0.0, hr=0.0, rr=0.0)
    except Exception:
        static_state = {}

    hridef_ref_m = max(0.0, float(static_state.get("front_ride_height", 30.0)) / 1000.0)
    hrider_ref_m = max(0.0, float(static_state.get("rear_ride_height", 40.0)) / 1000.0)
    if hridef_ref_m <= 1e-9:
        hridef_ref_m = 0.03
    if hrider_ref_m <= 1e-9:
        hrider_ref_m = 0.04

    hridef_vals = [float(v) for v in np.linspace(max(0.0, hridef_ref_m - 0.012), hridef_ref_m + 0.012, 9)]
    hrider_vals = [float(v) for v in np.linspace(max(0.0, hrider_ref_m - 0.012), hrider_ref_m + 0.012, 9)]

    def eval_aero(hridef: float, hrider: float, drs_on: bool) -> Dict[str, float]:
        vars_map = {
            "hRideF": float(hridef),
            "hRideR": float(hrider),
            "aFlapF": float(flap.get("aFlapF", 0.0)),
            "aFlapR": float(flap.get("aFlapR", 0.0)),
        }
        raw_clf = _eval_poly_terms(cl_f_poly, vars_map)
        raw_clr = _eval_poly_terms(cl_r_poly, vars_map)
        raw_cd = _eval_poly_terms(cd_poly, vars_map)
        dclf = _eval_poly_terms(drs_cl_f_poly, vars_map) if (drs_on and drs_enabled) else 0.0
        dclr = _eval_poly_terms(drs_cl_r_poly, vars_map) if (drs_on and drs_enabled) else 0.0
        dcd = _eval_poly_terms(drs_cd_poly, vars_map) if (drs_on and drs_enabled) else 0.0
        cl_front = ((raw_clf + cl_front_offset + cl_global_offset * front_global_share) * cl_front_factor) + dclf
        cl_rear = ((raw_clr + cl_rear_offset + cl_global_offset * rear_global_share) * cl_rear_factor) + dclr
        cd_total = ((raw_cd + cd_offset) * cd_factor) + dcd
        return {
            "hRideF": float(hridef),
            "hRideR": float(hrider),
            "cl_front": float(cl_front),
            "cl_rear": float(cl_rear),
            "cl_total": float(cl_front + cl_rear),
            "cd_total": float(cd_total),
        }

    aero_front_sweep = [eval_aero(hf, hrider_ref_m, drs_on=False) for hf in hridef_vals]
    aero_rear_sweep = [eval_aero(hridef_ref_m, hr, drs_on=False) for hr in hrider_vals]
    aero_map_rows = [eval_aero(hf, hr, drs_on=False) for hf in hridef_vals for hr in hrider_vals]
    aero_map_rows_drs = [eval_aero(hf, hr, drs_on=True) for hf in hridef_vals for hr in hrider_vals]

    fields: List[Dict[str, str]] = [
        {"key": "setup.name", "value": str(json_data.get("name", "Imported setup"))},
        {"key": "mass.mCar_kg", "value": _summarize_value(mass.get("mCar", ""))},
        {"key": "mass.front_weight_balance", "value": _summarize_value(mass.get("rWeightBalF", ""))},
        {"key": "aero.ARef_m2", "value": _summarize_value(aero.get("ARef", ""))},
        {"key": "aero.DRS_enabled", "value": "true" if drs_enabled else "false"},
        {"key": "tyres.global_grip", "value": _summarize_value(tyre_global_grip)},
        {"key": "vehicle.front_ride_height_ref_mm", "value": _summarize_value(static_state.get("front_ride_height", ""))},
        {"key": "vehicle.rear_ride_height_ref_mm", "value": _summarize_value(static_state.get("rear_ride_height", ""))},
        {"key": "vehicle.front_track_mm", "value": _summarize_value(static_state.get("front_track_mm", ""))},
        {"key": "vehicle.rear_track_mm", "value": _summarize_value(static_state.get("rear_track_mm", ""))},
    ]
    _flatten_for_view(cfg, "config", fields, limit=300)

    return {
        "fields": fields,
        "aero": {
            "front_reference_hRideF_m": float(hridef_ref_m),
            "rear_reference_hRideR_m": float(hrider_ref_m),
            "front_sweep": aero_front_sweep,
            "rear_sweep": aero_rear_sweep,
            "map_no_drs": aero_map_rows,
            "map_drs": aero_map_rows_drs,
        },
        "tyres": {
            "front": _build_tyre_preview(tyre_front, tyre_global_grip * tyre_front_grip),
            "rear": _build_tyre_preview(tyre_rear, tyre_global_grip * tyre_rear_grip),
        },
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "json_loaded": _state["json_data"] is not None,
        "pipeline_source": _PIPELINE_SOURCE,
        "upright_solver_loaded": solve_upright_for_zw is not None,
        "windows_perf_tuning": _json_clean(_state.get("windows_perf_tuning", {})),
    })


@app.route("/api/calibration_models")
def calibration_models():
    try:
        models = sorted([str(k) for k in VEHICLE_REGISTRY.keys()]) if isinstance(VEHICLE_REGISTRY, dict) and VEHICLE_REGISTRY else []
        if not models:
            models = list(_CALIBRATION_MODEL_FALLBACK)
        default_model = "F2_2026" if "F2_2026" in models else models[0]
        return jsonify({"status": "success", "models": models, "default_model": default_model})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/load_upload", methods=["POST"])
def load_upload():
    try:
        if "file" in request.files:
            raw = request.files["file"].read()
            json_data = _decode_json_bytes(raw)
        else:
            body = request.get_json(silent=True) or {}
            if isinstance(body.get("json_data"), dict):
                json_data = body["json_data"]
            else:
                return jsonify({"status": "error", "message": "Send file or json_data."}), 400

        _state["json_data"] = json_data
        _state["json_path"] = None
        _state["geometry_solver_cache"] = {}
        _state["body_attitude_ref_cache"] = {}
        _state["center_state_cache"] = {}
        _state["json_signature_cache"] = {}
        _state["platform_last_state"] = {}
        _ensure_json_path(json_data)
        return jsonify({"status": "success"})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/simulate_dynamic_aero", methods=["POST"])
def simulate_dynamic_aero():
    try:
        body = request.get_json(force=True) or {}
        json_data = body.get("json_data") if isinstance(body.get("json_data"), dict) else _state.get("json_data")
        if not json_data:
            return jsonify({"status": "error", "message": "No JSON loaded"}), 400

        data = _build_dynamic_aero_rows(json_data, body)
        _state["last_csv"] = data.get("csv", "")
        return jsonify({"status": "success", "data": _json_clean(data)})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/export_dynamic_csv")
def export_dynamic_csv():
    csv_text = str(_state.get("last_csv", "") or "")
    if not csv_text:
        return jsonify({"status": "error", "message": "No simulation executed yet."}), 400
    return app.response_class(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=performance_lab_simulation.csv"},
    )


@app.route("/api/sensitivity", methods=["POST"])
def sensitivity():
    try:
        body = request.get_json(force=True) or {}
        json_data = body.get("json_data") if isinstance(body.get("json_data"), dict) else _state.get("json_data")
        if not json_data:
            return jsonify({"status": "error", "message": "No JSON loaded"}), 400
        data = _build_sensitivity_data(json_data, body)
        _state["last_sensitivity_csv"] = data.get("csv", "")
        return jsonify({"status": "success", "data": _json_clean(data)})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/optimize_dynamic", methods=["POST"])
def optimize_dynamic():
    try:
        body = request.get_json(force=True) or {}
        json_data = body.get("json_data") if isinstance(body.get("json_data"), dict) else _state.get("json_data")
        if not isinstance(json_data, dict):
            return jsonify({"status": "error", "message": "No JSON loaded"}), 400

        problem = _parse_optimization_problem(body)
        static_ref = compute_center_antis_for_state(json_data, hf=0.0, rf=0.0, hr=0.0, rr=0.0)

        def _evaluator(sim_inputs: Dict[str, Any]) -> Dict[str, Any]:
            params = {
                "speed_kph": float(sim_inputs.get("speed_kph", problem.fixed_inputs.get("speed_kph", 180.0))),
                "air_density": float(sim_inputs.get("air_density", problem.fixed_inputs.get("air_density", 1.225))),
                "drs_on": bool(sim_inputs.get("drs_on", problem.fixed_inputs.get("drs_on", False))),
                "ax_g": float(sim_inputs.get("ax_g", problem.fixed_inputs.get("ax_g", 0.0))),
                "ay_g": float(sim_inputs.get("ay_g", problem.fixed_inputs.get("ay_g", 0.0))),
                "hf": float(sim_inputs.get("hf", 0.0)),
                "rf": float(sim_inputs.get("rf", 0.0)),
                "hr": float(sim_inputs.get("hr", 0.0)),
                "rr": float(sim_inputs.get("rr", 0.0)),
                "slip_angle_fl_deg": float(sim_inputs.get("slip_angle_fl_deg", problem.fixed_inputs.get("slip_angle_fl_deg", 4.0))),
                "slip_angle_fr_deg": float(sim_inputs.get("slip_angle_fr_deg", problem.fixed_inputs.get("slip_angle_fr_deg", 4.0))),
                "slip_angle_rl_deg": float(sim_inputs.get("slip_angle_rl_deg", problem.fixed_inputs.get("slip_angle_rl_deg", 4.0))),
                "slip_angle_rr_deg": float(sim_inputs.get("slip_angle_rr_deg", problem.fixed_inputs.get("slip_angle_rr_deg", 4.0))),
                "slip_ratio_fl": float(sim_inputs.get("slip_ratio_fl", problem.fixed_inputs.get("slip_ratio_fl", 0.0))),
                "slip_ratio_fr": float(sim_inputs.get("slip_ratio_fr", problem.fixed_inputs.get("slip_ratio_fr", 0.0))),
                "slip_ratio_rl": float(sim_inputs.get("slip_ratio_rl", problem.fixed_inputs.get("slip_ratio_rl", 0.0))),
                "slip_ratio_rr": float(sim_inputs.get("slip_ratio_rr", problem.fixed_inputs.get("slip_ratio_rr", 0.0))),
                "camber_fl_deg": float(sim_inputs.get("camber_fl_deg", problem.fixed_inputs.get("camber_fl_deg", 0.0))),
                "camber_fr_deg": float(sim_inputs.get("camber_fr_deg", problem.fixed_inputs.get("camber_fr_deg", 0.0))),
                "camber_rl_deg": float(sim_inputs.get("camber_rl_deg", problem.fixed_inputs.get("camber_rl_deg", 0.0))),
                "camber_rr_deg": float(sim_inputs.get("camber_rr_deg", problem.fixed_inputs.get("camber_rr_deg", 0.0))),
            }
            return _compute_operating_point(json_data, params, _static_ref=static_ref)

        result: OptimizationResult = run_optimization(problem, _evaluator)
        _state["last_opt_global_csv"] = candidates_to_csv(result.global_candidates)
        _state["last_opt_top_csv"] = candidates_to_csv(result.top_candidates)
        return jsonify({"status": "success", "data": json_safe(result)})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/export_optimize_csv")
def export_optimize_csv():
    kind = str(request.args.get("kind", "top")).strip().lower()
    if kind == "global":
        csv_text = str(_state.get("last_opt_global_csv", "") or "")
        filename = "optimization_global_candidates.csv"
    else:
        csv_text = str(_state.get("last_opt_top_csv", "") or "")
        filename = "optimization_top_candidates.csv"
    if not csv_text:
        return jsonify({"status": "error", "message": "No optimization executed yet."}), 400
    return app.response_class(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _gg_rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    io_csv = io.StringIO()
    io_csv.write(",".join(keys) + "\n")
    for r in rows:
        vals: List[str] = []
        for k in keys:
            v = r.get(k)
            text = "" if v is None else str(v)
            if ("," in text) or ('"' in text) or ("\n" in text):
                text = '"' + text.replace('"', '""') + '"'
            vals.append(text)
        io_csv.write(",".join(vals) + "\n")
    return io_csv.getvalue()


@app.route("/api/gg/frozen", methods=["POST"])
def gg_frozen():
    try:
        body = request.get_json(force=True) or {}
        json_data = body.get("json_data") if isinstance(body.get("json_data"), dict) else _state.get("json_data")
        if not isinstance(json_data, dict):
            return jsonify({"status": "error", "message": "No JSON loaded"}), 400

        base = _get_or_build_gg_base_geometry(json_data, body)
        calibrated = base.get("calibrated_json_data")
        if not isinstance(calibrated, dict):
            raise ValueError("Could not build calibrated geometry.")

        result = _compute_gg_static_map(calibrated_json_data=calibrated, body=body, mode="frozen")
        result["calibration"] = _json_clean(base.get("metadata", {}))
        result["csv"] = _gg_rows_to_csv(result.get("rows", []))
        return jsonify({"status": "success", "data": _json_clean(result)})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/gg/relaxed", methods=["POST"])
def gg_relaxed():
    return jsonify({"status": "error", "message": "Relaxed mode has been removed. Use /api/gg/frozen (Rigid Setup)."}), 410


@app.route("/api/gg/families", methods=["POST"])
def gg_families():
    return jsonify({"status": "error", "message": "State Families mode has been removed. Use /api/gg/frozen (Rigid Setup)."}), 410


@app.route("/api/platform_explorer/run", methods=["POST"])
def platform_explorer_run():
    try:
        body = request.get_json(force=True) or {}
        json_data = body.get("json_data") if isinstance(body.get("json_data"), dict) else _state.get("json_data")
        if not isinstance(json_data, dict):
            return jsonify({"status": "error", "message": "No JSON loaded"}), 400

        model_id = body.get("model_id")
        base = _build_calibrated_json_base(
            json_data=json_data,
            model_id=str(model_id).strip() if model_id not in (None, "") else None,
            force_rebuild=bool(body.get("force_rebuild_calibration", False)),
        )
        calibrated = base.get("calibrated_json_data")
        if not isinstance(calibrated, dict):
            raise ValueError("Could not build calibrated geometry.")

        base_geometry = compute_center_antis_for_state(calibrated, hf=0.0, rf=0.0, hr=0.0, rr=0.0, state_cache={})
        platform_params = _extract_platform_solver_parameters_from_json(calibrated)
        platform_params["base_geometry"] = base_geometry
        platform_params["base_front_track_mm"] = _safe_float(base_geometry.get("front_track_mm", 1200.0), 1200.0)
        platform_params["base_rear_track_mm"] = _safe_float(base_geometry.get("rear_track_mm", 1200.0), 1200.0)
        platform_params["base_hRideF_m"] = _safe_float(base_geometry.get("front_ride_height", 0.0), 0.0) / 1000.0
        platform_params["base_hRideR_m"] = _safe_float(base_geometry.get("rear_ride_height", 0.0), 0.0) / 1000.0
        platform_params["base_h_cg_mm"] = _safe_float(base_geometry.get("h_cg_mm", np.nan), np.nan)
        # Physical ride-height bounds around calibrated base:
        # - lower bound cannot go below ground (0 m)
        # - upper bound from droop travel
        base_hf = float(max(0.0, _safe_float(platform_params.get("base_hRideF_m", 0.0), 0.0)))
        base_hr = float(max(0.0, _safe_float(platform_params.get("base_hRideR_m", 0.0), 0.0)))
        front_comp = float(max(0.0, _safe_float(platform_params.get("front_max_compression_m", 0.0), 0.0)))
        rear_comp = float(max(0.0, _safe_float(platform_params.get("rear_max_compression_m", 0.0), 0.0)))
        front_droop = float(max(0.0, _safe_float(platform_params.get("front_max_droop_m", 0.0), 0.0)))
        rear_droop = float(max(0.0, _safe_float(platform_params.get("rear_max_droop_m", 0.0), 0.0)))

        # One-shot hard bounds (computed once per run):
        # - never go below ground (RH >= 0)
        # - never exceed bump/droop travel envelope
        front_comp_eff = float(min(front_comp, max(0.0, base_hf)))
        rear_comp_eff = float(min(rear_comp, max(0.0, base_hr)))
        front_droop_eff = float(max(0.0, front_droop))
        rear_droop_eff = float(max(0.0, rear_droop))
        platform_params["front_max_compression_m"] = front_comp_eff
        platform_params["rear_max_compression_m"] = rear_comp_eff
        platform_params["front_max_droop_m"] = front_droop_eff
        platform_params["rear_max_droop_m"] = rear_droop_eff

        platform_params["min_hRideF_m"] = float(max(0.0, base_hf - front_comp_eff))
        platform_params["min_hRideR_m"] = float(max(0.0, base_hr - rear_comp_eff))
        platform_params["max_hRideF_m"] = float(base_hf + front_droop_eff)
        platform_params["max_hRideR_m"] = float(base_hr + rear_droop_eff)

        result = _compute_platform_explorer_grid(
            calibrated_json_data=calibrated,
            calibration_meta=dict(base.get("metadata") or {}),
            platform_params=platform_params,
            body=body,
        )
        return jsonify({"status": "success", "data": _json_clean(result)})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/setup_overview")
def setup_overview():
    try:
        json_data = _state.get("json_data")
        if not isinstance(json_data, dict):
            return jsonify({"status": "error", "message": "No JSON loaded"}), 400
        data = _build_setup_overview(json_data)
        return jsonify({"status": "success", "data": _json_clean(data)})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


def main():
    _state["windows_perf_tuning"] = _configure_windows_performance_mode()
    port = int(os.environ.get("DYNAMIC_SIM_PORT", "6060"))
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
