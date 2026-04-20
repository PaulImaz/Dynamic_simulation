"""
Title: Body Attitude

Description:
  Python module in the SuspensionLab project.

Instructions:
  - Keep naming conventions and structure consistent with project standards.
  - Run via the main launcher (suspension_lab.py) or project tests as applicable.

Authors:
  - Paul Imaz (PI)

Versions:
  - 2.0.0 (15/04/2026 - PI): Standardized template-style file header in English.
"""


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


_ROLL_STRATEGIES = {"average_axle_roll", "front_only", "rear_only"}
_EPS = 1e-12


def _as_vec3(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} debe ser un vector de 3 elementos; recibido shape={arr.shape}.")
    return arr


def _json_to_tool(point_xyz: Any, scale: float = 1000.0) -> np.ndarray:
    p = _as_vec3(point_xyz, name="point_xyz")
    return np.array([-p[0] * scale, -p[1] * scale, -p[2] * scale], dtype=float)


def _get_cfg(json_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(json_data, dict):
        raise ValueError("json_data debe ser dict.")
    if "config" in json_data:
        cfg = json_data["config"]
        if not isinstance(cfg, dict):
            raise ValueError("json_data['config'] debe ser dict.")
        return cfg
    if "chassis" in json_data and "suspension" in json_data:
        return json_data
    raise ValueError("No se encontró 'config' ni estructura equivalente con 'chassis' y 'suspension'.")


def _get_chassis(cfg: Dict[str, Any]) -> Dict[str, Any]:
    chassis = cfg.get("chassis")
    if not isinstance(chassis, dict):
        raise ValueError("Falta config.chassis.")
    return chassis


def _get_pickups(cfg: Dict[str, Any], axle: str) -> Dict[str, Any]:
    try:
        return cfg["suspension"][axle]["external"]["pickUpPts"]
    except Exception as exc:
        raise ValueError(f"Faltan pickup points de suspensión para axle='{axle}'.") from exc


def _get_weight_balance_front(chassis: Dict[str, Any]) -> float:
    if "rWeightBalF" in chassis:
        return float(chassis["rWeightBalF"])
    crm = chassis.get("carRunningMass")
    if isinstance(crm, dict) and "rWeightBalF" in crm:
        return float(crm["rWeightBalF"])
    raise ValueError("Falta rWeightBalF en config.chassis o config.chassis.carRunningMass.")


def _infer_wheelbase_mm(cfg: Dict[str, Any]) -> float:
    chassis = _get_chassis(cfg)
    ride_f = chassis.get("rRideF")
    ride_r = chassis.get("rRideR")
    if ride_f is not None and ride_r is not None:
        x_f = float(_json_to_tool(ride_f)[0])
        x_r = float(_json_to_tool(ride_r)[0])
        wb = abs(x_f - x_r)
        if wb > _EPS:
            return wb

    front_pp = _get_pickups(cfg, "front")
    rear_pp = _get_pickups(cfg, "rear")
    if "rAxleC" in front_pp and "rAxleC" in rear_pp:
        x_f = float(_json_to_tool(front_pp["rAxleC"])[0])
        x_r = float(_json_to_tool(rear_pp["rAxleC"])[0])
        wb = abs(x_f - x_r)
        if wb > _EPS:
            return wb

    raise ValueError(
        "No se pudo inferir wheelbase_mm. Se requiere rRideF/rRideR o rAxleC delantero/trasero válidos."
    )


def _infer_track_mm(cfg: Dict[str, Any], axle: str) -> float:
    pp = _get_pickups(cfg, axle)
    for key in ("rAxleC", "rUserTCP", "rAxleAxis"):
        if key in pp:
            p = _as_vec3(pp[key], name=f"{axle}.{key}")
            track = abs(float(p[1])) * 2000.0
            if track > _EPS:
                return track
    raise ValueError(
        f"No se pudo inferir track_{axle}_mm. Se esperaba {axle}.external.pickUpPts con rAxleC/rUserTCP/rAxleAxis."
    )


def _infer_cg_body_mm(chassis: Dict[str, Any], wheelbase_mm: float) -> np.ndarray:
    if "rCoG" in chassis:
        return _json_to_tool(chassis["rCoG"])

    if "zCoG" not in chassis:
        raise ValueError("Falta zCoG en config.chassis.")
    z_cg_mm = float(-float(chassis["zCoG"]) * 1000.0)

    if "xCoG" in chassis:
        x_cg_mm = float(-float(chassis["xCoG"]) * 1000.0)
    else:
        wbf = _get_weight_balance_front(chassis)
        # Convención elegida:
        # x_cg medido desde el eje delantero (x=0) hacia el eje trasero (x=wheelbase),
        # por equilibrio estático:
        #   rWeightBalF = (wheelbase - x_cg) / wheelbase
        #   x_cg = (1 - rWeightBalF) * wheelbase
        x_cg_mm = float((1.0 - wbf) * wheelbase_mm)

    if "yCoG" in chassis:
        y_cg_mm = float(-float(chassis["yCoG"]) * 1000.0)
    else:
        y_cg_mm = 0.0

    return np.array([x_cg_mm, y_cg_mm, z_cg_mm], dtype=float)


def _roll_rotation_matrix(roll_rad: float) -> np.ndarray:
    c = math.cos(roll_rad)
    s = math.sin(roll_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def _pitch_rotation_matrix(pitch_rad: float) -> np.ndarray:
    c = math.cos(pitch_rad)
    s = math.sin(pitch_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _rotation_matrix(roll_rad: float, pitch_rad: float) -> np.ndarray:
    """
    Matriz de rotación cuerpo->global.

    Convención:
    - Roll alrededor de X (marco body).
    - Pitch alrededor de Y (marco body).
    - Composición: R = R_pitch @ R_roll
      (primero roll, después pitch sobre el vector body).
    """
    return _pitch_rotation_matrix(pitch_rad) @ _roll_rotation_matrix(roll_rad)


def _validate_state_4w(state_4w: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(state_4w, dict):
        raise ValueError("state_4w debe ser dict.")
    missing = [k for k in ("hf", "rf", "hr", "rr") if k not in state_4w]
    if missing:
        raise ValueError(f"Faltan claves en state_4w: {missing}")
    return {k: float(state_4w[k]) for k in ("hf", "rf", "hr", "rr")}


@dataclass
class BodyReference:
    wheelbase_mm: float
    track_front_mm: float
    track_rear_mm: float
    cg_body_mm: np.ndarray
    undertray_front_body_mm: Optional[np.ndarray] = None
    undertray_mid_body_mm: Optional[np.ndarray] = None
    undertray_rear_body_mm: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.wheelbase_mm = float(self.wheelbase_mm)
        self.track_front_mm = float(self.track_front_mm)
        self.track_rear_mm = float(self.track_rear_mm)
        if self.wheelbase_mm <= _EPS:
            raise ValueError(f"wheelbase_mm debe ser > 0; recibido {self.wheelbase_mm}")
        if self.track_front_mm <= _EPS:
            raise ValueError(f"track_front_mm debe ser > 0; recibido {self.track_front_mm}")
        if self.track_rear_mm <= _EPS:
            raise ValueError(f"track_rear_mm debe ser > 0; recibido {self.track_rear_mm}")
        self.cg_body_mm = _as_vec3(self.cg_body_mm, name="cg_body_mm")
        if self.undertray_front_body_mm is not None:
            self.undertray_front_body_mm = _as_vec3(self.undertray_front_body_mm, name="undertray_front_body_mm")
        if self.undertray_mid_body_mm is not None:
            self.undertray_mid_body_mm = _as_vec3(self.undertray_mid_body_mm, name="undertray_mid_body_mm")
        if self.undertray_rear_body_mm is not None:
            self.undertray_rear_body_mm = _as_vec3(self.undertray_rear_body_mm, name="undertray_rear_body_mm")


@dataclass
class BodyAttitudeState:
    heave_mm: float
    roll_rad: float
    roll_deg: float
    pitch_rad: float
    pitch_deg: float
    translation_mm: np.ndarray
    rotation_matrix: np.ndarray
    cg_global_mm: np.ndarray
    h_cg_mm: float
    undertray_front_global_mm: Optional[np.ndarray] = None
    undertray_mid_global_mm: Optional[np.ndarray] = None
    undertray_rear_global_mm: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.heave_mm = float(self.heave_mm)
        self.roll_rad = float(self.roll_rad)
        self.roll_deg = float(self.roll_deg)
        self.pitch_rad = float(self.pitch_rad)
        self.pitch_deg = float(self.pitch_deg)
        self.translation_mm = _as_vec3(self.translation_mm, name="translation_mm")
        self.cg_global_mm = _as_vec3(self.cg_global_mm, name="cg_global_mm")
        self.rotation_matrix = np.asarray(self.rotation_matrix, dtype=float)
        if self.rotation_matrix.shape != (3, 3):
            raise ValueError(f"rotation_matrix debe ser 3x3; recibido {self.rotation_matrix.shape}")
        self.h_cg_mm = float(self.h_cg_mm)
        if self.undertray_front_global_mm is not None:
            self.undertray_front_global_mm = _as_vec3(self.undertray_front_global_mm, name="undertray_front_global_mm")
        if self.undertray_mid_global_mm is not None:
            self.undertray_mid_global_mm = _as_vec3(self.undertray_mid_global_mm, name="undertray_mid_global_mm")
        if self.undertray_rear_global_mm is not None:
            self.undertray_rear_global_mm = _as_vec3(self.undertray_rear_global_mm, name="undertray_rear_global_mm")


def build_body_reference_from_json(json_data: Dict[str, Any]) -> BodyReference:
    """
    Construye la referencia rígida del cuerpo (mm, marco de herramienta).
    """
    cfg = _get_cfg(json_data)
    chassis = _get_chassis(cfg)

    wheelbase_mm = _infer_wheelbase_mm(cfg)
    track_front_mm = _infer_track_mm(cfg, "front")
    track_rear_mm = _infer_track_mm(cfg, "rear")
    cg_body_mm = _infer_cg_body_mm(chassis, wheelbase_mm)

    def _opt_undertray(name: str) -> Optional[np.ndarray]:
        val = chassis.get(name)
        if val is None:
            return None
        return _json_to_tool(val)

    return BodyReference(
        wheelbase_mm=wheelbase_mm,
        track_front_mm=track_front_mm,
        track_rear_mm=track_rear_mm,
        cg_body_mm=cg_body_mm,
        undertray_front_body_mm=_opt_undertray("rUndertrayFront"),
        undertray_mid_body_mm=_opt_undertray("rUndertrayMid"),
        undertray_rear_body_mm=_opt_undertray("rUndertrayRear"),
    )


def transform_body_point_to_global(body_point_mm: np.ndarray, attitude: BodyAttitudeState) -> np.ndarray:
    """
    Transforma un punto fijo del chasis (marco body) a coordenadas globales instantáneas.
    """
    p = _as_vec3(body_point_mm, name="body_point_mm")
    return attitude.translation_mm + attitude.rotation_matrix @ p


def compute_body_attitude_state(
    json_data: Dict[str, Any],
    state_4w: Dict[str, Any],
    ref: Optional[BodyReference] = None,
    roll_strategy: str = "average_axle_roll",
) -> BodyAttitudeState:
    """
    Calcula actitud y CG instantáneo para un estado 4-wheel.

    Parámetros:
    - state_4w en mm:
      hf/hr: heave por eje.
      rf/rr: desplazamiento diferencial equivalente de roll por eje.
    - roll_strategy:
      - average_axle_roll: promedio entre roll delantero y trasero.
      - front_only: usa solo roll delantero.
      - rear_only: usa solo roll trasero.

    Nota de signo:
    - Se usa translation_z = -heave_mm para mantener la convención habitual
      de +heave como compresión (chasis baja en Z global de herramienta).
    """
    if roll_strategy not in _ROLL_STRATEGIES:
        raise ValueError(f"roll_strategy inválida '{roll_strategy}'. Opciones: {sorted(_ROLL_STRATEGIES)}")

    vals = _validate_state_4w(state_4w)
    hf = vals["hf"]
    rf = vals["rf"]
    hr = vals["hr"]
    rr = vals["rr"]

    ref = ref if ref is not None else build_body_reference_from_json(json_data)

    roll_front = math.atan(rf / ref.track_front_mm)
    roll_rear = math.atan(rr / ref.track_rear_mm)

    if roll_strategy == "average_axle_roll":
        roll_body = 0.5 * (roll_front + roll_rear)
    elif roll_strategy == "front_only":
        roll_body = roll_front
    else:
        roll_body = roll_rear

    pitch_body = math.atan((hr - hf) / ref.wheelbase_mm)
    heave_body = 0.5 * (hf + hr)

    translation = np.array([0.0, 0.0, -heave_body], dtype=float)
    rotation = _rotation_matrix(roll_body, pitch_body)
    cg_global = translation + rotation @ ref.cg_body_mm

    attitude = BodyAttitudeState(
        heave_mm=heave_body,
        roll_rad=roll_body,
        roll_deg=math.degrees(roll_body),
        pitch_rad=pitch_body,
        pitch_deg=math.degrees(pitch_body),
        translation_mm=translation,
        rotation_matrix=rotation,
        cg_global_mm=cg_global,
        h_cg_mm=float(cg_global[2]),
    )

    if ref.undertray_front_body_mm is not None:
        attitude.undertray_front_global_mm = transform_body_point_to_global(ref.undertray_front_body_mm, attitude)
    if ref.undertray_mid_body_mm is not None:
        attitude.undertray_mid_global_mm = transform_body_point_to_global(ref.undertray_mid_body_mm, attitude)
    if ref.undertray_rear_body_mm is not None:
        attitude.undertray_rear_global_mm = transform_body_point_to_global(ref.undertray_rear_body_mm, attitude)

    return attitude


def compute_body_attitude_summary(
    json_data: Dict[str, Any],
    state_4w: Dict[str, Any],
    ref: Optional[BodyReference] = None,
    roll_strategy: str = "average_axle_roll",
) -> Dict[str, Any]:
    """
    Resumen serializable para integración rápida en APIs.
    """
    attitude = compute_body_attitude_state(
        json_data=json_data,
        state_4w=state_4w,
        ref=ref,
        roll_strategy=roll_strategy,
    )
    out: Dict[str, Any] = {
        "heave_mm": attitude.heave_mm,
        "roll_deg": attitude.roll_deg,
        "pitch_deg": attitude.pitch_deg,
        "cg_global_mm": attitude.cg_global_mm.tolist(),
        "h_cg_mm": attitude.h_cg_mm,
    }
    if attitude.undertray_front_global_mm is not None:
        out["undertray_front_global_mm"] = attitude.undertray_front_global_mm.tolist()
    if attitude.undertray_mid_global_mm is not None:
        out["undertray_mid_global_mm"] = attitude.undertray_mid_global_mm.tolist()
    if attitude.undertray_rear_global_mm is not None:
        out["undertray_rear_global_mm"] = attitude.undertray_rear_global_mm.tolist()
    return out
