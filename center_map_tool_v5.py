"""
Title: Center Map Tool V5

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


import json
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ============================================================
# TRANSFORMACIÓN DE COORDENADAS  (antes: json_to_suspension.py)
# ============================================================
#
# JSON (metros):             Herramienta (mm):
#   x → longitudinal +atrás    X = -x * 1000  (longitudinal +adelante)
#   y → lateral +derecha        Y = -y * 1000  (lateral +izquierda)
#   z → vertical +abajo         Z = -z * 1000  (vertical +arriba)
#
# Los puntos traseros del JSON están en coordenadas globales;
# se les resta rRideR para obtener coordenadas locales al eje.

def _json_to_tool(pt: list, scale: float = 1000.0) -> np.ndarray:
    x, y, z = pt
    return np.array([-x * scale, -y * scale, -z * scale], dtype=float)


def _json_to_tool_local(pt: list, origin: list, scale: float = 1000.0) -> np.ndarray:
    x, y, z = pt
    ox, oy, oz = origin
    return np.array([-(x-ox)*scale, -(y-oy)*scale, -(z-oz)*scale], dtype=float)


def _detect_pushrod_body(pp: dict) -> str:
    """
    Lee el campo 'name' de pickUpPts y devuelve 'lower_wishbone' o 'upright'.
    Si no se reconoce el texto, lanza un error descriptivo.
    """
    name = pp.get("name", "").lower()
    if "lower wishbone" in name:
        return "lower_wishbone"
    if "upright" in name:
        return "upright"
    raise ValueError(
        f"No se puede detectar el cuerpo del pushrod desde el nombre: '{pp.get('name')}'. "
        "Esperado 'lower wishbone' o 'upright' en el campo name de pickUpPts."
    )


def _extract_front_points(cfg: dict) -> Dict[str, np.ndarray]:
    pp = cfg["suspension"]["front"]["external"]["pickUpPts"]
    return {
        "P1":  _json_to_tool(pp["rFLWBI"]),
        "P2":  _json_to_tool(pp["rRLWBI"]),
        "P3":  _json_to_tool(pp["rFUWBI"]),
        "P4":  _json_to_tool(pp["rRUWBI"]),
        "P6":  _json_to_tool(pp["rFUWBO"]),
        "P7":  _json_to_tool(pp["rFLWBO"]),
        "P9":  _json_to_tool(pp["rUserTCP"]),
        "P10": _json_to_tool(pp["rAxleC"]),
        "P11": _json_to_tool(pp["rPRO"]),
        "P5":  _json_to_tool(pp["rTRI"]),
        "P8":  _json_to_tool(pp["rTRO"]),
    }


def _extract_rear_points_local(cfg: dict) -> Dict[str, np.ndarray]:
    pp     = cfg["suspension"]["rear"]["external"]["pickUpPts"]
    origin = cfg["chassis"]["rRideR"]

    def _p7_rear(pt: list, org: list, scale: float = 1000.0) -> np.ndarray:
        """
        P7 (lower wishbone outboard, rear) requiere que la componente X local
        se trate con el mismo convenio que en v4 (negada respecto a la
        transformación estándar) para que el IC en vista lateral XZ y, por
        tanto, la altura del pitch center coincidan con la referencia física.
        """
        x, y, z   = pt
        ox, oy, oz = org
        return np.array([(x - ox) * scale,          # X: signo positivo (igual a v4)
                         -(y - oy) * scale,
                         -(z - oz) * scale], dtype=float)

    return {
        "P1":  _json_to_tool_local(pp["rFLWBI"],   origin),
        "P2":  _json_to_tool_local(pp["rRLWBI"],   origin),
        "P3":  _json_to_tool_local(pp["rFUWBI"],   origin),
        "P4":  _json_to_tool_local(pp["rRUWBI"],   origin),
        "P6":  _json_to_tool_local(pp["rFUWBO"],   origin),
        "P7":  _p7_rear(pp["rFLWBO"],              origin),   # X corregido (convenio v4)
        "P9":  _json_to_tool_local(pp["rUserTCP"], origin),
        "P10": _json_to_tool_local(pp["rAxleC"],   origin),
        "P11": _json_to_tool_local(pp["rPRO"],     origin),
        "P5":  _json_to_tool_local(pp["rTRI"],     origin),
        "P8":  _json_to_tool_local(pp["rTRO"],     origin),
    }


def _extract_vehicle_params(cfg: dict) -> dict:
    chassis      = cfg["chassis"]
    wheelbase_mm = abs(chassis["rRideR"][0]) * 1000.0
    wbf          = chassis["carRunningMass"]["rWeightBalF"]
    return {
        "wheelbase_mm":     wheelbase_mm,
        "x_cg_mm":          wbf * wheelbase_mm,
        "z_cg_mm":          abs(chassis["zCoG"]) * 1000.0,
        "weight_balance_f": wbf,
        "mass_kg":          chassis["carRunningMass"]["mCar"],
        "ride_height_f_mm": chassis["hRideFSetup"] * 1000.0,
        "ride_height_r_mm": chassis["hRideRSetup"] * 1000.0,
    }


def load_suspension_from_json(path: str) -> Tuple[dict, dict, dict, str, str]:
    """
    Lee el JSON y devuelve:
      front_pts          : pickup points delanteros en mm (sistema herramienta, lado izquierdo)
      rear_local         : pickup points traseros en mm (coordenadas locales al eje)
      vehicle            : wheelbase, CdG, masas, ride heights
      front_pushrod_body : 'lower_wishbone' o 'upright'
      rear_pushrod_body  : 'lower_wishbone' o 'upright'
    """
    with open(path, "r") as f:
        data = json.load(f)
    cfg = data["config"]

    front_pp = cfg["suspension"]["front"]["external"]["pickUpPts"]
    rear_pp  = cfg["suspension"]["rear"]["external"]["pickUpPts"]

    return (
        _extract_front_points(cfg),
        _extract_rear_points_local(cfg),
        _extract_vehicle_params(cfg),
        _detect_pushrod_body(front_pp),
        _detect_pushrod_body(rear_pp),
    )


# ============================================================
# SOLVER CINEMÁTICO INVERSO  (antes: upright_solver.py)
# ============================================================

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _unit(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n < 1e-15:
        raise ValueError("Vector de norma casi cero.")
    return v / n


def _rodrigues(v: np.ndarray, axis_hat: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return v * c + np.cross(axis_hat, v) * s + axis_hat * (float(np.dot(axis_hat, v)) * (1.0 - c))


def _orthogonal_unit(n_hat: np.ndarray) -> np.ndarray:
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, n_hat))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = ref - float(np.dot(ref, n_hat)) * n_hat
    return _unit(u)


def _signed_angle_about_axis(v0: np.ndarray, v1: np.ndarray, axis_hat: np.ndarray) -> float:
    axis_hat = _unit(axis_hat)
    p0 = v0 - float(np.dot(v0, axis_hat)) * axis_hat
    p1 = v1 - float(np.dot(v1, axis_hat)) * axis_hat
    n0, n1 = _norm(p0), _norm(p1)
    if n0 < 1e-12 or n1 < 1e-12:
        raise ValueError("Vector degenerado al calcular ángulo alrededor del eje.")
    p0, p1 = p0 / n0, p1 / n1
    c = max(-1.0, min(1.0, float(np.dot(p0, p1))))
    s = float(np.dot(axis_hat, np.cross(p0, p1)))
    return math.atan2(s, c)


@dataclass
class _Circle3D:
    center:     np.ndarray
    normal_hat: np.ndarray
    radius:     float
    u_hat:      np.ndarray
    v_hat:      np.ndarray

    def point(self, angle: float) -> np.ndarray:
        return self.center + self.radius * (
            math.cos(angle) * self.u_hat + math.sin(angle) * self.v_hat
        )


def _circle_from_two_spheres(P1, P2, R1, R2, P0) -> _Circle3D:
    dvec  = P2 - P1
    d     = _norm(dvec)
    if d < 1e-12:
        raise ValueError("Centros de esfera coincidentes.")
    n_hat  = dvec / d
    a      = (R1*R1 - R2*R2 + d*d) / (2.0 * d)
    center = P1 + a * n_hat
    rho_sq = max(R1*R1 - a*a, 0.0)
    radius = math.sqrt(rho_sq)
    u      = P0 - center
    u      = u - float(np.dot(u, n_hat)) * n_hat
    u_hat  = _orthogonal_unit(n_hat) if _norm(u) < 1e-10 else _unit(u)
    v_hat  = _unit(np.cross(n_hat, u_hat))
    return _Circle3D(center=center, normal_hat=n_hat, radius=radius, u_hat=u_hat, v_hat=v_hat)


def _rotate_minimal(v, a0_hat, a1_hat) -> np.ndarray:
    a0_hat = _unit(a0_hat)
    a1_hat = _unit(a1_hat)
    cross  = np.cross(a0_hat, a1_hat)
    s      = _norm(cross)
    c      = float(np.dot(a0_hat, a1_hat))
    if s < 1e-12:
        if c > 0.0:
            return v.copy()
        return _rodrigues(v, _orthogonal_unit(a0_hat), math.pi)
    return _rodrigues(v, cross / s, math.atan2(s, c))


def _lca_frame(A, C, B) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A, C, B  = (np.asarray(x, dtype=float).reshape(3) for x in (A, C, B))
    axis_hat = _unit(C - A)
    proj     = A + float(np.dot(B - A, axis_hat)) * axis_hat
    lca_r    = B - proj
    if _norm(lca_r) < 1e-12:
        raise ValueError("B demasiado cerca del eje A-C.")
    e1 = _unit(lca_r)
    e2 = _unit(np.cross(axis_hat, e1))
    return e1, e2, axis_hat


@dataclass
class UprightKinematicsInput:
    A: np.ndarray;  C: np.ndarray;  D: np.ndarray;  F: np.ndarray;  T: np.ndarray
    B0: np.ndarray; E0: np.ndarray; S0: np.ndarray; W0: np.ndarray; N0: np.ndarray
    pushrod_wheel_body: str         = "upright"
    W_lca_offset_local: np.ndarray  = None

    def __post_init__(self):
        for name in ["A","C","D","F","T","B0","E0","S0","W0","N0"]:
            setattr(self, name, np.array(getattr(self, name), dtype=float).reshape(3))
        self.pushrod_wheel_body = str(self.pushrod_wheel_body).strip().lower()
        if self.pushrod_wheel_body not in ("upright", "lower_wishbone"):
            raise ValueError(f"pushrod_wheel_body inválido: {self.pushrod_wheel_body}")
        if self.W_lca_offset_local is None:
            self.W_lca_offset_local = np.zeros(3, dtype=float)
        else:
            self.W_lca_offset_local = np.array(self.W_lca_offset_local, dtype=float).reshape(3)

        self.L_AB = _norm(self.B0 - self.A);  self.L_CB = _norm(self.B0 - self.C)
        self.L_DE = _norm(self.E0 - self.D);  self.L_FE = _norm(self.E0 - self.F)
        self.L_BE = _norm(self.E0 - self.B0); self.L_ST = _norm(self.S0 - self.T)

        self.lower_circle = _circle_from_two_spheres(self.A, self.C, self.L_AB, self.L_CB, self.B0)
        self.upper_circle = _circle_from_two_spheres(self.D, self.F, self.L_DE, self.L_FE, self.E0)

        self.e1_0   = _unit(self.E0 - self.B0)
        s0_perp     = self.S0 - self.B0
        s0_perp     = s0_perp - float(np.dot(s0_perp, self.e1_0)) * self.e1_0
        self.e2_0   = _unit(s0_perp)
        self.e3_0   = _unit(np.cross(self.e1_0, self.e2_0))
        self.S_local = self._to_local(self.S0)
        self.W_local = self._to_local(self.W0)
        self.N_local = self._to_local(self.N0)
        self.lca_axis_hat = _unit(self.C - self.A)

    def _to_local(self, P: np.ndarray) -> np.ndarray:
        r = P - self.B0
        return np.array([float(np.dot(r, self.e1_0)),
                         float(np.dot(r, self.e2_0)),
                         float(np.dot(r, self.e3_0))], dtype=float)


def _build_upright_positions(inp: UprightKinematicsInput, state: np.ndarray) -> dict:
    theta, phi, psi = np.asarray(state, dtype=float).reshape(3)
    B  = inp.lower_circle.point(theta)
    E  = inp.upper_circle.point(phi)
    e1 = _unit(E - B)

    e2_base = _rotate_minimal(inp.e2_0, inp.e1_0, e1)
    e2_base = e2_base - float(np.dot(e2_base, e1)) * e1
    e2_base = _unit(e2_base)
    e2      = _rodrigues(e2_base, e1, psi)
    e2      = e2 - float(np.dot(e2, e1)) * e1
    e2      = _unit(e2)
    e3      = _unit(np.cross(e1, e2))

    def from_local(lv):
        return B + lv[0]*e1 + lv[1]*e2 + lv[2]*e3

    S = from_local(inp.S_local)
    N = from_local(inp.N_local)

    if inp.pushrod_wheel_body == "upright":
        W = from_local(inp.W_local)
    else:
        theta_lca = _signed_angle_about_axis(inp.B0 - inp.A, B - inp.A, inp.lca_axis_hat)
        W_base    = inp.A + _rodrigues(inp.W0 - inp.A, inp.lca_axis_hat, theta_lca)
        lca_e1, lca_e2, lca_e3 = _lca_frame(inp.A, inp.C, B)
        W = W_base + (inp.W_lca_offset_local[0]*lca_e1
                    + inp.W_lca_offset_local[1]*lca_e2
                    + inp.W_lca_offset_local[2]*lca_e3)

    return {"B": B, "E": E, "S": S, "W": W, "N": N,
            "frame_e1": e1, "frame_e2": e2, "frame_e3": e3}


def solve_upright_for_zw(
    inp: UprightKinematicsInput,
    zw_m: float,
    x0: Optional[np.ndarray] = None,
) -> Tuple[dict, np.ndarray, dict]:
    from scipy.optimize import least_squares

    z_target = inp.N0[2] - zw_m
    p0       = np.zeros(3) if x0 is None else np.array(x0, dtype=float).reshape(3)

    def residuals(p):
        pos = _build_upright_positions(inp, p)
        return np.array([
            _norm(pos["E"] - pos["B"]) - inp.L_BE,
            _norm(pos["S"] - inp.T)    - inp.L_ST,
            pos["N"][2]                - z_target,
        ], dtype=float)

    sol  = least_squares(residuals, p0, method="trf",
                         ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=300)
    pos  = _build_upright_positions(inp, sol.x)
    info = {"success": bool(sol.success), "message": str(sol.message),
            "nfev": int(sol.nfev), "residual_norm": float(np.linalg.norm(sol.fun))}
    return pos, sol.x, info


# Alias de compatibilidad
DoubleWishboneInput = UprightKinematicsInput
solve_pose_for_zw   = solve_upright_for_zw


# ============================================================
# CLASE PRINCIPAL
# ============================================================

class SuspensionGeometryExact:
    """
    Cálculo de:
      - Front-view instant centers (para roll center)
      - Side-view instant centers (para pitch center)
      - Roll center delantero y trasero
      - Pitch center
      - Anti-dive / Anti-squat
      - Mapa completo variando la posición vertical del centro de rueda
    """

    def __init__(self, json_path: str):
        (
            front_src,
            rear_local_src,
            self._vehicle_params,
            self.front_pushrod_body,
            self.rear_pushrod_body,
        ) = load_suspension_from_json(json_path)

        self.wheelbase = self._vehicle_params["wheelbase_mm"]

        self.front      = {k: v.copy() for k, v in front_src.items()}
        self.rear_local = {k: v.copy() for k, v in rear_local_src.items()}

        self.rear = {
            key: value + np.array([self.wheelbase, 0.0, 25.0], dtype=float)
            for key, value in self.rear_local.items()
        }

        self.front_base = deepcopy(self.front)
        self.rear_base  = deepcopy(self.rear)

        self.front_left_base  = deepcopy(self.front_base)
        self.front_right_base = self.mirror_points_y(self.front_base)
        self.rear_left_base   = deepcopy(self.rear_base)
        self.rear_right_base  = self.mirror_points_y(self.rear_base)

        self.front_left  = deepcopy(self.front_left_base)
        self.front_right = deepcopy(self.front_right_base)
        self.rear_left   = deepcopy(self.rear_left_base)
        self.rear_right  = deepcopy(self.rear_right_base)

        def _make_kin(base, pushrod_body):
            return {
                "A":  base["P1"].copy(),  "C":  base["P2"].copy(),
                "D":  base["P3"].copy(),  "F":  base["P4"].copy(),
                "B0": base["P7"].copy(),  "E0": base["P6"].copy(),
                "N0": base["P10"].copy(), "S0": base["P8"].copy(),
                "W0": base["P11"].copy(), "T":  base["P5"].copy(),
                "pushrod_wheel_body": pushrod_body,
                "W_lca_offset_local": np.zeros(3, dtype=float),
            }

        self.front_left_kin_points  = _make_kin(self.front_left_base, self.front_pushrod_body)
        self.rear_left_kin_points   = _make_kin(self.rear_left_base, self.rear_pushrod_body)
        self.front_right_kin_points = self.mirror_kin_points_y(self.front_left_kin_points)
        self.rear_right_kin_points  = self.mirror_kin_points_y(self.rear_left_kin_points)
        self.front_track_mm = float(abs(self.front_left_base["P9"][1] - self.front_right_base["P9"][1]))
        self.rear_track_mm = float(abs(self.rear_left_base["P9"][1] - self.rear_right_base["P9"][1]))
        self.front_cp_z_ref = 0.5 * (float(self.front_left_base["P9"][2]) + float(self.front_right_base["P9"][2]))
        self.rear_cp_z_ref = 0.5 * (float(self.rear_left_base["P9"][2]) + float(self.rear_right_base["P9"][2]))
        # OPK-like mode: optional coupling between roll angle and commanded wheel-center
        # vertical displacement while contact patches remain fixed. Front is kept pure by
        # default (0.0). Rear uses a mild coupling to match observed OPK rear RC trend.
        self.opk_like_roll_zw_ratio_front = 0.0
        self.opk_like_roll_zw_ratio_rear = 0.15

    # ============================================================
    # Utilidades geométricas
    # ============================================================

    @staticmethod
    def mirror_points_y(points):
        return {key: np.array([p[0], -p[1], p[2]], dtype=float) for key, p in points.items()}

    @staticmethod
    def mirror_kin_points_y(kin_points):
        out = {}
        for k, v in kin_points.items():
            if isinstance(v, np.ndarray) and v.shape == (3,):
                out[k] = np.array([v[0], -v[1], v[2]], dtype=float)
            else:
                out[k] = v
        return out

    @staticmethod
    def average_side_points(points_a, points_b):
        return {k: 0.5 * (np.asarray(points_a[k], dtype=float) +
                          np.asarray(points_b[k], dtype=float))
                for k in points_a}

    @staticmethod
    def project_point(point, plane):
        if plane == "yz":
            return np.array([point[1], point[2]], dtype=float)
        elif plane == "xz":
            return np.array([point[0], point[2]], dtype=float)
        raise ValueError("plane debe ser 'yz' o 'xz'")

    @staticmethod
    def line_intersection_2d(p1, d1, p2, d2, eps=1e-10):
        A   = np.column_stack((d1, -d2))
        det = np.linalg.det(A)
        if abs(det) < eps:
            return None
        b    = p2 - p1
        t, _ = np.linalg.solve(A, b)
        return p1 + t * d1

    @staticmethod
    def perpendicular_direction_2d(v2, eps=1e-10):
        n    = np.array([-v2[1], v2[0]], dtype=float)
        norm = np.linalg.norm(n)
        if norm < eps:
            return None
        return n / norm

    @staticmethod
    def axis_unit_vector(a, b, eps=1e-10):
        u    = b - a
        norm = np.linalg.norm(u)
        if norm < eps:
            raise ValueError("Dos puntos del eje coinciden; eje inválido")
        return u / norm

    def projected_ic_line_from_arm(self, inboard_a, inboard_b, outboard, plane):
        u  = self.axis_unit_vector(inboard_a, inboard_b)
        v3 = np.cross(u, outboard - inboard_a)
        v2 = self.project_point(v3, plane)
        p2 = self.project_point(outboard, plane)
        d2 = self.perpendicular_direction_2d(v2)
        if d2 is None:
            return None, None, None
        return p2, d2, v2

    # ============================================================
    # Instant centers
    # ============================================================

    def calculate_side_ic_front_view(self, points, label=""):
        lower_p, lower_d, lower_v = self.projected_ic_line_from_arm(
            points["P1"], points["P2"], points["P7"], plane="yz")
        upper_p, upper_d, upper_v = self.projected_ic_line_from_arm(
            points["P3"], points["P4"], points["P6"], plane="yz")
        if lower_p is None or upper_p is None:
            return None, None
        ic = self.line_intersection_2d(lower_p, lower_d, upper_p, upper_d)
        if ic is None:
            return None, None
        debug = {"lower_p": lower_p, "lower_d": lower_d, "lower_v": lower_v,
                 "upper_p": upper_p, "upper_d": upper_d, "upper_v": upper_v}
        return ic, debug

    def calculate_axle_ic_side_view(self, points, label=""):
        lower_p, lower_d, lower_v = self.projected_ic_line_from_arm(
            points["P1"], points["P2"], points["P7"], plane="xz")
        upper_p, upper_d, upper_v = self.projected_ic_line_from_arm(
            points["P3"], points["P4"], points["P6"], plane="xz")
        if lower_p is None or upper_p is None:
            return None, None
        ic = self.line_intersection_2d(lower_p, lower_d, upper_p, upper_d)
        if ic is None:
            return None, None
        debug = {"lower_p": lower_p, "lower_d": lower_d, "lower_v": lower_v,
                 "upper_p": upper_p, "upper_d": upper_d, "upper_v": upper_v}
        return ic, debug

    # ============================================================
    # Roll center
    # ============================================================

    def calculate_roll_center_from_sides(self, points_left, points_right, name=""):
        ic_left,  dbg_left  = self.calculate_side_ic_front_view(points_left,  label=f"{name} left")
        ic_right, dbg_right = self.calculate_side_ic_front_view(points_right, label=f"{name} right")
        if ic_left is None or ic_right is None:
            return None
        cp_left  = self.project_point(points_left["P9"],  "yz")
        cp_right = self.project_point(points_right["P9"], "yz")
        d_left = cp_left - ic_left
        d_right = cp_right - ic_right
        n_left = float(np.linalg.norm(d_left))
        n_right = float(np.linalg.norm(d_right))
        if n_left < 1e-10 or n_right < 1e-10:
            return None
        sin_theta = abs(float(d_left[0] * d_right[1] - d_left[1] * d_right[0])) / (n_left * n_right)

        rc = self.line_intersection_2d(ic_left, d_left, ic_right, d_right)
        if rc is None:
            return None
        # Reject only clearly non-physical near-parallel blow-ups, keeping normal regions.
        if sin_theta < 2e-2 and (abs(float(rc[0])) > 5000.0 or abs(float(rc[1])) > 5000.0):
            return None
        return {"rc_2d": rc, "ic_left_2d": ic_left, "ic_right_2d": ic_right,
                "cp_left_2d": cp_left, "cp_right_2d": cp_right,
                "dbg_left": dbg_left, "dbg_right": dbg_right}

    def calculate_roll_center(self, points_left, name=""):
        return self.calculate_roll_center_from_sides(
            points_left, self.mirror_points_y(points_left), name=name)

    def calculate_pitch_center(self):
        return self.calculate_pitch_center_from_sides(
            self.front_left, self.front_right, self.rear_left, self.rear_right)

    # ============================================================
    # Pitch center
    # ============================================================

    def calculate_pitch_center_from_sides(self, front_left, front_right, rear_left, rear_right):
        front_avg = self.average_side_points(front_left, front_right)
        rear_avg  = self.average_side_points(rear_left,  rear_right)

        ic_front, dbg_front = self.calculate_axle_ic_side_view(front_left, label="front_left")
        ic_rear,  dbg_rear  = self.calculate_axle_ic_side_view(rear_left,  label="rear_left")
        if ic_front is None or ic_rear is None:
            return None

        cp_front = self.project_point(front_left["P9"], "xz")
        cp_rear  = self.project_point(rear_left["P9"],  "xz")
        pc = self.line_intersection_2d(ic_front, cp_front - ic_front,
                                       ic_rear,  cp_rear  - ic_rear)
        if pc is None:
            pc = np.array([np.inf, np.inf])

        return {"pc_2d": pc, "ic_front_2d": ic_front, "ic_rear_2d": ic_rear,
                "cp_front_2d": cp_front, "cp_rear_2d": cp_rear,
                "dbg_front": dbg_front, "dbg_rear": dbg_rear,
                "front_avg": front_avg, "rear_avg": rear_avg}

    # ============================================================
    # Alturas
    # ============================================================

    @staticmethod
    def height_from_ground_z(z_value, contact_patch_z):
        return z_value - contact_patch_z

    # ============================================================
    # Antis
    # ============================================================

    def line_z_at_x(self, p1, p2, x_target, eps=1e-12):
        dx = p2[0] - p1[0]
        if abs(dx) < eps:
            return None
        t = (x_target - p1[0]) / dx
        return p1[1] + t * (p2[1] - p1[1])

    def calculate_front_anti_dive(self, x_cg, z_cg):
        ic_fl, _ = self.calculate_axle_ic_side_view(self.front_left, label="front_left")
        ic_fr, _ = self.calculate_axle_ic_side_view(self.front_right, label="front_right")
        if ic_fl is None or ic_fr is None:
            raise ValueError("No se ha podido calcular el IC delantero en vista lateral")
        ic_front = 0.5 * (ic_fl + ic_fr)
        cp_front = 0.5 * (
            self.project_point(self.front_left["P9"], "xz") +
            self.project_point(self.front_right["P9"], "xz")
        )
        z_at_cg  = self.line_z_at_x(cp_front, ic_front, x_cg)
        if z_at_cg is None:
            raise ValueError("La recta CP→IC delantera es vertical")
        h_line = z_at_cg - cp_front[1]
        h_cg   = z_cg    - cp_front[1]
        if h_cg <= 0:
            raise ValueError("El CG debe estar por encima del suelo delantero")
        return {"ic_front": ic_front, "cp_front": cp_front, "x_cg": x_cg, "z_cg": z_cg,
                "z_line_at_cg": z_at_cg, "h_line": h_line, "h_cg": h_cg,
                "anti_dive_pct": (h_line / h_cg) * 100.0}

    def calculate_rear_anti_squat(self, x_cg, z_cg):
        ic_rl, _ = self.calculate_axle_ic_side_view(self.rear_left, label="rear_left")
        ic_rr, _ = self.calculate_axle_ic_side_view(self.rear_right, label="rear_right")
        if ic_rl is None or ic_rr is None:
            raise ValueError("No se ha podido calcular el IC trasero en vista lateral")
        ic_rear = 0.5 * (ic_rl + ic_rr)
        cp_rear = 0.5 * (
            self.project_point(self.rear_left["P9"], "xz") +
            self.project_point(self.rear_right["P9"], "xz")
        )
        z_at_cg = self.line_z_at_x(cp_rear, ic_rear, x_cg)
        if z_at_cg is None:
            raise ValueError("La recta CP→IC trasera es vertical")
        h_line = z_at_cg - cp_rear[1]
        h_cg   = z_cg    - cp_rear[1]
        if h_cg <= 0:
            raise ValueError("El CG debe estar por encima del suelo trasero")
        return {"ic_rear": ic_rear, "cp_rear": cp_rear, "x_cg": x_cg, "z_cg": z_cg,
                "z_line_at_cg": z_at_cg, "h_line": h_line, "h_cg": h_cg,
                "anti_squat_pct": (h_line / h_cg) * 100.0}

    # ============================================================
    # Integración con el solver cinemático
    # ============================================================

    _SOLVER_RESIDUAL_TOL = 1e-6

    @staticmethod
    def _validate_kin_points(kin_points, axle_name="axle"):
        required = ["A","C","D","F","T","B0","E0","S0","W0","N0"]
        missing  = [k for k in required if k not in kin_points]
        if missing:
            raise ValueError(f"Faltan claves en {axle_name}: {missing}")
        for key in required:
            arr = np.asarray(kin_points[key], dtype=float).reshape(3)
            if np.allclose(arr, 0.0):
                raise ValueError(f"El punto '{key}' de {axle_name} está a cero.")

    def build_kin_input(self, kin_points, axle_name="axle"):
        self._validate_kin_points(kin_points, axle_name=axle_name)
        return UprightKinematicsInput(
            A  = np.asarray(kin_points["A"],  dtype=float),
            C  = np.asarray(kin_points["C"],  dtype=float),
            D  = np.asarray(kin_points["D"],  dtype=float),
            F  = np.asarray(kin_points["F"],  dtype=float),
            T  = np.asarray(kin_points["T"],  dtype=float),
            B0 = np.asarray(kin_points["B0"], dtype=float),
            E0 = np.asarray(kin_points["E0"], dtype=float),
            S0 = np.asarray(kin_points["S0"], dtype=float),
            W0 = np.asarray(kin_points["W0"], dtype=float),
            N0 = np.asarray(kin_points["N0"], dtype=float),
            pushrod_wheel_body = kin_points.get("pushrod_wheel_body", "lower_wishbone"),
            W_lca_offset_local = np.asarray(
                kin_points.get("W_lca_offset_local", np.zeros(3)), dtype=float),
        )

    @staticmethod
    def apply_solver_result_to_points(base_points, solver_pos, contact_patch_mode="static_offset"):
        pts        = {k: np.array(v, dtype=float).copy() for k, v in base_points.items()}
        pts["P7"]  = np.asarray(solver_pos["B"], dtype=float).copy()
        pts["P6"]  = np.asarray(solver_pos["E"], dtype=float).copy()
        pts["P10"] = np.asarray(solver_pos["N"], dtype=float).copy()
        if contact_patch_mode == "static_offset":
            cp_offset = (np.asarray(base_points["P9"],  dtype=float) -
                         np.asarray(base_points["P10"], dtype=float))
            pts["P9"] = pts["P10"] + cp_offset
        elif contact_patch_mode == "vertical_only":
            pts["P9"]    = np.asarray(base_points["P9"], dtype=float).copy()
            dz           = pts["P10"][2] - np.asarray(base_points["P10"], dtype=float)[2]
            pts["P9"][2] = np.asarray(base_points["P9"], dtype=float)[2] + dz
        elif contact_patch_mode == "fixed_point":
            pts["P9"] = np.asarray(base_points["P9"], dtype=float).copy()
        else:
            raise ValueError("contact_patch_mode debe ser 'static_offset', 'vertical_only' o 'fixed_point'")
        return pts

    @staticmethod
    def _rotate_point_about_x(point, angle_rad, z_axis_origin):
        x, y, z = np.asarray(point, dtype=float).reshape(3)
        y_rel = y
        z_rel = z - float(z_axis_origin)
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        y_new = y_rel * c - z_rel * s
        z_new = y_rel * s + z_rel * c + float(z_axis_origin)
        return np.array([x, y_new, z_new], dtype=float)

    def _build_corner_opk_like_inputs(self, base_points, kin_points, roll_deg, heave_mm, axle):
        if axle == "front":
            cp_z_ref = self.front_cp_z_ref
        elif axle == "rear":
            cp_z_ref = self.rear_cp_z_ref
        else:
            raise ValueError("axle debe ser 'front' o 'rear'")
        # +roll = compresion; para chasis rotando sobre ruedas fijas equivale a
        # una rotacion negativa del chasis alrededor del eje longitudinal.
        angle = -math.radians(float(roll_deg))
        inboard_keys = ("P1", "P2", "P3", "P4", "P5")
        out_base = {k: np.asarray(v, dtype=float).copy() for k, v in base_points.items()}
        for key in inboard_keys:
            rotated = self._rotate_point_about_x(out_base[key], angle, cp_z_ref)
            rotated[2] -= float(heave_mm)
            out_base[key] = rotated
        out_kin = {k: (np.asarray(v, dtype=float).copy() if isinstance(v, np.ndarray) else v)
                   for k, v in kin_points.items()}
        out_kin["A"] = out_base["P1"].copy()
        out_kin["C"] = out_base["P2"].copy()
        out_kin["D"] = out_base["P3"].copy()
        out_kin["F"] = out_base["P4"].copy()
        out_kin["T"] = out_base["P5"].copy()
        return out_base, out_kin

    def _solve_corner_geometry_fixed_contact_patch(self, base_points, kin_points, state0=None,
                                                   axle_name="corner", zw_seed=0.0):
        cp = np.asarray(base_points["P9"], dtype=float)
        n0 = np.asarray(base_points["P10"], dtype=float)
        target_radius = float(np.linalg.norm(n0 - cp))
        if target_radius < 1e-9:
            raise ValueError(f"Target tyre radius is degenerate for {axle_name}")
        zw = float(zw_seed)
        state_seed = None if state0 is None else np.asarray(state0, dtype=float).reshape(3)
        best = None

        for _ in range(7):
            solved, state_i, info_i, _ = self.solve_corner_geometry(
                base_points, kin_points, zw, state_seed,
                axle_name=axle_name, contact_patch_mode="fixed_point"
            )
            n_i = np.asarray(solved["P10"], dtype=float)
            err_i = float(np.linalg.norm(n_i - cp) - target_radius)
            score_i = abs(err_i) + 0.1 * float(info_i.get("residual_norm", 0.0))
            if (best is None) or (score_i < best[0]):
                best = (score_i, solved, state_i, info_i, zw, err_i)
            if abs(err_i) < 1e-3:
                break

            delta = 0.5
            zw_probe = zw + delta
            solved_p, state_p, info_p, _ = self.solve_corner_geometry(
                base_points, kin_points, zw_probe, state_i,
                axle_name=axle_name, contact_patch_mode="fixed_point"
            )
            n_p = np.asarray(solved_p["P10"], dtype=float)
            err_p = float(np.linalg.norm(n_p - cp) - target_radius)
            deriv = (err_p - err_i) / delta
            if abs(deriv) < 1e-6:
                break
            step = np.clip(err_i / deriv, -15.0, 15.0)
            zw = float(np.clip(zw - step, -120.0, 120.0))
            state_seed = np.asarray(state_i, dtype=float).reshape(3)

        if best is None:
            raise RuntimeError(f"Unable to satisfy fixed contact patch for {axle_name}")
        return best[1], best[2], best[3], best[4]

    def solve_corner_geometry(self, base_points, kin_points, zw, state0=None,
                              axle_name="corner", contact_patch_mode="static_offset"):
        kin_input = self.build_kin_input(kin_points, axle_name=axle_name)

        # Robust solve: try multiple seeds and keep the lowest residual solution.
        # This avoids branch flips near kinematic singularities and improves left/right symmetry.
        seeds = []
        if state0 is not None:
            seeds.append(np.asarray(state0, dtype=float).reshape(3))
        else:
            try:
                _, static_state, _ = solve_upright_for_zw(kin_input, 0.0, x0=None)
                seeds.append(np.asarray(static_state, dtype=float).reshape(3))
            except Exception:
                pass
            seeds.append(np.zeros(3, dtype=float))
            seeds.append(None)

        best = None
        for seed in seeds:
            try:
                solver_pos_i, state_i, info_i = solve_upright_for_zw(kin_input, zw, x0=seed)
                score = float(info_i.get("residual_norm", np.inf))
                if (best is None) or (score < best[0]):
                    best = (score, solver_pos_i, state_i, info_i)
            except Exception:
                continue

        if best is None:
            raise RuntimeError(f"Upright solver failed for {axle_name} at zw={zw}")

        _, solver_pos, state, info = best
        if info["residual_norm"] > self._SOLVER_RESIDUAL_TOL and state0 is not None:
            solver_pos, state, info = solve_upright_for_zw(kin_input, zw, x0=None)
        solved_points = self.apply_solver_result_to_points(base_points, solver_pos, contact_patch_mode)
        return solved_points, state, info, solver_pos

    def solve_axle_geometry(self, base_points, kin_points, zw, state0=None,
                            axle_name="axle", contact_patch_mode="static_offset"):
        return self.solve_corner_geometry(base_points, kin_points, zw, state0,
                                          axle_name, contact_patch_mode)

    def update_average_axles(self):
        self.front = self.average_side_points(self.front_left, self.front_right)
        self.rear  = self.average_side_points(self.rear_left,  self.rear_right)

    # ============================================================
    # Mapa de barrido completo
    # ============================================================

    def generate_4wheel_map(self, hf_values, hr_values, rf_values, rr_values,
                            x_cg, z_cg, contact_patch_mode="static_offset",
                            roll_mode="chassis_rotation", verbose=True):
        rows = []
        state_fl = state_fr = state_rl = state_rr = None
        zw_fl = zw_fr = zw_rl = zw_rr = 0.0
        roll_mode = str(roll_mode).strip().lower()
        if roll_mode != "chassis_rotation":
            raise ValueError("Only 'chassis_rotation' roll_mode is supported in this version.")

        for hf in hf_values:
            for hr in hr_values:
                for rf in rf_values:
                    for rr_val in rr_values:
                        rf_deg = float(rf)
                        rr_deg = float(rr_val)
                        rf_mm_equiv = max(self.front_track_mm, 1e-9) * math.tan(math.radians(rf_deg))
                        rr_mm_equiv = max(self.rear_track_mm, 1e-9) * math.tan(math.radians(rr_deg))
                        zfl = zfr = zrl = zrr = np.nan
                        try:
                            fl_base, fl_kin = self._build_corner_opk_like_inputs(
                                self.front_left_base, self.front_left_kin_points, rf_deg, hf, axle="front"
                            )
                            fr_base, fr_kin = self._build_corner_opk_like_inputs(
                                self.front_right_base, self.front_right_kin_points, rf_deg, hf, axle="front"
                            )
                            rl_base, rl_kin = self._build_corner_opk_like_inputs(
                                self.rear_left_base, self.rear_left_kin_points, rr_deg, hr, axle="rear"
                            )
                            rr_base, rr_kin = self._build_corner_opk_like_inputs(
                                self.rear_right_base, self.rear_right_kin_points, rr_deg, hr, axle="rear"
                            )
                            zfl_cmd = hf + 0.5 * rf_mm_equiv * self.opk_like_roll_zw_ratio_front
                            zfr_cmd = hf - 0.5 * rf_mm_equiv * self.opk_like_roll_zw_ratio_front
                            zrl_cmd = hr + 0.5 * rr_mm_equiv * self.opk_like_roll_zw_ratio_rear
                            zrr_cmd = hr - 0.5 * rr_mm_equiv * self.opk_like_roll_zw_ratio_rear
                            fl, state_fl, info_fl, _ = self.solve_corner_geometry(
                                fl_base, fl_kin, zfl_cmd, state_fl,
                                axle_name="front_left", contact_patch_mode="fixed_point"
                            )
                            fr, state_fr, info_fr, _ = self.solve_corner_geometry(
                                fr_base, fr_kin, zfr_cmd, state_fr,
                                axle_name="front_right", contact_patch_mode="fixed_point"
                            )
                            rl, state_rl, info_rl, _ = self.solve_corner_geometry(
                                rl_base, rl_kin, zrl_cmd, state_rl,
                                axle_name="rear_left", contact_patch_mode="fixed_point"
                            )
                            rr_pts, state_rr, info_rr, _ = self.solve_corner_geometry(
                                rr_base, rr_kin, zrr_cmd, state_rr,
                                axle_name="rear_right", contact_patch_mode="fixed_point"
                            )
                            zfl = float(fl["P10"][2] - self.front_left_base["P10"][2])
                            zfr = float(fr["P10"][2] - self.front_right_base["P10"][2])
                            zrl = float(rl["P10"][2] - self.rear_left_base["P10"][2])
                            zrr = float(rr_pts["P10"][2] - self.rear_right_base["P10"][2])

                            self.front_left  = fl;  self.front_right = fr
                            self.rear_left   = rl;  self.rear_right  = rr_pts
                            self.update_average_axles()

                            front_rc   = self.calculate_roll_center_from_sides(fl, fr, name="FRONT")
                            rear_rc    = self.calculate_roll_center_from_sides(rl, rr_pts, name="REAR")
                            pitch      = self.calculate_pitch_center_from_sides(fl, fr, rl, rr_pts)
                            anti_dive  = self.calculate_front_anti_dive(x_cg, z_cg)
                            anti_squat = self.calculate_rear_anti_squat(x_cg, z_cg)
                            # Use axle-ground reference (average contact patch Z left/right).
                            # Using only one side introduces artificial asymmetry under roll.
                            front_cp_z_ref = 0.5 * (fl["P9"][2] + fr["P9"][2])
                            rear_cp_z_ref  = 0.5 * (rl["P9"][2] + rr_pts["P9"][2])

                            rows.append({
                                "hf": hf, "hr": hr, "rf": rf, "rr": rr_val,
                                "zfl": zfl, "zfr": zfr, "zrl": zrl, "zrr": zrr,
                                "rf_deg": rf_deg, "rr_deg": rr_deg,
                                "rf_mm_equiv": rf_mm_equiv, "rr_mm_equiv": rr_mm_equiv,
                                "roll_mode": roll_mode,
                                "front_rc_y":      np.nan if front_rc is None else front_rc["rc_2d"][0],
                                "front_rc_z":      np.nan if front_rc is None else front_rc["rc_2d"][1],
                                "front_rc_height": np.nan if front_rc is None else self.height_from_ground_z(front_rc["rc_2d"][1], front_cp_z_ref),
                                "front_ic_left_y": np.nan if front_rc is None else front_rc["ic_left_2d"][0],
                                "front_ic_left_z": np.nan if front_rc is None else front_rc["ic_left_2d"][1],
                                "front_ic_right_y": np.nan if front_rc is None else front_rc["ic_right_2d"][0],
                                "front_ic_right_z": np.nan if front_rc is None else front_rc["ic_right_2d"][1],
                                "rear_rc_y":       np.nan if rear_rc  is None else rear_rc["rc_2d"][0],
                                "rear_rc_z":       np.nan if rear_rc  is None else rear_rc["rc_2d"][1],
                                "rear_rc_height":  np.nan if rear_rc  is None else self.height_from_ground_z(rear_rc["rc_2d"][1],  rear_cp_z_ref),
                                "rear_ic_left_y": np.nan if rear_rc is None else rear_rc["ic_left_2d"][0],
                                "rear_ic_left_z": np.nan if rear_rc is None else rear_rc["ic_left_2d"][1],
                                "rear_ic_right_y": np.nan if rear_rc is None else rear_rc["ic_right_2d"][0],
                                "rear_ic_right_z": np.nan if rear_rc is None else rear_rc["ic_right_2d"][1],
                                "pitch_x":         np.nan if pitch    is None else pitch["pc_2d"][0],
                                "pitch_z":         np.nan if pitch    is None else pitch["pc_2d"][1],
                                "pitch_ic_front_x": np.nan if pitch is None else pitch["ic_front_2d"][0],
                                "pitch_ic_front_z": np.nan if pitch is None else pitch["ic_front_2d"][1],
                                "pitch_ic_rear_x": np.nan if pitch is None else pitch["ic_rear_2d"][0],
                                "pitch_ic_rear_z": np.nan if pitch is None else pitch["ic_rear_2d"][1],
                                "anti_dive_pct":   anti_dive["anti_dive_pct"],
                                "anti_squat_pct":  anti_squat["anti_squat_pct"],
                                "success_fl": bool(info_fl["success"]), "success_fr": bool(info_fr["success"]),
                                "success_rl": bool(info_rl["success"]), "success_rr": bool(info_rr["success"]),
                                "res_fl": float(info_fl["residual_norm"]), "res_fr": float(info_fr["residual_norm"]),
                                "res_rl": float(info_rl["residual_norm"]), "res_rr": float(info_rr["residual_norm"]),
                            })

                            if verbose:
                                frc_h = np.nan if front_rc is None else self.height_from_ground_z(front_rc["rc_2d"][1], front_cp_z_ref)
                                rrc_h = np.nan if rear_rc  is None else self.height_from_ground_z(rear_rc["rc_2d"][1],  rear_cp_z_ref)
                                pc_z  = np.nan if pitch    is None else pitch["pc_2d"][1]
                                print(f"hf={hf:7.2f} hr={hr:7.2f} rf={rf:7.2f} rr={rr_val:7.2f} | "
                                      f"FRC={frc_h:8.3f} RRC={rrc_h:8.3f} PCz={pc_z:8.3f}")

                        except Exception as exc:
                            state_fl = state_fr = state_rl = state_rr = None
                            zw_fl = zw_fr = zw_rl = zw_rr = 0.0
                            rows.append({"hf": hf, "hr": hr, "rf": rf, "rr": rr_val,
                                         "zfl": zfl, "zfr": zfr, "zrl": zrl, "zrr": zrr,
                                         "rf_deg": rf_deg if "rf_deg" in locals() else np.nan,
                                         "rr_deg": rr_deg if "rr_deg" in locals() else np.nan,
                                         "rf_mm_equiv": rf_mm_equiv if "rf_mm_equiv" in locals() else np.nan,
                                         "rr_mm_equiv": rr_mm_equiv if "rr_mm_equiv" in locals() else np.nan,
                                         "roll_mode": roll_mode,
                                         "error": str(exc)})
                            if verbose:
                                print(f"[ERROR] hf={hf}, hr={hr}, rf={rf}, rr={rr_val} -> {exc}")

        return pd.DataFrame(rows)

    # ============================================================
    # Informes
    # ============================================================

    def run_report(self, manual_front_rc=None, manual_rear_rc=None):
        print("\n" + "=" * 72)
        print("CÁLCULO EXACTO DE ROLL CENTER Y PITCH CENTER")
        print("=" * 72)

        print("\n--- ROLL CENTER DELANTERO ---")
        front_rc = self.calculate_roll_center_from_sides(self.front_left, self.front_right, name="FRONT")
        if front_rc is None:
            print("No se pudo calcular el RC delantero")
        else:
            rcf     = front_rc["rc_2d"]
            h_front = self.height_from_ground_z(rcf[1], self.front["P9"][2])
            print(f"IC izquierdo (YZ): Y={front_rc['ic_left_2d'][0]:.3f}  Z={front_rc['ic_left_2d'][1]:.3f}")
            print(f"IC derecho   (YZ): Y={front_rc['ic_right_2d'][0]:.3f}  Z={front_rc['ic_right_2d'][1]:.3f}")
            print(f"RC delantero (YZ): Y={rcf[0]:.3f}  Z={rcf[1]:.3f}")
            print(f"Altura RC delantera sobre suelo = {h_front:.3f} mm")
            if manual_front_rc is not None:
                print(f"Manual = {manual_front_rc:.3f} mm  |  Diferencia = {h_front - manual_front_rc:.3f} mm")

        print("\n--- ROLL CENTER TRASERO ---")
        rear_rc = self.calculate_roll_center_from_sides(self.rear_left, self.rear_right, name="REAR")
        if rear_rc is None:
            print("No se pudo calcular el RC trasero")
        else:
            rcr    = rear_rc["rc_2d"]
            h_rear = self.height_from_ground_z(rcr[1], self.rear["P9"][2])
            print(f"IC izquierdo (YZ): Y={rear_rc['ic_left_2d'][0]:.3f}  Z={rear_rc['ic_left_2d'][1]:.3f}")
            print(f"IC derecho   (YZ): Y={rear_rc['ic_right_2d'][0]:.3f}  Z={rear_rc['ic_right_2d'][1]:.3f}")
            print(f"RC trasero   (YZ): Y={rcr[0]:.3f}  Z={rcr[1]:.3f}")
            print(f"Altura RC trasera sobre suelo = {h_rear:.3f} mm")
            if manual_rear_rc is not None:
                print(f"Manual = {manual_rear_rc:.3f} mm  |  Diferencia = {h_rear - manual_rear_rc:.3f} mm")

        print("\n--- PITCH CENTER ---")
        pitch = self.calculate_pitch_center_from_sides(
            self.front_left, self.front_right, self.rear_left, self.rear_right)
        if pitch is None:
            print("No se pudo calcular el pitch center")
        else:
            pc = pitch["pc_2d"]
            print(f"IC delantero (XZ): X={pitch['ic_front_2d'][0]:.3f}  Z={pitch['ic_front_2d'][1]:.3f}")
            print(f"IC trasero   (XZ): X={pitch['ic_rear_2d'][0]:.3f}  Z={pitch['ic_rear_2d'][1]:.3f}")
            print(f"PC           (XZ): X={pc[0]:.3f}  Z={pc[1]:.3f}")
            print(f"Altura PC respecto al suelo delantero = {self.height_from_ground_z(pc[1], self.front['P9'][2]):.3f} mm")
            print(f"Altura PC respecto al suelo trasero   = {self.height_from_ground_z(pc[1], self.rear['P9'][2]):.3f} mm")

        return front_rc, rear_rc, pitch

    def report_dive_squat(self, x_cg, z_cg):
        front = self.calculate_front_anti_dive(x_cg, z_cg)
        rear  = self.calculate_rear_anti_squat(x_cg, z_cg)
        print("\n" + "=" * 72)
        print("ANTI-DIVE / ANTI-SQUAT")
        print("=" * 72)
        print(f"\nCG = X {x_cg:.3f}, Z {z_cg:.3f}")
        print("\n--- FRONT ANTI-DIVE ---")
        print(f"IC front         = X {front['ic_front'][0]:.3f}, Z {front['ic_front'][1]:.3f}")
        print(f"CP front         = X {front['cp_front'][0]:.3f}, Z {front['cp_front'][1]:.3f}")
        print(f"Z line at CG     = {front['z_line_at_cg']:.3f}")
        print(f"h_line           = {front['h_line']:.3f} mm")
        print(f"h_CG             = {front['h_cg']:.3f} mm")
        print(f"Anti-dive %      = {front['anti_dive_pct']:.3f} %")
        print("\n--- REAR ANTI-SQUAT ---")
        print(f"IC rear          = X {rear['ic_rear'][0]:.3f}, Z {rear['ic_rear'][1]:.3f}")
        print(f"CP rear          = X {rear['cp_rear'][0]:.3f}, Z {rear['cp_rear'][1]:.3f}")
        print(f"Z line at CG     = {rear['z_line_at_cg']:.3f}")
        print(f"h_line           = {rear['h_line']:.3f} mm")
        print(f"h_CG             = {rear['h_cg']:.3f} mm")
        print(f"Anti-squat %     = {rear['anti_squat_pct']:.3f} %")
        return {"front": front, "rear": rear}

    # ============================================================
    # Plots
    # ============================================================

    def plot_roll_center(self, rc_result, points_left, title):
        if rc_result is None:
            print(f"No hay datos para graficar {title}"); return
        fig, ax = plt.subplots(figsize=(8, 6))
        points_right = self.mirror_points_y(points_left)
        cp_left  = rc_result["cp_left_2d"];  cp_right = rc_result["cp_right_2d"]
        ic_left  = rc_result["ic_left_2d"];  ic_right = rc_result["ic_right_2d"]
        rc       = rc_result["rc_2d"]
        p6l = self.project_point(points_left["P6"],  "yz"); p7l = self.project_point(points_left["P7"],  "yz")
        p6r = self.project_point(points_right["P6"], "yz"); p7r = self.project_point(points_right["P7"], "yz")
        dbg_l = rc_result["dbg_left"]; dbg_r = rc_result["dbg_right"]

        def draw_long_line(p, d, xlim):
            xs = np.array(xlim)
            if abs(d[0]) < 1e-10:
                ax.plot(np.full(2, p[0]), [p[1]-2000, p[1]+2000], "--", linewidth=1.5)
            else:
                t = (xs - p[0]) / d[0]
                ax.plot(xs, p[1] + t * d[1], "--", linewidth=1.5)

        xlim = [-9000, 9000]
        ax.scatter(p6l[0], p6l[1], c="purple", s=80, label="P6 L")
        ax.scatter(p7l[0], p7l[1], c="brown",  s=80, label="P7 L")
        ax.scatter(p6r[0], p6r[1], c="purple", s=80, marker="s", label="P6 R")
        ax.scatter(p7r[0], p7r[1], c="brown",  s=80, marker="s", label="P7 R")
        ax.scatter(cp_left[0],  cp_left[1],  c="black", s=140, marker="*", label="CP L")
        ax.scatter(cp_right[0], cp_right[1], c="gray",  s=140, marker="*", label="CP R")
        for dbg in (dbg_l, dbg_r):
            draw_long_line(dbg["lower_p"], dbg["lower_d"], xlim)
            draw_long_line(dbg["upper_p"], dbg["upper_d"], xlim)
        ax.plot([ic_left[0],  cp_left[0]],  [ic_left[1],  cp_left[1]],  "g-", linewidth=2)
        ax.plot([ic_right[0], cp_right[0]], [ic_right[1], cp_right[1]], "g-", linewidth=2)
        ax.scatter(ic_left[0],  ic_left[1],  c="green",     s=180, marker="X", label="IC L")
        ax.scatter(ic_right[0], ic_right[1], c="limegreen", s=180, marker="X", label="IC R")
        ax.scatter(rc[0], rc[1], c="red", s=220, marker="D", label="RC")
        h = self.height_from_ground_z(rc[1], points_left["P9"][2])
        ax.set_title(f"{title}\nAltura RC = {h:.3f} mm")
        ax.set_xlabel("Y [mm]"); ax.set_ylabel("Z [mm]")
        ax.grid(True, alpha=0.3); ax.axhline(y=0, color="k", alpha=0.2); ax.axvline(x=0, color="k", alpha=0.2)
        ax.legend(fontsize=8); plt.tight_layout(); plt.show()

    def plot_pitch_center(self, pitch_result):
        if pitch_result is None:
            print("No hay datos para graficar el pitch center"); return
        fig, ax = plt.subplots(figsize=(9, 6))
        ic_front = pitch_result["ic_front_2d"]; ic_rear  = pitch_result["ic_rear_2d"]
        cp_front = pitch_result["cp_front_2d"]; cp_rear  = pitch_result["cp_rear_2d"]
        pc = pitch_result["pc_2d"]
        dbg_f = pitch_result["dbg_front"]; dbg_r = pitch_result["dbg_rear"]
        p6f = self.project_point(self.front["P6"], "xz"); p7f = self.project_point(self.front["P7"], "xz")
        p6r = self.project_point(self.rear["P6"],  "xz"); p7r = self.project_point(self.rear["P7"],  "xz")

        def draw_long_line(p, d, xlim):
            xs = np.array(xlim)
            if abs(d[0]) < 1e-10:
                ax.plot(np.full(2, p[0]), [p[1]-2000, p[1]+2000], "--", linewidth=1.5)
            else:
                t = (xs - p[0]) / d[0]
                ax.plot(xs, p[1] + t * d[1], "--", linewidth=1.5)

        xlim = [-500, self.wheelbase + 1000]
        ax.scatter(p6f[0], p6f[1], c="purple", s=80, label="P6 front")
        ax.scatter(p7f[0], p7f[1], c="brown",  s=80, label="P7 front")
        ax.scatter(p6r[0], p6r[1], c="purple", s=80, marker="s", label="P6 rear")
        ax.scatter(p7r[0], p7r[1], c="brown",  s=80, marker="s", label="P7 rear")
        ax.scatter(cp_front[0], cp_front[1], c="black", s=140, marker="*", label="CP front")
        ax.scatter(cp_rear[0],  cp_rear[1],  c="gray",  s=140, marker="*", label="CP rear")
        for dbg in (dbg_f, dbg_r):
            draw_long_line(dbg["lower_p"], dbg["lower_d"], xlim)
            draw_long_line(dbg["upper_p"], dbg["upper_d"], xlim)
        ax.plot([ic_front[0], cp_front[0]], [ic_front[1], cp_front[1]], "g-", linewidth=2)
        ax.plot([ic_rear[0],  cp_rear[0]],  [ic_rear[1],  cp_rear[1]], "g-", linewidth=2)
        ax.scatter(ic_front[0], ic_front[1], c="green",     s=180, marker="X", label="IC front")
        ax.scatter(ic_rear[0],  ic_rear[1],  c="limegreen", s=180, marker="X", label="IC rear")
        ax.scatter(pc[0], pc[1], c="red", s=220, marker="D", label="Pitch Center")
        h_front = self.height_from_ground_z(pc[1], self.front["P9"][2])
        h_rear  = self.height_from_ground_z(pc[1], self.rear["P9"][2])
        ax.set_title(f"Pitch Center\nAltura suelo delantero = {h_front:.3f} mm | trasero = {h_rear:.3f} mm")
        ax.set_xlabel("X [mm]"); ax.set_ylabel("Z [mm]")
        ax.grid(True, alpha=0.3); ax.axhline(y=0, color="k", alpha=0.2)
        ax.legend(fontsize=8); plt.tight_layout(); plt.show()


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "input.json"

    print(f"Cargando geometría desde: {json_path}")
    susp    = SuspensionGeometryExact(json_path=json_path)
    vehicle = susp._vehicle_params

    x_cg = vehicle["x_cg_mm"]
    z_cg = vehicle["z_cg_mm"]

    print(f"  Wheelbase : {susp.wheelbase:.1f} mm")
    print(f"  Masa      : {vehicle['mass_kg']:.1f} kg")
    print(f"  Balance F : {vehicle['weight_balance_f']*100:.2f} %")
    print(f"  x_CG      : {x_cg:.1f} mm  |  z_CG : {z_cg:.1f} mm")
    print(f"  Ride F/R  : {vehicle['ride_height_f_mm']:.1f} / {vehicle['ride_height_r_mm']:.1f} mm")
    print(f"  Pushrod F : {susp.front_left_kin_points['pushrod_wheel_body']}")
    print(f"  Pushrod R : {susp.rear_left_kin_points['pushrod_wheel_body']}")

    susp.update_average_axles()

    front_rc, rear_rc, pitch = susp.run_report()
    susp.report_dive_squat(x_cg, z_cg)

    # Caso 1: roll delantero
    df_roll_front = susp.generate_4wheel_map(
        hf_values=[0.0], hr_values=[0.0],
        rf_values=np.arange(-40.0, 41.0, 5.0), rr_values=[0.0],
        x_cg=x_cg, z_cg=z_cg, contact_patch_mode="static_offset", verbose=True,
    )
    df_roll_front.to_csv("map_roll_front.csv", index=False)

    # Caso 2: roll trasero
    df_roll_rear = susp.generate_4wheel_map(
        hf_values=[0.0], hr_values=[0.0],
        rf_values=[0.0], rr_values=np.arange(-40.0, 41.0, 5.0),
        x_cg=x_cg, z_cg=z_cg, contact_patch_mode="static_offset", verbose=True,
    )
    df_roll_rear.to_csv("map_roll_rear.csv", index=False)

    # Caso 3: pitch
    df_pitch = susp.generate_4wheel_map(
        hf_values=np.arange(-20.0, 21.0, 5.0), hr_values=np.arange(-20.0, 21.0, 5.0),
        rf_values=[0.0], rr_values=[0.0],
        x_cg=x_cg, z_cg=z_cg, contact_patch_mode="static_offset", verbose=True,
    )
    df_pitch.to_csv("map_pitch.csv", index=False)

    print("\nGuardados: map_roll_front.csv, map_roll_rear.csv, map_pitch.csv")
