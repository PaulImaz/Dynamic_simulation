"""
Title: Motion Ratio Tool

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

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from upright_solver import (
    UprightKinematicsInput,
    compute_lower_wishbone_local_offset,
    solve_upright_for_zw,
)


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n < 1e-15:
        raise ValueError("Vector de norma casi cero: no se puede normalizar.")
    return v / n


def rotate_point_about_axis_rodrigues(
    X0: np.ndarray,
    O: np.ndarray,
    k_hat: np.ndarray,
    omega: float,
) -> np.ndarray:
    r = X0 - O
    c = math.cos(omega)
    s = math.sin(omega)
    r_rot = r * c + np.cross(k_hat, r) * s + k_hat * (float(np.dot(k_hat, r)) * (1.0 - c))
    return O + r_rot


@dataclass
class RockerGeometry:
    axis_point: np.ndarray
    axis_dir: np.ndarray
    pushrod_point_rocker_0: np.ndarray
    damper_point_rocker_0: np.ndarray
    damper_point_chassis: np.ndarray
    pushrod_point_upright_0: np.ndarray

    def __post_init__(self):
        self.axis_point = np.array(self.axis_point, dtype=float).reshape(3)
        self.axis_dir = np.array(self.axis_dir, dtype=float).reshape(3)
        self.pushrod_point_rocker_0 = np.array(self.pushrod_point_rocker_0, dtype=float).reshape(3)
        self.damper_point_rocker_0 = np.array(self.damper_point_rocker_0, dtype=float).reshape(3)
        self.damper_point_chassis = np.array(self.damper_point_chassis, dtype=float).reshape(3)
        self.pushrod_point_upright_0 = np.array(self.pushrod_point_upright_0, dtype=float).reshape(3)

    @property
    def O(self) -> np.ndarray:
        return self.axis_point

    @property
    def k_hat(self) -> np.ndarray:
        return unit(self.axis_dir)

    @property
    def Pb0(self) -> np.ndarray:
        return self.pushrod_point_rocker_0

    @property
    def Db0(self) -> np.ndarray:
        return self.damper_point_rocker_0

    @property
    def Dc(self) -> np.ndarray:
        return self.damper_point_chassis

    @property
    def Pu0(self) -> np.ndarray:
        return self.pushrod_point_upright_0

    @property
    def Lp(self) -> float:
        return norm(self.Pu0 - self.Pb0)

    def distance_to_axis(self, X: np.ndarray) -> float:
        X = np.array(X, dtype=float).reshape(3)
        r = X - self.O
        proj = float(np.dot(r, self.k_hat)) * self.k_hat
        perp = r - proj
        return norm(perp)


def point_on_rocker(geom: RockerGeometry, local_point_0: np.ndarray, omega: float) -> np.ndarray:
    return rotate_point_about_axis_rodrigues(local_point_0, geom.O, geom.k_hat, omega)


def pushrod_point_on_rocker(geom: RockerGeometry, omega: float) -> np.ndarray:
    return point_on_rocker(geom, geom.Pb0, omega)


def damper_point_on_rocker(geom: RockerGeometry, omega: float) -> np.ndarray:
    return point_on_rocker(geom, geom.Db0, omega)


def pushrod_length_error(omega: float, geom: RockerGeometry, Pu: np.ndarray) -> float:
    Pb = pushrod_point_on_rocker(geom, omega)
    return norm(Pu - Pb) - geom.Lp


def reachability_interval_for_Pb(
    geom: RockerGeometry,
    Pu: np.ndarray,
) -> Tuple[float, float, float, np.ndarray]:
    k_hat = geom.k_hat
    O = geom.O
    Pb0 = geom.Pb0

    w = Pb0 - O
    C = O + k_hat * float(np.dot(k_hat, w))
    r = norm(Pb0 - C)

    u = Pu - C
    u_par = k_hat * float(np.dot(k_hat, u))
    u_perp = u - u_par

    upar = norm(u_par)
    uperp = norm(u_perp)

    dmin = math.sqrt(upar ** 2 + (uperp - r) ** 2)
    dmax = math.sqrt(upar ** 2 + (uperp + r) ** 2)
    return dmin, dmax, r, C


def _wrap_near(target: float, angle: float) -> float:
    two_pi = 2.0 * math.pi
    return angle + two_pi * round((target - angle) / two_pi)


def instantaneous_pushrod_tangent(geom: RockerGeometry, omega: float) -> np.ndarray:
    Qw = pushrod_point_on_rocker(geom, omega)
    return np.cross(geom.k_hat, Qw - geom.O)


def choose_omega_branch(
    geom: RockerGeometry,
    Pu: np.ndarray,
    omega_candidates: Tuple[float, float],
    omega_guess: float,
    tol_m: float = 1e-6,
    Pu_prev: Optional[np.ndarray] = None,
    omega_prev: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    o1, o2 = omega_candidates
    o1n = _wrap_near(omega_guess, o1)
    o2n = _wrap_near(omega_guess, o2)

    f1 = abs(pushrod_length_error(o1n, geom, Pu))
    f2 = abs(pushrod_length_error(o2n, geom, Pu))
    c1 = abs(o1n - omega_guess)
    c2 = abs(o2n - omega_guess)

    sign_metric = 0.0
    expected_sign = 0.0
    sign1_ok = 1.0
    sign2_ok = 1.0

    if Pu_prev is not None and omega_prev is not None:
        dPu = np.array(Pu, dtype=float) - np.array(Pu_prev, dtype=float)
        tangent_prev = instantaneous_pushrod_tangent(geom, omega_prev)
        sign_metric = float(np.dot(dPu, tangent_prev))

        if abs(sign_metric) > 1e-12:
            expected_sign = math.copysign(1.0, sign_metric)
            d1 = o1n - omega_prev
            d2 = o2n - omega_prev
            sign1 = 0.0 if abs(d1) < 1e-12 else math.copysign(1.0, d1)
            sign2 = 0.0 if abs(d2) < 1e-12 else math.copysign(1.0, d2)
            sign1_ok = 1.0 if (sign1 == 0.0 or sign1 == expected_sign) else 0.0
            sign2_ok = 1.0 if (sign2 == 0.0 or sign2 == expected_sign) else 0.0

    if sign1_ok != sign2_ok:
        chosen = o1n if sign1_ok > sign2_ok else o2n
    else:
        if abs(f1 - f2) > tol_m:
            chosen = o1n if f1 < f2 else o2n
        else:
            chosen = o1n if c1 <= c2 else o2n

    info = {
        "candidate1_deg": math.degrees(o1n),
        "candidate2_deg": math.degrees(o2n),
        "closure_err_1_m": f1,
        "closure_err_2_m": f2,
        "continuity_cost_1_rad": c1,
        "continuity_cost_2_rad": c2,
        "sign_metric": sign_metric,
        "expected_sign": expected_sign,
        "candidate1_sign_ok": sign1_ok,
        "candidate2_sign_ok": sign2_ok,
        "chosen_deg": math.degrees(chosen),
    }
    return chosen, info


def solve_omega_analytic(
    geom: RockerGeometry,
    Pu: np.ndarray,
    omega_guess: float = 0.0,
    tol_m: float = 1e-6,
    Pu_prev: Optional[np.ndarray] = None,
    omega_prev: Optional[float] = None,
    return_info: bool = False,
):
    k_hat = geom.k_hat
    O = geom.O
    Pb0 = geom.Pb0
    Lp = geom.Lp

    dmin, dmax, r, C = reachability_interval_for_Pb(geom, Pu)
    if Lp < dmin - 1e-8 or Lp > dmax + 1e-8:
        raise RuntimeError(
            "No existe solución analítica: longitud del pushrod fuera del rango alcanzable. "
            f"Lp={Lp:.6e}, dmin={dmin:.6e}, dmax={dmax:.6e}"
        )

    if r < 1e-12:
        if return_info:
            return omega_guess, {"degenerate_radius": True, "r_m": r}
        return omega_guess

    e1 = (Pb0 - C) / r
    e2 = unit(np.cross(k_hat, e1))

    u = Pu - C
    a = float(np.dot(u, e1))
    b = float(np.dot(u, e2))
    R = math.sqrt(a * a + b * b)

    if R < 1e-12:
        if return_info:
            return omega_guess, {"degenerate_projection": True, "R_m": R}
        return omega_guess

    s = (norm(u) ** 2 + r ** 2 - Lp ** 2) / (2.0 * r)
    x = max(-1.0, min(1.0, s / R))

    phi = math.atan2(b, a)
    delta = math.acos(x)
    o1 = phi + delta
    o2 = phi - delta

    chosen, branch_info = choose_omega_branch(
        geom=geom,
        Pu=Pu,
        omega_candidates=(o1, o2),
        omega_guess=omega_guess,
        tol_m=tol_m,
        Pu_prev=Pu_prev,
        omega_prev=omega_prev,
    )

    info = {
        "dmin_m": dmin,
        "dmax_m": dmax,
        "orbit_radius_m": r,
        "circle_center_x_m": float(C[0]),
        "circle_center_y_m": float(C[1]),
        "circle_center_z_m": float(C[2]),
        "solver_phi_deg": math.degrees(phi),
        "solver_delta_deg": math.degrees(delta),
        **branch_info,
    }

    if return_info:
        return chosen, info
    return chosen


def damper_length(geom: RockerGeometry, omega: float) -> float:
    Db = damper_point_on_rocker(geom, omega)
    return norm(geom.Dc - Db)


def rocker_diagnostics(geom: RockerGeometry, omega: float, Pu: np.ndarray) -> Dict[str, float]:
    Qw = pushrod_point_on_rocker(geom, omega)
    Vw = damper_point_on_rocker(geom, omega)
    tangent_Q = instantaneous_pushrod_tangent(geom, omega)
    tangent_V = np.cross(geom.k_hat, Vw - geom.O)
    damper_vec = geom.Dc - Vw
    damper_len = norm(damper_vec)

    dLd_domega = float("nan")
    if damper_len > 1e-15:
        u_hat = damper_vec / damper_len
        dLd_domega = float(np.dot(u_hat, -tangent_V))

    pushrod_error = pushrod_length_error(omega, geom, Pu)

    return {
        "Q_x_m": float(Qw[0]),
        "Q_y_m": float(Qw[1]),
        "Q_z_m": float(Qw[2]),
        "V_x_m": float(Vw[0]),
        "V_y_m": float(Vw[1]),
        "V_z_m": float(Vw[2]),
        "pushrod_closure_error_m": float(pushrod_error),
        "tangent_Q_norm_m_per_rad": float(norm(tangent_Q)),
        "tangent_V_norm_m_per_rad": float(norm(tangent_V)),
        "dLd_domega_m_per_rad": dLd_domega,
    }


@dataclass
class MotionRatioMap:
    """
    Mapa tabulado para usar luego dentro de 7posrig.
    Base:
      - zw_grid_m: desplazamiento wheel-center respecto al estático importado
      - s_grid_m: desplazamiento damper
      - mr_ds_dzw_grid: ds/dzw
      - mr_dzw_ds_grid: dzw/ds
    """
    zw_grid_m: np.ndarray
    s_grid_m: np.ndarray
    mr_ds_dzw_grid: np.ndarray
    mr_dzw_ds_grid: np.ndarray

    def __post_init__(self):
        self.zw_grid_m = np.asarray(self.zw_grid_m, dtype=float)
        self.s_grid_m = np.asarray(self.s_grid_m, dtype=float)
        self.mr_ds_dzw_grid = np.asarray(self.mr_ds_dzw_grid, dtype=float)
        self.mr_dzw_ds_grid = np.asarray(self.mr_dzw_ds_grid, dtype=float)

        if len(self.zw_grid_m) < 2:
            raise ValueError("MotionRatioMap necesita al menos 2 puntos.")
        if not np.all(np.diff(self.zw_grid_m) > 0.0):
            raise ValueError("zw_grid_m debe estar estrictamente ordenado de menor a mayor.")

        self.zw_min_m = float(self.zw_grid_m[0])
        self.zw_max_m = float(self.zw_grid_m[-1])

    def _interp_clamped(self, x: float, xp: np.ndarray, fp: np.ndarray) -> float:
        x_clamped = min(max(float(x), self.zw_min_m), self.zw_max_m)
        return float(np.interp(x_clamped, xp, fp))

    def eval_s(self, zw_m: float) -> float:
        return self._interp_clamped(zw_m, self.zw_grid_m, self.s_grid_m)

    def eval_mr_ds_dzw(self, zw_m: float) -> float:
        return self._interp_clamped(zw_m, self.zw_grid_m, self.mr_ds_dzw_grid)

    def eval_mr_dzw_ds(self, zw_m: float) -> float:
        return self._interp_clamped(zw_m, self.zw_grid_m, self.mr_dzw_ds_grid)

    def eval_s_dot_from_zw_dot(self, zw_m: float, zw_dot_mps: float) -> float:
        return self.eval_mr_ds_dzw(zw_m) * float(zw_dot_mps)

    def eval_wheel_force_from_spring_force(self, zw_m: float, spring_force_n: float) -> float:
        return float(spring_force_n) * self.eval_mr_ds_dzw(zw_m)

    def eval_wheel_stiffness_from_spring_stiffness(self, zw_m: float, spring_stiffness_npm: float) -> float:
        jac = self.eval_mr_ds_dzw(zw_m)
        return float(spring_stiffness_npm) * jac * jac

    def to_dict(self) -> Dict[str, list]:
        return {
            "zw_grid_m": self.zw_grid_m.tolist(),
            "s_grid_m": self.s_grid_m.tolist(),
            "mr_ds_dzw_grid": self.mr_ds_dzw_grid.tolist(),
            "mr_dzw_ds_grid": self.mr_dzw_ds_grid.tolist(),
        }


def _as_point_m(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(3)
    if float(np.max(np.abs(arr))) > 20.0:
        arr = arr * 1e-3
    return arr


def _infer_pushrod_wheel_body(suspension_name: str) -> str:
    name = str(suspension_name).strip().lower()
    if "pushrod on upright" in name or "upright" in name:
        return "upright"
    if "lower wishbone" in name or "lower_wishbone" in name:
        return "lower_wishbone"
    return "upright"


def _compute_discrete_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    dydx = np.full(n, np.nan, dtype=float)
    if n < 2:
        return dydx
    if n == 2:
        dx = x[1] - x[0]
        if abs(dx) > 1e-15:
            dydx[:] = (y[1] - y[0]) / dx
        return dydx
    dx0 = x[1] - x[0]
    dx1 = x[-1] - x[-2]
    if abs(dx0) > 1e-15:
        dydx[0] = (y[1] - y[0]) / dx0
    if abs(dx1) > 1e-15:
        dydx[-1] = (y[-1] - y[-2]) / dx1
    for i in range(1, n - 1):
        dx = x[i + 1] - x[i - 1]
        if abs(dx) > 1e-15:
            dydx[i] = (y[i + 1] - y[i - 1]) / dx
    return dydx


def _polyfit_with_rms(x: np.ndarray, y: np.ndarray, degree: int):
    coeffs = np.polyfit(x, y, degree)
    y_fit = np.polyval(coeffs, x)
    rms = float(np.sqrt(np.mean((y - y_fit) ** 2)))
    return coeffs, rms


def _parse_axle_for_mr(data: dict, axle: str) -> Dict:
    sus = data["config"]["suspension"][axle]
    ext = sus["external"]["pickUpPts"]
    intl = sus["internal"]["pickUpPts"]

    suspension_name = str(ext.get("name", ""))
    pushrod_wheel_body = _infer_pushrod_wheel_body(suspension_name)
    fallback_mr_wd = sus.get("internal", {}).get("spring", {}).get("MR_WD", None)

    upright_points = {
        "A": _as_point_m(ext["rFLWBI"]),
        "B0": _as_point_m(ext["rFLWBO"]),
        "C": _as_point_m(ext["rRLWBI"]),
        "D": _as_point_m(ext["rFUWBI"]),
        "E0": _as_point_m(ext["rFUWBO"]),
        "F": _as_point_m(ext["rRUWBI"]),
        "T": _as_point_m(ext["rTRI"]),
        "S0": _as_point_m(ext["rTRO"]),
        "W0": _as_point_m(ext["rPRO"]),
        "N0": _as_point_m(ext["rUserTCP"]),
    }

    rocker_points = {
        "axis_point": _as_point_m(ext["rRockerC"]),
        "axis_dir_point": _as_point_m(ext["rRockerAxis"]),
        "pushrod_point_rocker_0": _as_point_m(ext["rPRI"]),
        "damper_point_rocker_0": _as_point_m(intl["rCornerDamper"]),
        "damper_point_chassis": _as_point_m(intl["rCornerDamperChassis"]),
    }

    if pushrod_wheel_body == "lower_wishbone":
        w_lca_offset = compute_lower_wishbone_local_offset(
            A=upright_points["A"], C=upright_points["C"],
            B0=upright_points["B0"], W0=upright_points["W0"],
        )
    else:
        w_lca_offset = np.zeros(3, dtype=float)

    return {
        "axle": axle,
        "suspension_name": suspension_name,
        "pushrod_wheel_body": pushrod_wheel_body,
        "fallback_mr_wd": fallback_mr_wd,
        "upright_points": upright_points,
        "rocker_points": rocker_points,
        "W_lca_offset_local_m": w_lca_offset,
    }


def _build_models_for_mr(parsed_axle: Dict):
    up = parsed_axle["upright_points"]
    rk = parsed_axle["rocker_points"]

    upright = UprightKinematicsInput(
        A=up["A"], C=up["C"], D=up["D"], F=up["F"], T=up["T"],
        B0=up["B0"], E0=up["E0"], S0=up["S0"], W0=up["W0"], N0=up["N0"],
        pushrod_wheel_body=parsed_axle["pushrod_wheel_body"],
        W_lca_offset_local=parsed_axle["W_lca_offset_local_m"],
    )

    pos0, state0, info0 = solve_upright_for_zw(upright, zw_m=0.0, x0=None)
    if not info0["success"]:
        raise RuntimeError(f"Upright solver failed at zw=0 for axle={parsed_axle['axle']}")

    axis_point = rk["axis_point"]
    axis_dir = rk["axis_dir_point"] - rk["axis_point"]
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    rocker = RockerGeometry(
        axis_point=axis_point,
        axis_dir=axis_dir,
        pushrod_point_rocker_0=rk["pushrod_point_rocker_0"],
        damper_point_rocker_0=rk["damper_point_rocker_0"],
        damper_point_chassis=rk["damper_point_chassis"],
        pushrod_point_upright_0=pos0["W"],
    )

    omega0, _omega0_info = solve_omega_analytic(
        rocker, pos0["W"], omega_guess=0.0, Pu_prev=None, omega_prev=None, return_info=True,
    )
    ld0 = damper_length(rocker, omega0)
    ref = {"pos0": pos0, "state0": state0, "omega0": omega0, "Ld0": ld0}
    return upright, rocker, ref


def _generate_mr_table(parsed_axle, upright, rocker, ref, zw_mm: np.ndarray) -> pd.DataFrame:
    zw_mm = np.asarray(zw_mm, dtype=float)
    zw_m = zw_mm * 1e-3
    z_solver_m = -zw_m

    rows = [None] * len(zw_m)
    state_prev = np.array(ref["state0"], dtype=float).copy()
    omega_prev = float(ref["omega0"])
    w_prev = np.array(ref["pos0"]["W"], dtype=float).copy()
    ld0 = float(ref["Ld0"])
    omega0 = float(ref["omega0"])

    solve_order = np.argsort(z_solver_m)
    for i in solve_order:
        z_solver = float(z_solver_m[i])
        pos, state_prev, info = solve_upright_for_zw(upright, zw_m=z_solver, x0=state_prev)
        if not info["success"]:
            raise RuntimeError(f"Upright solver failed at zw={zw_mm[i]} mm")

        omega, _omega_info = solve_omega_analytic(
            rocker, pos["W"], omega_guess=omega_prev,
            Pu_prev=w_prev, omega_prev=omega_prev, return_info=True,
        )
        ld = damper_length(rocker, omega)
        rows[i] = {
            "zw_mm": float(zw_mm[i]),
            "zw_m": float(zw_m[i]),
            "omega_rad_abs": float(omega),
            "omega_deg_rel": float(math.degrees(omega - omega0)),
            "Ld_m": float(ld),
            "s_m": float(ld - ld0),
        }
        omega_prev = float(omega)
        w_prev = np.array(pos["W"], dtype=float).copy()

    df = pd.DataFrame([r for r in rows if r is not None])
    zw_arr = df["zw_m"].to_numpy(dtype=float)
    s_arr = df["s_m"].to_numpy(dtype=float)
    mr_ds = _compute_discrete_derivative(zw_arr, s_arr)
    fallback = parsed_axle.get("fallback_mr_wd", None)
    try:
        fallback_sign = math.copysign(1.0, float(fallback)) if abs(float(fallback)) > 1e-12 else 1.0
    except Exception:
        fallback_sign = 1.0
    med = float(np.nanmedian(mr_ds)) if np.any(np.isfinite(mr_ds)) else 1.0
    if np.isfinite(med) and abs(med) > 1e-12 and math.copysign(1.0, med) != fallback_sign:
        mr_ds = -mr_ds
        df["s_m"] = -df["s_m"].to_numpy(dtype=float)
    df["MR_ds_dzw"] = mr_ds
    mr_inv = np.full_like(mr_ds, np.nan)
    mask = np.abs(mr_ds) > 1e-12
    mr_inv[mask] = 1.0 / mr_ds[mask]
    df["MR_dzw_ds"] = mr_inv
    return df


def _build_mr_summary(parsed_axle, df: pd.DataFrame, poly_degree: int) -> Dict:
    zw = df["zw_m"].to_numpy(dtype=float)
    mr_dzw_ds = df["MR_dzw_ds"].to_numpy(dtype=float)
    valid = np.isfinite(zw) & np.isfinite(mr_dzw_ds)
    if int(np.count_nonzero(valid)) >= int(poly_degree + 1):
        coeffs, rms = _polyfit_with_rms(zw[valid], mr_dzw_ds[valid], poly_degree)
    else:
        coeffs, rms = np.array([]), float("nan")
    return {
        "axle": parsed_axle["axle"],
        "suspension_name": parsed_axle["suspension_name"],
        "pushrod_wheel_body": parsed_axle["pushrod_wheel_body"],
        "zw_min_m": float(np.min(df["zw_m"])),
        "zw_max_m": float(np.max(df["zw_m"])),
        "num_points": int(len(df)),
        "poly_degree": int(poly_degree),
        "poly_coefficients_high_to_low": coeffs.tolist(),
        "poly_rms_error": float(rms),
        "MR_mean": float(df["MR_ds_dzw"].mean()),
        "MR_min": float(df["MR_ds_dzw"].min()),
        "MR_max": float(df["MR_ds_dzw"].max()),
    }


def run_motion_ratio(data: dict, zmin_mm: float, zmax_mm: float, step_mm: float, poly_degree: int = 3) -> Dict:
    n = int(round((float(zmax_mm) - float(zmin_mm)) / float(step_mm))) + 1
    zw_mm = np.linspace(float(zmin_mm), float(zmax_mm), max(3, n))
    results = {}
    for axle in ("front", "rear"):
        parsed = _parse_axle_for_mr(data, axle)
        upright, rocker, ref = _build_models_for_mr(parsed)
        df = _generate_mr_table(parsed, upright, rocker, ref, zw_mm)
        summary = _build_mr_summary(parsed, df, poly_degree)
        results[axle] = {
            "df": df,
            "summary": summary,
            "parsed": parsed,
            "upright": upright,
            "rocker": rocker,
            "ref": ref,
        }
    return results
