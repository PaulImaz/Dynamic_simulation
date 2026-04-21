"""
Title: Calibrator

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
import json, math, os, sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from suspension_model import VEHICLE_REGISTRY, AxleSetupParams
from upright_solver import (
    UprightKinematicsInput,
    build_upright_positions_from_state,
    compute_lower_wishbone_local_offset,
    norm, unit,
)


# ══════════════════════════════════════════════════════════════════
# ACTUADOR DE TOE POR MODELO / EJE
# ══════════════════════════════════════════════════════════════════

TOE_ACTUATOR: Dict[str, Dict[str, str]] = {
    "F2_2026":  {"front": "steering_arm", "rear": "tie_rod_shim"},
    "F4_T421":  {"front": "steering_arm", "rear": "steering_arm"},
    "D324_EF":  {"front": "steering_arm", "rear": "tie_rod_shim"},
    "F3_2026":  {"front": "steering_arm", "rear": "tie_rod_shim"},
}


# ══════════════════════════════════════════════════════════════════
# COORDENADAS
# ══════════════════════════════════════════════════════════════════

def _j2mm(pt) -> np.ndarray:
    x, y, z = pt
    return np.array([-x * 1000., -y * 1000., -z * 1000.], dtype=float)


def _j2mm_local(pt, origin) -> np.ndarray:
    x, y, z = pt; ox, oy, oz = origin
    return np.array([-(x-ox)*1000., -(y-oy)*1000., -(z-oz)*1000.], dtype=float)


def _mm2j(v: np.ndarray, origin_json=None) -> List[float]:
    jx, jy, jz = -v[0]/1000., -v[1]/1000., -v[2]/1000.
    if origin_json is not None:
        jx += origin_json[0]; jy += origin_json[1]; jz += origin_json[2]
    return [float(jx), float(jy), float(jz)]


# ══════════════════════════════════════════════════════════════════
# MEDICIÓN GEOMÉTRICA DESDE PUNTOS
# ══════════════════════════════════════════════════════════════════

def measure_geometry(pts: "AxleRawPoints") -> dict:
    """
    Mide RH, camber y toe directamente desde los puntos del JSON.

    RH     : rw_mm - N0_z  (sistema herramienta: Z+arriba)
    Camber : atan2(-spin_z, -spin_y)  — lado derecho, negativo = neg camber
    Toe    : atan2(spin_x,  -spin_y)  — positivo = toe-out

    spin = unit(Ax0 - N0) — spin axis base (sin shim, posición de diseño).
    """
    N0   = np.asarray(pts.N0,  dtype=float)
    Ax0  = np.asarray(pts.Ax0, dtype=float)
    spin = Ax0 - N0
    n    = float(np.linalg.norm(spin))
    if n < 1e-10:
        raise ValueError(f"[{pts.axle}] rAxleAxis coincide con rAxleC — spin axis inválido")
    spin = spin / n

    rh_mm      = pts.rw_mm - N0[2]
    camber_deg = math.degrees(math.atan2(-spin[2], -spin[1]))
    toe_deg    = math.degrees(math.atan2( spin[0], -spin[1]))

    return {"rh_mm": rh_mm, "camber_deg": camber_deg, "toe_deg": toe_deg}


# ══════════════════════════════════════════════════════════════════
# EXTRACCIÓN DEL JSON
# ══════════════════════════════════════════════════════════════════

@dataclass
class AxleRawPoints:
    """Pickup points de un eje en mm (sistema herramienta)."""
    # Chassis — fijos
    A:    np.ndarray   # rFLWBI  LWB inner front
    C:    np.ndarray   # rRLWBI  LWB inner rear
    D:    np.ndarray   # rFUWBI  UWB inner front
    F:    np.ndarray   # rRUWBI  UWB inner rear
    T:    np.ndarray   # rTRI    tie rod inner (chassis)
    P12:  np.ndarray   # rPRI    pushrod inner (rocker/chassis)
    # Outboard — a calibrar
    B0:   np.ndarray   # rFLWBO  LWB outboard
    E0:   np.ndarray   # rFUWBO  UWB outboard
    S0:   np.ndarray   # rTRO    tie rod outboard
    W0:   np.ndarray   # rPRO    pushrod outboard (upright o LCA)
    N0:   np.ndarray   # rAxleC  wheel center
    Ax0:  np.ndarray   # rAxleAxis  second point on spin axis (solidario a mangueta)
    # Meta
    pushrod_wheel_body: str
    rw_mm:       float
    axle:        str
    origin_json: list


def extract_points(data: dict, axle: str) -> AxleRawPoints:
    cfg = data["config"]
    ext = cfg["suspension"][axle]["external"]["pickUpPts"]
    pwb = ("lower_wishbone"
           if "lower wishbone" in ext.get("name", "").lower()
           else "upright")
    rw_mm = float(cfg["suspension"][axle]["rWheelDesign"]) * 1000.

    if axle == "front":
        T_ = lambda k: _j2mm(ext[k])
        origin_json = [0., 0., 0.]
    else:
        org = cfg["chassis"]["rRideR"]
        T_ = lambda k: _j2mm_local(ext[k], org)
        origin_json = list(org)

    return AxleRawPoints(
        A=T_("rFLWBI"), C=T_("rRLWBI"),
        D=T_("rFUWBI"), F=T_("rRUWBI"),
        T=T_("rTRI"),   P12=T_("rPRI"),
        B0=T_("rFLWBO"), E0=T_("rFUWBO"),
        S0=T_("rTRO"),   W0=T_("rPRO"),
        N0=T_("rAxleC"), Ax0=T_("rAxleAxis"),
        pushrod_wheel_body=pwb,
        rw_mm=rw_mm, axle=axle, origin_json=origin_json,
    )


def extract_target(data: dict, axle: str) -> dict:
    """RH (mm), camber (°) y toe (°) objetivo leídos del JSON."""
    cfg = data["config"]
    sus = cfg["suspension"]
    if axle == "front":
        return {
            "rh_mm":      float(cfg["chassis"]["hRideFSetup"]) * 1000.,
            "camber_deg": math.degrees(float(
                sus["front"]["external"]["aCamberSetupAlignment"]["aCamberSetup"])),
            "toe_deg":    math.degrees(float(
                sus["front"]["external"]["aToeSetupAlignment"]["aToeSetup"])),
        }
    return {
        "rh_mm":      float(cfg["chassis"]["hRideRSetup"]) * 1000.,
        "camber_deg": math.degrees(float(
            sus["rear"]["external"]["aCamberSetupAlignment"]["aCamberSetup"])),
        "toe_deg":    math.degrees(float(
            sus["rear"]["external"]["aToeSetupAlignment"]["aToeSetup"])),
    }


# ══════════════════════════════════════════════════════════════════
# FASE 1 — GAUSS-SEIDEL CON SENSIBILIDADES DEL MANUAL
# ══════════════════════════════════════════════════════════════════

def compute_actuator_deltas_gs(
    params:   "AxleSetupParams",
    baseline: dict,
    target:   dict,
    max_cycles: int = 50,
    tol: float = 1e-4,
    verbose: bool = False,
) -> Tuple[float, float, float, dict]:
    """
    Gauss-Seidel con las sensibilidades del manual.

    Cada ciclo:
      Paso 1 — camber + toe simultáneos (sistema 2×2)
      Paso 2 — pushrod ajusta RH residual

    Funciona con deltas grandes porque itera hasta convergencia,
    a diferencia del J⁻¹ directo que solo es preciso para deltas pequeños.

    Devuelve (δ_push_total, δ_cam_total, δ_toe_total, info_dict)
    """
    tgt_rh  = target["rh_mm"]
    tgt_cam = target["camber_deg"]
    tgt_toe = target["toe_deg"]

    # Estado actual (en espacio de sensibilidades del manual)
    rh  = baseline["rh_mm"]
    cam = baseline["camber_deg"]
    toe = baseline["toe_deg"]

    # Sensibilidades del manual
    dRH_push  = params.dRH_dPush;  dCam_push = params.dCam_dPush; dToe_push = params.dToe_dPush
    dRH_cam   = params.dRH_dCam;   dCam_cam  = params.dCam_dCam;  dToe_cam  = params.dToe_dCam
    dRH_toe   = params.dRH_dToe;   dCam_toe  = params.dCam_dToe;  dToe_toe  = params.dToe_dToe

    # Jacobiano 2×2 para camber+toe simultáneos
    J_ct = np.array([[dCam_cam, dCam_toe],
                     [dToe_cam, dToe_toe]], dtype=float)

    total_dp = total_dc = total_dt = 0.0
    converged = False
    steps = []
    n_cycles = 0

    for cycle in range(max_cycles):
        ec = cam - tgt_cam
        et = toe - tgt_toe
        er = rh  - tgt_rh

        if abs(ec) < tol and abs(et) < tol and abs(er) < tol:
            converged = True
            break

        n_cycles = cycle + 1

        # Paso 1: camber + toe simultáneos
        d_c = d_t = 0.0
        b_ct = np.array([-(cam - tgt_cam), -(toe - tgt_toe)], dtype=float)
        try:
            dct = np.linalg.solve(J_ct, b_ct)
            d_c, d_t = float(dct[0]), float(dct[1])
            rh  += d_c * dRH_cam  + d_t * dRH_toe
            cam += d_c * dCam_cam + d_t * dCam_toe
            toe += d_c * dToe_cam + d_t * dToe_toe
            total_dc += d_c
            total_dt += d_t
        except np.linalg.LinAlgError:
            pass

        steps.append({"cycle": cycle+1, "actuator": "cam+toe",
                      "d_cam": d_c, "d_toe": d_t,
                      "rh": rh, "cam": cam, "toe": toe,
                      "err_rh": rh-tgt_rh, "err_cam": cam-tgt_cam, "err_toe": toe-tgt_toe})

        # Paso 2: pushrod ajusta RH residual
        d_p = 0.0
        er  = rh - tgt_rh
        if abs(er) > tol and abs(dRH_push) > 1e-10:
            d_p  = -er / dRH_push
            rh  += d_p * dRH_push
            cam += d_p * dCam_push
            toe += d_p * dToe_push
            total_dp += d_p

        steps.append({"cycle": cycle+1, "actuator": "pushrod",
                      "d_push": d_p,
                      "rh": rh, "cam": cam, "toe": toe,
                      "err_rh": rh-tgt_rh, "err_cam": cam-tgt_cam, "err_toe": toe-tgt_toe})

        if verbose:
            print(f"    Ciclo {cycle+1}: dc={d_c:+.4f} dt={d_t:+.4f} dp={d_p:+.4f} "
                  f"→ RH={rh:.4f} Cam={cam:.4f} Toe={toe:.4f} "
                  f"(eRH={rh-tgt_rh:+.4f} eCam={cam-tgt_cam:+.4f} eToe={toe-tgt_toe:+.4f})")

    info = {
        "converged":              converged,
        "n_cycles":               n_cycles,
        "delta_rh_final":         rh  - baseline["rh_mm"],
        "delta_camber_final":     cam - baseline["camber_deg"],
        "delta_toe_final":        toe - baseline["toe_deg"],
        "rh_predicted":           rh,
        "camber_predicted":       cam,
        "toe_predicted":          toe,
        "jacobian_predicted_RH":  rh  - baseline["rh_mm"],
        "jacobian_predicted_camber": cam - baseline["camber_deg"],
        "jacobian_predicted_toe":    toe - baseline["toe_deg"],
        "steps":                  steps,
    }
    return total_dp, total_dc, total_dt, info


# ══════════════════════════════════════════════════════════════════
# FASE 2 — SOLVER CON PUSHROD + L_ST CONFIGURABLE
# ══════════════════════════════════════════════════════════════════

def solve_upright_zw_lst(
    pts:        AxleRawPoints,
    zw_mm:      float,
    L_ST_new_m: float,
    S0_input:   Optional[np.ndarray] = None,
    x0:         Optional[np.ndarray] = None,
) -> Tuple[dict, np.ndarray, dict]:
    """
    Solver del upright con zw (desplazamiento vertical del wheel center)
    y longitud de tie rod L_ST configurables.

    Ecuaciones:
        F1: |E - B|  = L_BE         (longitud UWB — invariante)
        F2: |S - T|  = L_ST_new     (tie rod — incluye cambio de toe)
        F3: N_z      = N0_z - zw_m  (ride height — incluye cambio de altura)

    zw_mm > 0 → wheel center sube → coche sube → +RH.

    S0_input: si se proporciona (steering_arm), se usa como S0 en el inp
              en lugar de pts.S0 (que ya fue movida para el toe).
    """
    def m(v): return np.asarray(v, dtype=float) / 1000.

    S0_use = m(S0_input) if S0_input is not None else m(pts.S0)

    W_lca = np.zeros(3)
    if pts.pushrod_wheel_body == "lower_wishbone":
        W_lca = compute_lower_wishbone_local_offset(
            m(pts.A), m(pts.C), m(pts.B0), m(pts.W0)
        )

    inp = UprightKinematicsInput(
        A=m(pts.A),   C=m(pts.C),
        D=m(pts.D),   F=m(pts.F),
        T=m(pts.T),
        B0=m(pts.B0), E0=m(pts.E0),
        S0=S0_use,
        W0=m(pts.W0),
        N0=m(pts.N0),
        pushrod_wheel_body=pts.pushrod_wheel_body,
        W_lca_offset_local=W_lca,
    )

    # zw_mm > 0 → wheel center sube → N_z aumenta (Z es +arriba en sistema herramienta)
    z_target = inp.N0[2] + zw_mm / 1000.

    p0 = np.zeros(3) if x0 is None else np.asarray(x0, dtype=float).reshape(3)

    def residuals(p: np.ndarray) -> np.ndarray:
        pos = build_upright_positions_from_state(inp, p)
        B = pos["B"]; E = pos["E"]
        S = pos["S"]; N = pos["N"]
        return np.array([
            norm(E - B)     - inp.L_BE,
            norm(S - inp.T) - L_ST_new_m,
            N[2]            - z_target,
        ], dtype=float)

    sol = least_squares(
        residuals, p0,
        method="trf",
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
        max_nfev=500,
    )

    pos = build_upright_positions_from_state(inp, sol.x)
    info = {
        "success":       bool(sol.success),
        "message":       str(sol.message),
        "nfev":          int(sol.nfev),
        "residual_norm": float(np.linalg.norm(sol.fun)),
        "cost":          float(sol.cost),
    }
    return pos, sol.x, info


# ══════════════════════════════════════════════════════════════════
# CALIBRACIÓN DE UN EJE
# ══════════════════════════════════════════════════════════════════

def calibrate_axle(
    pts:          AxleRawPoints,
    params:       AxleSetupParams,
    baseline:     dict,
    target:       dict,
    toe_actuator: str,
    model_id:     str,
    verbose:      bool = True,
) -> dict:
    """
    Calibra un eje completo.

    Fase 1: Gauss-Seidel con sensibilidades del manual → [δ_push, δ_cam, δ_toe]
    Fase 2a: Camber — shim físico, no toca coordenadas.
    Fase 2b: Toe → nueva L_ST (y nueva S0 si steering_arm).
    Fase 2c: Altura → zw_eff = target_rh − baseline_rh.
             Solver con F3: N_z = N0_z + zw_eff → da B, E, S, W, N nuevos.
    """
    # ── Fase 1 — Gauss-Seidel ──────────────────────────────────────
    δ_push, δ_cam, δ_toe, ph1 = compute_actuator_deltas_gs(
        params, baseline, target, verbose=verbose
    )

    if verbose:
        print(f"\n  [Fase 1 — Gauss-Seidel  {'CONV' if ph1['converged'] else 'NO CONV'} "
              f"({ph1['n_cycles']} ciclos)]")
        print(f"    δ_pushrod      = {δ_push:+.4f} mm")
        print(f"    δ_camber_shim  = {δ_cam:+.4f} mm  (referencia; no modifica puntos)")
        print(f"    δ_toe_adj      = {δ_toe:+.4f} mm  ({toe_actuator})")
        print(f"    Predice: ΔRH={ph1['jacobian_predicted_RH']:+.3f}mm  "
              f"ΔCam={ph1['jacobian_predicted_camber']:+.4f}°  "
              f"ΔToe={ph1['jacobian_predicted_toe']:+.4f}°")

    # ── Fase 2a: Camber ────────────────────────────────────────────
    if verbose:
        print(f"\n  [Fase 2a — Camber]")
        print(f"    δ_camber_shim = {δ_cam:+.4f} mm → shim físico, no modifica coordenadas")

    # ── Fase 2b: Toe → nueva L_ST ─────────────────────────────────
    S0 = np.asarray(pts.S0, dtype=float)
    T  = np.asarray(pts.T,  dtype=float)
    L_ST_orig_mm = float(np.linalg.norm(S0 - T))

    if toe_actuator == "steering_arm":
        # S0 se mueve δ_toe mm en la dirección S0→T
        d = T - S0
        d_len = float(np.linalg.norm(d))
        if d_len < 1e-10:
            raise ValueError(f"[{pts.axle}] S0 y T coinciden")
        S0_new_toe   = S0 + (d / d_len) * δ_toe
        L_ST_new_mm  = float(np.linalg.norm(S0_new_toe - T))
        S0_for_solver = S0_new_toe
    else:
        # tie_rod_shim: S0 no se mueve; L_ST cambia directamente
        S0_new_toe    = S0.copy()
        L_ST_new_mm   = L_ST_orig_mm + δ_toe
        S0_for_solver = None   # usará pts.S0 en el solver

    if verbose:
        print(f"\n  [Fase 2b — Toe ({toe_actuator})]")
        if toe_actuator == "steering_arm":
            dS = S0_new_toe - S0
            print(f"    ΔS0 = [{dS[0]:+.4f}, {dS[1]:+.4f}, {dS[2]:+.4f}] mm")
        print(f"    L_ST: {L_ST_orig_mm:.4f} → {L_ST_new_mm:.4f} mm  "
              f"(Δ={L_ST_new_mm - L_ST_orig_mm:+.4f} mm)")

    # ── Fase 2c: Altura ─────────────────────────────────────────────
    # El solver trabaja en su propio sistema de medida, que tiene un offset
    # fijo respecto al manual (los puntos del JSON son el baseline de diseño
    # pero el solver los mide con un RH diferente al del manual).
    # La solución correcta es aplicar solo el DELTA de RH pedido:
    #   zw_eff = target_rh - baseline_rh_manual   (lo que queremos cambiar)
    # que es exactamente lo que ya tiene δ_push × mr_pushrod ≈ ΔRH
    # Pero más preciso: usar el delta directo del Jacobiano en RH
    delta_RH = target["rh_mm"] - baseline["rh_mm"]
    zw_eff_mm = delta_RH   # mm de desplazamiento del wheel center

    if verbose:
        print(f"\n  [Fase 2c — Altura (pushrod)]")
        print(f"    ΔRH (target-baseline) = {delta_RH:+.4f} mm = zw_eff")
        print(f"    (δ_push={δ_push:+.4f}mm × MR={params.mr_pushrod:.4f} = "
              f"{δ_push * params.mr_pushrod:+.4f}mm — Jacobiano, ref only)")

    # ── Solver ────────────────────────────────────────────────────
    pos, state, sol_info = solve_upright_zw_lst(
        pts,
        zw_mm=zw_eff_mm,
        L_ST_new_m=L_ST_new_mm / 1000.,
        S0_input=S0_for_solver,
    )

    B0_new = np.asarray(pos["B"]) * 1000.
    E0_new = np.asarray(pos["E"]) * 1000.
    W0_orig = np.asarray(pts.W0, dtype=float)
    P12     = np.asarray(pts.P12, dtype=float)
    push_vec = P12 - W0_orig
    push_len = float(np.linalg.norm(push_vec))
    push_hat = push_vec / push_len
    W0_new = W0_orig - push_hat * δ_push

    # N0 (rAxleC): solo se mueve en Z — el wheel centre solo sube/baja.
    # El solver da pos["N"] que incluye movimiento lateral/longitudinal como
    # artefacto de la cinemática; lo descartamos y solo tomamos el delta en Z.
    N0_orig_mm = np.asarray(pts.N0, dtype=float)
    N0_solver  = np.asarray(pos["N"]) * 1000.
    delta_z    = N0_solver[2] - N0_orig_mm[2]
    N0_new     = N0_orig_mm.copy()
    N0_new[2] += delta_z

    # S0 (rTRO):
    #   steering_arm → S0 se mueve físicamente (ya calculado en S0_new_toe)
    #   tie_rod_shim → solo cambia L_ST, S0 no se mueve en absoluto
    if toe_actuator == "steering_arm":
        S0_new = S0_new_toe   # posición tras mover el brazo de dirección
    else:
        S0_new = np.asarray(pts.S0, dtype=float).copy()  # sin cambio

    # ── rAxleAxis: solidario a la mangueta ────────────────────────
    # El vector spin = Ax0 - N0 es global (típicamente (0, -L, 0)).
    # Al calibrar hay tres efectos:
    #   1. N0 se mueve a N0_new (cambio de altura — solo Z)
    #   2. Camber shim rota spin alrededor del eje X global
    #   3. Toe shim rota spin alrededor del eje Z global
    # Ambas rotaciones se aplican sobre el spin original (sin pasar por
    # el frame del upright, lo que introduciría artefactos).
    Ax0_mm   = np.asarray(pts.Ax0, dtype=float)
    N0_mm    = np.asarray(pts.N0,  dtype=float)
    spin_vec = Ax0_mm - N0_mm   # vector original en mm

    # Rotación de camber alrededor de X
    cam_rad = math.radians(target["camber_deg"])
    X_hat   = np.array([1.0, 0.0, 0.0])
    c, s    = math.cos(cam_rad), math.sin(cam_rad)
    cr      = np.cross(X_hat, spin_vec)
    dot_x   = float(np.dot(X_hat, spin_vec))
    spin_vec = spin_vec * c + cr * s + X_hat * dot_x * (1.0 - c)

    # Rotación de toe alrededor de Z
    toe_rad = math.radians(target["toe_deg"])
    Z_hat   = np.array([0.0, 0.0, 1.0])
    c, s    = math.cos(toe_rad), math.sin(toe_rad)
    cr      = np.cross(Z_hat, spin_vec)
    dot_z   = float(np.dot(Z_hat, spin_vec))
    spin_vec = spin_vec * c + cr * s + Z_hat * dot_z * (1.0 - c)

    Ax0_new = N0_new + spin_vec

    # RH conseguido: el solver aplica zw_eff como delta sobre N0_z baseline
    # RH_solver = rw_mm - N_z_new
    # RH_real = RH_solver - offset  donde offset = rh_solver_baseline - rh_manual_baseline
    # Pero más simple: RH_real = baseline_rh_manual + (N0_z_baseline - N_z_new)*1000
    # N0_z_baseline en herramienta:
    N0_z_baseline_mm = float(np.asarray(pts.N0, dtype=float)[2])
    rh_achieved = baseline["rh_mm"] - (N0_z_baseline_mm - float(N0_new[2]))
    rh_error    = rh_achieved - target["rh_mm"]
    converged   = sol_info["success"] and abs(rh_error) < 0.5

    if verbose:
        print(f"\n  [Resultado solver]")
        print(f"    {'✓ OK' if sol_info['success'] else '✗ NO CONV'}  "
              f"nfev={sol_info['nfev']}  residual={sol_info['residual_norm']:.2e}")
        print(f"    RH conseguido : {rh_achieved:.3f} mm  "
              f"(target {target['rh_mm']:.3f} mm  error {rh_error:+.3f} mm)")
        print(f"    {'✓ RH OK' if converged else '✗ RH ERROR > 0.5mm'}")

    return {
        "converged":            converged,
        "rh_achieved_mm":       rh_achieved,
        "rh_target_mm":         target["rh_mm"],
        "rh_error_mm":          rh_error,
        "delta_pushrod_mm":     δ_push,
        "delta_camber_shim_mm": δ_cam,
        "delta_toe_adj_mm":     δ_toe,
        "zw_eff_mm":            zw_eff_mm,
        "L_ST_orig_mm":         L_ST_orig_mm,
        "L_ST_new_mm":          L_ST_new_mm,
        "toe_actuator":         toe_actuator,
        "W0_orig": np.asarray(pts.W0,  dtype=float),
        "B0_orig": np.asarray(pts.B0,  dtype=float),
        "E0_orig": np.asarray(pts.E0,  dtype=float),
        "S0_orig": np.asarray(pts.S0,  dtype=float),
        "N0_orig": np.asarray(pts.N0,  dtype=float),
        "Ax0_orig": np.asarray(pts.Ax0, dtype=float),
        "W0_new":  W0_new,
        "B0_new":  B0_new,
        "E0_new":  E0_new,
        "S0_new":  S0_new,
        "N0_new":  N0_new,
        "Ax0_new": Ax0_new,
        "phase1_info": ph1,
        "solver_info": sol_info,
        "solver_state": state,
    }


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def calibrate(
    json_data: dict,
    model_id:  str,
    verbose:   bool = True,
) -> dict:
    """
    Calibra ambos ejes del JSON para el modelo indicado.
    Devuelve {"model_id", "model_desc", "front": {...}, "rear": {...}}.
    """
    if model_id not in VEHICLE_REGISTRY:
        raise ValueError(
            f"Modelo '{model_id}' no registrado. "
            f"Disponibles: {list(VEHICLE_REGISTRY.keys())}"
        )

    vm      = VEHICLE_REGISTRY[model_id]
    toe_map = TOE_ACTUATOR.get(model_id, {"front": "steering_arm", "rear": "tie_rod_shim"})
    result  = {"model_id": model_id, "model_desc": vm.description}

    for axle in ("front", "rear"):
        params   = vm.front if axle == "front" else vm.rear
        pts      = extract_points(json_data, axle)

        # Baseline:
        #   RH      → del manual por modelo (referencia física)
        #   Camber  → medido desde los puntos del JSON (posición real actual)
        #   Toe     → medido desde los puntos del JSON (posición real actual)
        measured = measure_geometry(pts)
        baseline = {
            "rh_mm":      params.rh_baseline_mm,
            "camber_deg": measured["camber_deg"],
            "toe_deg":    measured["toe_deg"],
        }
        target       = extract_target(json_data, axle)
        toe_actuator = toe_map[axle]

        if verbose:
            print(f"\n{'═'*62}")
            print(f"  EJE {axle.upper()}  [{vm.description}]")
            print(f"  Baseline : RH={baseline['rh_mm']:.2f}mm  "
                  f"Cam={baseline['camber_deg']:.4f}°  Toe={baseline['toe_deg']:.4f}°")
            print(f"  Target   : RH={target['rh_mm']:.2f}mm  "
                  f"Cam={target['camber_deg']:.4f}°  Toe={target['toe_deg']:.4f}°")

        result[axle] = calibrate_axle(
            pts=pts, params=params,
            baseline=baseline, target=target,
            toe_actuator=toe_actuator,
            model_id=model_id, verbose=verbose,
        )

    return result


# ══════════════════════════════════════════════════════════════════
# ESCRITURA AL JSON
# ══════════════════════════════════════════════════════════════════

def write_calibrated_json(json_data: dict, result: dict) -> dict:
    """
    Escribe los puntos calibrados de vuelta al JSON (metros, coords JSON).
    Modifica: rPRO, rFLWBO, rRLWBO, rFUWBO, rRUWBO, rTRO, rAxleC.
    """
    out = deepcopy(json_data)
    cfg = out["config"]

    for axle in ("front", "rear"):
        cal = result[axle]
        ext = cfg["suspension"][axle]["external"]["pickUpPts"]
        org = cfg["chassis"]["rRideR"] if axle == "rear" else None

        def back(pt_mm: np.ndarray, key: str):
            ext[key] = _mm2j(pt_mm, origin_json=org)

        back(cal["W0_new"], "rPRO")
        back(cal["B0_new"], "rFLWBO")
        back(cal["B0_new"], "rRLWBO")
        back(cal["E0_new"], "rFUWBO")
        back(cal["E0_new"], "rRUWBO")
        back(cal["S0_new"], "rTRO")
        back(cal["N0_new"], "rAxleC")
        back(cal["Ax0_new"], "rAxleAxis")

    return out


# ══════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════

def format_report(result: dict) -> str:
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║              CALIBRACIÓN DE PICKUP POINTS                   ║",
        "╚══════════════════════════════════════════════════════════════╝",
        f"  Modelo : {result['model_id']}  —  {result['model_desc']}",
    ]

    for axle in ("front", "rear"):
        cal = result[axle]
        tag = "✓ CONVERGIDO" if cal["converged"] else "✗ NO CONVERGIDO"
        lines += [
            f"\n── EJE {axle.upper()} ──  [{tag}]",
            f"   Toe actuator : {cal['toe_actuator']}",
            f"\n   Actuadores (Fase 1):",
            f"     δ Pushrod      : {cal['delta_pushrod_mm']:>+10.4f} mm",
            f"     δ Camber shim  : {cal['delta_camber_shim_mm']:>+10.4f} mm  (referencia)",
            f"     δ Toe adj      : {cal['delta_toe_adj_mm']:>+10.4f} mm",
            f"\n   Altura:",
            f"     RH actual → target : {cal['rh_achieved_mm'] - cal['rh_error_mm']:.3f} → {cal['rh_target_mm']:.3f} mm  "
            f"  zw_eff={cal['zw_eff_mm']:+.4f} mm",
            f"   Tie rod:",
            f"     L_ST : {cal['L_ST_orig_mm']:.3f} → {cal['L_ST_new_mm']:.3f} mm  "
            f"(Δ={cal['L_ST_new_mm']-cal['L_ST_orig_mm']:+.4f} mm)",
            f"\n   RH conseguido  : {cal['rh_achieved_mm']:.3f} mm  "
            f"(target {cal['rh_target_mm']:.3f} mm  "
            f"error {cal['rh_error_mm']:+.3f} mm)",
            f"\n   {'Punto':<6} {'Orig X':>9} {'Orig Y':>9} {'Orig Z':>9}   "
            f"{'New X':>9} {'New Y':>9} {'New Z':>9}   {'|Δ|mm':>7}",
            f"   {'─'*75}",
        ]
        for name in ("W0", "B0", "E0", "S0", "N0"):
            orig = cal[f"{name}_orig"]
            new  = cal[f"{name}_new"]
            d    = float(np.linalg.norm(new - orig))
            lines.append(
                f"   {name:<6} {orig[0]:>9.3f} {orig[1]:>9.3f} {orig[2]:>9.3f}   "
                f"{new[0]:>9.3f} {new[1]:>9.3f} {new[2]:>9.3f}   {d:>7.4f}"
            )

    lines.append("\n" + "═" * 64)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrador de pickup points — SuspensionLab"
    )
    parser.add_argument("json_path", help="Ruta al JSON de setup")
    parser.add_argument(
        "--model", default="F2_2026",
        choices=list(VEHICLE_REGISTRY.keys()),
        help=f"Modelo. Disponibles: {list(VEHICLE_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--output", default=None,
        help="Ruta de salida del JSON calibrado (opcional)"
    )
    args = parser.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        data = json.load(f)

    result = calibrate(data, model_id=args.model, verbose=True)
    print()
    print(format_report(result))

    if args.output:
        out_data = write_calibrated_json(data, result)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"\n  JSON calibrado escrito en: {args.output}")
