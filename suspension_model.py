"""
Title: Suspension Model

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
import math, json, os, sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from upright_solver import (
    UprightKinematicsInput,
    build_upright_positions_from_state,
    compute_lower_wishbone_local_offset,
    norm, unit,
)


# ══════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════

def _json2mm(pt) -> np.ndarray:
    x, y, z = pt
    return np.array([-x*1000., -y*1000., -z*1000.], dtype=float)

def _json2mm_local(pt, origin) -> np.ndarray:
    x, y, z = pt; ox, oy, oz = origin
    return np.array([-(x-ox)*1000., -(y-oy)*1000., -(z-oz)*1000.], dtype=float)

def _mm2json(v: np.ndarray, origin_json=None) -> List[float]:
    jx, jy, jz = -v[0]/1000., -v[1]/1000., -v[2]/1000.
    if origin_json is not None:
        jx += origin_json[0]; jy += origin_json[1]; jz += origin_json[2]
    return [float(jx), float(jy), float(jz)]


# ══════════════════════════════════════════════════════════════════
# PARÁMETROS DEL MODELO POR COCHE
# ══════════════════════════════════════════════════════════════════

@dataclass
class AxleSetupParams:
    """
    Parámetros de un eje de un modelo de coche.
    Todos los ángulos en grados, distancias en mm.

    Toe: convención del ACTUADOR FÍSICO (no ángulo geométrico puro):
      front: +1 barrel turn = +1.058mm steering arm = +0.603° toe → baseline = +0.603°
      rear:  +1mm shim = -0.375° toe → baseline = -0.375°

    Sensibilidades (tabla manual, por mm de actuador):
      Los valores de la diagonal son los que más importan.
      Las cruces son el acoplamiento que el solver aproxima.
    """
    # ── Baseline de diseño ──────────────────────────────────────
    rh_baseline_mm:     float   # RH de diseño (suelo → plano z0)
    camber_baseline_deg: float  # Camber de diseño (neg = neg camber)
    toe_baseline_deg:   float   # Toe de diseño en unidades del actuador
    caster_baseline_deg: float
    mechanical_trail_mm: float = 0.
    scrub_radius_mm:    float = 0.

    # ── Actuadores (tabla manual p.51 Dallara F2) ────────────────
    # PUSHROD: por mm de shim
    mr_pushrod:         float = 1.0   # MR = mm_RH / mm_pushrod
    dRH_dPush:          float = 0.    # mm RH / mm pushrod = mr_pushrod
    dCam_dPush:         float = 0.    # ° camber / mm pushrod
    dToe_dPush:         float = 0.    # ° toe / mm pushrod
    dCas_dPush:         float = 0.    # ° caster / mm pushrod

    # CAMBER SHIM: por mm de shim
    dRH_dCam:           float = 0.
    dCam_dCam:          float = 0.    # ° camber / mm shim
    dToe_dCam:          float = 0.
    dCas_dCam:          float = 0.

    # TOE ADJUSTER: por mm del actuador físico (steering arm / shim)
    dRH_dToe:           float = 0.    # mm RH / ° toe
    dCam_dToe:          float = 0.    # ° camber / ° toe
    dToe_dToe:          float = 0.    # ° toe / mm actuador

    # CASTER ADJUSTER: por mm del actuador físico
    dRH_dCas:           float = 0.
    dCam_dCas:          float = 0.
    dToe_dCas:          float = 0.
    dCas_dCas:          float = 0.

    # ── Ratios ───────────────────────────────────────────────────
    wheel_damper_ratio: float = 1.0   # mm wheel / mm damper
    wheel_tbar_ratio:   float = 1.0   # mm wheel / deg torsion bar (front only)


@dataclass
class VehicleModel:
    """Modelo completo del coche con parámetros por eje."""
    model_id:    str
    description: str
    front:       AxleSetupParams
    rear:        AxleSetupParams


# ── F2 2026 (Dallara, Manual Rev 3.1) ───────────────────────────
F2_2026 = VehicleModel(
    model_id    = "F2_2026",
    description = "Dallara F2 2026 – Manual Rev 3.1",
    front = AxleSetupParams(
        rh_baseline_mm      = 25.0,
        camber_baseline_deg = -3.8,
        toe_baseline_deg    = +0.603,    # 1mm OUT → +0.603°/barrel turn
        caster_baseline_deg = 6.18,
        mechanical_trail_mm = 18.72,
        mr_pushrod          = 5.09 / 2.117,
        dRH_dPush           = 5.09 / 2.117,
        dCam_dPush          = -0.035 / 2.117,
        dToe_dPush          = -0.028 / 2.117,
        dCas_dPush          = 0.,
        dRH_dCam            = 0.508,
        dCam_dCam           = +0.333,
        dToe_dCam           = +0.073,
        dCas_dCam           = 0.,
        dRH_dToe            = 0.079,            # mm RH per barrel turn
        dCam_dToe           = 0.065,            # ° cam per barrel turn
        dToe_dToe           = 0.603 / 1.058,   # °/mm steering arm
        dRH_dCas            = -0.743 / 1.270,
        dCam_dCas           = -0.158 / 1.270,
        dToe_dCas           = +0.042 / 1.270,
        dCas_dCas           = +0.501 / 1.270,
        wheel_damper_ratio  = 1.425,
        wheel_tbar_ratio    = 1.490,
    ),
    rear = AxleSetupParams(
        rh_baseline_mm      = 55.0,
        camber_baseline_deg = -1.5,
        toe_baseline_deg    = -0.375,    # 1mm IN → -0.375°/mm shim
        caster_baseline_deg = 14.79,
        mechanical_trail_mm = 46.77,
        mr_pushrod          = 5.63 / 2.117,
        dRH_dPush           = 5.63 / 2.117,
        dCam_dPush          = +0.098 / 2.117,
        dToe_dPush          = -0.005 / 2.117,
        dCas_dPush          = 0.,
        dRH_dCam            = 0.379,
        dCam_dCam           = +0.164,
        dToe_dCam           = +0.379,
        dCas_dCam           = 0.,
        dRH_dToe            = 0.184 / 0.375,
        dCam_dToe           = 0.098 / 0.375,
        dToe_dToe           = 0.375,            # °/mm shim
        dRH_dCas            = -0.346 / 1.270,
        dCam_dCas           = +0.079 / 1.270,
        dToe_dCas           = -0.034 / 1.270,
        dCas_dCas           = -0.655 / 1.270,
        wheel_damper_ratio  = 1.308,
    ),
)

# ── Tatuus T-421 (F4 Regional)  –  Manual Release 2.1.0 ─────────
F4_T421 = VehicleModel(
    model_id    = "F4_T421",
    description = "Tatuus T-421 (F4) – Manual Release 2.1.0",
    front = AxleSetupParams(
        rh_baseline_mm      = 18.0,
        camber_baseline_deg = -4.0,
        toe_baseline_deg    = +20.0 / 60.0,
        caster_baseline_deg = 9.4,
        mr_pushrod          = 5.4 / 1.5,
        dRH_dPush           = 5.4 / 1.5,
        dCam_dPush=0., dToe_dPush=0., dCas_dPush=0.,
        dRH_dCam=0., dCam_dCam=+0.42, dToe_dCam=0., dCas_dCam=0.,
        dRH_dToe=0., dCam_dToe=0.,
        dToe_dToe           = 1.12 / 1.5,
        dRH_dCas=0., dCam_dCas=0., dToe_dCas=0.,
        dCas_dCas           = 0.47 / 1.5,
        wheel_damper_ratio  = 1.10,
    ),
    rear = AxleSetupParams(
        rh_baseline_mm      = 38.0,
        camber_baseline_deg = -3.0,
        toe_baseline_deg    = 0.0,
        caster_baseline_deg = 0.,
        mr_pushrod          = 6.5 / 1.5,
        dRH_dPush           = 6.5 / 1.5,
        dCam_dPush=0., dToe_dPush=0., dCas_dPush=0.,
        dRH_dCam=0., dCam_dCam=+0.30, dToe_dCam=0., dCas_dCam=0.,
        dRH_dToe=0., dCam_dToe=0.,
        dToe_dToe           = 1.00 / 1.5,
        dRH_dCas=0., dCam_dCas=0., dToe_dCas=0., dCas_dCas=0.,
        wheel_damper_ratio  = 1.23,
    ),
)

# ── Dallara 324 (EuroFormula Open / Eurocup-3)  –  Manual Rev 1.7 ─
D324_EF = VehicleModel(
    model_id    = "D324_EF",
    description = "Dallara 324 (EuroFormula/Eurocup-3) – Manual Rev 1.7",
    front = AxleSetupParams(
        rh_baseline_mm      = 25.0,
        camber_baseline_deg = -3.5,
        toe_baseline_deg    = 0.0,
        caster_baseline_deg = 16.03,
        mr_pushrod          = 5.936 / 2.12,
        dRH_dPush           = 5.936 / 2.12,
        dCam_dPush          = -0.095 / 2.12,
        dToe_dPush=0., dCas_dPush=0.,
        dRH_dCam=0.190, dCam_dCam=+0.434, dToe_dCam=+0.071, dCas_dCam=0.,
        dRH_dToe=0., dCam_dToe=0.,
        dToe_dToe           = 0.620 / 1.06,
        dRH_dCas            = -0.922 / 1.06,
        dCam_dCas           = -0.169 / 1.06,
        dToe_dCas           = +0.050 / 1.06,
        dCas_dCas           = +0.516 / 1.06,
        wheel_damper_ratio  = 1.447,
        wheel_tbar_ratio    = 1.699,
    ),
    rear = AxleSetupParams(
        rh_baseline_mm      = 47.0,
        camber_baseline_deg = -2.0,
        toe_baseline_deg    = 0.0,
        caster_baseline_deg = 17.0,
        mr_pushrod          = 6.145 / 2.33,
        dRH_dPush           = 6.145 / 2.33,
        dCam_dPush          = +0.188 / 2.33,
        dToe_dPush=0., dCas_dPush=0.,
        dRH_dCam=1.964, dCam_dCam=+0.390, dToe_dCam=+0.205, dCas_dCam=0.,
        dRH_dToe=0., dCam_dToe=0.,
        dToe_dToe           = -0.765 / 2.33,
        dRH_dCas            = -1.546 / 1.27,
        dCam_dCas           = +0.025 / 1.27,
        dToe_dCas           = +0.039 / 1.27,
        dCas_dCas           = -0.779 / 1.27,
        wheel_damper_ratio  = 1.289,
    ),
)

# ── Dallara F325 (F3 2026)  –  User Manual Rev 2.1 ──────────────
#
# GEOMETRÍA BASELINE (§4.4 / §4.5, mm, origen = Z-0 monocoque ref plane):
#   Config estándar trasera: A2-B2-C2-D2 (§4.5).
#
# SETUP SUGERIDO (§4.1):
#   Front: RH=26.5mm (Z-0 ref), Camber=−3.4°, Toe=1mm OUT, Caster=9.8° STD
#   Rear:  RH=50.0mm (Z-0 ref), Camber=−1.5°, Toe=1mm IN
#
# SENSIBILIDADES (§4.3, tabla p.52, imagen):
#   Pushrod: +1 barrel turn = 2.117 mm (mismo paso front y rear)
#     Front: +5.188mm RH, −0.066° Cam, +0.005° Toe, −0.064° Cas → por mm:
#            dRH=2.4506, dCam=−0.0312, dToe=+0.0024, dCas=−0.0302
#     Rear:  +5.462mm RH, +0.118° Cam, +0.003° Toe, +0.048° Cas → por mm:
#            dRH=2.5801, dCam=+0.0557, dToe=+0.0014, dCas=+0.0227
#   Toe adjuster:
#     Front: +1 barrel turn = 1.058mm → +0.708° Toe, +0.155mm RH, +0.122° Cam
#            → por mm: dToe=0.6692, dRH=0.1465, dCam=0.1153
#     Rear:  +1mm shim → −0.375° Toe, +0.084mm RH, +0.001° Cam, −0.074° Cas
#   Camber shim +1mm:
#     Front: +0.356° Cam, +0.599mm RH, +0.062° Toe, +0.017° Cas
#     Rear:  +0.199° Cam, +0.455mm RH, +0.372° Toe, +0.074° Cas
#   Caster adjuster:
#     Front: +1 barrel turn = 1.27mm → +0.535° Cas, −0.85mm RH, −0.15° Cam, +0.083° Toe
#            → por mm: dCas=0.4213, dRH=−0.6693, dCam=−0.1181, dToe=+0.0654
#     Rear:  +1 barrel turn = 1.058mm → −0.368° Cas, −1.646mm RH, −0.135° Toe
#            → por mm: dCas=−0.3478, dRH=−1.5558, dToe=−0.1276
#   Wheel/Damper: Front 1.449, Rear 1.241
#   Wheel/Torsion bar (front): 1.516 mm/deg
F3_2026 = VehicleModel(
    model_id    = "F3_2026",
    description = "Dallara F325 (F3 2026) – Manual Rev 2.1",
    front = AxleSetupParams(
        rh_baseline_mm      = 26.5,
        camber_baseline_deg = -3.4,
        toe_baseline_deg    = +0.708,       # 1mm OUT → +0.708°/1.058mm × 1mm ≈ +0.669°
                                            # pero el manual dice "1mm OUT" como referencia
                                            # → usamos la sensibilidad directa: 1mm = dToe_dToe * 1mm
        caster_baseline_deg = 9.80,         # built-in caster §4.3
        mechanical_trail_mm = 21.71,        # §4.3 Castor table

        # Pushrod: 2.117 mm/turn
        mr_pushrod          = 5.188 / 2.117,        # 2.4506
        dRH_dPush           = 5.188 / 2.117,
        dCam_dPush          = -0.066 / 2.117,       # −0.0312°/mm
        dToe_dPush          = +0.005 / 2.117,       # +0.0024°/mm
        dCas_dPush          = -0.064 / 2.117,       # −0.0302°/mm

        # Camber shim +1mm
        dRH_dCam            = +0.599,
        dCam_dCam           = +0.356,
        dToe_dCam           = +0.062,
        dCas_dCam           = +0.017,

        # Toe adjuster: 1.058mm/turn → 0.708°/turn
        dRH_dToe            = 0.155 / 1.058,        # 0.1465 mm/mm
        dCam_dToe           = 0.122 / 1.058,        # 0.1153°/mm
        dToe_dToe           = 0.708 / 1.058,        # 0.6692°/mm
        dRH_dCas            = -0.85  / 1.27,        # −0.6693 mm/mm
        dCam_dCas           = -0.15  / 1.27,        # −0.1181°/mm
        dToe_dCas           = +0.083 / 1.27,        # +0.0654°/mm
        dCas_dCas           = +0.535 / 1.27,        # +0.4213°/mm

        wheel_damper_ratio  = 1.449,
        wheel_tbar_ratio    = 1.516,                # mm wheel / deg torsion bar §4.3
    ),
    rear = AxleSetupParams(
        rh_baseline_mm      = 50.0,
        camber_baseline_deg = -1.5,
        toe_baseline_deg    = -0.375,       # 1mm IN → −0.375°/mm × 1mm = −0.375°
        caster_baseline_deg = 9.21,         # built-in caster §4.3
        mechanical_trail_mm = 2.6,

        # Pushrod: 2.117 mm/turn
        mr_pushrod          = 5.462 / 2.117,        # 2.5801
        dRH_dPush           = 5.462 / 2.117,
        dCam_dPush          = +0.118 / 2.117,       # +0.0557°/mm
        dToe_dPush          = +0.003 / 2.117,       # +0.0014°/mm
        dCas_dPush          = +0.048 / 2.117,       # +0.0227°/mm

        # Camber shim +1mm
        dRH_dCam            = +0.455,
        dCam_dCam           = +0.199,
        dToe_dCam           = +0.372,
        dCas_dCam           = +0.074,

        # Toe adjuster: 1mm shim directo → −0.375°
        dRH_dToe            = +0.084,               # mm/mm shim
        dCam_dToe           = +0.001,               # °/mm shim
        dToe_dToe           = -0.375,               # °/mm shim (toe-in = negativo)
        dRH_dCas            = -1.646 / 1.058,       # −1.5558 mm/mm
        dCam_dCas           = 0.,                   # no tabulado en tabla rear caster
        dToe_dCas           = -0.135 / 1.058,       # −0.1276°/mm (de tabla rear caster Toe change)
        dCas_dCas           = -0.368 / 1.058,       # −0.3478°/mm

        wheel_damper_ratio  = 1.241,
    ),
)

# Registro global
VEHICLE_REGISTRY: Dict[str, VehicleModel] = {
    "F2_2026": F2_2026,
    "F4_T421": F4_T421,
    "D324_EF": D324_EF,
    "F3_2026": F3_2026,
}


# ══════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE PUNTOS DEL JSON
# ══════════════════════════════════════════════════════════════════

@dataclass
class AxleGeometry:
    """
    Todos los pickup points de un eje en sistema herramienta (mm).
    X→+adelante, Y→+izquierda, Z→+arriba.
    """
    # Puntos inboard (chasis, FIJOS)
    A:   np.ndarray   # LWB inboard front
    C:   np.ndarray   # LWB inboard rear
    D:   np.ndarray   # UWB inboard front
    F:   np.ndarray   # UWB inboard rear
    T:   np.ndarray   # toe link inboard
    PRI: np.ndarray   # pushrod inboard (rocker, FIJO)

    # Puntos outboard (upright, se mueven)
    B0:  np.ndarray   # LWB outboard
    E0:  np.ndarray   # UWB outboard
    S0:  np.ndarray   # toe link outboard
    W0:  np.ndarray   # pushrod outboard (en LCA si lower_wishbone)
    N0:  np.ndarray   # axle center (wheel center)

    # Metadatos
    rw_mm: float
    pushrod_wheel_body: str
    axle: str
    origin_json: List[float]

    # Offset local del pushrod en el LCA (solo lower_wishbone)
    W_lca: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def inp(self, E0_override=None, S0_override=None) -> UprightKinematicsInput:
        """Construye UprightKinematicsInput (en metros) con overrides opcionales."""
        E0 = E0_override if E0_override is not None else self.E0
        S0 = S0_override if S0_override is not None else self.S0
        def m(v): return np.asarray(v) / 1000.
        return UprightKinematicsInput(
            A=m(self.A), C=m(self.C), D=m(self.D), F=m(self.F), T=m(self.T),
            B0=m(self.B0), E0=m(E0), S0=m(S0), W0=m(self.W0), N0=m(self.N0),
            pushrod_wheel_body=self.pushrod_wheel_body,
            W_lca_offset_local=np.asarray(self.W_lca) / 1000.,
        )


def extract_geometry(json_data: dict, axle: str) -> AxleGeometry:
    """Extrae los pickup points del JSON al sistema herramienta (mm)."""
    cfg = json_data["config"]
    ext = cfg["suspension"][axle]["external"]["pickUpPts"]
    pwb = ("lower_wishbone"
           if "lower wishbone" in ext.get("name","").lower() else "upright")
    rw_mm = float(cfg["suspension"][axle]["rWheelDesign"]) * 1000.

    if axle == "front":
        T_ = _json2mm
        origin_json = [0., 0., 0.]
    else:
        org = cfg["chassis"]["rRideR"]
        def T_(k): return _json2mm_local(k, org)
        origin_json = list(org)

    def p(key): return T_(ext[key])

    A   = p("rFLWBI"); C = p("rRLWBI")
    D   = p("rFUWBI"); F_ = p("rRUWBI")
    T   = p("rTRI")
    B0  = p("rFLWBO"); E0 = p("rFUWBO")
    S0  = p("rTRO");   W0 = p("rPRO")
    N0  = p("rAxleC")
    PRI = p("rPRI")   # pushrod inboard (rocker side), FIJO

    W_lca = np.zeros(3)
    if pwb == "lower_wishbone":
        W_lca = compute_lower_wishbone_local_offset(A, C, B0, W0)

    return AxleGeometry(
        A=A, C=C, D=D, F=F_, T=T, PRI=PRI,
        B0=B0, E0=E0, S0=S0, W0=W0, N0=N0,
        rw_mm=rw_mm, pushrod_wheel_body=pwb,
        axle=axle, origin_json=origin_json, W_lca=W_lca,
    )


def parse_setup(json_data: dict, axle: str) -> dict:
    cfg = json_data["config"]; sus = cfg["suspension"]
    if axle == "front":
        return {
            "rh_mm":       float(cfg["chassis"]["hRideFSetup"]) * 1000.,
            "camber_deg":  math.degrees(float(sus["front"]["external"]["aCamberSetupAlignment"]["aCamberSetup"])),
            "toe_deg":     math.degrees(float(sus["front"]["external"]["aToeSetupAlignment"]["aToeSetup"])),
        }
    return {
        "rh_mm":       float(cfg["chassis"]["hRideRSetup"]) * 1000.,
        "camber_deg":  math.degrees(float(sus["rear"]["external"]["aCamberSetupAlignment"]["aCamberSetup"])),
        "toe_deg":     math.degrees(float(sus["rear"]["external"]["aToeSetupAlignment"]["aToeSetup"])),
    }


# ══════════════════════════════════════════════════════════════════
# SOLVER CON L_ST CONFIGURABLE
# ══════════════════════════════════════════════════════════════════

def solve_upright(
    inp: UprightKinematicsInput,
    zw_m: float,
    L_ST_override: Optional[float] = None,
    x0: Optional[np.ndarray] = None,
) -> Tuple[dict, np.ndarray, dict]:
    """
    Extiende solve_upright_for_zw con L_ST configurable.
    L_ST_override es la longitud del tie rod en METROS.
    """
    L_ST = L_ST_override if L_ST_override is not None else inp.L_ST
    z_target = inp.N0[2] - zw_m
    p0 = np.zeros(3) if x0 is None else np.array(x0).reshape(3)

    def residuals(p):
        pos = build_upright_positions_from_state(inp, p)
        return np.array([
            norm(pos["E"] - pos["B"]) - inp.L_BE,
            norm(pos["S"] - inp.T)    - L_ST,
            pos["N"][2]               - z_target,
        ])

    sol = least_squares(residuals, p0, method="trf",
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=400)
    pos  = build_upright_positions_from_state(inp, sol.x)
    info = {"success": bool(sol.success), "residual_norm": float(np.linalg.norm(sol.fun)),
            "nfev": int(sol.nfev)}
    return pos, sol.x, info


# ══════════════════════════════════════════════════════════════════
# MEDICIÓN CINEMÁTICA
# ══════════════════════════════════════════════════════════════════

def _measure_raw(pos: dict, rw_mm: float) -> Tuple[float, float, float]:
    """
    Mide RH, camber y toe en coordenadas internas (raw).

    RH     = rw_mm − N_z                         (mm)
    Camber = ángulo del kingpin (e1) en plano YZ  (°)
    Toe    = ángulo del eje longitudinal (e3) en plano XY  (°)
    """
    N  = np.asarray(pos["N"]) * 1000.
    e1 = np.asarray(pos["frame_e1"])
    e3 = np.asarray(pos["frame_e3"])

    rh_raw = rw_mm - float(N[2])

    e1_yz = np.array([0., e1[1], e1[2]])
    n = float(np.linalg.norm(e1_yz))
    if n > 1e-10: e1_yz /= n
    ang = math.degrees(math.atan2(abs(e1_yz[1]), abs(e1_yz[2])))
    # e1[1] > 0 en herramienta izquierda = parte superior hacia el interior → camber negativo
    cam_raw = -ang if e1[1] > 0 else ang

    e3_xy = np.array([e3[0], e3[1], 0.])
    n2 = float(np.linalg.norm(e3_xy))
    if n2 > 1e-10: e3_xy /= n2
    toe_raw = math.degrees(math.atan2(e3_xy[0], abs(e3_xy[1])))

    return rh_raw, cam_raw, toe_raw


# ══════════════════════════════════════════════════════════════════
# OFFSETS  raw ↔ real
# ══════════════════════════════════════════════════════════════════

@dataclass
class Offsets:
    """
    Transformación entre el espacio interno (raw) y el espacio físico real.

    Calculados midiendo los puntos del manual con el solver y comparando
    con los valores reales del manual (que son el baseline de diseño).

        offset = raw_medido(pts_manual) − real_baseline_manual

    Uso:
        raw_target  = real_target + offset
        real_medido = raw_medido  − offset
    """
    axle: str
    rh_off:  float
    cam_off: float
    toe_off: float
    # Valores que los generaron (para comparación y reporte)
    baseline_rh:  float
    baseline_cam: float
    baseline_toe: float
    raw_rh:  float
    raw_cam: float
    raw_toe: float

    def to_raw(self, rh, cam, toe):
        return rh+self.rh_off, cam+self.cam_off, toe+self.toe_off

    def to_real(self, rh, cam, toe):
        return rh-self.rh_off, cam-self.cam_off, toe-self.toe_off


def compute_offsets(geo: AxleGeometry, params: AxleSetupParams) -> Offsets:
    """
    Mide los puntos del manual con el solver en zw=0 y calcula los offsets
    respecto al baseline del manual.
    """
    inp = geo.inp()
    pos, _, _ = solve_upright(inp, zw_m=0.)
    rh_raw, cam_raw, toe_raw = _measure_raw(pos, geo.rw_mm)

    return Offsets(
        axle=geo.axle,
        rh_off  = rh_raw  - params.rh_baseline_mm,
        cam_off = cam_raw - params.camber_baseline_deg,
        toe_off = toe_raw - params.toe_baseline_deg,
        baseline_rh  = params.rh_baseline_mm,
        baseline_cam = params.camber_baseline_deg,
        baseline_toe = params.toe_baseline_deg,
        raw_rh  = rh_raw,
        raw_cam = cam_raw,
        raw_toe = toe_raw,
    )


def measure(geo: AxleGeometry, offs: Offsets,
            zw_mm: float = 0., x0=None,
            L_ST_override_mm: Optional[float] = None,
            E0_override: Optional[np.ndarray] = None,
            ) -> Tuple[dict, np.ndarray, dict]:
    """
    Resuelve la cinemática y devuelve mediciones en valores REALES.
    L_ST_override_mm: longitud del tie rod en mm (None = usar la del baseline).
    E0_override: nuevo punto UWB outboard en mm herramienta.
    """
    L_ST_m = (L_ST_override_mm / 1000.) if L_ST_override_mm is not None else None
    inp = geo.inp(E0_override=E0_override)
    pos, state, info = solve_upright(inp, zw_m=zw_mm/1000., L_ST_override=L_ST_m, x0=x0)
    rh_raw, cam_raw, toe_raw = _measure_raw(pos, geo.rw_mm)
    rh, cam, toe = offs.to_real(rh_raw, cam_raw, toe_raw)

    meas = {
        "rh_mm":       rh,
        "camber_deg":  cam,
        "toe_deg":     toe,
        "_raw": {"rh": rh_raw, "cam": cam_raw, "toe": toe_raw},
    }
    return meas, state, info


# ══════════════════════════════════════════════════════════════════
# ACTUADORES FÍSICOS
# ══════════════════════════════════════════════════════════════════

@dataclass
class ActuatorState:
    """Estado de los tres actuadores físicos en mm."""
    delta_pushrod_mm:    float = 0.   # >0 = alarga pushrod = coche sube = +RH
    delta_camber_shim_mm: float = 0.  # >0 = camber más negativo (exterior)
    delta_toe_adj_mm:    float = 0.   # >0 = dirección según signo dToe_dToe


def _camber_shim_dir(geo: AxleGeometry) -> np.ndarray:
    """Dirección normalizada del shim de camber: ⊥ al eje UWB, con componente -Y (exterior)."""
    D  = np.asarray(geo.D, dtype=float)
    F_ = np.asarray(geo.F, dtype=float)
    uwb_hat = unit(F_ - D)
    raw = np.array([0., -1., 0.]) - float(np.dot([0., -1., 0.], uwb_hat)) * uwb_hat
    n = float(np.linalg.norm(raw))
    return raw / n if n > 1e-10 else np.array([0., -1., 0.])


def apply_actuators(geo: AxleGeometry, params: AxleSetupParams,
                    act: ActuatorState) -> Tuple[dict, dict]:
    """
    Aplica los actuadores físicos y devuelve los parámetros para el solver:
        {zw_mm, L_ST_mm, E0_mm}  — parámetros a pasar a measure()
    y los deltas de puntos outboard para reporte.

    Las modificaciones son:
      Pushrod: zw_eff = δ_push × MR  (mm de desplazamiento del wheel center)
      Camber:  E0_new = E0 + shim_dir × δ_cam  (mm herramienta)
      Toe:     L_ST_new = L_ST_baseline + δ_toe  (mm)

    Los puntos del JSON no se modifican durante la medición —
    los "puntos calibrados" se calculan aparte en calibrate().
    """
    # ── Pushrod → zw ────────────────────────────────────────────
    zw_eff_mm = act.delta_pushrod_mm * params.mr_pushrod

    # ── Camber shim → E0 ────────────────────────────────────────
    shim_dir = _camber_shim_dir(geo)
    E0_new = np.asarray(geo.E0, dtype=float) + shim_dir * act.delta_camber_shim_mm

    # ── Toe adjuster → L_ST ─────────────────────────────────────
    # L_ST baseline (mm)
    S0_m = np.asarray(geo.S0) / 1000.
    T_m  = np.asarray(geo.T)  / 1000.
    L_ST_baseline_mm = float(np.linalg.norm(S0_m - T_m)) * 1000.

    # Signo: dToe_dToe para front es positivo (+ = toe-out).
    # El toe adjuster físico: +δ_toe mm de actuador → dToe_dToe × δ_toe grados de cambio.
    # Para que el solver produzca ese cambio, L_ST debe aumentar en:
    #   δ_L_ST = δ_toe × mm_de_TS_por_unidad_de_actuador
    # Sensibilidad del solver: ΔToe/ΔL_ST ≈ lo que hemos medido.
    # Usamos la sensibilidad del manual como escala de referencia.
    L_ST_new_mm = L_ST_baseline_mm + act.delta_toe_adj_mm

    solver_params = {
        "zw_mm":     zw_eff_mm,
        "L_ST_mm":   L_ST_new_mm,
        "E0_mm":     E0_new,
    }

    # Deltas de puntos outboard para reporte (en mm herramienta)
    pts_delta = {
        "E0": E0_new - np.asarray(geo.E0),
        "S0": np.zeros(3),   # S0 no se mueve directamente (L_ST cambia, no S0)
        "N0": np.zeros(3),   # el desplazamiento de N0 lo calcula el solver
        "B0": np.zeros(3),
        "W0": np.zeros(3),
    }
    return solver_params, pts_delta


# ══════════════════════════════════════════════════════════════════
# CALIBRACIÓN
# ══════════════════════════════════════════════════════════════════

@dataclass
class CalibrationResult:
    axle: str
    model_id: str
    converged: bool
    iterations: int
    cost: float

    # Valores en espacio real
    baseline: dict       # {"rh_mm", "camber_deg", "toe_deg"}
    target: dict
    achieved: dict
    residuals: dict

    # Actuadores en unidades físicas
    actuators: ActuatorState

    # Sensitivity comparison (solver vs manual)
    jacobian_solver: np.ndarray   # 3×3: [RH, Cam, Toe] × [push, cam, toe]
    jacobian_manual: np.ndarray

    # Geometry
    geo_orig: AxleGeometry
    offsets: Offsets
    solver_params: dict   # {zw_mm, L_ST_mm, E0_mm} aplicados

    messages: List[str] = field(default_factory=list)
    steps:    List[dict] = field(default_factory=list)




def calibrate(
    geo, params, offs, target_real, max_cycles=30, verbose=False,
):
    # Calibracion Gauss-Seidel con sensibilidades del manual (tabla p.51).
    # Cada ciclo: Paso 1 = camber+toe simultaneos (sistema 2x2)
    #             Paso 2 = pushrod ajusta RH residual
    # Al converger, aplica deltas 3D via upright_solver.
    msgs = []; steps = []
    tol = 0.001
    tgt_rh = target_real["rh_mm"]
    tgt_cam = target_real["camber_deg"]
    tgt_toe = target_real["toe_deg"]
    rh = offs.baseline_rh; cam = offs.baseline_cam; toe = offs.baseline_toe
    msgs.append(f"[Baseline]  RH={rh:.4f}mm  Cam={cam:.4f}  Toe={toe:.4f}")
    msgs.append(f"[Target]    RH={tgt_rh:.4f}mm  Cam={tgt_cam:.4f}  Toe={tgt_toe:.4f}")
    if verbose:
        print(f"  INICIO: RH={rh:.4f}mm  Cam={cam:.4f}  Toe={toe:.4f}")
        print(f"  TARGET: RH={tgt_rh:.4f}mm  Cam={tgt_cam:.4f}  Toe={tgt_toe:.4f}")
    dCam_cam=params.dCam_dCam; dRH_cam=params.dRH_dCam; dToe_cam=params.dToe_dCam
    dToe_toe=params.dToe_dToe; dRH_toe=params.dRH_dToe; dCam_toe=params.dCam_dToe
    dRH_push=params.dRH_dPush; dCam_push=params.dCam_dPush; dToe_push=params.dToe_dPush
    J_ct = np.array([[dCam_cam, dCam_toe],[dToe_cam, dToe_toe]])
    total_dp=total_dc=total_dt=0.; converged=False; n_cycles=0
    for cycle in range(max_cycles):
        ec=cam-tgt_cam; et=toe-tgt_toe; er=rh-tgt_rh
        if abs(ec)<tol and abs(et)<tol and abs(er)<tol:
            converged=True; break
        n_cycles=cycle+1
        if verbose:
            print(f"\n  Ciclo {cycle+1}:")
        # Paso 1: camber+toe simultaneos
        b_ct=np.array([-(cam-tgt_cam),-(toe-tgt_toe)]); d_c=d_t=0.
        try:
            dct=np.linalg.solve(J_ct,b_ct); d_c=float(dct[0]); d_t=float(dct[1])
            rh+=d_c*dRH_cam+d_t*dRH_toe
            cam+=d_c*dCam_cam+d_t*dCam_toe
            toe+=d_c*dToe_cam+d_t*dToe_toe
            total_dc+=d_c; total_dt+=d_t
        except Exception: pass
        steps.append({"cycle":cycle+1,"actuator":"cam+toe","delta_mm":d_c,"delta_toe":d_t,
                       "rh":rh,"cam":cam,"toe":toe,
                       "err_rh":rh-tgt_rh,"err_cam":cam-tgt_cam,"err_toe":toe-tgt_toe})
        msgs.append(f"[C{cycle+1} Cam+Toe dc={d_c:>+8.4f}mm dt={d_t:>+8.4f}mm] "
                    f"RH={rh:.4f}  Cam={cam:.4f}  Toe={toe:.4f}  "
                    f"(dCam={cam-tgt_cam:+.4f}  dToe={toe-tgt_toe:+.4f})")
        if verbose:
            print(f"    Cam+Toe dc={d_c:>+8.4f}mm dt={d_t:>+8.4f}mm -> "
                  f"RH={rh:>9.4f}  Cam={cam:>9.4f}  Toe={toe:>9.4f}  "
                  f"(dCam={cam-tgt_cam:+.4f} dToe={toe-tgt_toe:+.4f})")
        # Paso 2: pushrod
        d_p=0.; er=rh-tgt_rh
        if abs(er)>tol:
            d_p=-er/dRH_push
            rh+=d_p*dRH_push; cam+=d_p*dCam_push; toe+=d_p*dToe_push
            total_dp+=d_p
        steps.append({"cycle":cycle+1,"actuator":"pushrod","delta_mm":d_p,
                       "rh":rh,"cam":cam,"toe":toe,
                       "err_rh":rh-tgt_rh,"err_cam":cam-tgt_cam,"err_toe":toe-tgt_toe})
        msgs.append(f"[C{cycle+1} Pushrod  d={d_p:>+8.4f}mm] "
                    f"RH={rh:.4f}  Cam={cam:.4f}  Toe={toe:.4f}  "
                    f"(dRH={rh-tgt_rh:+.4f}mm)")
        if verbose:
            print(f"    Pushrod   d={d_p:>+8.4f}mm          -> "
                  f"RH={rh:>9.4f}  Cam={cam:>9.4f}  Toe={toe:>9.4f}  "
                  f"(dRH={rh-tgt_rh:+.4f}mm)")
    if verbose:
        print(f"\n  {'CONV' if converged else 'NO CONV'} ({n_cycles} ciclos)")
    mf={"rh_mm":rh,"camber_deg":cam,"toe_deg":toe}
    resid={k:mf[k]-target_real[k] for k in ("rh_mm","camber_deg","toe_deg")}
    def _aau(geo_cur,act):
        sp,_=apply_actuators(geo_cur,params,act)
        E0n=np.asarray(sp["E0_mm"],dtype=float)
        inp=geo_cur.inp(E0_override=E0n)
        pos,_,_=solve_upright(inp,zw_m=sp["zw_mm"]/1000.,L_ST_override=sp["L_ST_mm"]/1000.)
        Bn=np.asarray(pos["B"])*1000.; En=np.asarray(pos["E"])*1000.; Nn=np.asarray(pos["N"])*1000.
        e1=np.asarray(pos["frame_e1"]); e2=np.asarray(pos["frame_e2"]); e3=np.asarray(pos["frame_e3"])
        Sl=geo_cur.inp().S_local; Sn=Bn+(Sl[0]*e1+Sl[1]*e2+Sl[2]*e3)*1000.
        Ap=np.asarray(geo_cur.A,dtype=float); Cp=np.asarray(geo_cur.C,dtype=float)
        B0=np.asarray(geo_cur.B0,dtype=float); W0=np.asarray(geo_cur.W0,dtype=float)
        ax=unit(Cp-Ap)
        def perp(P): r=P-Ap; return r-float(np.dot(r,ax))*ax
        b0p=perp(B0); bcp=perp(Bn)
        n0=float(np.linalg.norm(b0p)); nc=float(np.linalg.norm(bcp))
        if n0>1e-10 and nc>1e-10:
            b0h=b0p/n0; bch=bcp/nc
            ca=max(-1.,min(1.,float(np.dot(b0h,bch)))); sa=float(np.dot(ax,np.cross(b0h,bch)))
            th=math.atan2(sa,ca); rW=W0-Ap
            Wn=(Ap+rW*math.cos(th)+np.cross(ax,rW)*math.sin(th)+ax*float(np.dot(ax,rW))*(1.-math.cos(th)))
        else: Wn=W0.copy()
        gn=deepcopy(geo_cur); gn.B0=Bn; gn.E0=En; gn.N0=Nn; gn.S0=Sn; gn.W0=Wn
        if gn.pushrod_wheel_body=="lower_wishbone":
            from upright_solver import compute_lower_wishbone_local_offset as _c
            gn.W_lca=_c(gn.A,gn.C,gn.B0,gn.W0)
        return gn
    at=ActuatorState(delta_pushrod_mm=total_dp,delta_camber_shim_mm=total_dc,delta_toe_adj_mm=total_dt)
    gc=_aau(geo,at)
    Jm=np.array([[params.dRH_dPush,params.dRH_dCam,params.dRH_dToe*params.dToe_dToe],
                  [params.dCam_dPush,params.dCam_dCam,params.dCam_dToe*params.dToe_dToe],
                  [params.dToe_dPush,params.dToe_dCam,params.dToe_dToe]])
    mid=next((k for k,v in VEHICLE_REGISTRY.items() if v.front is params or v.rear is params),"unknown")
    LST=float(np.linalg.norm(np.asarray(gc.S0)/1000.-np.asarray(gc.T)/1000.))*1000.
    tag=f"[{'CONV' if converged else 'NO CONV'} ({n_cycles} ciclos)]"
    msgs.append(f"{tag}  dpush={total_dp:+.4f}mm  dcam={total_dc:+.4f}mm  dtoe={total_dt:+.4f}mm")
    msgs.append(f"[Alcanzado]  RH={mf['rh_mm']:.4f}mm  Cam={mf['camber_deg']:.4f}  Toe={mf['toe_deg']:.4f}")
    msgs.append(f"[Residuos]   dRH={resid['rh_mm']:+.5f}mm  dCam={resid['camber_deg']:+.5f}  dToe={resid['toe_deg']:+.5f}")
    return CalibrationResult(
        axle=geo.axle,model_id=mid,converged=converged,iterations=n_cycles*2,
        cost=float(resid["rh_mm"]**2+resid["camber_deg"]**2+resid["toe_deg"]**2),
        baseline={"rh_mm":offs.baseline_rh,"camber_deg":offs.baseline_cam,"toe_deg":offs.baseline_toe},
        target=target_real,achieved=mf,residuals=resid,
        actuators=ActuatorState(delta_pushrod_mm=total_dp,delta_camber_shim_mm=total_dc,delta_toe_adj_mm=total_dt),
        jacobian_solver=Jm.copy(),jacobian_manual=Jm,
        geo_orig=gc,offsets=offs,
        solver_params={"zw_mm":0.,"L_ST_mm":LST,"E0_mm":gc.E0},
        messages=msgs,steps=steps,
    )


# ══════════════════════════════════════════════════════════════════
# API DE ALTO NIVEL
# ══════════════════════════════════════════════════════════════════

def calibrate_json(
    json_data: dict,
    model_id: str = "F2_2026",
    target_front: Optional[dict] = None,
    target_rear:  Optional[dict] = None,
    verbose: bool = True,
) -> Dict[str, CalibrationResult]:
    """
    Calibra ambos ejes de un JSON de setup.
    target_front/rear: None = usar setup del JSON como target.
    """
    if model_id not in VEHICLE_REGISTRY:
        raise ValueError(f"Modelo '{model_id}' no registrado. Disponibles: {list(VEHICLE_REGISTRY)}")
    vm = VEHICLE_REGISTRY[model_id]

    results = {}
    for axle in ("front", "rear"):
        params = vm.front if axle == "front" else vm.rear
        geo    = extract_geometry(json_data, axle)
        offs   = compute_offsets(geo, params)
        tgt    = (target_front if axle=="front" else target_rear) or parse_setup(json_data, axle)

        if verbose:
            print(f"\n{'='*68}")
            print(f"  EJE {axle.upper()}  [{vm.description}]")
            print(f"  Baseline manual: RH={params.rh_baseline_mm:.2f}mm  "
                  f"Cam={params.camber_baseline_deg:.4f}°  Toe={params.toe_baseline_deg:.4f}°")
            print(f"  Target:          RH={tgt['rh_mm']:.4f}mm  "
                  f"Cam={tgt['camber_deg']:.4f}°  Toe={tgt['toe_deg']:.4f}°")
            raw_t = offs.to_raw(tgt["rh_mm"], tgt["camber_deg"], tgt["toe_deg"])
            print(f"  Target raw:      RH={raw_t[0]:.4f}mm  Cam={raw_t[1]:.4f}°  Toe={raw_t[2]:.4f}°")
            print(f"{'='*68}")

        results[axle] = calibrate(geo, params, offs, tgt, verbose=verbose)

    return results


def write_calibrated(json_data: dict,
                     cal_front: CalibrationResult,
                     cal_rear:  CalibrationResult) -> dict:
    """
    Escribe los pickup points calibrados de vuelta al JSON (en metros).

    PUNTOS QUE CAMBIAN Y CÓMO SE CALCULAN:

    B0 (LWB outboard), E0 (UWB outboard), N0 (axle center):
        Posiciones directas del upright_solver en el estado calibrado
        (zw_eff, L_ST_new, E0_shim). Son la posición de diseño del upright
        en ese setup — al leer con zw=0 el solver parte de ahí.

    S0 (toe link outboard / steering arm):
        NO es S_solver (que está en zw_eff). Es la posición que S tiene
        en el upright calibrado expresada en coordenadas globales, calculada
        usando las coordenadas locales de S en el frame del upright calibrado:
            S0_new = B_cal + S_local[0]*e1_cal + S_local[1]*e2_cal + S_local[2]*e3_cal
        Esto garantiza que |S0_new - T| = L_ST_new (longitud del brazo alargado)
        y que el twist del upright quede correctamente definido.

    W0 (pushrod outboard, en el LCA):
        W0 rota solidario con el LCA alrededor del eje A-C cuando B0 se desplaza.
        Se calcula rotando W0_orig el mismo ángulo que ha rotado B0→B_cal.
        La nueva longitud |W0_cal - PRI| ≈ |W0_orig - PRI| + δ_push_mm.

    PUNTOS QUE NO CAMBIAN:
        Todos los inboard: A, C, D, F, T, PRI, rRockerC, rRockerAxis.
    """
    out = deepcopy(json_data)
    cfg = out["config"]

    def write_axle(cal: CalibrationResult, axle: str):
        ext = cfg["suspension"][axle]["external"]["pickUpPts"]
        geo = cal.geo_orig   # ya tiene los puntos calibrados tras Gauss-Seidel
        org = cfg["chassis"]["rRideR"] if axle == "rear" else None

        # geo_orig contiene los puntos outboard calibrados — escribir directamente
        ext["rFLWBO"] = _mm2json(np.asarray(geo.B0), origin_json=org)
        ext["rRLWBO"] = _mm2json(np.asarray(geo.B0), origin_json=org)
        ext["rFUWBO"] = _mm2json(np.asarray(geo.E0), origin_json=org)
        ext["rRUWBO"] = _mm2json(np.asarray(geo.E0), origin_json=org)
        ext["rAxleC"] = _mm2json(np.asarray(geo.N0), origin_json=org)
        ext["rTRO"]   = _mm2json(np.asarray(geo.S0), origin_json=org)
        ext["rPRO"]   = _mm2json(np.asarray(geo.W0), origin_json=org)

    write_axle(cal_front, "front")
    write_axle(cal_rear,  "rear")
    return out


def get_available_models() -> List[Dict[str, str]]:
    return [{"id": k, "name": v.model_id, "description": v.description}
            for k, v in VEHICLE_REGISTRY.items()]


def format_jacobian(J: np.ndarray, label: str) -> str:
    rows = ["    Δ→       pushrod    cam_shim   toe_adj"]
    for i, row_name in enumerate(["ΔRH (mm) ", "ΔCam (°) ", "ΔToe (°) "]):
        rows.append(f"    {row_name}  {J[i,0]:>+9.4f}  {J[i,1]:>+9.4f}  {J[i,2]:>+9.4f}")
    return f"  Jacobiano {label}:\n" + "\n".join(rows)


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _path = os.path.join(_HERE, "input.json")
    with open(_path) as _f: data = json.load(_f)

    vm = VEHICLE_REGISTRY["F2_2026"]

    print("═"*68)
    print("FASE 1: Verificación baseline y Jacobianos")
    print("═"*68)

    for axle in ("front","rear"):
        params = vm.front if axle=="front" else vm.rear
        geo    = extract_geometry(data, axle)
        offs   = compute_offsets(geo, params)

        m0, _, info0 = measure(geo, offs)
        print(f"\n{axle.upper()} (solver res={info0['residual_norm']:.1e}):")
        print(f"  Baseline manual: RH={params.rh_baseline_mm:.4f}mm  Cam={params.camber_baseline_deg:.4f}°  Toe={params.toe_baseline_deg:.4f}°")
        print(f"  Medido:          RH={m0['rh_mm']:.4f}mm  Cam={m0['camber_deg']:.4f}°  Toe={m0['toe_deg']:.4f}°")
        ok = (abs(m0["rh_mm"]-params.rh_baseline_mm)<1e-4 and
              abs(m0["camber_deg"]-params.camber_baseline_deg)<1e-4 and
              abs(m0["toe_deg"]-params.toe_baseline_deg)<1e-4)
        print(f"  {'✓ Offsets correctos' if ok else '✗ ERROR en offsets'}")

        # Jacobiano del solver
        J_s = np.zeros((3,3))
        m0_r = np.array(offs.to_raw(m0["rh_mm"], m0["camber_deg"], m0["toe_deg"]))
        for j, dx in enumerate([np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])]):
            act_t = ActuatorState(delta_pushrod_mm=float(dx[0]),delta_camber_shim_mm=float(dx[1]),delta_toe_adj_mm=float(dx[2]))
            sp_t, _ = apply_actuators(geo, params, act_t)
            m_t, _, _ = measure(geo, offs, zw_mm=sp_t["zw_mm"],
                                L_ST_override_mm=sp_t["L_ST_mm"], E0_override=sp_t["E0_mm"])
            J_s[:,j] = np.array(offs.to_raw(m_t["rh_mm"],m_t["camber_deg"],m_t["toe_deg"])) - m0_r

        J_m = np.array([
            [params.dRH_dPush, params.dRH_dCam, params.dRH_dToe*params.dToe_dToe],
            [params.dCam_dPush,params.dCam_dCam,params.dCam_dToe*params.dToe_dToe],
            [params.dToe_dPush,params.dToe_dCam,params.dToe_dToe],
        ])
        print(format_jacobian(J_s, "solver (numérico)"))
        print(format_jacobian(J_m, "manual (tabla p.51)"))

    print("\n\n" + "═"*68)
    print("FASE 2: Calibración completa")
    print("═"*68)
    results = calibrate_json(data, model_id="F2_2026", verbose=True)

    for axle, cal in results.items():
        print(f"\n{'─'*68}")
        print(f"{axle.upper()}: converged={cal.converged}  nfev={cal.iterations}  cost={cal.cost:.2e}")
        print(f"  Baseline: RH={cal.baseline['rh_mm']:.4f}mm  Cam={cal.baseline['camber_deg']:.4f}°  Toe={cal.baseline['toe_deg']:.4f}°")
        print(f"  Target:   RH={cal.target['rh_mm']:.4f}mm  Cam={cal.target['camber_deg']:.4f}°  Toe={cal.target['toe_deg']:.4f}°")
        print(f"  Alcanzado:RH={cal.achieved['rh_mm']:.4f}mm  Cam={cal.achieved['camber_deg']:.4f}°  Toe={cal.achieved['toe_deg']:.4f}°")
        print(f"  Errores:  ΔRH={cal.residuals['rh_mm']:+.5f}mm  ΔCam={cal.residuals['camber_deg']:+.5f}°  ΔToe={cal.residuals['toe_deg']:+.5f}°")
        print(f"  Actuadores:")
        print(f"    δ Pushrod:    {cal.actuators.delta_pushrod_mm:>+8.4f} mm")
        print(f"    δ Camber shim:{cal.actuators.delta_camber_shim_mm:>+8.4f} mm")
        print(f"    δ Toe adjuster:{cal.actuators.delta_toe_adj_mm:>+8.4f} mm")
        print(f"  Solver params: zw={cal.solver_params['zw_mm']:.4f}mm  L_ST={cal.solver_params['L_ST_mm']:.4f}mm")
