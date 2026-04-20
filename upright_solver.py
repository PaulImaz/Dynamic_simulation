"""
Title: Upright Solver

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
from scipy.optimize import least_squares


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n < 1e-15:
        raise ValueError("Vector de norma casi cero.")
    return v / n


def rodrigues_rotate_vector(v: np.ndarray, axis_hat: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return v * c + np.cross(axis_hat, v) * s + axis_hat * (float(np.dot(axis_hat, v)) * (1.0 - c))


def orthogonal_unit_vector(n_hat: np.ndarray) -> np.ndarray:
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, n_hat))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = ref - float(np.dot(ref, n_hat)) * n_hat
    return unit(u)


def signed_angle_about_axis(v0: np.ndarray, v1: np.ndarray, axis_hat: np.ndarray) -> float:
    """
    Ángulo firmado que lleva v0 a v1 alrededor de axis_hat.
    """
    axis_hat = unit(axis_hat)

    p0 = v0 - float(np.dot(v0, axis_hat)) * axis_hat
    p1 = v1 - float(np.dot(v1, axis_hat)) * axis_hat

    n0 = norm(p0)
    n1 = norm(p1)
    if n0 < 1e-12 or n1 < 1e-12:
        raise ValueError("Vector degenerado al calcular ángulo alrededor del eje.")

    p0 = p0 / n0
    p1 = p1 / n1

    c = max(-1.0, min(1.0, float(np.dot(p0, p1))))
    s = float(np.dot(axis_hat, np.cross(p0, p1)))

    return math.atan2(s, c)


@dataclass
class Circle3D:
    center: np.ndarray
    normal_hat: np.ndarray
    radius: float
    u_hat: np.ndarray
    v_hat: np.ndarray

    def point(self, angle: float) -> np.ndarray:
        return self.center + self.radius * (
            math.cos(angle) * self.u_hat + math.sin(angle) * self.v_hat
        )


def circle_from_two_spheres(
    P1: np.ndarray,
    P2: np.ndarray,
    R1: float,
    R2: float,
    P0: np.ndarray,
) -> Circle3D:
    dvec = P2 - P1
    d = norm(dvec)
    if d < 1e-12:
        raise ValueError("Centros de esfera coincidentes.")

    n_hat = dvec / d
    a = (R1 * R1 - R2 * R2 + d * d) / (2.0 * d)
    center = P1 + a * n_hat

    rho_sq = R1 * R1 - a * a
    if rho_sq < -1e-10:
        raise ValueError("Las dos esferas no se cortan en una circunferencia real.")
    rho_sq = max(rho_sq, 0.0)
    radius = math.sqrt(rho_sq)

    u = P0 - center
    u = u - float(np.dot(u, n_hat)) * n_hat
    if norm(u) < 1e-10:
        u_hat = orthogonal_unit_vector(n_hat)
    else:
        u_hat = unit(u)

    v_hat = unit(np.cross(n_hat, u_hat))

    return Circle3D(center=center, normal_hat=n_hat, radius=radius, u_hat=u_hat, v_hat=v_hat)


def rotate_vector_minimal(v: np.ndarray, a0_hat: np.ndarray, a1_hat: np.ndarray) -> np.ndarray:
    a0_hat = unit(a0_hat)
    a1_hat = unit(a1_hat)

    cross = np.cross(a0_hat, a1_hat)
    s = norm(cross)
    c = float(np.dot(a0_hat, a1_hat))

    if s < 1e-12:
        if c > 0.0:
            return v.copy()
        axis_hat = orthogonal_unit_vector(a0_hat)
        return rodrigues_rotate_vector(v, axis_hat, math.pi)

    axis_hat = cross / s
    angle = math.atan2(s, c)
    return rodrigues_rotate_vector(v, axis_hat, angle)


def build_lower_wishbone_frame(
    A: np.ndarray,
    C: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Frame local del lower wishbone para expresar offsets cerca del pickup exterior B.

    e1: radial desde el eje A-C hasta B
    e2: tangencial
    e3: eje A-C
    """
    A = np.asarray(A, dtype=float).reshape(3)
    C = np.asarray(C, dtype=float).reshape(3)
    B = np.asarray(B, dtype=float).reshape(3)

    axis_hat = unit(C - A)
    B_axis_proj = A + float(np.dot(B - A, axis_hat)) * axis_hat
    lca_r = B - B_axis_proj
    if norm(lca_r) < 1e-12:
        raise ValueError("B demasiado cerca del eje A-C para definir el frame del lower wishbone.")

    e1 = unit(lca_r)
    e2 = unit(np.cross(axis_hat, e1))
    e3 = axis_hat
    return e1, e2, e3


def compute_lower_wishbone_local_offset(
    A: np.ndarray,
    C: np.ndarray,
    B0: np.ndarray,
    W0: np.ndarray,
) -> np.ndarray:
    """
    Devuelve el offset local de W0 respecto al frame local del lower wishbone en ride height.
    """
    e1, e2, e3 = build_lower_wishbone_frame(A, C, B0)
    r = np.asarray(W0, dtype=float).reshape(3) - np.asarray(B0, dtype=float).reshape(3)
    return np.array([
        float(np.dot(r, e1)),
        float(np.dot(r, e2)),
        float(np.dot(r, e3)),
    ], dtype=float)


@dataclass
class UprightKinematicsInput:
    # Chasis fijos
    A: np.ndarray
    C: np.ndarray
    D: np.ndarray
    F: np.ndarray
    T: np.ndarray

    # Sistema en ride height
    B0: np.ndarray
    E0: np.ndarray
    S0: np.ndarray
    W0: np.ndarray
    N0: np.ndarray

    # "upright" o "lower_wishbone"
    pushrod_wheel_body: str = "upright"

    # Offset local del pickup W respecto al lower wishbone (m)
    # Solo se usa si pushrod_wheel_body == "lower_wishbone"
    W_lca_offset_local: np.ndarray = None

    def __post_init__(self):
        for name in ["A", "C", "D", "F", "T", "B0", "E0", "S0", "W0", "N0"]:
            setattr(self, name, np.array(getattr(self, name), dtype=float).reshape(3))

        self.pushrod_wheel_body = str(self.pushrod_wheel_body).strip().lower()
        if self.pushrod_wheel_body not in ("upright", "lower_wishbone"):
            raise ValueError(
                f"pushrod_wheel_body no válido: {self.pushrod_wheel_body}. "
                "Usa 'upright' o 'lower_wishbone'."
            )

        if self.W_lca_offset_local is None:
            self.W_lca_offset_local = np.zeros(3, dtype=float)
        else:
            self.W_lca_offset_local = np.array(self.W_lca_offset_local, dtype=float).reshape(3)

        self.L_AB = norm(self.B0 - self.A)
        self.L_CB = norm(self.B0 - self.C)
        self.L_DE = norm(self.E0 - self.D)
        self.L_FE = norm(self.E0 - self.F)
        self.L_BE = norm(self.E0 - self.B0)
        self.L_ST = norm(self.S0 - self.T)

        self.lower_circle = circle_from_two_spheres(
            self.A, self.C, self.L_AB, self.L_CB, self.B0
        )
        self.upper_circle = circle_from_two_spheres(
            self.D, self.F, self.L_DE, self.L_FE, self.E0
        )

        # Frame local inicial del upright con origen en B0
        self.e1_0 = unit(self.E0 - self.B0)

        s0 = self.S0 - self.B0
        s0_perp = s0 - float(np.dot(s0, self.e1_0)) * self.e1_0
        self.e2_0 = unit(s0_perp)
        self.e3_0 = unit(np.cross(self.e1_0, self.e2_0))

        self.S_local = self._to_local(self.S0)
        self.W_local = self._to_local(self.W0)
        self.N_local = self._to_local(self.N0)

        self.lca_axis_hat = unit(self.C - self.A)

    def _to_local(self, P: np.ndarray) -> np.ndarray:
        r = P - self.B0
        return np.array([
            float(np.dot(r, self.e1_0)),
            float(np.dot(r, self.e2_0)),
            float(np.dot(r, self.e3_0)),
        ], dtype=float)


def build_upright_positions_from_state(
    inp: UprightKinematicsInput,
    state: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Construye la pose del upright a partir del estado [theta, phi, psi]:
      - B y E definen el eje principal BE
      - psi define el twist alrededor de BE
      - S y N se transportan desde el frame local del upright
      - W depende del body seleccionado:
          * upright
          * lower_wishbone (+ offset local opcional)
    """
    theta, phi, psi = np.asarray(state, dtype=float).reshape(3)

    B = inp.lower_circle.point(theta)
    E = inp.upper_circle.point(phi)

    # Eje principal del upright
    e1 = unit(E - B)

    # Transportar la dirección transversal inicial con la rotación mínima
    e2_base = rotate_vector_minimal(inp.e2_0, inp.e1_0, e1)
    e2_base = e2_base - float(np.dot(e2_base, e1)) * e1
    e2_base = unit(e2_base)

    # Twist alrededor de BE fijado por psi
    e2 = rodrigues_rotate_vector(e2_base, e1, psi)
    e2 = e2 - float(np.dot(e2, e1)) * e1
    e2 = unit(e2)

    e3 = unit(np.cross(e1, e2))

    def from_local(local_vec: np.ndarray) -> np.ndarray:
        return B + local_vec[0] * e1 + local_vec[1] * e2 + local_vec[2] * e3

    S = from_local(inp.S_local)
    N = from_local(inp.N_local)

    if inp.pushrod_wheel_body == "upright":
        W = from_local(inp.W_local)

    elif inp.pushrod_wheel_body == "lower_wishbone":
        # Ángulo real del lower arm a partir de B0 -> B alrededor del eje A-C
        theta_lca = signed_angle_about_axis(
            inp.B0 - inp.A,
            B - inp.A,
            inp.lca_axis_hat,
        )

        # Pickup geométrico base
        W_base = inp.A + rodrigues_rotate_vector(
            inp.W0 - inp.A,
            inp.lca_axis_hat,
            theta_lca,
        )

        # Frame actual del lower wishbone para aplicar offset local
        lca_e1, lca_e2, lca_e3 = build_lower_wishbone_frame(inp.A, inp.C, B)

        offset_global = (
            inp.W_lca_offset_local[0] * lca_e1
            + inp.W_lca_offset_local[1] * lca_e2
            + inp.W_lca_offset_local[2] * lca_e3
        )

        W = W_base + offset_global

    else:
        raise ValueError(
            f"pushrod_wheel_body no soportado: {inp.pushrod_wheel_body}"
        )

    return {
        "B": B,
        "E": E,
        "S": S,
        "W": W,
        "N": N,
        "frame_e1": e1,
        "frame_e2": e2,
        "frame_e3": e3,
    }


def solve_upright_for_zw(
    inp: UprightKinematicsInput,
    zw_m: float,
    x0: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict]:
    """
    Incógnitas:
      theta -> posición de B en su circunferencia
      phi   -> posición de E en su circunferencia
      psi   -> twist del upright alrededor del eje BE

    Ecuaciones:
      F1: |E-B| = const
      F2: |S-T| = const
      F3: N_z = N0_z + zw_m
    """
    z_target = inp.N0[2] - zw_m

    if x0 is None:
        p0 = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        p0 = np.array(x0, dtype=float).reshape(3)

    def residuals(p: np.ndarray) -> np.ndarray:
        pos = build_upright_positions_from_state(inp, p)
        B = pos["B"]
        E = pos["E"]
        S = pos["S"]
        N = pos["N"]

        return np.array([
            norm(E - B) - inp.L_BE,
            norm(S - inp.T) - inp.L_ST,
            N[2] - z_target,
        ], dtype=float)

    sol = least_squares(
        residuals,
        p0,
        method="trf",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=300,
    )

    pos = build_upright_positions_from_state(inp, sol.x)

    info = {
        "success": bool(sol.success),
        "message": str(sol.message),
        "nfev": int(sol.nfev),
        "residual_norm": float(np.linalg.norm(sol.fun)),
    }
    return pos, sol.x, info


DoubleWishboneInput = UprightKinematicsInput
solve_pose_for_zw = solve_upright_for_zw