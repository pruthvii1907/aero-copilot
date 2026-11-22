"""Pure-Python aerospace helper functions (no MCP / LLM dependencies).

Kept separate from the MCP server so they can be unit-tested on their own.
All inputs and outputs are SI unless the name says otherwise.
"""
from __future__ import annotations

import cmath
import math

G0 = 9.80665          # m/s^2
R_AIR = 287.05287     # J/(kg K)
GAMMA = 1.4
T0 = 288.15           # K   (sea level, ISA)
P0 = 101325.0         # Pa  (sea level, ISA)
LAPSE = -0.0065       # K/m (troposphere)
H_TROPOPAUSE = 11_000.0
H_MAX = 20_000.0
C_LIGHT = 299_792_458.0


def isa_atmosphere(altitude_m: float) -> dict:
    """International Standard Atmosphere, 0 to 20,000 m geopotential altitude."""
    h = float(altitude_m)
    if not 0.0 <= h <= H_MAX:
        raise ValueError(
            f"altitude {h:g} m is outside the supported ISA range (0 to 20,000 m)"
        )
    exponent = -G0 / (LAPSE * R_AIR)  # ~5.2559
    if h <= H_TROPOPAUSE:
        temp = T0 + LAPSE * h
        pres = P0 * (temp / T0) ** exponent
    else:
        t11 = T0 + LAPSE * H_TROPOPAUSE
        p11 = P0 * (t11 / T0) ** exponent
        temp = t11
        pres = p11 * math.exp(-G0 * (h - H_TROPOPAUSE) / (R_AIR * t11))
    rho = pres / (R_AIR * temp)
    a = math.sqrt(GAMMA * R_AIR * temp)
    mu = 1.458e-6 * temp**1.5 / (temp + 110.4)  # Sutherland's law
    return {
        "altitude_m": h,
        "temperature_k": temp,
        "pressure_pa": pres,
        "density_kg_m3": rho,
        "speed_of_sound_m_s": a,
        "dynamic_viscosity_pa_s": mu,
    }


def mach_number(velocity_m_s: float, altitude_m: float = 0.0) -> dict:
    if velocity_m_s < 0:
        raise ValueError("velocity must be non-negative")
    atm = isa_atmosphere(altitude_m)
    a = atm["speed_of_sound_m_s"]
    return {
        "mach": velocity_m_s / a,
        "speed_of_sound_m_s": a,
        "velocity_m_s": velocity_m_s,
        "altitude_m": altitude_m,
    }


def reynolds_number(velocity_m_s: float, length_m: float, altitude_m: float = 0.0) -> dict:
    if velocity_m_s < 0:
        raise ValueError("velocity must be non-negative")
    if length_m <= 0:
        raise ValueError("reference length must be positive")
    atm = isa_atmosphere(altitude_m)
    re = atm["density_kg_m3"] * velocity_m_s * length_m / atm["dynamic_viscosity_pa_s"]
    return {
        "reynolds": re,
        "density_kg_m3": atm["density_kg_m3"],
        "dynamic_viscosity_pa_s": atm["dynamic_viscosity_pa_s"],
        "velocity_m_s": velocity_m_s,
        "length_m": length_m,
        "altitude_m": altitude_m,
    }


def sandwich_panel_stiffness(
    face_modulus_gpa: float,
    face_thickness_mm: float,
    core_thickness_mm: float,
    core_shear_modulus_mpa: float,
    core_modulus_mpa: float = 0.0,
    span_m: float = 0.0,
    center_line_load_n_per_m: float = 0.0,
) -> dict:
    """Symmetric 3-layer sandwich idealisation, per metre of width.

    D = Ef*tf*d^2/2 + Ef*tf^3/6 + Ec*tc^3/12      (flexural rigidity, N*m)
    S = Gc*d^2/tc                                  (shear rigidity, N/m)
    d = tc + tf (distance between face mid-planes).
    If span and load are given: simply-supported centre deflection
    w = P L^3/(48 D) + P L/(4 S), with P a line load across the width (N/m).
    NOTE: a real C-sandwich radome (5+ layers) needs a layered analysis.
    """
    if min(face_modulus_gpa, face_thickness_mm, core_thickness_mm, core_shear_modulus_mpa) <= 0:
        raise ValueError("moduli and thicknesses must be positive")
    ef = face_modulus_gpa * 1e9
    tf = face_thickness_mm * 1e-3
    tc = core_thickness_mm * 1e-3
    gc = core_shear_modulus_mpa * 1e6
    ec = core_modulus_mpa * 1e6
    d = tc + tf
    big_d = ef * tf * d**2 / 2 + ef * tf**3 / 6 + ec * tc**3 / 12
    big_s = gc * d**2 / tc
    out = {
        "flexural_rigidity_n_m": big_d,
        "shear_rigidity_n_per_m": big_s,
        "face_mid_plane_distance_m": d,
    }
    if span_m > 0 and center_line_load_n_per_m > 0:
        bending = center_line_load_n_per_m * span_m**3 / (48 * big_d)
        shear = center_line_load_n_per_m * span_m / (4 * big_s)
        out.update(
            {
                "span_m": span_m,
                "center_deflection_mm": (bending + shear) * 1e3,
                "bending_part_mm": bending * 1e3,
                "shear_part_mm": shear * 1e3,
            }
        )
    return out


def slab_transmission(
    relative_permittivity: float,
    loss_tangent: float,
    thickness_mm: float,
    frequency_ghz: float,
) -> dict:
    """Normal-incidence power transmission through one flat dielectric slab in air.

    Uses the ABCD (transmission-matrix) method with a complex refractive index.
    Single homogeneous layer only; oblique incidence and multilayer
    (A/B/C-sandwich) stacks are NOT modelled.
    """
    if relative_permittivity < 1 or loss_tangent < 0:
        raise ValueError("need relative_permittivity >= 1 and loss_tangent >= 0")
    if thickness_mm <= 0 or frequency_ghz <= 0:
        raise ValueError("thickness and frequency must be positive")
    n = cmath.sqrt(relative_permittivity * (1 - 1j * loss_tangent))
    k0 = 2 * math.pi * frequency_ghz * 1e9 / C_LIGHT
    theta = k0 * n * thickness_mm * 1e-3
    z = 1 / n
    a = cmath.cos(theta)
    b = 1j * z * cmath.sin(theta)
    c = 1j * cmath.sin(theta) / z
    d = a
    denom = a + b + c + d
    t = abs(2 / denom) ** 2
    r = abs((a + b - c - d) / denom) ** 2
    absorbed = max(0.0, 1 - t - r)
    half_wave_mm = C_LIGHT / (2 * frequency_ghz * 1e9 * math.sqrt(relative_permittivity)) * 1e3
    return {
        "transmission_percent": 100 * t,
        "reflection_percent": 100 * r,
        "absorbed_percent": 100 * absorbed,
        "insertion_loss_db": -10 * math.log10(t),
        "half_wave_thickness_mm": half_wave_mm,
        "frequency_ghz": frequency_ghz,
        "thickness_mm": thickness_mm,
    }


def tvc_torque(thrust_n: float, arm_m: float, gimbal_angle_deg: float) -> dict:
    """Control torque from a gimballed thruster about the vehicle CG."""
    if thrust_n < 0 or arm_m <= 0:
        raise ValueError("thrust must be >= 0 and arm must be > 0")
    if abs(gimbal_angle_deg) > 90:
        raise ValueError("gimbal angle must be within +/-90 degrees")
    ang = math.radians(gimbal_angle_deg)
    side = thrust_n * math.sin(ang)
    return {
        "torque_n_m": side * arm_m,
        "side_force_n": side,
        "axial_thrust_n": thrust_n * math.cos(ang),
        "gimbal_angle_deg": gimbal_angle_deg,
    }


def tvc_required_angle(thrust_n: float, arm_m: float, torque_n_m: float) -> dict:
    """Gimbal angle needed to produce a requested torque."""
    if thrust_n <= 0 or arm_m <= 0:
        raise ValueError("thrust and arm must be positive")
    ratio = torque_n_m / (thrust_n * arm_m)
    if abs(ratio) > 1:
        raise ValueError(
            f"requested torque exceeds what {thrust_n:g} N at {arm_m:g} m can produce "
            f"(max {thrust_n * arm_m:g} N*m)"
        )
    ang = math.degrees(math.asin(ratio))
    return {"gimbal_angle_deg": ang, "max_torque_n_m": thrust_n * arm_m}
