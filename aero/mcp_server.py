"""MCP server exposing aerospace engineering tools over stdio.

Run standalone:   python -m aero.mcp_server
Inspect:          npx @modelcontextprotocol/inspector python -m aero.mcp_server
Any MCP client (Claude Desktop, Cursor, your own agent) can use it.

IMPORTANT: stdout carries the JSON-RPC stream. Never print() to stdout here;
use stderr for debugging.
"""
from mcp.server.fastmcp import FastMCP

from . import physics as ph

mcp = FastMCP("AeroEngineering")


def _safe(fn, **kwargs) -> dict:
    """Return errors as readable data so the LLM can explain them (Module 2, Run 3)."""
    try:
        return fn(**kwargs)
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def isa_atmosphere(altitude_m: float) -> dict:
    """International Standard Atmosphere properties at an altitude in metres (0 to 20000).
    Returns temperature (K), pressure (Pa), density (kg/m^3), speed of sound (m/s)
    and dynamic viscosity (Pa*s)."""
    return _safe(ph.isa_atmosphere, altitude_m=altitude_m)


@mcp.tool()
def mach_number(velocity_m_s: float, altitude_m: float = 0.0) -> dict:
    """Mach number for a true airspeed (m/s) at an altitude (m, ISA, 0 to 20000)."""
    return _safe(ph.mach_number, velocity_m_s=velocity_m_s, altitude_m=altitude_m)


@mcp.tool()
def reynolds_number(velocity_m_s: float, length_m: float, altitude_m: float = 0.0) -> dict:
    """Reynolds number (dimensionless) for airspeed (m/s), reference length (m,
    e.g. wing chord) and altitude (m, ISA, 0 to 20000)."""
    return _safe(
        ph.reynolds_number,
        velocity_m_s=velocity_m_s,
        length_m=length_m,
        altitude_m=altitude_m,
    )


@mcp.tool()
def sandwich_panel_stiffness(
    face_modulus_gpa: float,
    face_thickness_mm: float,
    core_thickness_mm: float,
    core_shear_modulus_mpa: float,
    core_modulus_mpa: float = 0.0,
    span_m: float = 0.0,
    center_line_load_n_per_m: float = 0.0,
) -> dict:
    """Flexural rigidity D (N*m per m width) and shear rigidity S (N/m) of a symmetric
    3-layer sandwich panel (identical thin faces + core). Optionally gives centre
    deflection (mm) of a simply supported strip under a central line load if span_m
    and center_line_load_n_per_m are both > 0. Idealisation only; multilayer
    C-sandwich needs a layered analysis."""
    return _safe(
        ph.sandwich_panel_stiffness,
        face_modulus_gpa=face_modulus_gpa,
        face_thickness_mm=face_thickness_mm,
        core_thickness_mm=core_thickness_mm,
        core_shear_modulus_mpa=core_shear_modulus_mpa,
        core_modulus_mpa=core_modulus_mpa,
        span_m=span_m,
        center_line_load_n_per_m=center_line_load_n_per_m,
    )


@mcp.tool()
def slab_transmission(
    relative_permittivity: float,
    loss_tangent: float,
    thickness_mm: float,
    frequency_ghz: float,
) -> dict:
    """Radome-style check: normal-incidence power transmission, reflection and absorption
    (percent) and insertion loss (dB) of a single flat dielectric slab in air at a
    frequency in GHz. Also returns the half-wave thickness (mm). Single layer only."""
    return _safe(
        ph.slab_transmission,
        relative_permittivity=relative_permittivity,
        loss_tangent=loss_tangent,
        thickness_mm=thickness_mm,
        frequency_ghz=frequency_ghz,
    )


@mcp.tool()
def tvc_torque(thrust_n: float, arm_m: float, gimbal_angle_deg: float) -> dict:
    """Thrust-vector-control torque (N*m) about the CG for a thrust (N), moment arm (m)
    from gimbal pivot to CG, and gimbal deflection angle (degrees). Also returns side
    force and axial thrust (N)."""
    return _safe(
        ph.tvc_torque, thrust_n=thrust_n, arm_m=arm_m, gimbal_angle_deg=gimbal_angle_deg
    )


@mcp.tool()
def tvc_required_angle(thrust_n: float, arm_m: float, torque_n_m: float) -> dict:
    """Gimbal angle (degrees) needed to produce a requested control torque (N*m) with a
    given thrust (N) and moment arm (m). Errors if the torque is unreachable."""
    return _safe(
        ph.tvc_required_angle, thrust_n=thrust_n, arm_m=arm_m, torque_n_m=torque_n_m
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
