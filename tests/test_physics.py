import math

import pytest

from aero import physics as ph


def test_isa_sea_level():
    a = ph.isa_atmosphere(0)
    assert a["temperature_k"] == pytest.approx(288.15)
    assert a["pressure_pa"] == pytest.approx(101325)
    assert a["density_kg_m3"] == pytest.approx(1.225, rel=1e-3)
    assert a["speed_of_sound_m_s"] == pytest.approx(340.29, rel=1e-3)


def test_isa_tropopause():
    a = ph.isa_atmosphere(11000)
    assert a["temperature_k"] == pytest.approx(216.65)
    assert a["pressure_pa"] == pytest.approx(22632, rel=1e-3)
    assert a["density_kg_m3"] == pytest.approx(0.3639, rel=2e-3)


def test_isa_continuous_across_tropopause():
    below = ph.isa_atmosphere(10999.9)["pressure_pa"]
    above = ph.isa_atmosphere(11000.1)["pressure_pa"]
    assert below == pytest.approx(above, rel=1e-4)


def test_isa_out_of_range():
    with pytest.raises(ValueError):
        ph.isa_atmosphere(25000)


def test_mach_at_speed_of_sound_is_one():
    a = ph.isa_atmosphere(0)["speed_of_sound_m_s"]
    assert ph.mach_number(a, 0)["mach"] == pytest.approx(1.0)


def test_reynolds_sea_level():
    re = ph.reynolds_number(50, 1.0, 0)["reynolds"]
    assert re == pytest.approx(3.42e6, rel=0.01)


def test_tvc_round_trip():
    t = ph.tvc_torque(25, 0.2, 7.0)["torque_n_m"]
    ang = ph.tvc_required_angle(25, 0.2, t)["gimbal_angle_deg"]
    assert ang == pytest.approx(7.0)


def test_tvc_unreachable_torque():
    with pytest.raises(ValueError):
        ph.tvc_required_angle(10, 0.1, 5)


def test_sandwich_known_values():
    r = ph.sandwich_panel_stiffness(70, 0.5, 10, 50)
    assert r["flexural_rigidity_n_m"] == pytest.approx(1930.8, rel=1e-3)
    assert r["shear_rigidity_n_per_m"] == pytest.approx(551_250, rel=1e-3)


def test_sandwich_deflection_present():
    r = ph.sandwich_panel_stiffness(70, 0.5, 10, 50, span_m=0.5, center_line_load_n_per_m=100)
    assert r["center_deflection_mm"] > 0


def test_half_wave_slab_is_transparent():
    half = ph.slab_transmission(4.0, 0.0, 1.0, 10.0)["half_wave_thickness_mm"]
    r = ph.slab_transmission(4.0, 0.0, half, 10.0)
    assert r["transmission_percent"] == pytest.approx(100, abs=1e-6)


def test_slab_energy_conservation():
    r = ph.slab_transmission(4.0, 0.02, 5.0, 10.0)
    total = r["transmission_percent"] + r["reflection_percent"] + r["absorbed_percent"]
    assert total == pytest.approx(100, abs=1e-6)
    assert r["absorbed_percent"] > 0


def test_slab_loss_reduces_transmission():
    lossless = ph.slab_transmission(4.0, 0.0, 7.5, 10.0)["transmission_percent"]
    lossy = ph.slab_transmission(4.0, 0.05, 7.5, 10.0)["transmission_percent"]
    assert lossy < lossless
    assert not math.isnan(lossy)
