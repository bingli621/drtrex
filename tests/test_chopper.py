import pytest
import scipp as sc
import scipp.constants as const
from scippneutron.tof import chopper_cascade
from trex.chopper import Chopper
from trex.instrument import Instrument
from trex.params import chopper_params


def test_create_bw_chopper(trex):
    bw1_params, *_ = chopper_params
    bw1 = Chopper.from_parameters(bw1_params, instrument=trex)
    assert type(bw1) is Chopper


def test_angle_offset(trex):
    _, _, ps1_params, ps2_params, _, _ = chopper_params

    angle_offset = Chopper.get_angle_offset(
        ps1_params.slit_center, ps1_params.beam_position, trex.chopper_mode
    )
    assert angle_offset == sc.scalar(-235.0, unit="deg")  # CCW

    angle_offset = Chopper.get_angle_offset(
        ps2_params.slit_center, ps2_params.beam_position, trex.chopper_mode
    )
    assert angle_offset == sc.scalar(-235.0, unit="deg")  # CW

    # making sure Chopper.get_phase do not change ChopperParameters.centers
    p2 = Chopper.from_parameters(ps2_params, trex)
    angle_offset = p2.get_angle_offset(
        ps2_params.slit_center, ps2_params.beam_position, trex.chopper_mode
    )
    assert angle_offset == sc.scalar(-235.0, unit="deg")  # CW


def test_open_close_times(trex):
    bw1_params, _, ps1_params, *_ = chopper_params
    mn_over_h = const.m_n / const.h

    bw1 = Chopper.from_parameters(bw1_params, trex)
    t_open, t_close = bw1.open_close_times()
    t_center = mn_over_h * bw1.distance * trex.wavelength.to(unit="m")
    t_center += trex.t_offset.to(unit="s")
    assert sc.any(sc.isclose(t_center.to(unit="us"), ((t_open + t_close) / 2)))

    ps1 = Chopper.from_parameters(ps1_params, trex)
    t_open, t_close = ps1.open_close_times()
    t_center = mn_over_h * ps1.distance * trex.wavelength.to(unit="m")
    t_center += trex.t_offset.to(unit="s")
    assert sc.any(sc.isclose(t_center.to(unit="us"), ((t_open + t_close) / 2)))


def test_chopper_cascade(trex):
    bw1_params, *_ = chopper_params
    bw1 = Chopper.from_parameters(bw1_params, trex)
    bw1_chopper = bw1.to_chopper_cascade()
    assert type(bw1_chopper) is chopper_cascade.Chopper


def test_chopper_frequency(trex):
    bw1_params, _, ps1_params, _, m1_params, _ = chopper_params
    f_bw = Chopper.from_parameters(bw1_params, trex)._calculate_frequency(
        bw1_params, rrm=trex.rrm, source_frequency=trex.source.frequency
    )
    f_ps = Chopper.from_parameters(ps1_params, trex)._calculate_frequency(
        ps1_params, rrm=trex.rrm, source_frequency=trex.source.frequency
    )
    f_m = Chopper.from_parameters(m1_params, trex)._calculate_frequency(
        m1_params, rrm=trex.rrm, source_frequency=trex.source.frequency
    )
    assert sc.allclose(f_bw, 14.0 * sc.Unit("Hz"))
    assert sc.allclose(f_m, 14.0 * 4 * sc.Unit("Hz"))
    assert sc.allclose(f_ps, 14.0 * 4 * 3 / 4 * sc.Unit("Hz"))


def test_chopper_frequency_and_phase():

    T_OFFSET = sc.scalar(1.7, unit="ms")
    central_wavelength = sc.scalar(1.0, unit="Å")
    rrm: int = 12  # repetition rate multiplication factor
    mode = "High Flux"  # Chopper mode

    trex = Instrument(
        wavelength=central_wavelength, rrm=rrm, mode=mode, t_offset=T_OFFSET
    )
    bw1 = trex.choppers["Bandwidth Chopper 1"]
    bw2 = trex.choppers["Bandwidth Chopper 2"]

    assert sc.allclose(bw1.frequency, 14.0 * sc.Unit("Hz"))
    assert sc.allclose(bw1.phase, 49.2902 * sc.Unit("deg"))
    assert sc.allclose(bw2.phase, 59.5116 * sc.Unit("deg"))

    ps1 = trex.choppers["Pulse Shaping Chopper 1"]
    ps2 = trex.choppers["Pulse Shaping Chopper 2"]
    assert sc.allclose(ps1.frequency, 126.0 * sc.Unit("Hz"))
    assert sc.allclose(ps2.frequency, -126.0 * sc.Unit("Hz"))
    assert sc.allclose(
        ps1.phase, (359.8698 - 360) * sc.Unit("deg"), rtol=sc.scalar(1e-3)
    )
    assert sc.allclose(ps2.phase, (248.984 - 360) * sc.Unit("deg"))

    m1 = trex.choppers["Monochromatic Chopper 1"]
    m2 = trex.choppers["Monochromatic Chopper 2"]
    assert sc.allclose(m1.frequency, 168.0 * sc.Unit("Hz"))
    assert sc.allclose(m2.frequency, -168.0 * sc.Unit("Hz"))
    assert sc.allclose(m1.phase, 64.4018 * sc.Unit("deg"))
    assert sc.allclose(m2.phase, 125.445 * sc.Unit("deg"))


@pytest.fixture
def trex():
    T_OFFSET = sc.scalar(1.7, unit="ms")
    central_wavelength = sc.scalar(1.0, unit="Å")
    rrm = 4
    trex = Instrument(wavelength=central_wavelength, rrm=rrm, t_offset=T_OFFSET)
    return trex
