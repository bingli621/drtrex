import pytest
import scipp as sc
import tof
from trex.chopper import Chopper, ChopperParameters
import scipp.constants as const
from scippneutron.tof import chopper_cascade


def test_create_bw_chopper(bw1_params):
    bw1 = Chopper(bw1_params)
    assert type(bw1) is Chopper


def test_angle_offset(ps1_params, ps2_params):

    angle_offset = Chopper.get_angle_offset(ps1_params.centers, ps1_params.mode)
    assert angle_offset == sc.scalar(-55.0, unit="deg")  # CCW

    angle_offset = Chopper.get_angle_offset(ps2_params.centers, ps2_params.mode)
    assert angle_offset == sc.scalar(-55.0, unit="deg")  # CW

    # making sure Chopper.get_phase do not change ChopperParameters.centers
    p2 = Chopper(ps2_params)
    angle_offset = p2.get_angle_offset(ps2_params.centers, ps2_params.mode)
    assert angle_offset == sc.scalar(-55.0, unit="deg")  # CW


def test_open_close_times(bw1_params, ps1_params):
    mn_over_h = const.m_n / const.h

    bw1 = Chopper(bw1_params)
    t_open, t_close = bw1.open_close_times()
    t_center = mn_over_h * bw1.distance * bw1_params.wavelength.to(unit="m")
    t_center += bw1_params.time_shift.to(unit="s")
    assert sc.allclose(t_center.to(unit="us"), ((t_open + t_close) / 2)[0])

    ps1 = Chopper(ps1_params)
    t_open, t_close = ps1.open_close_times()
    t_center = mn_over_h * ps1.distance * ps1_params.wavelength.to(unit="m")
    t_center += ps1_params.time_shift.to(unit="s")
    assert sc.allclose(
        t_center.to(unit="us"), ((t_open + t_close) / 2)[2]
    )  # 2 for High Flux mode, 3 for High Resolutio mode


def test_chopper_cascade(bw1_params):
    bw1 = Chopper(bw1_params)
    bw1_chopper = bw1.to_chopper_cascade()
    assert type(bw1_chopper) is chopper_cascade.Chopper


@pytest.fixture
def bw1_params():

    T_OFFSET = sc.scalar(1.7, unit="ms")
    central_wavelength = sc.scalar(1.0, unit="Å")
    frequency = sc.scalar(14.0, unit="Hz")

    bw1_params = ChopperParameters(
        name="Bandwidth Chopper 1",
        wavelength=central_wavelength,
        frequency=frequency,
        distance=sc.scalar(31.964, unit="m"),  # Source to BW chopper 1,
        centers=sc.array(dims=["cutouts"], values=(0.0,), unit="deg"),
        widths=sc.array(dims=["cutouts"], values=(61.4,), unit="deg"),
        time_shift=T_OFFSET,
        direction=tof.AntiClockwise,
    )

    return bw1_params


@pytest.fixture
def ps1_params():

    T_OFFSET = sc.scalar(1.7, unit="ms")
    central_wavelength = sc.scalar(1.0, unit="Å")
    frequency = sc.scalar(14.0, unit="Hz")

    L_P1 = sc.scalar(107.95, unit="m")  # Distance to the P-chopper 1
    THETA_POS_P = sc.array(dims=["cutouts"], values=(0, -55, -180, -235), unit="deg")
    THETA_WIDTH_P = sc.array(dims=["cutouts"], values=(20, 35, 20, 35), unit="deg")

    ps1_params = ChopperParameters(
        name="Pulse Shapping Chopper 1",
        wavelength=central_wavelength,
        frequency=frequency,
        distance=L_P1,
        centers=THETA_POS_P,
        widths=THETA_WIDTH_P,
        time_shift=T_OFFSET,
        mode="High Flux",
        direction=tof.AntiClockwise,
    )

    return ps1_params


@pytest.fixture
def ps2_params():

    T_OFFSET = sc.scalar(1.7, unit="ms")
    central_wavelength = sc.scalar(1.0, unit="Å")
    frequency = sc.scalar(14.0, unit="Hz")

    L_P2 = sc.scalar(108.05, unit="m")  # Distance to the P-chopper 2
    THETA_POS_P = sc.array(dims=["cutouts"], values=(0, -55, -180, -235), unit="deg")
    THETA_WIDTH_P = sc.array(dims=["cutouts"], values=(20, 35, 20, 35), unit="deg")

    ps2_params = ChopperParameters(
        name="Pulse Shapping Chopper 2",
        wavelength=central_wavelength,
        frequency=frequency,
        distance=L_P2,
        centers=THETA_POS_P,
        widths=THETA_WIDTH_P,
        time_shift=T_OFFSET,
        mode="High Flux",
        direction=tof.Clockwise,
    )

    return ps2_params
