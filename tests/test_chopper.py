import pytest
import scipp as sc
import tof
from trex.chopper import Chopper, ChopperParameters


def test_create_bw_chopper(bw1_params):
    bw1 = Chopper(bw1_params)
    assert type(bw1) is Chopper


def test_angle_offset(p1_params, p2_params):

    angle_offset = Chopper.get_angle_offset(p1_params.centers, p1_params.mode)
    assert angle_offset == sc.scalar(-55.0, unit="deg")  # CCW

    angle_offset = Chopper.get_angle_offset(p2_params.centers, p2_params.mode)
    assert angle_offset == sc.scalar(-55.0, unit="deg")  # CW

    # making sure Chopper.get_phase do not change ChopperParameters.centers
    p2 = Chopper(p2_params)
    angle_offset = p2.get_angle_offset(p2_params.centers, p2_params.mode)
    assert angle_offset == sc.scalar(-55.0, unit="deg")  # CW


@pytest.fixture
def bw1_params():

    T_OFFSET = sc.scalar(1.7, unit="ms")
    central_wavelength = sc.scalar(1.0, unit="Å")
    frequency = sc.scalar(14.0, unit="Hz")

    L_BW1 = sc.scalar(31.964, unit="m")  # Source to BW chopper 1
    THETA_POS_BW1 = sc.array(dims=["cutout"], values=(0.0,), unit="deg")
    THETA_WIDTH_BW1 = sc.array(dims=["cutout"], values=(61.4,), unit="deg")

    bw1_params = ChopperParameters(
        name="Bandwidth Chopper 1",
        wavelength=central_wavelength,
        frequency=frequency,
        distance=L_BW1,
        centers=THETA_POS_BW1,
        widths=THETA_WIDTH_BW1,
        time_shift=T_OFFSET,
        direction=tof.AntiClockwise,
    )

    return bw1_params


@pytest.fixture
def p1_params():

    T_OFFSET = sc.scalar(1.7, unit="ms")
    central_wavelength = sc.scalar(1.0, unit="Å")
    frequency = sc.scalar(14.0, unit="Hz")

    L_P1 = sc.scalar(107.95, unit="m")  # Distance to the P-chopper 1
    THETA_POS_P = sc.array(dims=["cutout"], values=(0, -55, -180, -235), unit="deg")
    THETA_WIDTH_P = sc.array(dims=["cutout"], values=(20, 35, 20, 35), unit="deg")

    p1_params = ChopperParameters(
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

    return p1_params


@pytest.fixture
def p2_params():

    T_OFFSET = sc.scalar(1.7, unit="ms")
    central_wavelength = sc.scalar(1.0, unit="Å")
    frequency = sc.scalar(14.0, unit="Hz")

    L_P2 = sc.scalar(108.05, unit="m")  # Distance to the P-chopper 2
    THETA_POS_P = sc.array(dims=["cutout"], values=(0, -55, -180, -235), unit="deg")
    THETA_WIDTH_P = sc.array(dims=["cutout"], values=(20, 35, 20, 35), unit="deg")

    p2_params = ChopperParameters(
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

    return p2_params
