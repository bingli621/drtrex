import pytest
import scipp as sc
import tof
from instrument.chopper import Chopper


@pytest.fixture
def p_chopper1():
    mode = "High Flux"
    T_OFFSET = sc.scalar(0.0, unit="s")
    central_wavelength = 1 * sc.Unit("Å")
    rrm: int = 8

    L_P1 = sc.scalar(value=107.95, unit="m")  # Distance to the P-chopper 1
    THETA_POS_P = sc.array(dims=["cutout"], values=(0, -55, -180, -235), unit="deg")
    THETA_WIDTH_P = sc.array(dims=["cutout"], values=(20, 35, 20, 35), unit="deg")
    angle_offset = THETA_POS_P[1] if mode == "High Flux" else THETA_POS_P[0]

    return Chopper(
        frequency=self.source.frequency,
        centers=THETA_POS_BW1,
        widths=THETA_WIDTH_BW1,
        phase=phase,
        direction=tof.AntiClockwise,
        distance=L_BW1,
        name="Bandwidth Chopper 1",
    )


def test_create_chopper(p_chopper1):
    T_OFFSET = sc.scalar(0.0, unit="s")
    central_wavelength = 1 * sc.Unit("Å")
    rrm: int = 8
    assert True


def test_phase():
    pass
