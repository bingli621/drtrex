import pytest
import scipp as sc

from drtrex.instrument import Instrument
from drtrex.components.utils import calculate_frame_at, acceptance_paths


def test_calculate_frame_at(trex):
    # choppers
    frame_bw2 = calculate_frame_at("Bandwidth Chopper 2", trex)
    assert frame_bw2.subbounds().sizes["subframe"] == 1
    frame_ps2 = calculate_frame_at("Pulse Shaping Chopper 2", trex)
    assert frame_ps2.subbounds().sizes["subframe"] == 7
    frame_m2 = calculate_frame_at("Monochromatic Chopper 2", trex)
    assert frame_m2.subbounds().sizes["subframe"] == 7
    # monitors
    frame_mon1 = calculate_frame_at("Monitor 1", trex)
    assert frame_mon1.subbounds().sizes["subframe"] == 1
    bw_min, bw_max = frame_mon1.subbounds()["subframe", 0]["wavelength"]
    assert sc.allclose(bw_min, 1.6 * sc.Unit("Å"), rtol=sc.scalar(0.05))
    assert sc.allclose(bw_max, 3.2 * sc.Unit("Å"), rtol=sc.scalar(0.05))

    frame_mon2 = calculate_frame_at("Monitor 2", trex)
    assert frame_mon2.subbounds().sizes["subframe"] == 7
    frame_mon3 = calculate_frame_at("Monitor 3", trex)
    assert frame_mon3.subbounds().sizes["subframe"] == 7


def test_acceptance_path(trex):
    frame_m2 = calculate_frame_at("Monochromatic Chopper 2", trex)
    subframe_vortices = acceptance_paths(frame=frame_m2)
    assert len(subframe_vortices) == 7


@pytest.fixture
def trex():
    central_wavelength = sc.scalar(2.5, unit="Å")
    rrm: int = 8  # repetition rate multiplication factor
    T_OFFSET = sc.scalar(1.7, unit="ms")
    trex = Instrument(wavelength=central_wavelength, rrm=rrm, t_offset=T_OFFSET)
    return trex
