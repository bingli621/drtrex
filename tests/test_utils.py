import pytest
import scipp as sc
from trex.instrument import Instrument
from trex.utils import calculate_frame_at


def test_calculate_frame_at(trex):
    # choppers
    frames_bw2 = calculate_frame_at("Bandwidth Chopper 2", trex)
    assert frames_bw2.subbounds().sizes["subframe"] == 1
    frames_ps2 = calculate_frame_at("Pulse Shaping Chopper 2", trex)
    assert frames_ps2.subbounds().sizes["subframe"] == 7
    frames_m2 = calculate_frame_at("Monochromatic Chopper 2", trex)
    assert frames_m2.subbounds().sizes["subframe"] == 7
    # monitors
    frames_mon1 = calculate_frame_at("Monitor 1", trex)
    assert frames_mon1.subbounds().sizes["subframe"] == 1
    bw_min, bw_max = frames_mon1.subbounds()["subframe", 0]["wavelength"]
    assert sc.allclose(bw_min, 1.6 * sc.Unit("Å"), rtol=sc.scalar(0.05))
    assert sc.allclose(bw_max, 3.2 * sc.Unit("Å"), rtol=sc.scalar(0.05))

    frames_mon2 = calculate_frame_at("Monitor 2", trex)
    assert frames_mon2.subbounds().sizes["subframe"] == 7
    frames_mon3 = calculate_frame_at("Monitor 3", trex)
    assert frames_mon3.subbounds().sizes["subframe"] == 7


@pytest.fixture
def trex():
    central_wavelength = sc.scalar(2.5, unit="Å")
    rrm: int = 8  # repetition rate multiplication factor
    T_OFFSET = sc.scalar(1.7, unit="ms")
    trex = Instrument(wavelength=central_wavelength, rrm=rrm, t_offset=T_OFFSET)
    return trex
