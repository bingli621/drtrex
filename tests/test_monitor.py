import scipp as sc
import pytest
from trex.instrument import Instrument


def test_calculate_bandwidth(trex):
    bw_min, bw_max = trex.mon_sample.calculate_bandwidth()
    assert sc.allclose(
        ((bw_min + bw_max) / 2)[3], 2.5 * sc.Unit("Å"), rtol=sc.scalar(0.001)
    )


def test_calculate_toa_range(trex):
    t_min, t_max = trex.mon_sample.calculate_toa_range(unit="s")
    assert len(t_min) == 7
    assert sc.allclose(
        ((t_min + t_max) / 2)[3], 0.105212 * sc.Unit("s"), rtol=sc.scalar(0.001)
    )


def test_calculate_toa(trex):
    toa = trex.mon_sample.calculate_toa()
    assert sc.allclose(toa[3], 105212.0 * sc.Unit("us"), rtol=sc.scalar(0.001))


def test_estimate_toa_centroid(trex):
    res = trex.model.run()
    toa_m3 = trex.mon3.estimate_toa_centroid(model_result=res)
    assert len(toa_m3) == 7
    assert sc.allclose(toa_m3[0].data, 77842.5 * sc.Unit("us"), rtol=sc.scalar(0.01))

    toa_m1 = trex.mon1.estimate_toa_centroid(model_result=res)
    assert len(toa_m1) == 1
    assert sc.allclose(toa_m1.data, trex.mon1.calculate_toa(), rtol=sc.scalar(0.01))


@pytest.fixture
def trex():
    central_wavelength = sc.scalar(2.5, unit="Å")
    rrm: int = 8  # repetition rate multiplication factor
    T_OFFSET = sc.scalar(1.7, unit="ms")
    trex = Instrument(wavelength=central_wavelength, rrm=rrm, t_offset=T_OFFSET)
    return trex
