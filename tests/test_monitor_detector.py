import scipp as sc
import pytest
from drtrex.instrument import Instrument
from drtrex.components.source import Source


def test_calculate_bandwidth(trex):
    bw_min, bw_max = trex.monitors["Monitor at Sample"].calculate_bandwidth()
    assert sc.allclose(
        ((bw_min + bw_max) / 2)[3], 2.5 * sc.Unit("Å"), rtol=sc.scalar(0.001)
    )


def test_calculate_toa_range(trex):
    t_min, t_max = trex.monitors["Monitor at Sample"].calculate_toa_range(unit="s")
    assert len(t_min) == 7
    assert sc.allclose(
        ((t_min + t_max) / 2)[3], 0.105212 * sc.Unit("s"), rtol=sc.scalar(0.001)
    )


def test_calculate_toa(trex):
    toa = trex.monitors["Monitor at Sample"].calculate_toa()
    assert sc.allclose(toa[3], 105212.0 * sc.Unit("us"), rtol=sc.scalar(0.001))


def test_estimate_toa_centroid(trex):
    res = trex.run()
    toa_m3 = trex.monitors["Monitor 3"].estimate_toa_centroid(model_result=res)
    assert len(toa_m3) == 7
    assert sc.allclose(toa_m3[0].data, 77842.5 * sc.Unit("us"), rtol=sc.scalar(0.01))

    toa_m1 = trex.monitors["Monitor 1"].estimate_toa_centroid(model_result=res)
    assert len(toa_m1) == 1
    assert sc.allclose(
        toa_m1.data, trex.monitors["Monitor 1"].calculate_toa(), rtol=sc.scalar(0.01)
    )


def test_wrap_unwrap_frame(trex):
    res = trex.run()
    assert res["Monitor 3"].data.coords["toa"].max() > trex.period
    assert res["Detector"].data.coords["toa"].max() > trex.period

    trex.monitors["Monitor 3"].wrap_frame(res)
    assert res["Monitor 3"].data.coords["toa"].max() <= trex.period
    trex.detectors["Detector"].wrap_frame(res)
    assert res["Detector"].data.coords["toa"].max() <= trex.period

    wavelength_lower_bound = trex._calculate_wavelength_lower_bound()
    trex.monitors["Monitor 3"].unwrap_frame(res, wavelength_lower_bound)
    assert res["Monitor 3"].data.coords["toa"].max() > trex.period
    trex.detectors["Detector"].unwrap_frame(res, ei_ef_ratio=0.5)
    assert res["Detector"].data.coords["toa"].max() > trex.period


def test_toa_to_energy(trex):
    res = trex.run()
    det = trex.detectors["Detector"]
    det.wrap_frame(res)
    toa_bin_edges, ei, toa_sample = det.unwrap_frame(res, ei_ef_ratio=0.2)
    det.toa_to_energy(res, toa_bin_edges, ei, toa_sample)


def test_energy_transfer_range(trex):
    res = trex.run()
    det = trex.detectors["Detector"]
    det.wrap_frame(res)
    toa_bin_edges, ei, toa_sample = det.unwrap_frame(res, ei_ef_ratio=0.2)
    en_min, en_max = det.energy_transfer_ranges(toa_bin_edges, res)
    assert sc.all(en_max <= ei)


@pytest.fixture
def trex():
    central_wavelength = sc.scalar(2.5, unit="Å")
    rrm: int = 8  # repetition rate multiplication factor
    T_OFFSET = sc.scalar(1.7, unit="ms")
    trex = Instrument(wavelength=central_wavelength, rrm=rrm, t_offset=T_OFFSET)
    trex.source = Source(facility="ess", neutrons=1_000_000, pulses=1)
    return trex
