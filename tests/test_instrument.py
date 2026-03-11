import pytest
import numpy as np
import scipp as sc
from trex.instrument import Instrument


def test_calculate_delta_lambda():

    central_wavelength = sc.scalar(2.5, unit="Å")
    rrm: int = 8  # repetition rate multiplication factor
    trex = Instrument(wavelength=central_wavelength, rrm=rrm)
    delta_lambda = trex.calculate_delta_lambda()
    assert sc.allclose(delta_lambda, 0.218035 * sc.Unit("Å"))


def test_calculate_incoming_wavelength(trex):
    lambda_i = trex.calculate_incoming_wavelength()
    assert len(lambda_i) == 7
    assert sc.allclose(
        sc.min(lambda_i),
        trex.wavelength - 3 * trex.calculate_delta_lambda(),
        rtol=sc.scalar(0.01),
    )
    assert sc.allclose(
        sc.max(lambda_i),
        trex.wavelength + 3 * trex.calculate_delta_lambda(),
        rtol=sc.scalar(0.01),
    )


def test_calculate_incoming_wavelength_bounds(trex):
    lambda_i_bounds_low, lambda_i_bounds_high = (
        trex.calculate_incoming_wavelength_bounds()
    )
    assert len(lambda_i_bounds_low) == 7
    assert len(lambda_i_bounds_high) == 7


def test_calculate_incoming_energy(trex):
    ei = trex.calculate_incoming_energy()
    assert len(ei) == 7
    assert sc.allclose(ei[3], 13.0 * sc.Unit("meV"), rtol=sc.scalar(0.05))


def test_estimate_incoming_wavelength(trex):
    res = trex.model.run()
    lambda_in = trex.estimate_incoming_wavelength(res)
    lambda_expected = trex.calculate_incoming_wavelength()
    assert sc.allclose(lambda_in, lambda_expected, rtol=sc.scalar(0.1))


def test_estimate_incoming_energy(trex):
    res = trex.model.run()
    ei = trex.estimate_incoming_energy(res)
    ei_expected = trex.calculate_incoming_energy()
    assert sc.allclose(ei, ei_expected, rtol=sc.scalar(0.1))


def test_estimate_toa_at(trex):
    res = trex.model.run()
    toa_m3 = trex.monitors["Monitor 3"].estimate_toa_centroid(res)
    toa = trex.estimate_toa_at("Monitor 3", res)
    assert sc.allclose(toa, toa_m3.data)

    toa_sample = trex.estimate_toa_at("Monitor at Sample", res)
    assert np.all((toa_sample - toa).values)


def test_wrap_unwrap_frame(trex):
    res = trex.model.run()
    assert res["Monitor 3"].data.coords["toa"].max() > trex.period
    assert res["Detector"].data.coords["toa"].max() > trex.period
    trex.wrap_frame(res)
    assert res["Monitor 3"].data.coords["toa"].max() <= trex.period
    assert res["Detector"].data.coords["toa"].max() <= trex.period
    trex.unwrap_frame(res)
    assert res["Monitor 3"].data.coords["toa"].max() > trex.period
    assert res["Detector"].data.coords["toa"].max() > trex.period


@pytest.fixture
def trex():
    central_wavelength = sc.scalar(2.5, unit="Å")
    rrm: int = 8  # repetition rate multiplication factor
    T_OFFSET = sc.scalar(1.7, unit="ms")
    trex = Instrument(wavelength=central_wavelength, rrm=rrm, t_offset=T_OFFSET)
    return trex
