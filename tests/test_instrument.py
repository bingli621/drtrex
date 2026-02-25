import pytest
import scipp as sc
from trex.instrument import Instrument
from trex.chopper import Chopper


def test_get_chopper_frequency():

    f_bw, f_ps, f_m = Instrument.calculate_chopper_frequency(
        source_frequency=sc.scalar(14.0, unit="Hz"), rrm=12
    )
    assert sc.allclose(f_bw, 14.0 * sc.Unit("Hz"))
    assert sc.allclose(f_m, 14.0 * 12 * sc.Unit("Hz"))
    assert sc.allclose(f_ps, 14.0 * 12 * 3 / 4 * sc.Unit("Hz"))

    with pytest.raises(ValueError, match=r".* 336 Hz"):
        Instrument.calculate_chopper_frequency(
            source_frequency=sc.scalar(14.0, unit="Hz"), rrm=32
        )


def test_chopper_frequency_and_phase():

    T_OFFSET = sc.scalar(1.7, unit="ms")
    central_wavelength = sc.scalar(1.0, unit="Å")
    rrm: int = 12  # repetition rate multiplication factor
    mode = "High Flux"  # Chopper mode

    trex = Instrument(
        wavelength=central_wavelength, rrm=rrm, mode=mode, t_offset=T_OFFSET
    )
    assert sc.allclose(trex.bw1.frequency, 14.0 * sc.Unit("Hz"))
    assert sc.allclose(trex.bw1.phase, 49.2902 * sc.Unit("deg"))
    assert sc.allclose(trex.bw2.phase, 59.5116 * sc.Unit("deg"))

    assert sc.allclose(trex.ps1.frequency, 126.0 * sc.Unit("Hz"))
    assert sc.allclose(trex.ps1.phase, 179.8698 * sc.Unit("deg"))
    assert sc.allclose(trex.ps2.phase, 291.0164 * sc.Unit("deg"))

    assert sc.allclose(trex.m1.frequency, 168.0 * sc.Unit("Hz"))
    assert sc.allclose(trex.m1.phase, 64.4018 * sc.Unit("deg"))
    assert sc.allclose(trex.m2.phase, 234.5545 * sc.Unit("deg"))


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


@pytest.fixture
def trex():
    central_wavelength = sc.scalar(2.5, unit="Å")
    rrm: int = 8  # repetition rate multiplication factor
    T_OFFSET = sc.scalar(1.7, unit="ms")
    trex = Instrument(wavelength=central_wavelength, rrm=rrm, t_offset=T_OFFSET)
    return trex
