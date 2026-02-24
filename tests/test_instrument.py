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


def test_validate_component(trex_cold):
    bw1 = trex_cold._validate_component("Bandwidth Chopper 1")
    assert type(bw1) is Chopper

    with pytest.raises(AttributeError):
        trex_cold._validate_component("BandwidthChopper1")


def test_calculate_frame_at(trex_cold):
    # choppers
    frames_bw2 = trex_cold._calculate_frame_at("Bandwidth Chopper 2")
    assert frames_bw2.subbounds().sizes["subframe"] == 1
    frames_ps2 = trex_cold._calculate_frame_at("Pulse Shaping Chopper 2")
    assert frames_ps2.subbounds().sizes["subframe"] == 7
    frames_m2 = trex_cold._calculate_frame_at("Monochromatic Chopper 2")
    assert frames_m2.subbounds().sizes["subframe"] == 7
    # monitors
    frames_mon1 = trex_cold._calculate_frame_at("Monitor 1")
    assert frames_mon1.subbounds().sizes["subframe"] == 1
    bw_min, bw_max = frames_mon1.subbounds()["subframe", 0]["wavelength"]
    assert sc.allclose(bw_min, 1.6 * sc.Unit("Å"), rtol=sc.scalar(0.05))
    assert sc.allclose(bw_max, 3.2 * sc.Unit("Å"), rtol=sc.scalar(0.05))

    frames_mon2 = trex_cold._calculate_frame_at("Monitor 2")
    assert frames_mon2.subbounds().sizes["subframe"] == 7
    frames_mon3 = trex_cold._calculate_frame_at("Monitor 3")
    assert frames_mon3.subbounds().sizes["subframe"] == 7


def test_calculate_bandwidth_at(trex_cold):
    bw_min, bw_max = trex_cold.calculate_bandwidth_at(component_name="Sample")
    assert sc.allclose(
        ((bw_min + bw_max) / 2)[3], 2.5 * sc.Unit("Å"), rtol=sc.scalar(0.001)
    )


def test_calculate_toa_range_at(trex_cold):
    t_min, t_max = trex_cold.calculate_toa_range_at(component_name="Sample", unit="s")
    assert sc.allclose(
        ((t_min + t_max) / 2)[3], 0.105212 * sc.Unit("s"), rtol=sc.scalar(0.001)
    )


def test_calculate_toa_at(trex_cold):
    toa = trex_cold.calculate_toa_at(component_name="Sample")
    assert sc.allclose(toa[3], 105212.0 * sc.Unit("us"), rtol=sc.scalar(0.001))


def test_calculate_incoming_wavelength(trex_cold):
    lambda_i = trex_cold.calculate_incoming_wavelength()
    assert len(lambda_i) == 7
    assert sc.allclose(
        sc.min(lambda_i),
        trex_cold.wavelength - 3 * trex_cold.calculate_delta_lambda(),
        rtol=sc.scalar(0.01),
    )
    assert sc.allclose(
        sc.max(lambda_i),
        trex_cold.wavelength + 3 * trex_cold.calculate_delta_lambda(),
        rtol=sc.scalar(0.01),
    )


def test_calculate_incoming_wavelength_bounds(trex_cold):
    lambda_i_bounds_low, lambda_i_bounds_high = (
        trex_cold.calculate_incoming_wavelength_bounds()
    )
    assert len(lambda_i_bounds_low) == 7
    assert len(lambda_i_bounds_high) == 7


def test_calculate_incoming_energy(trex_cold):
    ei = trex_cold.calculate_incoming_energy()
    assert len(ei) == 7
    assert sc.allclose(ei[3], 13.0 * sc.Unit("meV"), rtol=sc.scalar(0.05))


def test_estimate_toa_centroid_at(trex_cold):
    res = trex_cold.model.run()
    toa_m3 = trex_cold.estimate_toa_centroid_at("Monitor 3", model_result=res)
    assert len(toa_m3) == 7
    assert sc.allclose(toa_m3[0].data, 77842.5 * sc.Unit("us"), rtol=sc.scalar(0.01))

    toa_m1 = trex_cold.estimate_toa_centroid_at("Monitor 1", model_result=res)
    assert len(toa_m1) == 1
    assert sc.allclose(
        toa_m1.data, trex_cold.calculate_toa_at("Monitor 1"), rtol=sc.scalar(0.01)
    )


def test_estimate_incoming_wavelength(trex_cold):
    res = trex_cold.model.run()
    lambda_in = trex_cold.estimate_incoming_wavelength(res)
    lambda_expected = trex_cold.calculate_incoming_wavelength()
    assert sc.allclose(lambda_in, lambda_expected, rtol=sc.scalar(0.1))


def test_estimate_incoming_energy(trex_cold):
    res = trex_cold.model.run()
    ei = trex_cold.estimate_incoming_energy(res)
    ei_expected = trex_cold.calculate_incoming_energy()
    assert sc.allclose(ei, ei_expected, rtol=sc.scalar(0.1))


@pytest.fixture
def trex_cold():
    central_wavelength = sc.scalar(2.5, unit="Å")
    rrm: int = 8  # repetition rate multiplication factor
    T_OFFSET = sc.scalar(1.7, unit="ms")
    trex = Instrument(wavelength=central_wavelength, rrm=rrm, t_offset=T_OFFSET)
    return trex
