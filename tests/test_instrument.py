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


def test_calculate_bandwidth():
    central_wavelength = sc.scalar(2.5, unit="Å")
    rrm: int = 8  # repetition rate multiplication factor
    T_OFFSET = sc.scalar(1.7, unit="ms")
    trex = Instrument(wavelength=central_wavelength, rrm=rrm, t_offset=T_OFFSET)
    bw_min, bw_max = trex._calculate_bandwidth(
        source_time_range=(sc.scalar(0.2, unit="ms"), sc.scalar(3.0, unit="ms")),
        source_wavelength_range=(sc.scalar(0.25, unit="Å"), sc.scalar(7.5, unit="Å")),
    )
    assert sc.allclose(bw_min, 1.75011 * sc.Unit("Å"))
    assert sc.allclose(bw_max, 3.26968 * sc.Unit("Å"))


def test_validate_component(trex_cold):
    bw1 = trex_cold._validate_component("Bandwidth Chopper 1")
    assert type(bw1) is Chopper

    with pytest.raises(AttributeError):
        trex_cold._validate_component("BandwidthChopper1")


# TODO
def test_calculate_frame_at(trex_cold):

    source_time_range = (sc.scalar(0.0, unit="ms"), sc.scalar(4.0, unit="ms"))
    source_wavelength_range = (sc.scalar(0.0, unit="Å"), sc.scalar(4.0, unit="Å"))

    frames_bw2 = trex_cold._calculate_frame_at(
        "Bandwidth Chopper 2", source_time_range, source_wavelength_range
    )
    frames_ps2 = trex_cold._calculate_frame_at(
        "Pulse Shaping Chopper 2", source_time_range, source_wavelength_range
    )
    frames_m2 = trex_cold._calculate_frame_at(
        "Monochromatic Chopper 2", source_time_range, source_wavelength_range
    )

    pass


def test_calculate_bandwidth_at():

    bw_min, bw_max = trex_cold.calculate_bandwidth_at(
        component_name="monitor_sample",
        source_time_range=(sc.scalar(0.0, unit="ms"), sc.scalar(4.0, unit="ms")),
        source_wavelength_range=(sc.scalar(0.0, unit="Å"), sc.scalar(4.0, unit="Å")),
    )
    assert sc.allclose(bw_min, 1.75011 * sc.Unit("Å"))
    assert sc.allclose(bw_max, 3.26968 * sc.Unit("Å"))


def test_calculate_incoming_wavelength(trex_cold):
    # default bandwidth range
    lambda_i = trex_cold.calculate_incoming_wavelength()
    assert len(lambda_i) == 7
    assert sc.allclose(sc.min(lambda_i), 1.84589 * sc.Unit("Å"))
    assert sc.allclose(sc.max(lambda_i), 3.15411 * sc.Unit("Å"))

    # limit the bandwith
    lambda_i = trex_cold.calculate_incoming_wavelength(
        bandwidth=(1.9 * sc.Unit("Å"), 3.1 * sc.Unit("Å"))
    )
    assert len(lambda_i) == 5


def test_calculate_incoming_energy(trex_cold):
    bw = trex_cold._calculate_bandwidth()
    ei = trex_cold.calculate_incoming_energy(bw)
    assert len(ei) == 7
    assert sc.allclose(ei[3], 13.08867 * sc.Unit("meV"))


# #TODO
# def test_calculate_toa_at():
#     central_wavelength = sc.scalar(1, unit="Å")
#     rrm: int = 8  # repetition rate multiplication factor
#     T_OFFSET = sc.scalar(1.7, unit="ms")
#     trex = Instrument(wavelength=central_wavelength, rrm=rrm, t_offset=T_OFFSET)

#     toa = trex.calculate_toa_at("monitor_sample")
#     assert sc.allclose(toa, 43105.1 * sc.Unit("us"))

#     toa_rrm = trex.calculate_toa_at("monitor_sample", RRM=True)
#     assert len(toa_rrm) == 7
#     assert sc.allclose(toa_rrm[3], 43105.1 * sc.Unit("us"))


@pytest.fixture
def trex_cold():
    central_wavelength = sc.scalar(2.5, unit="Å")
    rrm: int = 8  # repetition rate multiplication factor
    T_OFFSET = sc.scalar(1.7, unit="ms")
    trex = Instrument(wavelength=central_wavelength, rrm=rrm, t_offset=T_OFFSET)
    return trex
