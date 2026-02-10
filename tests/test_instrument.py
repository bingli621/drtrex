import pytest
import scipp as sc
from trex.instrument import Instrument


def test_get_chopper_frequency():

    f_bw, f_ps, f_m = Instrument.get_chopper_frequency(
        source_frequency=sc.scalar(14.0, unit="Hz"), rrm=12
    )
    assert sc.allclose(f_bw, 14.0 * sc.Unit("Hz"))
    assert sc.allclose(f_m, 14.0 * 12 * sc.Unit("Hz"))
    assert sc.allclose(f_ps, 14.0 * 12 * 3 / 4 * sc.Unit("Hz"))

    with pytest.raises(ValueError, match=r".* 336 Hz"):
        Instrument.get_chopper_frequency(
            source_frequency=sc.scalar(14.0, unit="Hz"), rrm=32
        )


def test_trex():

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
