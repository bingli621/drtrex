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
