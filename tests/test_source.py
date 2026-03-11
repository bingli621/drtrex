import pytest
import scipp as sc
from trex.instrument import Instrument
from trex.source import Source


def test_time_wavelength_tange(trex):
    t_min, t_max = trex.source.calculate_time_range(number_of_sigma=3)
    assert t_min > sc.scalar(0.0, unit="us")
    assert t_max < sc.scalar(5000.0, unit="us")
    l_min, l_max = trex.source.calculate_wavelength_range(number_of_sigma=1.5)
    assert l_min > sc.scalar(0.0, unit="Å")
    assert l_max < sc.scalar(6, unit="Å")


def test_apply_mask(trex):
    mask = trex.mask_from_choppers("Monochromatic Chopper 2")

    source = Source(facility="ess", neutrons=1_000_000)
    num = source.data.shape[1]
    source.apply_mask(mask)
    num_masked = source.data.shape[1]
    assert num_masked < num


@pytest.fixture
def trex():
    T_OFFSET = sc.scalar(1.7, unit="ms")
    central_wavelength = sc.scalar(1.0, unit="Å")
    rrm = 4
    trex = Instrument(wavelength=central_wavelength, rrm=rrm, t_offset=T_OFFSET)
    return trex
