import tof
import scipp as sc
from trex.utils import centers_to_edges
from typing import Tuple, TYPE_CHECKING
from trex.utils import calculate_variable_range_at

if TYPE_CHECKING:
    from trex.instrument import Instrument
    from trex.params import MonitorParameters


class Monitor(tof.Detector):  # type: ignore
    def __init__(self, parameters: "MonitorParameters", instrument: "Instrument"):
        self.instrument = instrument
        super().__init__(name=parameters.name, distance=parameters.distance)

    def calculate_bandwidth(self, unit="Å") -> Tuple[sc.Variable, sc.Variable]:
        """Calculate the bandwidth"""
        return calculate_variable_range_at(
            component_name=self.name,
            variable_name="wavelength",
            unit=unit,
            instrument=self.instrument,
        )

    def calculate_toa_range(self, unit="us") -> Tuple[sc.Variable, sc.Variable]:
        """Calculate time of arrival at a given component"""
        return calculate_variable_range_at(
            component_name=self.name,
            variable_name="time",
            rename_dims_to="toa",
            unit=unit,
            instrument=self.instrument,
        )

    def calculate_toa(self, unit="us") -> sc.Variable:
        """Calculate time of arrival at a given component"""
        t_min, t_max = calculate_variable_range_at(
            component_name=self.name,
            variable_name="time",
            rename_dims_to="toa",
            unit=unit,
            instrument=self.instrument,
        )
        return (t_min + t_max) / 2

    def calculate_toa_bin_edges(self, unit="us") -> sc.Variable:
        toa = self.calculate_toa()
        if len(toa) == 1:
            freq = self.instrument.source.frequency
            half_period = (0.5 / freq).to(unit=toa.unit)
            return sc.concat([toa - half_period, toa + half_period], dim="toa")
        else:
            return centers_to_edges(toa)

    def estimate_toa_centroid(self, model_result: "tof.result.Result") -> sc.DataArray:

        event = model_result[self.name].data.squeeze()
        event_masked = event[~event.masks["blocked_by_others"]]

        toa_edges = self.calculate_toa_bin_edges()
        toa_binned = event_masked.bin(toa=toa_edges).drop_coords("distance")
        toa_centers = (
            toa_binned.bins.data * toa_binned.bins.coords["toa"]
        ).bins.sum() / toa_binned.bins.sum()

        return toa_centers
