import numpy as np
import tof
import scipp as sc
from typing import List, TYPE_CHECKING


if TYPE_CHECKING:
    from tof.result import Result
    from trex.instrument import Instrument
    from trex.params import DetectorParameters


class Detector(tof.Detector):  # type: ignore
    def __init__(self, parameters: "DetectorParameters", instrument: "Instrument"):
        self.instrument = instrument
        super().__init__(name=parameters.name, distance=parameters.distance)

    def wrap_frame(self, model_result: "Result"):
        model_result[self.name].data.coords["toa"] %= self.instrument.period

    def unwrap_frame(self, model_result: "Result", ei_ef_ratio=0.0) -> sc.Variable:
        """Unwrap frame and return TOA bin edges for conversion to energy"""

        instrument = self.instrument
        toa_sample = instrument.estimate_toa_at("Monitor at Sample", model_result)

        wavelength = instrument.estimate_incoming_wavelength(model_result)
        speed_in = tof.utils.wavelength_to_speed(wavelength)
        speed_out = speed_in / np.sqrt(ei_ef_ratio)
        sample_to_det_distance = (
            self.distance - instrument.monitors["Monitor at Sample"].distance
        )
        toa_det = toa_sample + (sample_to_det_distance / speed_out).rename_dims(
            {"wavelength": "toa"}
        ).to(unit="us")
        # detemine bin edges
        toa_det_min = toa_det[0]
        period = instrument.period.to(unit="us")
        toa_det_bin_edges = sc.concat([toa_det, toa_det_min + period], dim="toa")
        # unwarp frame
        num_period = toa_det_min // period
        remainder = toa_det_min % period
        # Shift TOAs into correct pulse and apply absolute offset
        data = model_result[self.name].data
        toa = data.coords["toa"]["pulse", 0]
        toa_shifted = sc.where(toa < remainder, toa + period, toa)
        data.coords["toa"]["pulse", 0] = toa_shifted + num_period * period

        return toa_det_bin_edges

    def toa_to_energy(
        self, toa_bin_edges, model_result: "Result"
    ) -> List[sc.DataArray]:

        instrument = self.instrument
        sample_det_distance = (
            self.distance - instrument.monitors["Monitor at Sample"].distance
        )
        energy_in = instrument.estimate_incoming_energy(model_result)
        toa_sample = instrument.estimate_toa_at("Monitor at Sample", model_result)

        data = model_result[self.name].data["pulse", 0]
        data_sorted = sc.sort(data, key="toa")
        data_list = []
        for i, ei_i in enumerate(energy_in):
            data_sel = data_sorted["toa", toa_bin_edges[i] : toa_bin_edges[i + 1]]
            time = data_sel.coords["toa"] - toa_sample[i]
            speed = sample_det_distance / time
            ef = tof.utils.speed_to_energy(speed)

            data_en = data_sel.assign_coords({"ei": ei_i, "ef": ef, "en": ei_i - ef})
            sizes = {"pulse": 1, "event": data_en.size}
            data_en = data_en.broadcast(sizes=sizes)
            for name, coord in data_en.coords.items():
                if name in ("distance", "ei"):
                    continue
                data_en.coords[name] = coord.broadcast(sizes=sizes)
            for name, mask in data_en.masks.items():
                data_en.masks[name] = mask.broadcast(sizes=sizes)
            data_list.append(data_en)

        return data_list
