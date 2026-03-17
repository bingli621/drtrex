import numpy as np
import tof
import scipp as sc
from typing import List, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from tof.result import Result
    from trex.instrument import Instrument
    from trex.params import DetectorParameters


class Detector(tof.Detector):  # type: ignore
    def __init__(self, parameters: "DetectorParameters", instrument: "Instrument"):
        self.instrument = instrument
        super().__init__(name=parameters.name, distance=parameters.distance)

    def wrap_frame(self, model_result: "Result"):
        model_result[self.name].data.coords["toa"] %= self.instrument.period  # type: ignore

    def unwrap_frame(
        self, model_result: "Result", ei_ef_ratio: float = 0.0
    ) -> Tuple[sc.Variable, sc.Variable, sc.Variable]:
        """Unwrap frame and return TOA bin edges for conversion to energy"""

        instrument = self.instrument
        toa_sample = instrument.estimate_toa_at("Monitor at Sample", model_result)

        ei = instrument.estimate_ei(model_result)
        speed_in = tof.utils.energy_to_speed(ei)
        speed_out = speed_in / np.sqrt(ei_ef_ratio)
        sample_to_det_distance = (
            self.distance - instrument.monitors["Monitor at Sample"].distance
        )
        toa_det = toa_sample + (sample_to_det_distance / speed_out).to(
            unit=toa_sample.unit
        )
        # detemine bin edges
        toa_det_min = toa_det[0]
        period = instrument.period.to(unit="us")
        toa_edges = sc.concat([toa_det, toa_det_min + period], dim="rrm")
        # unwarp frame
        num_period = toa_det_min // period
        remainder = toa_det_min % period
        # Shift TOAs into correct pulse and apply absolute offset
        data = model_result[self.name].data  # type: ignore
        toa = data.coords["toa"]["pulse", 0]
        toa_shifted = sc.where(toa < remainder, toa + period, toa)
        data.coords["toa"]["pulse", 0] = toa_shifted + num_period * period

        return (toa_edges, ei, toa_sample)

    def toa_to_energy(
        self,
        model_result: "Result",
        toa_edges: sc.Variable,
        ei: sc.Variable,
        toa_sample: sc.Variable,
    ) -> List[sc.DataArray]:

        instrument = self.instrument
        sample_det_distance = (
            self.distance - instrument.monitors["Monitor at Sample"].distance
        )
        data = model_result[self.name].data["pulse", 0]  # type: ignore
        # filter NaN in coords
        data = data[~sc.isnan(data.coords["toa"])]
        data_sorted = sc.sort(data, key="toa")
        data_list = []
        for i, ei_i in enumerate(ei):  # type: ignore
            data_sel = data_sorted["toa", toa_edges[i] : toa_edges[i + 1]]
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

    def energy_transfer_ranges(
        self, toa_bin_edges: sc.Variable, model_result: "Result"
    ) -> Tuple[sc.Variable, sc.Variable]:
        """Return (en_min, en_max)"""
        instrument = self.instrument
        sample_det_distance = (
            self.distance - instrument.monitors["Monitor at Sample"].distance
        )
        ei = instrument.estimate_ei(model_result)
        toa_sample = instrument.estimate_toa_at("Monitor at Sample", model_result)

        time_gain = toa_bin_edges[:-1] - toa_sample
        speed = sample_det_distance / time_gain
        ef_max = tof.utils.speed_to_energy(speed)
        time_loss = toa_bin_edges[1:] - toa_sample
        speed = sample_det_distance / time_loss
        ef_min = tof.utils.speed_to_energy(speed)
        return (ei - ef_max, ei - ef_min)
