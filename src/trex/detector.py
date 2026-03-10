import tof
import scipp as sc
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from trex.instrument import Instrument
    from trex.params import DetectorParameters


class Detector(tof.Detector):  # type: ignore
    def __init__(self, parameters: "DetectorParameters", instrument: "Instrument"):
        self.instrument = instrument
        super().__init__(name=parameters.name, distance=parameters.distance)

    def wrap_frame(self, model_result: "tof.result.Result"):
        model_result[self.name].data.coords["toa"] %= self.instrument.period

    def unwrap_frame_detectors(self, model_result: tof.result.Result, ei_ef_ratio=0.0):
        pass
