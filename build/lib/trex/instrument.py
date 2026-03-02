from typing import Literal, Dict, Tuple
import scipp as sc
import numpy as np
import tof

from trex.source import Source
from trex.chopper import Chopper
from trex.monitor import Monitor
import scipp.constants as const
from trex.params import chopper_params, monitor_params


class Instrument(object):
    def __init__(
        self,
        wavelength,
        rrm: int,
        mode: Literal["High Flux", "High Resolution"] = "High Flux",
        t_offset=sc.scalar(0.0, unit="s"),
        source=Source(facility="ess", neutrons=1_000_000, pulses=1),  # type: ignore
    ) -> None:
        """Initialize instrument with central wavelength and repitition rate RRM"""

        self.wavelength = wavelength
        self.rrm: int = rrm
        self.chopper_mode: str = mode
        self.t_offset = t_offset

        self.source = source

        self.choppers = {param.name: Chopper(param, self) for param in chopper_params}

        monitors = [Monitor(param, self) for param in monitor_params]
        # [self.mon1, self.mon2, self.mon3, self.mon_sample, self.mon_beamstop]
        self.monitors = {monitor.name: monitor for monitor in monitors}

        detectors = [self.detector]
        self.detectors = {detector.name: detector for detector in detectors}

        # DEL_L = sc.scalar(0.02, unit="m")  # Effective flight path uncertainty

    def __str__(self) -> str:
        return (
            f"T-Rex running in {self.chopper_mode} mode, "
            + f"with cententral wavelength = {self.wavelength.value:.2f} Å, "
            + f"RRM = {self.rrm}.\n"
            + f"Pulse shaping chopper frequency = {self.choppers['Pulse Shaping Chopper 1'].frequency.value:3g} Hz, "
            + f"Monochromatic chopper frequency = {self.choppers['Monochromatic Chopper 1'].frequency.value:3g} Hz"
        )

    def _calculate_time_limit(self, distance):
        """Time needed for the slowest incomnig beam out of the RRM to
        propagate to a given distance, e.g. detector position."""

        wavelength_max = self.wavelength + self.calculate_delta_lambda() * self.rrm / 2
        speed_min = tof.utils.wavelength_to_speed(wavelength_max)
        time_max = distance / speed_min
        return time_max

    def _validate_component(self, component_name):
        component = (self.choppers | self.monitors | self.detectors).get(
            component_name, None
        )
        if component is None:
            raise AttributeError(f"{component_name} does not exist.")
        return component

    @property
    def chopper_cascade(self) -> Dict:
        time_limit = self._calculate_time_limit(
            self.monitors["Beamstop Monitor"].distance
        )
        return {
            name: chopper.to_chopper_cascade(time_limit)
            for name, chopper in self.choppers.items()
        }

    @property
    def detector(self):
        L_DETECTOR = sc.scalar(166.8, unit="m")  # Source to sample in m
        return tof.Detector(distance=L_DETECTOR, name="Detector")  # type: ignore

    @property
    def model(self):
        components = list(self.choppers.values()) + list(
            (self.monitors | self.detectors).values()
        )
        # if sample is not None:
        #     components.append(sample)
        return tof.Model(source=self.source, components=components)  # type: ignore

    def calculate_delta_lambda(self) -> sc.Variable:
        """Calculate step in wavelength selected by monochromatic choppers"""
        m1 = self.choppers["Monochromatic Chopper 1"]
        m2 = self.choppers["Monochromatic Chopper 2"]
        h_over_mn = (const.Planck / const.m_n).to(unit="Å*m/s")  # 3956 Å*m/s
        m_chopper_position = (m1.axle_position.fields.z + m2.axle_position.fields.z) / 2
        delta_lambda = h_over_mn / m1.frequency / m_chopper_position.to(unit="m")
        return delta_lambda.to(unit="Å")

    def calculate_incoming_wavelength_bounds(self) -> Tuple[sc.Variable, sc.Variable]:
        bw_min, bw_max = self.monitors["Monitor at Sample"].calculate_bandwidth()
        return (bw_min, bw_max)

    def calculate_incoming_wavelength(self) -> sc.Variable:
        bw_min, bw_max = self.monitors["Monitor at Sample"].calculate_bandwidth()
        bw = sc.empty_like(bw_min)
        del_lambda = self.calculate_delta_lambda()
        idx_0 = np.where(
            ((bw_min - self.wavelength) * (bw_max - self.wavelength)).values < 0
        )[0][0]
        for i, idx in enumerate(range(-idx_0, -idx_0 + len(bw_min))):
            bw[i] = idx * del_lambda
        return bw + self.wavelength

    def calculate_incoming_energy(self) -> sc.Variable:
        wavelength_array = self.calculate_incoming_wavelength()

        speed = tof.utils.wavelength_to_speed(wavelength_array)
        energy = tof.utils.speed_to_energy(speed)
        energy_array = sc.array(dims=["energy"], values=energy.values, unit=energy.unit)
        return energy_array

    def estimate_incoming_wavelength(
        self, model_result: tof.result.Result
    ) -> sc.DataArray:
        mon3 = self.monitors["Monitor 3"]
        mon_beamstop = self.monitors["Beamstop Monitor"]
        toa_m3 = mon3.estimate_toa_centroid(model_result=model_result)
        toa_det = mon_beamstop.estimate_toa_centroid(model_result=model_result)
        time = toa_det.data - toa_m3.data
        distance = mon_beamstop.distance - mon3.distance
        speed = distance / time
        wavelength = tof.utils.speed_to_wavelength(speed)
        return wavelength.rename_dims({"toa": "wavelength"})

    def estimate_incoming_energy(self, model_result: tof.result.Result) -> sc.DataArray:
        wavelength = self.estimate_incoming_wavelength(model_result)
        speed = tof.utils.wavelength_to_speed(wavelength)
        return tof.utils.speed_to_energy(speed).rename_dims({"wavelength": "energy"})
