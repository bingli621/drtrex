from typing import Literal, Dict, Tuple
import scipp as sc
import numpy as np
import tof

from trex.source import Source
from trex.chopper import Chopper
from trex.monitor import Monitor
from trex.detector import Detector
import scipp.constants as const
from trex.params import chopper_params, monitor_params, detector_params
from trex.utils import calculate_frame_at, acceptance_paths


class Instrument(object):
    """Class to model the instrument"""

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
        self.period = (1.0 / self.source.frequency).to(unit="us")

        self.choppers = {
            param.name: Chopper.from_parameters(parameters=param, instrument=self)
            for param in chopper_params
        }
        self.monitors = {param.name: Monitor(param, self) for param in monitor_params}
        self.detectors = {
            param.name: Detector(param, self) for param in detector_params
        }

    def __str__(self) -> str:
        return (
            f"T-Rex running in {self.chopper_mode} mode, "
            + f"with cententral wavelength = {self.wavelength.value:.2f} Å, "
            + f"RRM = {self.rrm}.\n"
            + f"Pulse shaping chopper frequency = {self.choppers['Pulse Shaping Chopper 1'].frequency.value:3g} Hz, "
            + f"Monochromatic chopper frequency = {self.choppers['Monochromatic Chopper 1'].frequency.value:3g} Hz"
        )

    # -----------------------------------------------------------------------
    # properties
    # -----------------------------------------------------------------------

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
        choppers = [
            tof.Chopper.from_diskchopper(diskchopper, name=name)
            for name, diskchopper in self.choppers.items()
        ]
        detectors = list((self.monitors | self.detectors).values())
        # if sample is not None:
        #     components.append(sample)
        return tof.Model(source=self.source, choppers=choppers, detectors=detectors)  # type: ignore

    # -----------------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------------

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

    def _calculate_wavelength_lower_bound(self):
        # give a margin of 0.1
        return (
            self.calculate_incoming_wavelength()[0]
            - 0.5 * self.calculate_delta_lambda()
        )

    # -----------------------------------------------------------------------
    # class methods: analytical calculation
    # -----------------------------------------------------------------------

    def calculate_delta_lambda(self) -> sc.Variable:
        """Calculate step in wavelength selected by monochromatic choppers"""
        m1 = self.choppers["Monochromatic Chopper 1"]
        m2 = self.choppers["Monochromatic Chopper 2"]
        h_over_mn = (const.Planck / const.m_n).to(unit="Å*m/s")  # 3956 Å*m/s
        m_chopper_position = (m1.distance + m2.distance) / 2
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

    # -----------------------------------------------------------------------
    # class methods: from Monte-Carlo samplings in ToF
    # -----------------------------------------------------------------------

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

    # TODO
    def estimate_toa_at(self, component_name: str, model_result: tof.result.Result):
        toa_m3_min = (
            self.monitors["Monitor 3"].estimate_toa_centroid(model_result).data[0]
        )

        wavelength_max = self.estimate_incoming_wavelength(model_result)[0]
        speed_in_max = tof.utils.wavelength_to_speed(wavelength_max)

        mon3_distance = self.monitors["Monitor at Sample"].distance
        sample_distance = self.monitors["Monitor 3"].distance
        mon3_to_sample = sample_distance - mon3_distance

        toa_sample_min = toa_m3_min + (mon3_to_sample / speed_in_max).to(unit="us")

    # -----------------------------------------------------------------------
    # class methods: mask
    # -----------------------------------------------------------------------

    def mask_from_choppers(self, last_chopper_name: str = "Monochromatic Chopper 2"):
        frame = calculate_frame_at(last_chopper_name, self)
        mask_vortices = acceptance_paths(frame=frame)
        return mask_vortices

    # -----------------------------------------------------------------------
    # class methods: wrap/unwrap frame
    # -----------------------------------------------------------------------

    def wrap_frame(self, model_result: tof.result.Result):
        """Mimic the wrapped frame in data"""
        for component in (self.monitors | self.detectors).values():
            component.wrap_frame(model_result)

    # TODO
    def unwrap_frame_detectors(self, model_result: tof.result.Result, ei_ef_ratio=0.0):
        """ei_ef_ration is defined as ei/ef, this value should be between 0 and 1."""

        toa_m3_min = (
            self.monitors["Monitor 3"].estimate_toa_centroid(model_result).data[0]
        )

        wavelength_max = self.estimate_incoming_wavelength(model_result)[0]
        speed_in_max = tof.utils.wavelength_to_speed(wavelength_max)
        speed_out_max = speed_in_max / np.sqrt(ei_ef_ratio)
        mon3_distance = self.monitors["Monitor at Sample"].distance
        sample_distance = self.monitors["Monitor 3"].distance
        mon3_to_sample = sample_distance - mon3_distance

        period = self.period.to(unit="us")
        for name, component in self.detectors.items():
            det_distance = component.distance
            sample_to_det = det_distance - sample_distance

            toa_det_min = (
                toa_m3_min
                + (mon3_to_sample / speed_in_max).to(unit="us")
                + (sample_to_det / speed_out_max).to(unit="us")
            )
            # Determine pulse wrapping
            num_period = toa_det_min // period
            remainder = toa_det_min % period
            # Shift TOAs into correct pulse and apply absolute offset
            data = model_result[name].data
            toa = data.coords["toa"]["pulse", 0]
            toa_shifted = sc.where(toa < remainder, toa + period, toa)
            data.coords["toa"]["pulse", 0] = toa_shifted + num_period * period

    def unwrap_frame(
        self,
        model_result: tof.result.Result,
        wavelength_lower_bound=None,
        ei_ef_ratio=0.0,
    ):
        wavelength_lower_bound = (
            self._calculate_wavelength_lower_bound()
            if wavelength_lower_bound is None
            else wavelength_lower_bound
        )
        for component in self.monitors.values():
            component.unwrap_frame(model_result, wavelength_lower_bound)
        self.unwrap_frame_detectors(model_result, ei_ef_ratio)
