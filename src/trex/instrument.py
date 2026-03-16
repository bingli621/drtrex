from typing import Literal, Dict, Tuple, List
import scipp as sc
import numpy as np
import tof

from trex.components.source import Source
from trex.components.chopper import Chopper
from trex.components.monitor import Monitor
from trex.components.detector import Detector
import scipp.constants as const
from trex.params import chopper_params, monitor_params, detector_params
from trex.components.utils import calculate_frame_at, acceptance_paths


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

    def calculate_ei(self) -> sc.Variable:
        wavelength_array = self.calculate_incoming_wavelength()
        speed = tof.utils.wavelength_to_speed(wavelength_array)
        energy = tof.utils.speed_to_energy(speed)
        energy_array = sc.array(dims=["rrm"], values=energy.values, unit=energy.unit)
        return energy_array

    # -----------------------------------------------------------------------
    # class methods: from Monte-Carlo samplings in ToF
    # -----------------------------------------------------------------------

    def estimate_incoming_wavelength(
        self, model_result: tof.result.Result
    ) -> sc.DataArray:
        """Estimate incomnig wavelength from a simulation using ToF."""
        mon3 = self.monitors["Monitor 3"]
        mon_beamstop = self.monitors["Beamstop Monitor"]
        toa_m3 = mon3.estimate_toa_centroid(model_result=model_result)
        toa_det = mon_beamstop.estimate_toa_centroid(model_result=model_result)
        time = toa_det.data - toa_m3.data
        distance = mon_beamstop.distance - mon3.distance
        speed = distance / time
        wavelength = tof.utils.speed_to_wavelength(speed)
        return wavelength

    def estimate_ei(self, model_result: tof.result.Result) -> sc.DataArray:
        """Estimate incomnig energy Ei from a simulation using ToF."""
        wavelength = self.estimate_incoming_wavelength(model_result)
        speed = tof.utils.wavelength_to_speed(wavelength)
        return tof.utils.speed_to_energy(speed)

    def estimate_toa_at(
        self, component_name: str, model_result: tof.result.Result
    ) -> sc.DataArray:
        """Estimate TOA at a given component from a simulation using ToF."""
        m3 = self.monitors["Monitor 3"]
        toa_m3 = m3.estimate_toa_centroid(model_result).data

        wavelength = self.estimate_incoming_wavelength(model_result)
        speed = tof.utils.wavelength_to_speed(wavelength)
        component = self._validate_component(component_name)
        time = ((component.distance - m3.distance) / speed).to(unit=toa_m3.unit)
        return toa_m3 + time

    def estimate_qe_coverage(
        self, model_result: tof.result.Result, ei_ef_ratio=0.0
    ) -> Dict[str, sc.DataArray]:
        wavelength = self.estimate_incoming_wavelength(model_result)
        # ei = self.estimate_ei(model_result)
        ki = 2 * const.pi / wavelength
        detector = self.detectors["Detector"]
        toa_bin_edges = detector.unwrap_frame(model_result, ei_ef_ratio)
        en_min, en_max = detector.energy_transfer_ranges(toa_bin_edges, model_result)

        q = sc.linspace("momentum transfer", 0.0 * ki.unit, 2.5 * ki.max(), 200)
        prefactor = const.hbar**2 / 2 / const.m_n
        ei = (prefactor * ki**2).to(unit="meV")
        upper_bound = (prefactor * q * (2 * ki - q)).to(unit="meV")
        upper_bound = sc.where(upper_bound > en_max, en_max, upper_bound)
        lower_bound = (prefactor * q * (-2 * ki - q)).to(unit="meV")
        lower_bound = sc.where(lower_bound < en_min, en_min, lower_bound)

        qe_coverage = {}
        for i, ei_i in enumerate(ei):
            # mask q values if lower bound larger than upper bound
            lower_bound_i = lower_bound["rrm", i]
            upper_bound_i = upper_bound["rrm", i]
            q_mask = lower_bound_i < upper_bound_i
            lower_bound_masked = lower_bound_i[q_mask]
            upper_bound_masked = upper_bound_i[q_mask]
            coords = sc.concat(
                [q[q_mask], q[q_mask]],
                dim="momentum transfer",
            )
            data = sc.concat(
                [lower_bound_masked, upper_bound_masked], dim="momentum transfer"
            )
            qe_coverage[f"Ei = {ei_i.value:.3g} (meV)"] = sc.DataArray(
                data=data, coords={"momentum transfer": coords}
            )

        return qe_coverage

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

    def unwrap_frame(
        self,
        model_result: tof.result.Result,
        wavelength_lower_bound=None,
        ei_ef_ratio=0.0,
    ) -> List[sc.DataArray]:
        if wavelength_lower_bound is None:
            wavelength_lower_bound = self._calculate_wavelength_lower_bound()

        for monitor in self.monitors.values():
            monitor.unwrap_frame(model_result, wavelength_lower_bound)
        detector = self.detectors["Detector"]
        toa_bin_edges = detector.unwrap_frame(model_result, ei_ef_ratio)
        reduced_list = detector.toa_to_energy(toa_bin_edges, model_result)
        return reduced_list
