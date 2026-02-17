from typing import Literal
import scipp as sc

import tof
from trex.chopper import ChopperParameters, Chopper
import scipp.constants as const
from scippneutron.tof import chopper_cascade


class Instrument(object):
    def __init__(
        self,
        wavelength,
        rrm: int,
        mode: Literal["High Flux", "High Resolution"] = "High Flux",
        t_offset=sc.scalar(0.0, unit="s"),
        source=tof.Source(facility="ess", neutrons=1_000_000, pulses=1),  # type: ignore
    ) -> None:
        """Initialize instrument with central wavelength and repitition rate RRM"""

        self.wavelength = wavelength
        self.rrm: int = rrm
        self.mode: str = mode
        self.t_offset = t_offset

        self.source = source

        self.bw_frequency, self.ps_frequency, self.m_frequency = (
            self.calculate_chopper_frequency(source.frequency, rrm)
        )
        choppers = [self.bw1, self.bw2, self.ps1, self.ps2, self.m1, self.m2]
        self.choppers = {chopper.name: chopper for chopper in choppers}

        detectors = [self.mon1, self.mon2, self.mon3, self.mon_sample, self.detector]
        self.detectors = {detector.name: detector for detector in detectors}

        # DEL_L = sc.scalar(0.02, unit="m")  # Effective flight path uncertainty

    def __str__(self) -> str:
        return (
            f"T-Rex running in {self.mode} mode, "
            + f"with cententral wavelength = {self.wavelength.value:.2f} Å, "
            + f"RRM = {self.rrm}.\n"
            + f"Pulse shaping chopper frequency = {self.ps1.frequency.value:3g} Hz, "
            + f"Monochromatic chopper frequency = {self.m1.frequency.value:3g} Hz"
        )

    def _calculate_time_limit(self, distance):
        """Time needed for the slowest incomnig beam out of the RRM to
        propagate to detector position."""
        wavelength_max = self.wavelength + self.calculate_delta_lambda() * self.rrm / 2
        speed_min = tof.utils.wavelength_to_speed(wavelength_max)
        time_max = distance / speed_min
        return time_max

    def _validate_component(self, component_name):
        component = (self.choppers | self.detectors).get(component_name, None)
        if component is None:
            raise AttributeError(f"{component_name} does not exist.")
        return component

    @staticmethod
    def calculate_chopper_frequency(source_frequency, rrm: int):
        """Get the frequencies of BW, PS and M-choppers.
        Note:
            fP = L_M/L_PS * fM /2, L_PS / L_M = 3/4 for P choppers have two sets of slits.
            P chopper can be further slowed down by deviding an integer. However the frequency
            must be multiles of BW frequency (14 Hz for ESS). Therefore, RRM needs to be
            multiples of 4.
            RRM should be smaller than 24.
        """
        m_frequency_max = sc.scalar(336, unit="Hz")  # Max frequency of PS choppers
        ps_frequency_max = sc.scalar(252, unit="Hz")

        if not (rrm // 4):
            raise ValueError(f"RRM = {rrm} needs to be multiples of 4.")

        bw_frequency = source_frequency
        m_frequency = bw_frequency * rrm
        ps_frequency = m_frequency * 0.75 / 1

        if m_frequency > m_frequency_max:
            raise ValueError(
                f"""Monochromatic chopper frequency = {m_frequency.value:.5g} Hz is 
                    arger than the maximum frequency of {m_frequency_max.value} Hz"""
            )

        if ps_frequency > ps_frequency_max:
            raise ValueError(
                f"""Pulse shaping chopper frequency = {ps_frequency.value:.5g} Hz is 
                    larger than the maximum frequency of {ps_frequency_max.value} Hz"""
            )

        return (bw_frequency, ps_frequency, m_frequency)

    @property
    def bw1(self):
        params = ChopperParameters(
            name="Bandwidth Chopper 1",
            wavelength=self.wavelength,
            frequency=self.bw_frequency,
            distance=sc.scalar(31.964, unit="m"),  # Source to BW chopper 1,
            centers=sc.array(dims=["cutouts"], values=(0.0,), unit="deg"),
            widths=sc.array(dims=["cutouts"], values=(61.4,), unit="deg"),
            time_shift=self.t_offset,
            direction=tof.AntiClockwise,
        )
        return Chopper(params)

    @property
    def bw2(self):
        params = ChopperParameters(
            name="Bandwidth Chopper 2",
            wavelength=self.wavelength,
            frequency=self.bw_frequency,
            distance=sc.scalar(39.987, unit="m"),
            centers=sc.array(dims=["cutouts"], values=(0.0,), unit="deg"),
            widths=sc.array(dims=["cutouts"], values=(63.3,), unit="deg"),
            time_shift=self.t_offset,
            direction=tof.AntiClockwise,
        )
        return Chopper(params)

    @property
    def ps1(self):
        params = ChopperParameters(
            name="Pulse Shaping Chopper 1",
            mode=self.mode,
            wavelength=self.wavelength,
            frequency=self.ps_frequency,
            distance=sc.scalar(107.95, unit="m"),
            centers=sc.array(dims=["cutouts"], values=(0, -55, -180, -235), unit="deg"),
            widths=sc.array(dims=["cutouts"], values=(20, 35, 20, 35), unit="deg"),
            time_shift=self.t_offset,
            direction=tof.AntiClockwise,
        )
        return Chopper(params)

    @property
    def ps2(self):
        params = ChopperParameters(
            name="Pulse Shaping Chopper 2",
            mode=self.mode,
            wavelength=self.wavelength,
            frequency=self.ps_frequency,
            distance=sc.scalar(108.05, unit="m"),
            centers=sc.array(dims=["cutouts"], values=(0, -55, -180, -235), unit="deg"),
            widths=sc.array(dims=["cutouts"], values=(20, 35, 20, 35), unit="deg"),
            time_shift=self.t_offset,
            direction=tof.Clockwise,
        )
        return Chopper(params)

    @property
    def m1(self):
        params = ChopperParameters(
            name="Monochromatic Chopper 1",
            mode=self.mode,
            wavelength=self.wavelength,
            frequency=self.m_frequency,
            distance=sc.scalar(161.995, unit="m"),
            centers=sc.array(dims=["cutouts"], values=(-180, +5), unit="deg"),
            widths=sc.array(dims=["cutouts"], values=(2.5, 4.3), unit="deg"),
            time_shift=self.t_offset,
            direction=tof.AntiClockwise,
        )
        return Chopper(params)

    @property
    def m2(self):
        params = ChopperParameters(
            name="Monochromatic Chopper 2",
            mode=self.mode,
            wavelength=self.wavelength,
            frequency=self.m_frequency,
            distance=sc.scalar(162.005, unit="m"),
            centers=sc.array(dims=["cutouts"], values=(0, -175), unit="deg"),
            widths=sc.array(dims=["cutouts"], values=(2.5, 4.3), unit="deg"),
            time_shift=self.t_offset,
            direction=tof.Clockwise,
        )
        return Chopper(params)

    @property
    def chopper_cascade(self):
        time_limit = self._calculate_time_limit(self.detector.distance)
        return sc.DataGroup(
            {
                name: chopper.to_chopper_cascade(time_limit)[name]
                for name, chopper in self.choppers.items()
            }
        )

    @property
    def mon1(self):
        L_BM1 = sc.scalar(41.98786, unit="m")  # Position of Beam monitor 1
        return tof.Detector(distance=L_BM1, name="Monitor 1")  # type: ignore

    @property
    def mon2(self):
        L_BM2 = sc.scalar(110.99, unit="m")  # Position of Beam monitor 2
        return tof.Detector(distance=L_BM2, name="Monitor 2")  # type: ignore

    @property
    def mon3(self):
        L_BM3 = sc.scalar(163.2, unit="m")  # Tentative position of Beam monitor 3
        return tof.Detector(distance=L_BM3, name="Monitor 3")  # type: ignore

    @property
    def mon_sample(self):
        L_SAMPLE = sc.scalar(163.8, unit="m")  # Source to sample in m
        return tof.Detector(distance=L_SAMPLE, name="Sample")  # type: ignore

    @property
    def detector(self):
        L_DETECTOR = sc.scalar(166.8, unit="m")  # Source to sample in m
        return tof.Detector(distance=L_DETECTOR, name="Detector")  # type: ignore

    @property
    def model(self):
        return tof.Model(
            source=self.source, detectors=self.detectors, choppers=self.choppers
        )  # type: ignore

    def calculate_delta_lambda(self) -> sc.Variable:
        """Calculate step in wavelength selected by monochromatic choppers"""
        h_over_mn = (const.Planck / const.m_n).to(unit="Å*m/s")  # 3956 Å*m/s
        m_chopper_position = (self.m1.distance + self.m2.distance) / 2
        delta_lambda = h_over_mn / self.m_frequency / m_chopper_position.to(unit="m")
        return delta_lambda.to(unit="Å")

    # TODO not needed
    def _calculate_bandwidth(
        self,
        source_time_range=(sc.scalar(0.0, unit="ms"), sc.scalar(4.0, unit="ms")),
        source_wavelength_range=(sc.scalar(0.25, unit="Å"), sc.scalar(7.5, unit="Å")),
    ):
        """Calculate the bandwidth determined by the Bandwidth choppers"""

        wavelength_min, wavelength_max = source_wavelength_range
        time_min, time_max = source_time_range

        open_times, close_times = self.bw1.open_close_times()
        open_bw1, close_bw1 = open_times[0], close_times[0]
        open_times, close_times = self.bw2.open_close_times()
        open_bw2, close_bw2 = open_times[0], close_times[0]
        wavelength_max = min(
            wavelength_max,
            tof.utils.speed_to_wavelength(
                self.bw1.distance / (close_bw1 - time_min.to(unit="us"))
            ),
            tof.utils.speed_to_wavelength(
                self.bw2.distance / (close_bw2 - time_min.to(unit="us"))
            ),
        )
        wavelength_min = max(
            wavelength_min,
            tof.utils.speed_to_wavelength(
                self.bw1.distance / (open_bw1 - time_max.to(unit="us"))
            ),
            tof.utils.speed_to_wavelength(
                self.bw2.distance / (open_bw2 - time_max.to(unit="us"))
            ),
        )

        return (wavelength_min, wavelength_max)

    def calculate_incoming_wavelength(self, bandwidth=None) -> sc.Variable:
        bandwidth = self._calculate_bandwidth() if bandwidth is None else bandwidth
        bw_min, bw_max = bandwidth
        del_lambda = self.calculate_delta_lambda()
        wavelength_list = [self.wavelength.value]
        for i in range(1, int(self.rrm / 2 + 1)):
            if (w_plus := self.wavelength + i * del_lambda) < bw_max:
                wavelength_list.append(w_plus.value)
            if (w_minus := self.wavelength - i * del_lambda) > bw_min:
                wavelength_list.append(w_minus.value)
        wavelength_list.sort()
        return sc.array(dims=["wavelength"], values=wavelength_list, unit="Å")

    def calculate_incoming_energy(self, bandwidth=None) -> sc.Variable:
        wavelength_array = self.calculate_incoming_wavelength(bandwidth)

        speed = tof.utils.wavelength_to_speed(wavelength_array)
        energy = tof.utils.speed_to_energy(speed)
        energy_array = sc.array(dims=["energy"], values=energy.values, unit=energy.unit)
        return energy_array

    def _calculate_frame_at(
        self, component_name: str, source_time_range, source_wavelength_range
    ):
        wavelength_min, wavelength_max = source_wavelength_range
        time_min, time_max = source_time_range

        component = self._validate_component(component_name)
        frames = chopper_cascade.FrameSequence.from_source_pulse(
            time_min=time_min,
            time_max=time_max,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )
        frames = frames.chop(self.chopper_cascade.values())
        at_component = frames.propagate_to(component.distance)
        frame = at_component[-1]  # ignore previous choppers
        return frame

    def calculate_bandwidth_at(
        self,
        component_name: str,
        source_time_range=(sc.scalar(0.0, unit="ms"), sc.scalar(4.0, unit="ms")),
        # source_wavelength_range=(sc.scalar(0.25, unit="Å"), sc.scalar(7.5, unit="Å")),
        source_wavelength_range=(sc.scalar(0.0, unit="Å"), sc.scalar(4, unit="Å")),
    ):
        """Calculate the bandwidth at a given component"""

        frame = self._calculate_frame_at(
            component_name, source_time_range, source_wavelength_range
        )
        pass

    # TODO
    def calculate_toa_at(self, component_name: str, RRM=False, wavelength_range=None):
        """Calculate time of arrival at a given component in microseconds
        Use wavelength_range = (min,max) to limit the range of interest"""

        component = self._validate_component(component_name)
        distance = component.distance
        central_wavelength = self.wavelength
        wavelength_array = (
            self.calculate_incoming_wavelength(wavelength_range)
            if RRM
            else sc.array(
                dims=["wavelength"], values=[central_wavelength.value], unit="Å"
            )
        )
        speed = tof.utils.wavelength_to_speed(wavelength_array)
        toa = (distance / speed).to(unit="us") + self.t_offset.to(unit="us")
        toa_array = sc.array(dims=["toa"], values=toa.values, unit=toa.unit)
        return toa_array
