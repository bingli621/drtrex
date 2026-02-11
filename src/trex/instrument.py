from typing import Literal, List
import scipp as sc
import tof
from trex.chopper import ChopperParameters, Chopper
import scipp.constants as const


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
        self.choppers = [self.bw1, self.bw2, self.ps1, self.ps2, self.m1, self.m2]
        self.detectors = [
            self.monitor1,
            self.monitor2,
            self.monitor3,
            self.monitor_sample,
            self.detector,
        ]
        # DEL_L = sc.scalar(0.02, unit="m")  # Effective flight path uncertainty

    def __str__(self) -> str:
        return (
            f"T-Rex running in {self.mode} mode, "
            + f"with cententral wavelength = {self.wavelength.value:.2f} Å, "
            + f"RRM = {self.rrm}.\n"
            + f"Pulse shaping chopper frequency = {self.ps1.frequency.value:3g} Hz, "
            + f"Monochromatic chopper frequency = {self.m1.frequency.value:3g} Hz"
        )

    @staticmethod
    def calculate_chopper_frequency(source_frequency, rrm: int):
        """Get the frequencies of BW, PS and M-choppers. RRM needs to be multiples of 4
        Note:
            L_PS / L_M = 3/4"""
        if not (rrm // 4):
            raise ValueError(f"RRM = {rrm} needs to be multiples of 4.")

        bw_frequency = source_frequency
        m_frequency = bw_frequency * rrm
        m_frequency_max = sc.scalar(336, unit="Hz")  # Max frequency of PS choppers
        if m_frequency > m_frequency_max:
            raise ValueError(
                f"""Monochromatic chopper frequency = {m_frequency.value:.5g} Hz is 
                             larger than the maximum frequency of {m_frequency_max.value} Hz"""
            )
        # fp = LM/LP * fM /2, for P choppers have two sets of slits
        # P chopper can be further slowed down by deviding an integer
        # however the frequency must be multiles of BW frequency = 14 Hz
        ps_frequency = m_frequency * 0.75 / 1
        if ps_frequency > (ps_frequency_max := sc.scalar(252, unit="Hz")):
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
            centers=sc.array(dims=["cutout"], values=(0.0,), unit="deg"),
            widths=sc.array(dims=["cutout"], values=(61.4,), unit="deg"),
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
            centers=sc.array(dims=["cutout"], values=(0.0,), unit="deg"),
            widths=sc.array(dims=["cutout"], values=(63.3,), unit="deg"),
            time_shift=self.t_offset,
            direction=tof.AntiClockwise,
        )
        return Chopper(params)

    @property
    def ps1(self):
        params = ChopperParameters(
            name="Pulse Shapping Chopper 1",
            mode=self.mode,
            wavelength=self.wavelength,
            frequency=self.ps_frequency,
            distance=sc.scalar(107.95, unit="m"),
            centers=sc.array(dims=["cutout"], values=(0, -55, -180, -235), unit="deg"),
            widths=sc.array(dims=["cutout"], values=(20, 35, 20, 35), unit="deg"),
            time_shift=self.t_offset,
            direction=tof.AntiClockwise,
        )
        return Chopper(params)

    @property
    def ps2(self):
        params = ChopperParameters(
            name="Pulse Shapping Chopper 2",
            mode=self.mode,
            wavelength=self.wavelength,
            frequency=self.ps_frequency,
            distance=sc.scalar(108.05, unit="m"),
            centers=sc.array(dims=["cutout"], values=(0, -55, -180, -235), unit="deg"),
            widths=sc.array(dims=["cutout"], values=(20, 35, 20, 35), unit="deg"),
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
            centers=sc.array(dims=["cutout"], values=(-180, +5), unit="deg"),
            widths=sc.array(dims=["cutout"], values=(2.5, 4.3), unit="deg"),
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
            centers=sc.array(dims=["cutout"], values=(0, -175), unit="deg"),
            widths=sc.array(dims=["cutout"], values=(2.5, 4.3), unit="deg"),
            time_shift=self.t_offset,
            direction=tof.Clockwise,
        )
        return Chopper(params)

    @property
    def monitor1(self):
        L_BM1 = sc.scalar(41.98786, unit="m")  # Position of Beam monitor 1
        return tof.Detector(distance=L_BM1, name="monitor 1")  # type: ignore

    @property
    def monitor2(self):
        L_BM2 = sc.scalar(110.99, unit="m")  # Position of Beam monitor 2
        return tof.Detector(distance=L_BM2, name="monitor 2")  # type: ignore

    @property
    def monitor3(self):
        L_BM3 = sc.scalar(163.2, unit="m")  # Tentative position of Beam monitor 3
        return tof.Detector(distance=L_BM3, name="monitor 3")  # type: ignore

    @property
    def monitor_sample(self):
        L_SAMPLE = sc.scalar(163.8, unit="m")  # Source to sample in m
        return tof.Detector(distance=L_SAMPLE, name="sample")  # type: ignore

    @property
    def detector(self):
        L_DETECTOR = sc.scalar(166.8, unit="m")  # Source to sample in m
        return tof.Detector(distance=L_DETECTOR, name="detector")  # type: ignore

    @property
    def model(self):
        return tof.Model(source=self.source, detectors=self.detectors, choppers=self.choppers)  # type: ignore

    def calculate_delta_lambda(self) -> sc.Variable:
        """Calculate step in wavelength selected by monochromatic choppers"""
        h_over_mn = (const.Planck / const.m_n).to(unit="Å*m/s")  # 3956 Å*m/s
        m_chopper_position = (self.m1.distance + self.m2.distance) / 2
        delta_lambda = h_over_mn / self.m_frequency / m_chopper_position.to(unit="m")
        return delta_lambda.to(unit="Å")

    def calculate_bandwidth(
        self,
        source_time_range=(sc.scalar(0.2, unit="ms"), sc.scalar(3, unit="ms")),
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

    def calculate_incoming_wavelength(self, bandwidth=None) -> List[sc.Variable]:
        bandwidth = self.calculate_bandwidth() if bandwidth is None else bandwidth
        bw_min, bw_max = bandwidth
        del_lambda = self.calculate_delta_lambda()
        wavelength_list = [self.wavelength]
        for i in range(1, int(self.rrm / 2 + 1)):
            if (w_plus := self.wavelength + i * del_lambda) < bw_max:
                wavelength_list.append(w_plus)
            if (w_minus := self.wavelength - i * del_lambda) > bw_min:
                wavelength_list.append(w_minus)
        wavelength_list.sort()
        return wavelength_list

    # TODO
    def calculate_incoming_energy(self, bandwidth=None) -> List[sc.Variable]:
        wavelength_list = self.calculate_incoming_wavelength(bandwidth)
        energy_list = []
        for wavelength in wavelength_list:
            speed = tof.utils.wavelength_to_speed(wavelength)
            energy = tof.utils.speed_to_energy(speed) / 2  # TOF missed a factor of 2
            energy_list.append(energy)
        return energy_list

    # TODO
    def calculate_toa_at(self, component_name: str, RRM=False, wavelength_range=None):
        """Calculate time of arrival at a given component in microseconds
        Use wavelength_range = (min,max) to limit the range of interest"""
        try:
            component = getattr(self, component_name)
        except AttributeError:
            print(f"{component_name} does not exist.")
            return None

        distance = component.distance
        central_wavelength = self.wavelength
        delta_lambda = self.calculate_delta_lambda()
        if RRM:
            n = int(self.rrm / 2)
            wavelength_array = sc.empty(
                dims=["wavelength"], shape=[2 * n + 1], unit="Å"
            )
            for i in range(2 * n + 1):
                wavelength_array[i] = central_wavelength + (i - n) * delta_lambda

            wavelength_min, wavelength_max = (
                self.calculate_bandwidth()
                if wavelength_range is None
                else wavelength_range
            )

            wavelength_array = wavelength_array[
                (wavelength_array > wavelength_min)
                & (wavelength_array < wavelength_max)
            ]

        else:
            wavelength_array = sc.array(
                dims=["wavelength"], values=[central_wavelength.value], unit="Å"
            )

        speed_array = tof.utils.wavelength_to_speed(wavelength_array)
        toa_array = (distance / speed_array + self.t_offset).to(unit="us")
        return sc.array(dims=["toa"], values=toa_array.values, unit=toa_array.unit)

    # TODO
    @staticmethod
    def centers_to_edges(centers, dim=None):
        """return bin edges with the given center values"""
        dim = centers.dim if dim is None else dim
        mid = (centers[:-1] + centers[1:]) / 2
        first = centers[0] - (mid[0] - centers[0])
        last = centers[-1] + (centers[-1] - mid[-1])
        return sc.concat([first, mid, last], dim=dim)
