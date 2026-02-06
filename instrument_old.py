import scipp as sc
import numpy as np
from typing import Literal
import tof
import scipp.constants as const


class Instrument(object):

    def __init__(
        self,
        wavelength,
        rrm: int,
        mode: Literal["High Flux", "High Resolution"] = "High Flux",
        t_offset=sc.scalar(0.0, unit="s"),
        source=tof.Source(facility="ess", neutrons=1_000_000, pulses=1),  # type: ignore
    ):
        self.wavelength = wavelength
        self.rrm: int = rrm
        self.mode: str = mode
        self.t_offset = t_offset

        self.source = source
        # self.source_wavelength_range = sc.array(dims=['wavelength'], values=[],unit='Å')
        self.m_frequency = self.source.frequency * self.rrm
        # fp = LM/LP * fM /2, for P choppers have two sets of slits
        # P chopper can be further slowed down by deviding an integer
        # however the frequency must be multiles of BW frequency = 14 Hz
        self.p_frequency = self.m_frequency * 0.75 / 1

        self.p_frequency_max = sc.scalar(252, unit="Hz")  # Max frequency of PS choppers
        self.m_frequency_max = sc.scalar(336, unit="Hz")  # Max frequency of PS choppers

        DEL_L = sc.scalar(0.02, unit="m")  # Effective flight path uncertainty

    def __str__(self) -> str:
        return f"""T-Rex running with cententral wavelength = {self.wavelength:.5g} Å, RRM = {self.rrm}."""

    @staticmethod
    def centers_to_edges(centers, dim=None):
        """return bin edges with the given center values"""
        dim = centers.dim if dim is None else dim
        mid = (centers[:-1] + centers[1:]) / 2
        first = centers[0] - (mid[0] - centers[0])
        last = centers[-1] + (centers[-1] - mid[-1])
        return sc.concat([first, mid, last], dim=dim)

    def bw_chopper1(self):
        L_BW1 = sc.scalar(31.964, unit="m")  # Source to BW chopper 1
        THETA_POS_BW1 = sc.array(dims=["cutout"], values=(0.0,), unit="deg")
        THETA_WIDTH_BW1 = sc.array(dims=["cutout"], values=(61.4,), unit="deg")
        phase = self.get_chopper_phase(
            position=L_BW1,
            wavelength=self.wavelength,
            frequency=self.source.frequency,
            time_shift=self.t_offset,
        )

        return tof.Chopper(
            frequency=self.source.frequency,
            centers=THETA_POS_BW1,
            widths=THETA_WIDTH_BW1,
            phase=phase,
            direction=tof.AntiClockwise,
            distance=L_BW1,
            name="Bandwidth Chopper 1",
        )  # type: ignore

    def bw_chopper2(self):

        L_BW2 = sc.scalar(39.987, unit="m")  # Source to BW chopper 1
        THETA_POS_BW2 = sc.array(dims=["cutout"], values=(0.0,), unit="deg")  # Wi
        THETA_WIDTH_BW2 = sc.array(dims=["cutout"], values=(63.3,), unit="deg")

        return tof.Chopper(
            frequency=self.source.frequency,
            centers=THETA_POS_BW2,
            widths=THETA_WIDTH_BW2,
            phase=self.get_chopper_phase(
                position=L_BW2,
                wavelength=self.wavelength,
                frequency=self.source.frequency,
                time_shift=self.t_offset,
            ),
            direction=tof.AntiClockwise,
            distance=L_BW2,
            name="Bandwidth Chopper 2",
        )  # type: ignore

    def p_chopper1(self):
        L_P1 = sc.scalar(107.95, unit="m")  # Distance to the P-chopper 1
        THETA_POS_P = sc.array(dims=["cutout"], values=(0, -55, -180, -235), unit="deg")
        THETA_WIDTH_P = sc.array(dims=["cutout"], values=(20, 35, 20, 35), unit="deg")

        angle_offset = THETA_POS_P[1] if self.mode == "High Flux" else THETA_POS_P[0]
        phase = self.get_chopper_phase(
            L_P1,
            self.wavelength,
            self.p_frequency,
            angle_offset=angle_offset,  # negative to advance in time
            time_shift=self.t_offset,
        )

        return tof.Chopper(
            frequency=self.p_frequency,
            centers=THETA_POS_P,
            widths=THETA_WIDTH_P,
            phase=phase,
            direction=tof.AntiClockwise,
            distance=L_P1,
            name="Pulse Shaping Chopper 1 CCW",
        )  # type: ignore

    def p_chopper2(self):
        L_P2 = sc.scalar(108.05, unit="m")  # Distance to the P-chopper 2
        THETA_POS_P = sc.array(dims=["cutout"], values=(0, -55, -180, -235), unit="deg")
        THETA_WIDTH_P = sc.array(dims=["cutout"], values=(20, 35, 20, 35), unit="deg")

        angle_offset = THETA_POS_P[1] if self.mode == "High Flux" else THETA_POS_P[0]
        phase = self.get_chopper_phase(
            L_P2,
            self.wavelength,
            self.p_frequency,
            angle_offset=angle_offset * (-1),  # positive to delay in time
            time_shift=self.t_offset,
        )

        return tof.Chopper(
            frequency=self.p_frequency,
            centers=THETA_POS_P,
            widths=THETA_WIDTH_P,
            phase=phase,
            distance=L_P2,
            name="Pulse Shaping Chopper 2 CW",
        )  # type: ignore

    def m_chopper1(self):
        L_M1 = sc.scalar(161.995, unit="m")  # Distance to the M-chopper 1
        THETA_POS_M = sc.array(dims=["cutout"], values=(-180, +5), unit="deg")
        THETA_WIDTH_M = sc.array(dims=["cutout"], values=(2.5, 4.3), unit="deg")

        angle_offset = THETA_POS_M[1] if self.mode == "High Flux" else THETA_POS_M[0]
        phase = self.get_chopper_phase(
            L_M1,
            self.wavelength,
            self.m_frequency,
            angle_offset=angle_offset,  # positive to delay in time
            time_shift=self.t_offset,
        )

        return tof.Chopper(
            frequency=self.m_frequency,
            centers=THETA_POS_M,
            widths=THETA_WIDTH_M,
            phase=phase,
            direction=tof.AntiClockwise,
            distance=L_M1,
            name="Monochromatic Chopper 1 CCW",
        )  # type: ignore

    def m_chopper2(self):
        L_M2 = sc.scalar(162.005, unit="m")  # Distance to the M-chopper 2
        THETA_POS_M = sc.array(dims=["cutout"], values=(0, -175), unit="deg")
        THETA_WIDTH_M = sc.array(dims=["cutout"], values=(2.5, 4.3), unit="deg")

        angle_offset = THETA_POS_M[1] if self.mode == "High Flux" else THETA_POS_M[0]
        phase = self.get_chopper_phase(
            L_M2,
            self.wavelength,
            self.m_frequency,
            angle_offset=angle_offset * (-1),  # positive to delay in time
            time_shift=self.t_offset,
        )

        return tof.Chopper(
            frequency=self.m_frequency,
            centers=THETA_POS_M,
            widths=THETA_WIDTH_M,
            phase=phase,
            distance=L_M2,
            name="Monochromatic Chopper 2 CW",
        )  # type: ignore

    def monitor1(self):
        L_BM1 = sc.scalar(41.98786, unit="m")  # Position of Beam monitor 1
        return tof.Detector(distance=L_BM1, name="monitor 1")  # type: ignore

    def monitor2(self):
        L_BM2 = sc.scalar(110.99, unit="m")  # Position of Beam monitor 2
        return tof.Detector(distance=L_BM2, name="monitor 2")  # type: ignore

    def monitor3(self):
        L_BM3 = sc.scalar(163.2, unit="m")  # Tentative position of Beam monitor 3
        return tof.Detector(distance=L_BM3, name="monitor 3")  # type: ignore

    def monitor_sample(self):
        L_SAMPLE = sc.scalar(163.8, unit="m")  # Source to sample in m
        return tof.Detector(distance=L_SAMPLE, name="sample")  # type: ignore

    def detector(self):
        L_DETECTOR = sc.scalar(166.8, unit="m")  # Source to sample in m
        return tof.Detector(distance=L_DETECTOR, name="detector")  # type: ignore

    @property
    def choppers(self):
        return (
            self.bw_chopper1(),
            self.bw_chopper2(),
            self.p_chopper1(),
            self.p_chopper2(),
            self.m_chopper1(),
            self.m_chopper2(),
        )

    @property
    def detectors(self):
        return (
            self.monitor1(),
            self.monitor2(),
            self.monitor3(),
            self.monitor_sample(),
            self.detector(),
        )

    @property
    def model(self):
        return tof.Model(source=self.source, detectors=self.detectors, choppers=self.choppers)  # type: ignore

    def calculate_bandwidth(
        self,
        source_duration=sc.scalar(3, unit="ms"),
        source_wavelength_min=sc.scalar(0.25, unit="Å"),
        source_wavelength_max=sc.scalar(7.5, unit="Å"),
    ):
        """Calculate the bandwidth determined by the Bandwidth choppers"""
        open_times, close_times = self.bw_chopper1().open_close_times()
        open_bw1, close_bw1 = open_times[0], close_times[0]
        open_times, close_times = self.bw_chopper2().open_close_times()
        open_bw2, close_bw2 = open_times[0], close_times[0]
        wavelength_max = min(
            source_wavelength_max,
            tof.utils.speed_to_wavelength(self.bw_chopper1().distance / close_bw1),
            tof.utils.speed_to_wavelength(self.bw_chopper2().distance / close_bw2),
        )
        wavelength_min = max(
            source_wavelength_min,
            tof.utils.speed_to_wavelength(
                self.bw_chopper1().distance / (open_bw1 - source_duration.to(unit="us"))
            ),
            tof.utils.speed_to_wavelength(
                self.bw_chopper2().distance / (open_bw2 - source_duration.to(unit="us"))
            ),
        )

        return (wavelength_min, wavelength_max)

    def calculate_delta_lambda(self):
        """Calculate step in wavelength selected by monochromatic choppers"""
        h_over_mn = (const.Planck / const.m_n).to(unit="Å*m/s")  # 3956 Å*m/s
        m_chopper_position = (
            self.m_chopper1().distance + self.m_chopper2().distance
        ) / 2
        delta_lambda = h_over_mn / self.m_frequency / m_chopper_position.to(unit="m")
        return delta_lambda.to(unit="Å")

    def calculate_toa_at(self, component_name: str, RRM=False, wavelength_range=None):
        """Calculate time of arrival at a given component in microseconds
        Use wavelength_range = (min,max) to limit the range of interest"""
        try:
            component = getattr(self, component_name)()
        except AttributeError:
            print(f"{component_name} does not exist.")
            return None

        distance = component.distance
        central_wavelength = self.wavelength
        delta_lambda = self.calculate_delta_lambda()
        if RRM:
            n = int(np.ceil(self.rrm / 2))
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

    def find_ei_from_monitors(self, hist_monitor3, hist_detector):
        """Calculate incoming energy from the histogram of counts vs. TOA at monitor 3 and detector"""
        pass
        return None


if __name__ == "__main__":
    print(Instrument(wavelength=1, rrm=12))
