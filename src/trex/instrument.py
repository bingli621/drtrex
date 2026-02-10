from typing import Literal
import scipp as sc
import tof
from trex.chopper import ChopperParameters, Chopper


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
            self.get_chopper_frequency(source.frequency, rrm)
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
            + f"RRM = {self.rrm}."
        )

    @staticmethod
    def get_chopper_frequency(source_frequency, rrm: int):
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
