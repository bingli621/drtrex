import scipp as sc
import numpy as np
from typing import Literal
import tof
import scipp.constants as const


class Chopper(tof.Chopper):  # pyright: ignore[reportGeneralTypeIssues]

    def __init__(
        self,
        name,
        wavelength,
        frequency,
        mode: Literal["High Resolution", "High Flux"] = "High Resolution",
        time_shift=sc.scalar(0.0, unit="s"),
    ):
        distance = 1
        theta_pos = 1
        theta_width = 1
        direction = tof.AntiClockwise
        angle_offset = sc.scalar(0.0, unit="deg")

        phase = self.get_phase(
            distance=distance,
            wavelength=wavelength,
            frequency=frequency,
            time_shift=time_shift,
            angle_offset=angle_offset,
        )
        super().__init__(
            frequency=frequency,
            centers=theta_pos,
            widths=theta_width,
            phase=phase,
            direction=direction,
            distance=distance,
            name=name,
        )

    # TODO
    def str(self):
        print(f"{self.name} running at 1 Hz")

    @staticmethod
    def get_phase(
        distance,
        wavelength,
        frequency,
        time_shift=sc.scalar(0.0, unit="s"),
        angle_offset=sc.scalar(0.0, unit="deg"),
    ) -> float:
        """Positive time shift/ angle offset delays the time"""

        h = const.Planck  # Planck constant in J.s
        mn = const.m_n  # Neutron mass in kg
        h_over_mn = (h / mn).to(unit="Å*m/s")  # 3956 Å*m/s
        degs = sc.scalar(360.0, unit="deg")
        velocity = h_over_mn / wavelength.to(unit="Å")  # in m/s
        time = distance.to(unit="m") / velocity + time_shift.to(unit="s")  # in s
        angle = time * frequency * degs
        return (angle + angle_offset) % degs
