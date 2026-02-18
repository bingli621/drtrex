from dataclasses import dataclass
from typing import Literal, Optional

import tof
import scipp as sc
import scipp.constants as const
from scippneutron.tof import chopper_cascade


@dataclass(frozen=True)
class ChopperParameters:
    """centers are defined using the side facing the incoming beam, CCW is positive"""

    name: str
    wavelength: sc.Variable
    frequency: sc.Variable
    distance: sc.Variable
    centers: sc.Variable
    widths: sc.Variable
    time_shift: sc.Variable
    direction: Literal[tof.AntiClockwise, tof.Clockwise]
    mode: Optional[str] = None


class Chopper(tof.Chopper):  # type: ignore
    def __init__(self, parameters: ChopperParameters):

        angle_offset = self.get_angle_offset(parameters.centers, parameters.mode)
        phase = self.get_phase(parameters, angle_offset)

        super().__init__(
            frequency=parameters.frequency,
            centers=parameters.centers,
            widths=parameters.widths,
            phase=phase,
            direction=parameters.direction,
            distance=parameters.distance,
            name=parameters.name,
        )

    @staticmethod
    def get_angle_offset(centers: sc.Variable, mode: Optional[str]):
        """Get angle offset for the given mode, assuming the first set of slits are for
        the 'High Resolution' mode, and the secend sets are for the 'High Flux' mode.

        Note:
        Increasing values goes CCW.

        Note:
        scipp variables are mutable, use copy to avoid changing of centers when operating
        on angle_offset

        """
        if mode is None:
            angle_offset = centers[0].copy()
        elif mode == "High Resolution":
            angle_offset = centers[0].copy()
        elif mode == "High Flux":
            angle_offset = centers[1].copy()
        else:
            raise ValueError(
                f"Unrecogonized mode={mode}. Needs to be 'High Resolution' or 'High Flux'."
            )
        return angle_offset

    @staticmethod
    def get_phase(
        parameters: ChopperParameters, angle_offset: sc.Variable
    ) -> sc.Variable:
        """Positive time shift/ angle offset delays the time"""

        h = const.Planck  # Planck constant in J.s
        mn = const.m_n  # Neutron mass in kg
        h_over_mn = (h / mn).to(unit="Å*m/s")  # 3956 Å*m/s
        two_pi = sc.scalar(360.0, unit="deg")

        velocity = h_over_mn / parameters.wavelength.to(unit="Å")  # in m/s
        time = parameters.distance.to(unit="m") / velocity + parameters.time_shift.to(
            unit="s"
        )
        angle = time * parameters.frequency * two_pi

        if parameters.direction == tof.Clockwise:
            angle_offset *= -1

        return (angle + angle_offset) % two_pi

    def open_close_times(self, *arg, **kwarg):
        return super().open_close_times(*arg, **kwarg)

    def to_chopper_cascade(
        self, time_limit=sc.scalar(1, unit="s")
    ) -> chopper_cascade.Chopper:
        time_open, time_close = self.open_close_times(time_limit=time_limit, unit="s")
        return chopper_cascade.Chopper(
            distance=self.distance, time_open=time_open, time_close=time_close
        )
