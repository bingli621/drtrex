from typing import TYPE_CHECKING
import numpy as np
import tof
import scipp as sc
import scipp.constants as const
from scippneutron.tof import chopper_cascade
from scippneutron.chopper.disk_chopper import DiskChopper
from drcspec.components.utils import calculate_frame_at

if TYPE_CHECKING:
    from drcspec.instrument import Instrument
    from drcspec.params import ChopperParameters


class Chopper(DiskChopper):
    name: str
    distance: sc.Variable
    instrument: "Instrument"

    @classmethod
    def from_parameters(
        cls, parameters: "ChopperParameters", instrument: "Instrument"
    ) -> "Chopper":

        frequency = cls._calculate_frequency(
            parameters,
            f_M=instrument.f_M,
            n=instrument.n,
            source_frequency=instrument.source.frequency,
        )
        slit_begin, slit_end = cls._calculate_slit_openings(parameters)
        phase = cls._calculate_phase(
            parameters,
            frequency=frequency,
            wavelength=instrument.wavelength,
            t_offset=instrument.t_offset,
        )

        new_chopper = cls(
            axle_position=parameters.axle_position,
            frequency=frequency,
            beam_position=parameters.beam_position,
            radius=parameters.radius,
            slit_begin=slit_begin,
            slit_end=slit_end,
            slit_height=parameters.slit_height,
            phase=phase,
        )
        new_chopper.name = parameters.name
        new_chopper.distance = parameters.axle_position.fields.z
        new_chopper.instrument = instrument

        return new_chopper

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_angle_offset(centers: sc.Variable, beam_position: sc.Variable):
        """Return angle offset from beam position for the given chopper mode.

        - First slit set  → High Resolution
        - Second slit set → High Flux
        - CCW is positive
        """

        slit_center = centers[0]
        return slit_center - beam_position

    @staticmethod
    def _calculate_phase(
        parameters: "ChopperParameters",
        frequency: sc.Variable,
        wavelength: sc.Variable,
        t_offset: sc.Variable,
    ) -> sc.Variable:
        """Positive time shift/ angle offset delays the time"""

        angle_offset = Chopper.get_angle_offset(
            parameters.slit_center, parameters.beam_position
        )

        h = const.Planck  # Planck constant in J.s
        mn = const.m_n  # Neutron mass in kg
        h_over_mn = (h / mn).to(unit="Å*m/s")  # 3956 Å*m/s
        two_pi = sc.scalar(360.0, unit="deg")

        velocity = h_over_mn / wavelength.to(unit="Å")  # in m/s
        time = parameters.axle_position.fields.z.to(unit="m") / velocity + t_offset.to(
            unit="s"
        )
        angle = time * frequency * two_pi

        phase = (angle + angle_offset) % two_pi
        # if phase > sc.scalar(180.0, unit="deg"):
        #     # subtract 2 pi to avoid an issue of missing openings at
        #     # small time in ToF
        #     phase -= two_pi
        return phase

    @staticmethod
    def _calculate_frequency(parameters: "ChopperParameters", f_M, n, source_frequency):
        """Get the frequencies of BW, PS, RRM and M-choppers."""

        match name := parameters.name:
            case s if s.startswith("Bandwidth"):
                freq = source_frequency
            case "Pulse Shaping Chopper 1":
                freq = f_M / 2

            case "Pulse Shaping Chopper 2":
                freq = f_M / 2 * (-1)

            case "RRM Chopper":
                freq = f_M / n

            case "Monochromatic Chopper 1":
                freq = f_M

            case "Monochromatic Chopper 2":
                freq = -f_M

            case _:
                raise ValueError(f"Unrecognized chopper name: {name}")

        # if sc.abs(freq) > parameters.frequency_max:
        #     raise ValueError(
        #         f"{name} frequency = {freq.value:.5g} Hz exceeds "
        #         f"maximum {parameters.frequency_max.value:.5g} Hz"
        #     )
        return freq

    @staticmethod
    def _calculate_slit_openings(parameters: "ChopperParameters"):
        slit_begin = parameters.slit_center - 0.5 * parameters.slit_width
        slit_end = parameters.slit_center + 0.5 * parameters.slit_width
        return (slit_begin, slit_end)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------
    def open_close_times(self, *arg, **kwarg):
        tof_chopper = tof.Chopper.from_diskchopper(self)
        return tof_chopper.open_close_times(*arg, **kwarg)

    def to_chopper_cascade(
        self, time_limit=sc.scalar(1, unit="s")
    ) -> chopper_cascade.Chopper:
        time_open, time_close = self.open_close_times(time_limit=time_limit, unit="s")
        return chopper_cascade.Chopper(
            distance=self.axle_position.fields.z,
            time_open=time_open,
            time_close=time_close,
        )

    def calculate_frame(self) -> chopper_cascade.Frame:
        return calculate_frame_at(self.name, instrument=self.instrument)
