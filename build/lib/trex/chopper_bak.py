from typing import TYPE_CHECKING

import tof
import scipp as sc
import scipp.constants as const
from scippneutron.tof import chopper_cascade


if TYPE_CHECKING:
    from trex.instrument import Instrument
    from trex.params import ChopperParameters


class Chopper(tof.Chopper):  # type: ignore
    def __init__(self, parameters: "ChopperParameters", instrument: "Instrument"):

        self.instrument = instrument
        frequency = self._calculate_frequency(parameters)

        super().__init__(
            frequency=frequency,
            centers=parameters.slit_center,
            widths=parameters.slit_width,
            phase=self._calculate_phase(parameters, frequency),
            direction=parameters.direction,
            distance=parameters.axle_position,
            name=parameters.name,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_angle_offset(centers: sc.Variable, mode: str | None):
        """Return angle offset for the given chopper mode.

        - First slit set  → High Resolution
        - Second slit set → High Flux
        - CCW is positive

        Note:
        scipp variables are mutable, use copy to avoid changing of centers
        when operating on angle_offset

        """
        if mode not in (None, "High Resolution", "High Flux"):
            raise ValueError(
                f"Unrecognized mode={mode}. Expected 'High Resolution' or 'High Flux'."
            )

        return centers[1 if mode == "High Flux" and len(centers) > 1 else 0].copy()

    # ------------------------------------------------------------------
    # Internal calculations
    # ------------------------------------------------------------------

    def _calculate_phase(
        self, parameters: "ChopperParameters", frequency
    ) -> sc.Variable:
        """Positive time shift/ angle offset delays the time"""

        angle_offset = self.get_angle_offset(
            parameters.slit_center, self.instrument.chopper_mode
        )

        h = const.Planck  # Planck constant in J.s
        mn = const.m_n  # Neutron mass in kg
        h_over_mn = (h / mn).to(unit="Å*m/s")  # 3956 Å*m/s
        two_pi = sc.scalar(360.0, unit="deg")

        velocity = h_over_mn / self.instrument.wavelength.to(unit="Å")  # in m/s
        time = parameters.axle_position.to(
            unit="m"
        ) / velocity + self.instrument.t_offset.to(unit="s")
        angle = time * frequency * two_pi

        if parameters.direction == tof.Clockwise:
            angle_offset *= -1

        return (angle + angle_offset) % two_pi

    def _calculate_frequency(self, parameters: "ChopperParameters"):
        """Get the frequencies of BW, PS and M-choppers.
        Note:
            fP = L_M/L_PS * fM /2, L_PS / L_M = 3/4 for P choppers have two sets of slits.
            P chopper can be further slowed down by deviding an integer. However the frequency
            must be multiles of BW frequency (14 Hz for ESS). Therefore, RRM needs to be
            multiples of 4.
            RRM should be smaller than 24.
        """
        rrm = self.instrument.rrm
        if rrm % 4 != 0:
            raise ValueError(f"RRM = {rrm} needs to be multiples of 4.")

        source_freq = self.instrument.source.frequency
        match name := parameters.name:
            case s if s.startswith("Bandwidth"):
                freq = source_freq
            case s if s.startswith("Pulse Shaping"):
                freq = source_freq * rrm * 0.75 / 1
            case s if s.startswith("Monochromatic"):
                freq = source_freq * rrm
            case _:
                raise ValueError(f"Unrecognized chopper name: {name}")
        if freq > parameters.frequency_max:
            raise ValueError(
                f"{name} frequency = {freq.value:.5g} Hz exceeds "
                f"maximum {parameters.frequency_max.value:.5g} Hz"
            )
        return freq

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    def open_close_times(self, *arg, **kwarg):
        return super().open_close_times(*arg, **kwarg)

    def to_chopper_cascade(
        self, time_limit=sc.scalar(1, unit="s")
    ) -> chopper_cascade.Chopper:
        time_open, time_close = self.open_close_times(time_limit=time_limit, unit="s")
        return chopper_cascade.Chopper(
            distance=self.distance, time_open=time_open, time_close=time_close
        )
