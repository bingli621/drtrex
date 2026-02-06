import json


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
    ) -> None:
        """Initialize instrument with central wavelength and repitition rate RRM"""

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
    def load_parameter(path_to_file="trex_params.json"):
        with open(path_to_file) as params:
            d = json.load(params)
        return d
