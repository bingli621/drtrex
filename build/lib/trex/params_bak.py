from dataclasses import dataclass, field
from typing import Literal
import scipp as sc
import tof


# ------------------------------------------------------------------
# Choppers
# ------------------------------------------------------------------
@dataclass()
class ChopperParameters:
    """First opeing is High Resolution, second opening is High Flux.
    Centers are defined using the side facing the incoming beam, CCW is positive"""

    name: str
    distance: sc.Variable
    centers: sc.Variable
    widths: sc.Variable
    direction: Literal[tof.AntiClockwise, tof.Clockwise]  # type: ignore
    frequency_max: sc.Variable = field(default_factory=lambda: sc.scalar(15, unit="Hz"))


bw1 = ChopperParameters(
    name="Bandwidth Chopper 1",
    direction=tof.AntiClockwise,
    distance=sc.scalar(31.964, unit="m"),  # Source to BW chopper 1,
    centers=sc.array(dims=["cutouts"], values=(0.0,), unit="deg"),
    widths=sc.array(dims=["cutouts"], values=(61.4,), unit="deg"),
)


bw2 = ChopperParameters(
    name="Bandwidth Chopper 2",
    direction=tof.AntiClockwise,
    distance=sc.scalar(39.987, unit="m"),
    centers=sc.array(dims=["cutouts"], values=(0.0,), unit="deg"),
    widths=sc.array(dims=["cutouts"], values=(63.3,), unit="deg"),
)


ps1 = ChopperParameters(
    name="Pulse Shaping Chopper 1",
    direction=tof.AntiClockwise,
    distance=sc.scalar(107.95, unit="m"),
    centers=sc.array(dims=["cutouts"], values=(0, -55, -180, -235), unit="deg"),
    widths=sc.array(dims=["cutouts"], values=(20, 35, 20, 35), unit="deg"),
    frequency_max=sc.scalar(252, unit="Hz"),
)


ps2 = ChopperParameters(
    name="Pulse Shaping Chopper 2",
    direction=tof.Clockwise,
    distance=sc.scalar(108.05, unit="m"),
    centers=sc.array(dims=["cutouts"], values=(0, -55, -180, -235), unit="deg"),
    widths=sc.array(dims=["cutouts"], values=(20, 35, 20, 35), unit="deg"),
    frequency_max=sc.scalar(252, unit="Hz"),
)

m1 = ChopperParameters(
    name="Monochromatic Chopper 1",
    distance=sc.scalar(161.995, unit="m"),
    centers=sc.array(dims=["cutouts"], values=(-180, +5), unit="deg"),
    widths=sc.array(dims=["cutouts"], values=(2.5, 4.3), unit="deg"),
    direction=tof.AntiClockwise,
    frequency_max=sc.scalar(336, unit="Hz"),
)

m2 = ChopperParameters(
    name="Monochromatic Chopper 2",
    distance=sc.scalar(162.005, unit="m"),
    centers=sc.array(dims=["cutouts"], values=(0, -175), unit="deg"),
    widths=sc.array(dims=["cutouts"], values=(2.5, 4.3), unit="deg"),
    direction=tof.Clockwise,
    frequency_max=sc.scalar(336, unit="Hz"),
)

chopper_params = [bw1, bw2, ps1, ps2, m1, m2]

# ------------------------------------------------------------------
# Monitors
# ------------------------------------------------------------------


@dataclass()
class MonitorParameters:
    name: str
    distance: sc.Variable


mon1 = MonitorParameters(
    distance=sc.scalar(41.98786, unit="m"),
    name="Monitor 1",
)


mon2 = MonitorParameters(
    name="Monitor 2",
    distance=sc.scalar(110.99, unit="m"),
)


mon3 = MonitorParameters(
    name="Monitor 3",
    distance=sc.scalar(163.2, unit="m"),  # Tentative position of Beam monitor 3
)


mon_sample = MonitorParameters(
    name="Monitor at Sample",
    distance=sc.scalar(163.8, unit="m"),  # Source to sample in m
)

mon_beamstop = MonitorParameters(
    name="Beamstop Monitor",
    distance=sc.scalar(166.8, unit="m"),
)


# detector = MonitorParameters(
#     name="Detector",
#     distance=sc.scalar(166.8, unit="m"),
# )

monitor_params = [mon1, mon2, mon3, mon_sample, mon_beamstop]
