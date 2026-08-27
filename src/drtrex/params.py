from dataclasses import dataclass, field
import scipp as sc


# ------------------------------------------------------------------
# Choppers
# ------------------------------------------------------------------
@dataclass(frozen=True)
class ChopperDesign:
    """First opeing is High Resolution, second opening is High Flux.
    Centers are defined using the side facing the incoming beam, CCW is positive"""

    name: str
    axle_position: sc.Variable
    slit_center: sc.Variable
    slit_width: sc.Variable
    slit_height: sc.Variable
    frequency_max: sc.Variable
    radius: sc.Variable = field(default_factory=lambda: sc.scalar(0.35, unit="m"))
    beam_position: sc.Variable = field(
        default_factory=lambda: sc.scalar(0.0, unit="deg")
    )


BW1_DESIGN = ChopperDesign(  # CCW
    name="Bandwidth Chopper 1",
    axle_position=sc.vector([0, -0.3075, 31.964], unit="m"),  # Source to BW chopper 1,
    slit_center=sc.array(dims=["cutouts"], values=(0.0,), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(61.4,), unit="deg"),
    slit_height=sc.scalar(0.35, unit="m"),
    frequency_max=sc.scalar(15, unit="Hz"),
)


BW2_DESIGN = ChopperDesign(  # CCW
    name="Bandwidth Chopper 2",
    axle_position=sc.vector((0, -0.3075, 39.987), unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0.0,), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(63.3,), unit="deg"),
    slit_height=sc.scalar(0.35, unit="m"),
    frequency_max=sc.scalar(15, unit="Hz"),
)


PS1_DESIGN = ChopperDesign(  # CCW
    name="Pulse Shaping Chopper 1",
    beam_position=sc.scalar(180.0, unit="deg"),
    axle_position=sc.vector((0, 0.305, 107.95), unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0, -55, -180, -235), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(20, 35, 20, 35), unit="deg"),
    slit_height=sc.scalar(0.095, unit="m"),
    frequency_max=sc.scalar(252, unit="Hz"),
)


PS2_DESIGN = ChopperDesign(  # CW
    name="Pulse Shaping Chopper 2",
    beam_position=sc.scalar(180.0, unit="deg"),
    axle_position=sc.vector((0, 0.305, 108.05), unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0, -55, -180, -235), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(20, 35, 20, 35), unit="deg"),
    slit_height=sc.scalar(0.095, unit="m"),
    frequency_max=sc.scalar(252, unit="Hz"),
)

MC1_DESIGN = ChopperDesign(  # CCW
    name="Monochromatic Chopper 1",
    beam_position=sc.scalar(180.0, unit="deg"),
    axle_position=sc.vector((0, 0.325, 161.995), unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0, -175), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(2.5, 4.3), unit="deg"),
    slit_height=sc.scalar(0.045, unit="m"),
    frequency_max=sc.scalar(336, unit="Hz"),
)

MC2_DESIGN = ChopperDesign(  # CW
    name="Monochromatic Chopper 2",
    axle_position=sc.vector((0, -0.325, 162.005), unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0, -175), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(2.5, 4.3), unit="deg"),
    slit_height=sc.scalar(0.045, unit="m"),
    frequency_max=sc.scalar(336, unit="Hz"),
)


class ChopperSettings:
    """Runtime operating parameters for a given design."""

    design: ChopperDesign
    frequency: sc.Variable
    phase: sc.Variable = field(init=False)
    delay: sc.Variable = field(init=False)

    def __post_init__(self):
        if self.frequency > design.frequency_max:
            raise ValueError(
                f"frequency ({self.frequency.value}) exceeds "
                f"frequency_max ({design.frequency_max.value}) for chopper {design.name!r}"
            )

        # Angular offset between beam position and slit center
        # (placeholder — replace with your actual phase relation)
        self.phase = (self.beam_position - self.slit_center).to(unit="deg")

        # Time delay corresponding to that phase, given the rotation period
        period = sc.reciprocal(self.rotation_speed).to(unit="s")
        self.delay = (self.phase / sc.scalar(360.0, unit="deg")) * period


chopper_params = [
    BW1_DESIGN,
    BW2_DESIGN,
    PS1_DESIGN,
    PS2_DESIGN,
    MC1_DESIGN,
    MC2_DESIGN,
]

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

monitor_params = [mon1, mon2, mon3, mon_sample, mon_beamstop]


# ------------------------------------------------------------------
# Detectors
# ------------------------------------------------------------------


@dataclass()
class DetectorParameters:
    name: str
    distance: sc.Variable


detector = DetectorParameters(
    name="Detector",
    distance=sc.scalar(166.8, unit="m"),
)

detector_params = [detector]

# ------------------------------------------------------------------
# Other parameters
# ------------------------------------------------------------------

# DEL_L = sc.scalar(0.02, unit="m")  # Effective flight path uncertainty
