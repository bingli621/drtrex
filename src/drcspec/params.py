from dataclasses import dataclass, field
import scipp as sc


# ------------------------------------------------------------------
# Choppers
# ------------------------------------------------------------------
@dataclass()
class ChopperParameters:
    """Centers are defined using the side facing the incoming beam, CCW is positive"""

    name: str
    axle_position: sc.Variable
    slit_center: sc.Variable
    slit_width: sc.Variable
    slit_height: sc.Variable = field(
        default_factory=lambda: sc.scalar(0.1, unit="m"),
    )
    radius: sc.Variable = field(default_factory=lambda: sc.scalar(0.35, unit="m"))
    beam_position: sc.Variable = field(
        default_factory=lambda: sc.scalar(0.0, unit="deg")
    )
    # frequency_max: sc.Variable = field(default_factory=lambda: sc.scalar(15, unit="Hz"))


bw1 = ChopperParameters(  # CCW
    name="Bandwidth Chopper 1",
    beam_position=sc.scalar(180.0, unit="deg"),
    axle_position=sc.vector([0, 0.31913, 14.95], unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0.0,), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(40.7,), unit="deg"),
    # slit_height=sc.scalar(0.35, unit="m"),
)


bw2 = ChopperParameters(  # CCW
    name="Bandwidth Chopper 2",
    beam_position=sc.scalar(180.0, unit="deg"),
    axle_position=sc.vector((0, 0.31913, 20.47), unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0.0,), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(41.7,), unit="deg"),
    # slit_height=sc.scalar(0.35, unit="m"),
)


bw3 = ChopperParameters(  # CCW
    name="Bandwidth Chopper 3",
    beam_position=sc.scalar(180.0, unit="deg"),
    axle_position=sc.vector((0, 0.30818, 104.59), unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0.0,), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(193.0,), unit="deg"),
    # slit_height=sc.scalar(0.35, unit="m"),
)


psc1 = ChopperParameters(  # CCW
    name="Pulse Shaping Chopper 1",
    beam_position=sc.scalar(180.0, unit="deg"),
    axle_position=sc.vector((0, 0.31064, 105.67), unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0, 120, 240), unit="deg"),
    slit_width=sc.array(
        dims=["cutouts"],
        values=(24.23, 24.23, 24.23),
        unit="deg",
    ),
    # slit_height=sc.scalar(0.35, unit="m"),
)


psc2 = ChopperParameters(  # CW
    name="Pulse Shaping Chopper 2",
    axle_position=sc.vector((0, -0.31064, 105.68), unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0, 120, 240), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(24.23, 24.23, 24.23), unit="deg"),
    # slit_height=sc.scalar(0.35, unit="m"),
)

rrm = ChopperParameters(  # CW
    name="RRM Chopper",
    axle_position=sc.vector((0, -0.3215, 158.45), unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0,), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(4.45,), unit="deg"),
    # slit_height=sc.scalar(0.35, unit="m"),
)

mc1 = ChopperParameters(  # CCW
    name="Monochromatic Chopper 1",
    beam_position=sc.scalar(180.0, unit="deg"),
    axle_position=sc.vector((0, 0.3215, 158.50), unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0,), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(4.45,), unit="deg"),
    # slit_height=sc.scalar(0.35, unit="m"),
)

mc2 = ChopperParameters(  # CW
    name="Monochromatic Chopper 2",
    axle_position=sc.vector((0, -0.3215, 158.51), unit="m"),
    slit_center=sc.array(dims=["cutouts"], values=(0,), unit="deg"),
    slit_width=sc.array(dims=["cutouts"], values=(4.45,), unit="deg"),
    # slit_height=sc.scalar(0.35, unit="m"),
)

chopper_params = [bw1, bw2, bw3, psc1, psc2, rrm, mc1, mc2]

# ------------------------------------------------------------------
# Monitors
# ------------------------------------------------------------------


# @dataclass()
# class MonitorParameters:
#     name: str
#     distance: sc.Variable


# mon1 = MonitorParameters(
#     distance=sc.scalar(41.98786, unit="m"),
#     name="Monitor 1",
# )


# mon2 = MonitorParameters(
#     name="Monitor 2",
#     distance=sc.scalar(110.99, unit="m"),
# )


# mon3 = MonitorParameters(
#     name="Monitor 3",
#     distance=sc.scalar(163.2, unit="m"),  # Tentative position of Beam monitor 3
# )


# mon_sample = MonitorParameters(
#     name="Monitor at Sample",
#     distance=sc.scalar(163.8, unit="m"),  # Source to sample in m
# )

# mon_beamstop = MonitorParameters(
#     name="Beamstop Monitor",
#     distance=sc.scalar(166.8, unit="m"),
# )

# monitor_params = [mon1, mon2, mon3, mon_sample, mon_beamstop]

monitor_params = []
# ------------------------------------------------------------------
# Detectors
# ------------------------------------------------------------------


# @dataclass()
# class DetectorParameters:
#     name: str
#     distance: sc.Variable


# detector = DetectorParameters(
#     name="Detector",
#     distance=sc.scalar(166.8, unit="m"),
# )

# detector_params = [detector]
detector_params = []

# ------------------------------------------------------------------
# Other parameters
# ------------------------------------------------------------------

# DEL_L = sc.scalar(0.02, unit="m")  # Effective flight path uncertainty
