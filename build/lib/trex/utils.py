import scipp as sc
from scippneutron.tof import chopper_cascade
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from trex.instrument import Instrument


def centers_to_edges(centers, dim=None):
    """return bin edges with the given center values"""
    dim = centers.dim if dim is None else dim
    mid = (centers[:-1] + centers[1:]) / 2
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return sc.concat([first, mid, last], dim=dim)


def calculate_frame_at(component_name: str, instrument: "Instrument"):
    source = instrument.source
    choppers = instrument.chopper_cascade
    component = instrument._validate_component(component_name)
    wavelength_min, wavelength_max = source.wavelength_range
    time_min, time_max = source.time_range

    frames = chopper_cascade.FrameSequence.from_source_pulse(
        time_min=time_min,
        time_max=time_max,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
    )
    relevant_choppers = [
        chopper
        for chopper in choppers.values()
        if chopper.distance <= component.distance
    ]  # find all choppers before the given component

    frames = frames.chop(relevant_choppers)
    at_component = frames.propagate_to(component.distance)
    frame = at_component[-1]
    return frame


def calculate_variable_range_at(
    component_name: str,
    variable_name: str,
    instrument: "Instrument",
    rename_dims_to: Optional[str] = None,
    unit: Optional[str] = None,
) -> Tuple[sc.Variable, sc.Variable]:
    """Calculate the range of a variable at a given component"""

    frame = calculate_frame_at(component_name, instrument)
    bounds = frame.subbounds().get(variable_name)
    if bounds is None:
        raise ValueError(f"{variable_name} is not a valid vairable name.")
    var_bounds = bounds.rename_dims({"subframe": variable_name})
    var_min = sc.sort(var_bounds["bound", 0], key=variable_name)
    var_max = sc.sort(var_bounds["bound", 1], key=variable_name)
    if rename_dims_to is not None:
        var_min = var_min.rename_dims({variable_name: rename_dims_to})
        var_max = var_max.rename_dims({variable_name: rename_dims_to})
    if unit is not None:
        var_min = var_min.to(unit=unit)
        var_max = var_max.to(unit=unit)
    return (var_min, var_max)
