from dataclasses import dataclass
import scipp as sc
from scippneutron.tof.chopper_cascade import FrameSequence, Frame
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from tof.result import Result
    from trex.instrument import Instrument


def centers_to_edges(centers, dim=None):
    """return bin edges with the given center values"""
    dim = centers.dim if dim is None else dim
    mid = (centers[:-1] + centers[1:]) / 2
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return sc.concat([first, mid, last], dim=dim)


# -----------------------------------------------------------------------
# Frame propagation
# -----------------------------------------------------------------------


def calculate_frame_at(component_name: str, instrument: "Instrument") -> Frame:
    """Get a Frame at the given chopper/monitor/detector"""
    source = instrument.source
    choppers = instrument.chopper_cascade
    component = instrument._validate_component(component_name)
    wavelength_min, wavelength_max = source.wavelength_range
    time_min, time_max = source.time_range

    frames = FrameSequence.from_source_pulse(
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


# -----------------------------------------------------------------------
# Gegnerating masks from the chopper acceptance diagram
# -----------------------------------------------------------------------


@dataclass
class SubframeVortex:
    distance: sc.Variable
    time: sc.Variable
    wavelength: sc.Variable


def acceptance_paths(
    frame: Frame, time_unit: str = "us", wavelength_unit: str = "Å"
) -> list[SubframeVortex]:
    frame_at_source = frame.propagate_to(distance=sc.scalar(0.0, unit="m"))
    subframe_paths = []
    for subframe in frame_at_source.subframes:
        vortex = SubframeVortex(
            distance=frame.distance,
            time=subframe.time.to(unit=time_unit, copy=False),
            wavelength=subframe.wavelength.to(unit=wavelength_unit, copy=False),
        )
        subframe_paths.append(vortex)

    return subframe_paths


def coord_centers(da: sc.DataArray, name: str) -> sc.Variable:
    """
    Return coordinate values suitable for point representation.

    If the coordinate is 1-D edges, return bin midpoints.
    Otherwise return the coordinate as-is.
    """
    try:
        coord = da.coords[name]
    except KeyError as exc:
        raise KeyError(f"DataArray has no coordinate '{name}'") from exc

    if coord.ndim == 1 and da.coords.is_edges(name):
        return sc.midpoints(coord)

    return coord


def get_points(
    da: sc.DataArray,
    *,
    xcoord_name: str = "wavelength",
    ycoord_name: str = "birth_time",
) -> np.ndarray:
    """
    Return an (N, 2) NumPy array of (x, y) points from two DataArray coordinates.

    Coordinates are broadcast to a common shape before flattening.
    Bin-edge coordinates are converted to midpoints.
    Units are stripped explicitly before conversion to NumPy.
    """
    x_coord = coord_centers(da, xcoord_name)
    y_coord = coord_centers(da, ycoord_name)
    # broadcast
    sizes = {**x_coord.sizes, **y_coord.sizes}
    x_coord = x_coord.broadcast(sizes=sizes).copy(deep=True)
    y_coord = y_coord.broadcast(sizes=sizes).copy(deep=True)
    # Overwriting unit to make a stack using scipp operator...
    x_coord.unit = "dimensionless"
    y_coord.unit = "dimensionless"
    xy_stack = (
        sc.concat([x_coord, y_coord], dim="i")
        .flatten(dims=sizes.keys(), to="pos")  # type: ignore
        .transpose()
    )
    return xy_stack.values


# -----------------------------------------------------------------------
# Frame reconstruction
# -----------------------------------------------------------------------


def reconstruct_frame_at(
    component_name: str,
    model_result: "Result",
    wavelength_lower_bound,
    distance,
    period,
):
    pass
