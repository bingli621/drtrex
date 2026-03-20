import tof
import scipp as sc
from matplotlib.path import Path
import numpy as np
from drtrex.components.utils import SubframeVortex, get_points


class Source(tof.Source):  # type: ignore
    def __init__(self, time_range=None, wavelength_range=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.time_range = (
            self.calculate_time_range() if time_range is None else time_range
        )
        self.wavelength_range = (
            self.calculate_wavelength_range()
            if wavelength_range is None
            else wavelength_range
        )

    def __str__(self):
        t_min, t_max = self.time_range
        w_min, w_max = self.wavelength_range
        return (
            super.__str__(self)
            + f"\n  time range=({t_min.values:.3f}, {t_max.values:.3f}) [{t_min.unit}]"
            + f"\n  wavelength range=({w_min.values:.3f}, {w_max.values:.3f}) [{w_min.unit}]"
        )

    @staticmethod
    def calculate_range(data, number_of_sigma: float = 3.0):
        mean = sc.mean(data)
        std = sc.std(data, ddof=0)
        min_value = max(data.min(), mean - number_of_sigma * std)
        max_value = mean + number_of_sigma * std
        return (min_value, max_value)

    def calculate_time_range(self, number_of_sigma=3.0):
        data = self.data.coords["birth_time"]
        return self.calculate_range(data, number_of_sigma)

    def calculate_wavelength_range(self, number_of_sigma=2):
        data = self.data.coords["wavelength"]
        return self.calculate_range(data, number_of_sigma)

    def apply_mask(self, vortices: list[SubframeVortex]):
        da = self.data
        wav_time_points = get_points(da)
        if len(vortices) < 1:
            wavelengths = [vortices[0].wavelength]
            birth_times = [vortices[0].time.to(unit=da.coords["birth_time"].unit)]
        else:
            wavelengths = [vertex.wavelength for vertex in vortices]
            birth_times = [
                vortex.time.to(unit=da.coords["birth_time"].unit) for vortex in vortices
            ]

        inside = sc.zeros_like(da.data).astype(bool)
        inside.unit = None

        for birth_time, wavelength in zip(birth_times, wavelengths):
            vx = wavelength.values
            vy = birth_time.values
            verts = np.column_stack([vx, vy])
            path = Path(verts)
            sizes = da.sizes
            dims = sizes.keys()
            inside = inside | sc.array(
                dims=dims,
                values=path.contains_points(wav_time_points).reshape(
                    tuple(sizes.values())
                ),
            )

        filtered = da.squeeze()[inside.squeeze()]
        final_sizes = {"pulse": 1, **filtered.sizes}
        filtered = filtered.broadcast(sizes=final_sizes)
        for coord in filtered.coords:
            filtered.coords[coord] = filtered.coords[coord].broadcast(sizes=final_sizes)
        self._data = filtered
        # sself.neutrons = filtered.sizes["event"]
        return self
