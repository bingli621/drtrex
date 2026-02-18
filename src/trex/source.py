import tof
import scipp as sc


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

    @staticmethod
    def calculate_range(data, number_of_sigma=1):
        mean = sc.mean(data)
        std = sc.std(data, ddof=0)
        min_value = max(data.min(), mean - number_of_sigma * std)
        max_value = mean + number_of_sigma * std
        return (min_value, max_value)

    def calculate_time_range(self):
        data = self.data.coords["birth_time"]
        return self.calculate_range(data, number_of_sigma=3)

    def calculate_wavelength_range(self):
        data = self.data.coords["wavelength"]
        return self.calculate_range(data, number_of_sigma=3)
