from functools import partial
import numpy as np
from typing import Dict, Tuple
import scipp as sc
import tof
from .params import mon_sample


def uniform(ei, low, high):
    rng = np.random.default_rng(seed=83)
    """Uniform sampling between low and high"""
    de = sc.array(
        dims=ei.dims, values=rng.uniform(low, high, size=ei.shape), unit="meV"
    )
    return ei.to(unit="meV", copy=False) - de


def choice(ei, a):
    """Choose in an array, or if a is an integer, choose in np.arange(a)"""
    rng = np.random.default_rng(seed=83)
    de = sc.array(dims=ei.dims, values=rng.choice(a, size=ei.shape), unit="meV")
    return ei.to(unit="meV", copy=False) - de


def normal(ei, loc, scale):
    """Normal distribution, loc is center mu, scale is standard deviation sigma"""
    rng = np.random.default_rng(seed=83)
    de = sc.array(
        dims=ei.dims, values=rng.normal(loc, scale, size=ei.shape), unit="meV"
    )
    return ei.to(unit="meV", copy=False) - de


class Sample(tof.InelasticSample):  # type: ignore
    def __init__(self, en: Dict[str, Tuple[float, float]], name: str = "sample"):
        """Sample class

        Argument en defines the energy transfer, it need to be one of the three:
        en = {'uniform': (low, high)}
        en = {'choice': a} where a is an array or an integer
        en = {'normal': (loc, scale)}
        """
        ((key, value),) = en.items()
        en_dict = {
            "uniform": partial(uniform, low=value[0], high=value[1]),
            "choice": partial(choice, a=value),
            "normal": partial(normal, loc=value[0], scale=value[1]),
        }

        try:
            func = en_dict[key]
        except KeyError:
            raise ValueError(
                f"Unknown energy transfer type: {key}. "
                + "Needs to be 'uniform', 'choice' or 'normal'."
            )

        super().__init__(
            distance=mon_sample.distance,
            name=name,
            func=func,
        )
