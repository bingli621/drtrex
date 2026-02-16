import scipp as sc


def centers_to_edges(centers, dim=None):
    """return bin edges with the given center values"""
    dim = centers.dim if dim is None else dim
    mid = (centers[:-1] + centers[1:]) / 2
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    return sc.concat([first, mid, last], dim=dim)
