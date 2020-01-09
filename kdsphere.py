### https://github.com/jakevdp/kdsphere
### taken from Jake VanderPlas' kdsphere module, but don't need the whole thing.

import numpy as np
from scipy.spatial import cKDTree

def spherical_to_cartesian(data, return_radius=False):
    """Convert spherical coordinates to cartesian coordinates

    Parameters
    ----------
    data : array, shape (N, 2) or (N, 3)
    a collection of (lon, lat) or (lon, lat, time) coordinates.
    lon and lat should be in radians.
    If times are not specified, all are set to 1.0.

    Returns
    -------
    data3D : array, shape (N, 3)
    A 3D Cartesian view of the data. If time is provided, it is mapped to
    the radius.
    """
    data = np.asarray(data, dtype=float)

    # Data should be two-dimensional
    if data.ndim != 2:
        raise ValueError("data.shape = {0} should be "
                        "(N, 2) or (N, 3)".format(data.shape))

    # Data should have 2 or 3 columns
    if data.shape[1] == 2:
        lon, lat = data.T
        r = 1.0
    elif data.shape[1] == 3:
        lon, lat, r = data.T
    else:
        raise ValueError("data.shape = {0} should be "
                        "(N, 2) or (N, 3)".format(data.shape))

    data3d = np.array([r * np.cos(lat) * np.cos(lon),
                   r * np.cos(lat) * np.sin(lon),
                   r * np.sin(lat)]).T

    if return_radius:
        return data3d, r
    else:
        return data3d

class KDSphere(object):
    """KD Tree for Spherical Data, built on scipy's cKDTree

    Parameters
    ----------
    data : array_like, shape (N, 2)
        (lon, lat) pairs measured in radians
    **kwargs :
        Additional arguments are passed to cKDTree
    """
    def __init__(self, data, **kwargs):
        self.data = np.asarray(data)
        self.data3d = spherical_to_cartesian(self.data)
        self.kdtree_ = cKDTree(self.data3d, **kwargs)

    def query(self, data, k=1, eps=0, **kwargs):
        """Query for k-nearest neighbors

        Parameters
        ----------
        data : array_like, shape (N, 2)
            (lon, lat) pairs measured in radians
        k : integer
            The number of nearest neighbors to return.
        eps : non-negative float
            Return approximate nearest neighbors; the k-th returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real k-th nearest neighbor.

        Returns
        -------
        d : array_like, float, shape=(N, k)
            The distances to the nearest neighbors
        i : array_like, int, shape=(N, k)
            The indices of the neighbors
        """
        data_3d, r = spherical_to_cartesian(data, return_radius=True)
        dist_3d, ind = self.kdtree_.query(data_3d, k=k, eps=eps, **kwargs)
        dist_2d = 2 * np.arcsin(dist_3d * 0.5 / r)
        return dist_2d, ind
