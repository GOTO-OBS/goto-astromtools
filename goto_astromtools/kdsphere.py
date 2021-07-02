### https://github.com/jakevdp/kdsphere
### taken from Jake VanderPlas' kdsphere module, but don't need the whole thing.

# Copyright (c) 2016, Jake Vanderplas
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
    :param return_radius:
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
