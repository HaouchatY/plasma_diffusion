"""Preprocessing of TCV experimental data.

Turns the raw content of a shot file (measurements, etendues, good-channel
list, LIUQE equilibria) into the quantities needed by the samplers: the
measurement vector in normalized geometry-matrix units and the magnetic
equilibrium interpolated onto the reconstruction grid.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["interpolate_equilibria", "build_measurement"]

# TCV reconstruction-grid extent (m)
TCV_ZMIN, TCV_ZMAX = -0.75, 0.75
TCV_RMIN, TCV_RMAX = 0.624, 1.1376


def interpolate_equilibria(shot_data, nz: int = 120, nr: int = 40) -> np.ndarray:
    """Interpolate the LIUQE poloidal-flux maps onto the reconstruction grid.

    Parameters
    ----------
    shot_data:
        The ``data`` record of a shot ``.mat`` file, with fields ``psi``
        (LIUQE flux maps, shape ``(nr_liuqe, nz_liuqe, n_times)``),
        ``liuqe_rs`` and ``liuqe_zs`` (LIUQE grid coordinates).
    nz, nr:
        Vertical/radial size of the reconstruction grid.

    Returns
    -------
    Array of shape ``(n_times, nz, nr)`` with the interpolated equilibria,
    following the image convention of the training profiles (row 0 = top of
    the vessel, negative flux inside the core).
    """
    psi_raw = shot_data["psi"]
    rs = shot_data["liuqe_rs"][:, 0]
    zs = shot_data["liuqe_zs"][:, 0]
    dr_liuqe = float(np.mean(np.diff(rs.flatten())))
    dz_liuqe = float(np.mean(np.diff(zs.flatten())))

    dz = (TCV_ZMAX - TCV_ZMIN) / nz
    dr = (TCV_RMAX - TCV_RMIN) / nr
    grid_z = np.linspace(TCV_ZMIN, TCV_ZMAX, nz + 1, endpoint=True)[:-1] + dz / 2
    grid_r = np.linspace(TCV_RMIN, TCV_RMAX, nr + 1, endpoint=True)[:-1] + dr / 2

    # reconstruction-grid coordinates expressed in LIUQE image coordinates
    r_pos = (grid_r - rs.min()) / dr_liuqe
    z_pos = -(np.flip(grid_z) - zs.max()) / dz_liuqe
    coords = np.zeros((2, nz * nr))
    coords[0] = np.repeat(z_pos, nr)
    coords[1] = np.tile(r_pos, nz)

    n_times = psi_raw.shape[2]
    equilibria = np.zeros((n_times, nz, nr))
    for i in range(n_times):
        equilibria[i] = ndimage.map_coordinates(
            np.flip(psi_raw[:, :, i].T, axis=0), coords,
            order=2, mode="nearest", prefilter=True,
        ).reshape(nz, nr)
    return equilibria


def build_measurement(shot_data, frame: int, geometry_max: float):
    """Measurement vector of one time frame in normalized-geometry units.

    The raw detector signals are converted to line-integrated units
    (``measurements * etendue * 4 pi``) and divided by the maximum entry of
    the *raw* geometry matrix, so that they match a geometry matrix that was
    normalized to unit maximum. Faulty channels are removed.

    Returns
    -------
    y:
        Measurement vector restricted to the good channels.
    good_channels:
        Zero-based indices of the good channels (rows of the geometry matrix
        to keep).
    """
    good_channels = shot_data["good_channels"].flatten() - 1
    etendues = shot_data["etendues"].flatten()
    y = np.asarray(shot_data["measurements"][frame, :], dtype=np.float64)
    y = y * etendues * 4.0 * np.pi / geometry_max
    return y[good_channels], good_channels
