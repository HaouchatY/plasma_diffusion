"""Equilibrium-informed reaction-diffusion (RD) operator for DiffRD.

The DiffRD sampler augments the DiffPIR proximal subproblem with the
anisotropic smoothness term ``||L x||^2``, where ``L = sqrt(D(psi, alpha)) G``
combines the discrete gradient ``G`` with the anisotropic diffusion tensor
``D`` built from the magnetic equilibrium ``psi`` (smoothing is promoted along
rather than across flux surfaces).

Building ``L`` requires the optional dependency `pyxu-diffops
<https://github.com/pyxu-org/pyxu-diffops>`_ (the RD framework of Hamm et al.,
PPCF 2025). A precomputed operator for the tutorial equilibrium ships in
``data/rd_operator_example.npz``, so the tutorial runs without this extra.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["build_rd_operator", "save_rd_operator", "load_rd_operator"]


def build_rd_operator(psi: np.ndarray, anis_param: float = 1e-2,
                      sampling: float = 0.0125):
    """Build the anisotropic gradient matrix ``L`` for a given equilibrium.

    Parameters
    ----------
    psi:
        Poloidal-flux map of shape ``(H, W)`` on the reconstruction grid.
    anis_param:
        Anisotropy parameter ``alpha`` in ``(0, 1]``: the smoothing intensity
        across flux surfaces relative to the (unit) intensity along them.
    sampling:
        Grid spacing (m).

    Returns
    -------
    L:
        Sparse matrix with ``H W`` columns, such that the anisotropic
        smoothness term reads ``||L x||^2``.
    lipschitz:
        Lipschitz constant of the associated diffusion operator (twice the
        squared spectral norm bound used for step sizes).

    Raises
    ------
    ImportError:
        If ``pyxu-diffops`` is not installed.
    """
    try:
        from pyxu_diffops.operator import AnisDiffusionOp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Building RD operators requires the optional dependency "
            "'pyxu-diffops' (pip install pyxu-diffops)."
        ) from exc

    h, w = psi.shape
    op = AnisDiffusionOp(dim_shape=(1, h, w), alpha=anis_param,
                         diff_method_struct_tens="fd", freezing_arr=psi,
                         sampling=sampling, matrix_based_impl=True)
    L = sp.csr_matrix(op._grad_matrix_based.mat)
    return L, float(op.diff_lipschitz)


def save_rd_operator(path, L: sp.spmatrix, lipschitz: float) -> None:
    """Save an RD operator to a single ``.npz`` file."""
    L = sp.csr_matrix(L)
    np.savez_compressed(path, data=L.data, indices=L.indices, indptr=L.indptr,
                        shape=L.shape, lipschitz=lipschitz)


def load_rd_operator(path):
    """Load an RD operator saved by :func:`save_rd_operator`.

    Returns ``(L, lipschitz)`` with ``L`` a CSR sparse matrix.
    """
    with np.load(path) as f:
        L = sp.csr_matrix((f["data"], f["indices"], f["indptr"]),
                          shape=tuple(f["shape"]))
        lipschitz = float(f["lipschitz"])
    return L, lipschitz
