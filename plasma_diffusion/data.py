"""Data utilities: amplitude normalization and noise estimation.

The diffusion prior is trained on profiles with a fixed amplitude scale, so
measured data must be brought to that scale before posterior sampling. The
normalization coefficient ``alpha`` is estimated by projecting the measurement
vector onto the closest (shape-wise) measurement of the training set; the
reconstruction is multiplied back by ``alpha`` afterwards.
"""

from __future__ import annotations

import numpy as np
import pywt

__all__ = [
    "closest_training_measurement",
    "normalization_coefficient",
    "estimate_noise_std",
    "fit_affine_noise_model",
]


def closest_training_measurement(y: np.ndarray, y_train: np.ndarray,
                                 comparison: str = "norm") -> np.ndarray:
    """Training measurement whose *shape* is closest to ``y``.

    Both ``y`` (shape ``(M,)``) and the rows of ``y_train`` (shape ``(K, M)``)
    are normalized (by Euclidean norm, or by max if ``comparison='max'``)
    before the nearest-neighbour search, so only the profile shape matters.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y_train = np.asarray(y_train, dtype=np.float64)
    if comparison == "norm":
        y_n = y / np.linalg.norm(y)
        t_n = y_train / np.linalg.norm(y_train, axis=1, keepdims=True)
    elif comparison == "max":
        y_n = y / np.max(np.abs(y))
        t_n = y_train / np.max(np.abs(y_train), axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown comparison: {comparison!r}")
    idx = int(np.argmin(np.linalg.norm(t_n - y_n, axis=1)))
    return y_train[idx]


def normalization_coefficient(y: np.ndarray, y_train: np.ndarray,
                              comparison: str = "norm") -> float:
    """Amplitude scale ``alpha`` such that ``y / alpha`` matches the training
    amplitude distribution.

    ``alpha`` is the least-squares projection coefficient of ``y`` onto its
    closest training measurement: ``alpha = <y, y*> / <y*, y*>``.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y_closest = closest_training_measurement(y, y_train, comparison)
    return float(np.dot(y, y_closest) / np.dot(y_closest, y_closest))


def estimate_noise_std(y: np.ndarray) -> float:
    """Robust noise-floor estimate from the finest-scale wavelet details.

    A median-absolute-deviation estimator applied to the smallest 80% of the
    finest-scale ``db4`` detail coefficients of the measurement vector.

    Notes
    -----
    This estimates the *additive* noise floor only. For diagnostics whose
    noise grows with the signal (e.g. photon statistics, relative calibration
    errors), consider the affine per-channel model fitted by
    :func:`fit_affine_noise_model` from time-resolved data.
    """
    d = pywt.wavedec(np.asarray(y, dtype=np.float64), "db4", level=2,
                     mode="periodization")[-1]
    d = d[np.abs(d) <= np.percentile(np.abs(d), 80)]
    return float(np.median(np.abs(d)) / 0.6745)


def fit_affine_noise_model(measurements: np.ndarray) -> tuple[float, float]:
    """Fit the signal-dependent noise model ``sigma_ch = a + b |y_ch|``.

    Parameters
    ----------
    measurements:
        Time-resolved measurements of shape ``(n_times, n_channels)`` sampled
        fast enough that the signal barely evolves between frames. The
        per-channel noise std is measured from frame-to-frame differences and
        regressed against the mean absolute signal.

    Returns
    -------
    (a, b):
        Additive floor and relative (signal-proportional) noise coefficient,
        in the units of ``measurements``.
    """
    m = np.asarray(measurements, dtype=np.float64)
    sigma_ch = np.std(np.diff(m, axis=0), axis=0) / np.sqrt(2.0)
    y_mean = np.abs(m).mean(axis=0)
    design = np.vstack([np.ones_like(y_mean), y_mean]).T
    (a, b), *_ = np.linalg.lstsq(design, sigma_ch, rcond=None)
    return float(a), float(b)
