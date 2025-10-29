"""Helper metrics for the discrete scale recurrence simulations."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def compute_fidelity(reference: np.ndarray, state: np.ndarray) -> float:
    """Return the fidelity |<reference|state>|^2, normalized by vector norms."""
    ref_flat = np.asarray(reference).ravel()
    state_flat = np.asarray(state).ravel()
    denom = np.linalg.norm(ref_flat) * np.linalg.norm(state_flat)
    if denom == 0.0:
        return np.nan
    overlap = np.vdot(ref_flat, state_flat)
    return float(np.abs(overlap) ** 2 / (denom ** 2))


def spectral_entropy(state: np.ndarray, eps: float = 1e-15) -> float:
    """
    Normalized spectral entropy of the wavefunction amplitude.

    The entropy is computed from the squared modulus of the Fourier spectrum
    and normalized by log(N) so it lies in [0, 1].
    """
    spectrum = np.abs(np.fft.fftn(state)) ** 2
    power = spectrum / (np.sum(spectrum) + eps)
    power = power.reshape(-1)
    power = power[power > 0]
    if power.size == 0:
        return 0.0
    entropy = -np.sum(power * np.log(power))
    max_entropy = np.log(state.size)
    if max_entropy <= 0:
        return 0.0
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


@dataclass
class StructureFunctionResult:
    """Summary statistics for the q-th order structure function."""

    q: float
    scales: np.ndarray
    moments: np.ndarray
    slope: float


def structure_function_exponent(
    field: np.ndarray,
    q: float = 2.0,
    scales: Optional[Sequence[int]] = None,
    eps: float = 1e-16,
) -> StructureFunctionResult:
    """
    Estimate the scaling exponent of the q-th structure function.

    The function uses periodic shifts along every axis and averages the moment
    of order q over these increments. A linear fit in log-log yields the slope.
    """
    data = np.asarray(field)
    min_dim = min(data.shape)
    if min_dim < 4:
        raise ValueError("grid dimensions are too small for structure analysis")
    if scales is None:
        max_scale = max(2, min_dim // 4)
        raw = np.geomspace(1, max_scale, num=4)
        scales = np.unique(np.clip(raw.astype(int), 1, None))
    scales = np.array(sorted(set(int(s) for s in scales if s >= 1)))
    if scales.size == 0:
        raise ValueError("no valid scales provided for structure function")

    moments: List[float] = []
    for shift in scales:
        accum = 0.0
        for axis in range(data.ndim):
            rolled = np.roll(data, -shift, axis=axis)
            diff = data - rolled
            accum += np.mean(np.abs(diff) ** q)
        moments.append(accum / data.ndim)
    log_scales = np.log(scales)
    log_moments = np.log(np.array(moments) + eps)
    slope, _ = np.polyfit(log_scales, log_moments, deg=1)
    return StructureFunctionResult(
        q=q,
        scales=scales.astype(int),
        moments=np.array(moments),
        slope=float(slope),
    )


def detect_revivals(
    fidelity_series: Sequence[float],
    threshold: float = 0.95,
    min_separation: int = 1,
) -> List[Tuple[int, float]]:
    """
    Identify local maxima in the fidelity series above a threshold.

    Returns a list of (step, fidelity) pairs sorted by step.
    """
    series = np.asarray(fidelity_series, dtype=float).reshape(-1)
    revivals: List[Tuple[int, float]] = []
    if series.size == 0:
        return revivals
    n = series.size

    def push_or_replace(step: int, value: float) -> None:
        if revivals and step - revivals[-1][0] < min_separation:
            if value > revivals[-1][1]:
                revivals[-1] = (step, float(value))
        else:
            revivals.append((step, float(value)))

    for idx in range(1, n - 1):
        val = series[idx]
        if val < threshold:
            continue
        if val < series[idx - 1] or val < series[idx + 1]:
            continue
        push_or_replace(idx, val)

    if (
        n >= 2
        and series[-1] >= threshold
        and series[-1] >= series[-2]
    ):
        push_or_replace(n - 1, series[-1])
    return revivals


def rational_approximation(
    value: float,
    max_denominator: int = 256,
) -> Tuple[int, int, float, float]:
    """
    Approximate *value* by a rational number with denominator <= max_denominator.

    Returns (numerator, denominator, fraction_as_float, absolute_error).
    """
    frac = Fraction(value).limit_denominator(max_denominator)
    approx = float(frac)
    error = abs(value - approx)
    return frac.numerator, frac.denominator, approx, error
