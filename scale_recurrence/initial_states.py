"""Factories for common initial states on a periodic lattice."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def _normalize(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state.ravel())
    if norm == 0.0:
        return state
    return state / norm


def gaussian_wavepacket(
    grid_shape: Sequence[int],
    center: Optional[Sequence[float]] = None,
    width: float = 0.1,
    momentum: Optional[Sequence[float]] = None,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """
    Gaussian wavepacket on a unit torus with optional momentum tilt.

    *center* and *momentum* are expressed in units of the domain size.
    """
    grid_shape = tuple(int(n) for n in grid_shape)
    dims = len(grid_shape)
    if center is None:
        center = [0.5] * dims
    if momentum is None:
        momentum = [0.0] * dims
    axes = [
        np.linspace(0.0, 1.0, num=n, endpoint=False, dtype=float) for n in grid_shape
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    exponent = np.zeros(grid_shape, dtype=float)
    phase = np.zeros(grid_shape, dtype=float)
    for axis, coord, mu, k in zip(range(dims), mesh, center, momentum):
        delta = np.minimum(np.abs(coord - mu), 1.0 - np.abs(coord - mu))
        exponent += (delta ** 2) / (2.0 * width ** 2)
        phase += (2.0 * np.pi * k) * coord
    state = np.exp(-(exponent) + 1j * phase)
    return _normalize(state.astype(dtype, copy=False))


def dirac_comb(
    grid_shape: Sequence[int],
    period: Sequence[int],
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """
    Periodic comb of Dirac-like peaks spaced by *period* along each axis.
    """
    grid_shape = tuple(int(n) for n in grid_shape)
    period = tuple(int(p) for p in period)
    if len(period) != len(grid_shape):
        raise ValueError("period must have the same dimensionality as grid_shape")
    state = np.zeros(grid_shape, dtype=dtype)
    idx = [slice(None, None, p) for p in period]
    state[tuple(idx)] = 1.0
    return _normalize(state)


def random_phase_state(
    grid_shape: Sequence[int],
    seed: Optional[int] = None,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """
    Random phase state with uniform amplitude and random phases in [-pi, pi).
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, size=tuple(grid_shape))
    state = np.exp(1j * phases)
    return _normalize(state.astype(dtype, copy=False))
