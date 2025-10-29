"""Core simulator for the discrete scale recurrence dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import (
    StructureFunctionResult,
    compute_fidelity,
    detect_revivals,
    spectral_entropy,
    structure_function_exponent,
)


@dataclass
class NoiseModel:
    """
    Gaussian noise model applied at every iteration.

    alpha_std: multiplicative jitter on alpha, applied as alpha*(1 + N(0, alpha_std))
    phase_std: additive phase noise standard deviation (radians)
    phase_space: 'spatial' applies noise in real space, 'fourier' applies it in k-space
    seed: RNG seed for reproducibility
    """

    alpha_std: float = 0.0
    phase_std: float = 0.0
    phase_space: str = "spatial"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.alpha_std < 0.0 or self.phase_std < 0.0:
            raise ValueError("noise standard deviations must be non-negative")
        if self.phase_space not in {"spatial", "fourier"}:
            raise ValueError("phase_space must be either 'spatial' or 'fourier'")


@dataclass
class SimulationResult:
    """Container for the data collected during a simulation run."""

    fidelities: np.ndarray
    alphas: np.ndarray
    spectral_entropy: Optional[np.ndarray]
    structure_slopes: Optional[np.ndarray]
    structure_details: Optional[List[Optional[StructureFunctionResult]]]
    stored_states: Optional[np.ndarray]
    stored_steps: Sequence[int]
    revivals: List[Tuple[int, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepState:
    """Snapshot of the system at a given iteration."""

    step: int
    fidelity: float
    spectral_entropy: Optional[float]
    structure: Optional[StructureFunctionResult]
    alpha: Optional[float]
    state: Optional[np.ndarray] = None


class DiscreteScaleRecurrenceSimulator:
    """
    Simulate repeated applications of U_alpha = exp(i * alpha * Delta) on a periodic grid.
    """

    def __init__(
        self,
        grid_shape: Sequence[int],
        normalize_each_step: bool = True,
        dtype: np.dtype = np.complex128,
    ) -> None:
        self.grid_shape = tuple(int(n) for n in grid_shape)
        if np.prod(self.grid_shape) <= 1:
            raise ValueError("grid must contain at least two points")
        self.normalize_each_step = normalize_each_step
        self.dtype = np.dtype(dtype)
        self.k2 = self._build_k2(self.grid_shape)

    @staticmethod
    def _build_k2(shape: Sequence[int]) -> np.ndarray:
        """Squared wave-number magnitude for each lattice site in Fourier space."""
        axes = [np.fft.fftfreq(n, d=1.0 / n) for n in shape]
        grids = np.meshgrid(*axes, indexing="ij")
        k2 = np.zeros(shape, dtype=float)
        for axis in grids:
            k2 += axis ** 2
        return k2

    @staticmethod
    def _normalize(state: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(state.ravel())
        if norm == 0.0:
            return state
        return state / norm

    def step_stream(
        self,
        psi0: np.ndarray,
        alpha: float,
        num_steps: int,
        *,
        noise: Optional[NoiseModel] = None,
        metrics: Sequence[str] = ("fidelity", "spectral_entropy", "structure_exponent"),
        structure_scales: Optional[Sequence[int]] = None,
        return_state: bool = False,
    ) -> Iterator[StepState]:
        """
        Yield a :class:`StepState` for each iteration, starting at step 0 (initial state).

        If ``return_state`` is True, each snapshot contains a copy of the current
        complex state. Otherwise, ``state`` is left as ``None``.
        """
        metrics_set = set(metrics)
        compute_entropy = "spectral_entropy" in metrics_set
        compute_structure = "structure_exponent" in metrics_set

        psi = np.asarray(psi0, dtype=self.dtype).copy()
        if psi.shape != self.grid_shape:
            raise ValueError(f"psi0 shape {psi.shape} does not match grid {self.grid_shape}")
        if self.normalize_each_step:
            psi = self._normalize(psi)
        initial_state = psi.copy()

        current_hat = np.fft.fftn(psi)

        rng = np.random.default_rng(noise.seed) if noise is not None else None

        initial_fidelity = compute_fidelity(initial_state, psi)
        initial_entropy = spectral_entropy(psi) if compute_entropy else None
        initial_structure = (
            structure_function_exponent(np.abs(psi) ** 2, q=2.0, scales=structure_scales)
            if compute_structure
            else None
        )
        yield StepState(
            step=0,
            fidelity=initial_fidelity,
            spectral_entropy=initial_entropy,
            structure=initial_structure,
            alpha=None,
            state=psi.copy() if return_state else None,
        )

        for step in range(1, num_steps + 1):
            alpha_eff = alpha
            if noise is not None and noise.alpha_std > 0.0:
                alpha_eff *= 1.0 + rng.normal(scale=noise.alpha_std)

            phase = np.exp(-1j * alpha_eff * self.k2)
            current_hat *= phase
            psi = np.fft.ifftn(current_hat)

            if noise is not None and noise.phase_std > 0.0:
                if noise.phase_space == "spatial":
                    phase_noise = np.exp(
                        1j * rng.normal(scale=noise.phase_std, size=self.grid_shape)
                    )
                    psi *= phase_noise
                    current_hat = np.fft.fftn(psi)
                else:
                    phase_noise = np.exp(
                        1j * rng.normal(scale=noise.phase_std, size=self.grid_shape)
                    )
                    current_hat *= phase_noise
                    psi = np.fft.ifftn(current_hat)

            if self.normalize_each_step:
                psi = self._normalize(psi)
                current_hat = np.fft.fftn(psi)

            fidelity = compute_fidelity(initial_state, psi)
            ent = spectral_entropy(psi) if compute_entropy else None
            struct = (
                structure_function_exponent(np.abs(psi) ** 2, q=2.0, scales=structure_scales)
                if compute_structure
                else None
            )

            yield StepState(
                step=step,
                fidelity=fidelity,
                spectral_entropy=ent,
                structure=struct,
                alpha=alpha_eff,
                state=psi.copy() if return_state else None,
            )

    def run(
        self,
        psi0: np.ndarray,
        alpha: float,
        num_steps: int,
        *,
        noise: Optional[NoiseModel] = None,
        metrics: Sequence[str] = ("fidelity", "spectral_entropy", "structure_exponent"),
        structure_scales: Optional[Sequence[int]] = None,
        store_states: bool = True,
        store_stride: int = 1,
        revival_threshold: float = 0.95,
        revival_min_separation: int = 1,
    ) -> SimulationResult:
        """
        Run the simulation for ``num_steps`` iterations and collect diagnostics.
        """
        metrics_set = set(metrics)
        compute_entropy = "spectral_entropy" in metrics_set
        compute_structure = "structure_exponent" in metrics_set

        fidelities = np.empty(num_steps + 1, dtype=float)
        entropies = np.empty_like(fidelities) if compute_entropy else None
        structure_slopes = np.empty_like(fidelities) if compute_structure else None
        structure_details: Optional[List[Optional[StructureFunctionResult]]] = (
            [None] * (num_steps + 1) if compute_structure else None
        )
        stored_states: List[np.ndarray] = []
        stored_steps: List[int] = []
        alphas = np.empty(num_steps, dtype=float)

        stream = self.step_stream(
            psi0=psi0,
            alpha=alpha,
            num_steps=num_steps,
            noise=noise,
            metrics=metrics,
            structure_scales=structure_scales,
            return_state=store_states,
        )

        for snapshot in stream:
            step = snapshot.step
            fidelities[step] = snapshot.fidelity
            if entropies is not None:
                entropies[step] = snapshot.spectral_entropy if snapshot.spectral_entropy is not None else np.nan
            if compute_structure and structure_slopes is not None and structure_details is not None:
                result = snapshot.structure
                if result is not None:
                    structure_slopes[step] = result.slope
                    structure_details[step] = result
                else:
                    structure_slopes[step] = np.nan
                    structure_details[step] = None
            if step > 0:
                alphas[step - 1] = snapshot.alpha if snapshot.alpha is not None else alpha
            if store_states and snapshot.state is not None:
                if step % max(store_stride, 1) == 0 or step == num_steps:
                    stored_states.append(snapshot.state.copy())
                    stored_steps.append(step)

        revivals = detect_revivals(
            fidelities,
            threshold=revival_threshold,
            min_separation=revival_min_separation,
        )

        metadata: Dict[str, Any] = {
            "grid_shape": self.grid_shape,
            "normalize_each_step": self.normalize_each_step,
            "metrics": tuple(metrics_set),
            "store_stride": store_stride,
            "alpha_requested": alpha,
        }
        if noise is not None:
            metadata["noise"] = {
                "alpha_std": noise.alpha_std,
                "phase_std": noise.phase_std,
                "phase_space": noise.phase_space,
                "seed": noise.seed,
            }

        stored_array = np.stack(stored_states, axis=0) if store_states else None
        return SimulationResult(
            fidelities=fidelities,
            alphas=alphas,
            spectral_entropy=entropies,
            structure_slopes=structure_slopes,
            structure_details=structure_details,
            stored_states=stored_array,
            stored_steps=stored_steps,
            revivals=revivals,
            metadata=metadata,
        )
