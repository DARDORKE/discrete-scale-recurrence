#!/usr/bin/env python3
"""
Command-line helper to explore the discrete scale recurrence simulator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from scale_recurrence import (
    DiscreteScaleRecurrenceSimulator,
    NoiseModel,
    initial_states,
    metrics,
)


def parse_grid(value: str) -> Tuple[int, ...]:
    tokens = value.lower().replace("x", ",").split(",")
    dims = [int(tok.strip()) for tok in tokens if tok.strip()]
    if not dims:
        raise ValueError("grid specification is empty")
    return tuple(dims)


def parse_float_list(value: Optional[str], dims: int) -> Optional[List[float]]:
    if value is None:
        return None
    tokens = value.replace("x", ",").split(",")
    values = [float(tok.strip()) for tok in tokens if tok.strip()]
    if len(values) != dims:
        raise ValueError(f"expected {dims} entries, got {len(values)} in '{value}'")
    return values


def parse_int_list(value: Optional[str], dims: int) -> Optional[List[int]]:
    if value is None:
        return None
    tokens = value.replace("x", ",").split(",")
    values = [int(tok.strip()) for tok in tokens if tok.strip()]
    if len(values) != dims:
        raise ValueError(f"expected {dims} entries, got {len(values)} in '{value}'")
    return values


def parse_scales(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    tokens = value.split(",")
    return [max(1, int(tok.strip())) for tok in tokens if tok.strip()]


def resolve_alpha(alpha: Optional[float], ratio: Optional[str]) -> float:
    if alpha is not None:
        return float(alpha)
    if ratio is None:
        raise ValueError("either --alpha or --alpha-ratio must be provided")
    if "/" in ratio:
        numer, denom = ratio.split("/", 1)
        frac = float(int(numer)) / float(int(denom))
    else:
        frac = float(ratio)
    return 2.0 * math.pi * frac


def build_initial_state(args: argparse.Namespace, grid_shape: Tuple[int, ...]) -> np.ndarray:
    dims = len(grid_shape)
    if args.initial == "gaussian":
        center = parse_float_list(args.center, dims) if args.center else None
        momentum = parse_float_list(args.momentum, dims) if args.momentum else None
        return initial_states.gaussian_wavepacket(
            grid_shape=grid_shape,
            center=center,
            width=args.width,
            momentum=momentum,
            dtype=np.dtype(args.dtype),
        )
    if args.initial == "comb":
        period = parse_int_list(args.period, dims)
        if period is None:
            raise ValueError("--period must be provided for comb initial state")
        return initial_states.dirac_comb(
            grid_shape=grid_shape,
            period=period,
            dtype=np.dtype(args.dtype),
        )
    if args.initial == "random":
        return initial_states.random_phase_state(
            grid_shape=grid_shape,
            seed=args.random_seed,
            dtype=np.dtype(args.dtype),
        )
    raise ValueError(f"unknown initial state '{args.initial}'")


def summarize(result, alpha, grid_shape, save_path=None) -> None:
    dims = len(grid_shape)
    print(f"Grid: {grid_shape} ({dims}D) | steps: {result.fidelities.size - 1}")
    avg_alpha = float(np.mean(result.alphas)) if result.alphas.size else alpha
    std_alpha = float(np.std(result.alphas)) if result.alphas.size else 0.0
    print(f"alpha target: {alpha:.6g} | mean(alpha_n): {avg_alpha:.6g} ± {std_alpha:.3g}")
    ratio = alpha / (2.0 * math.pi)
    p, q, approx, error = metrics.rational_approximation(ratio)
    print(f"alpha / 2π ≈ {p}/{q} = {approx:.6g} (error {error:.2e})")
    print(f"max fidelity: {np.max(result.fidelities):.6f}")
    if result.spectral_entropy is not None:
        print(
            f"spectral entropy: start {result.spectral_entropy[0]:.4f} → "
            f"end {result.spectral_entropy[-1]:.4f}"
        )
    if result.structure_slopes is not None:
        print(
            f"structure slope: start {result.structure_slopes[0]:.4f} → "
            f"end {result.structure_slopes[-1]:.4f}"
        )

    if result.revivals:
        print("Detected revivals (step : fidelity):")
        for step, fidelity in result.revivals:
            print(f"  {step:5d} : {fidelity:.6f}")
    else:
        print("No revivals above the threshold were detected.")

    if save_path is not None:
        payload = {
            "fidelities": result.fidelities,
            "alphas": result.alphas,
        }
        if result.spectral_entropy is not None:
            payload["spectral_entropy"] = result.spectral_entropy
        if result.structure_slopes is not None:
            payload["structure_slopes"] = result.structure_slopes
        if result.stored_states is not None:
            payload["stored_states"] = result.stored_states
            payload["stored_steps"] = np.array(result.stored_steps, dtype=int)
        payload["metadata"] = np.array([json.dumps(result.metadata)], dtype=object)
        np.savez(save_path, **payload)
        print(f"Saved arrays to {save_path}")


def _final_state_projection(state: np.ndarray) -> Tuple[np.ndarray, str]:
    prob = np.abs(state) ** 2
    if prob.ndim == 1:
        return prob, "|psi|^2 (final state)"
    if prob.ndim == 2:
        return prob, "|psi|^2 (final state)"
    collapsed = prob.copy()
    while collapsed.ndim > 2:
        collapsed = collapsed.sum(axis=-1)
    return collapsed, "|psi|^2 (sum over trailing axes)"


def plot_results(result, grid_shape: Tuple[int, ...], plot_file: Optional[Path] = None) -> None:
    try:
        if plot_file is not None:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required for --plot or --plot-file. "
            "Install it via `python3 -m pip install matplotlib`."
        ) from exc

    steps = np.arange(result.fidelities.size)
    sections = ["fidelity"]
    if result.spectral_entropy is not None:
        sections.append("entropy")
    if result.structure_slopes is not None:
        sections.append("structure")
    if result.stored_states is not None:
        sections.append("state")

    fig, axes = plt.subplots(len(sections), 1, figsize=(8, 2.6 * len(sections)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax_idx = 0
    ax = axes[ax_idx]
    ax.plot(steps, result.fidelities, color="tab:blue", label="Fidelity")
    for step, _ in result.revivals:
        ax.axvline(step, color="tab:orange", linestyle="--", alpha=0.3)
    ax.set_ylabel("Fidelity")
    ax.set_title("Fidelity over iterations")
    ax.grid(True, alpha=0.2)
    ax.legend()

    if result.spectral_entropy is not None:
        ax_idx += 1
        ax = axes[ax_idx]
        ax.plot(steps, result.spectral_entropy, color="tab:green")
        ax.set_ylabel("Entropy")
        ax.set_title("Normalized spectral entropy")
        ax.grid(True, alpha=0.2)

    if result.structure_slopes is not None:
        ax_idx += 1
        ax = axes[ax_idx]
        ax.plot(steps, result.structure_slopes, color="tab:red")
        ax.set_ylabel("Slope")
        ax.set_title("Structure function exponent (q=2)")
        ax.grid(True, alpha=0.2)

    if result.stored_states is not None:
        ax_idx += 1
        ax = axes[ax_idx]
        state = result.stored_states[-1]
        data, label = _final_state_projection(state)
        if data.ndim == 1:
            ax.plot(np.arange(data.size), data, color="tab:purple")
            ax.set_ylabel(label)
        else:
            im = ax.imshow(data, origin="lower", cmap="viridis")
            ax.set_ylabel(label)
            fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Final profile")
        ax.grid(False)

    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()

    if plot_file is not None:
        fig.savefig(plot_file, dpi=160)
        print(f"Saved figure to {plot_file}")
        plt.close(fig)
    else:
        plt.show()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Simulate the discrete scale recurrence generated by U_alpha.",
    )
    parser.add_argument("--grid", default="256", help="grid shape, e.g. 256 or 128x128")
    parser.add_argument("--steps", type=int, default=256, help="number of iterations")
    parser.add_argument("--alpha", type=float, help="value of alpha in radians")
    parser.add_argument(
        "--alpha-ratio",
        type=str,
        help="fraction p/q interpreted as alpha = 2π * (p/q); overrides --alpha if given",
    )
    parser.add_argument(
        "--initial",
        choices=["gaussian", "comb", "random"],
        default="gaussian",
        help="initial state profile",
    )
    parser.add_argument("--width", type=float, default=0.08, help="gaussian width")
    parser.add_argument("--center", type=str, help="comma-separated center coordinates")
    parser.add_argument("--momentum", type=str, help="comma-separated momentum tilt")
    parser.add_argument("--period", type=str, help="period for comb initial state")
    parser.add_argument("--random-seed", type=int, help="seed for random phases")
    parser.add_argument(
        "--metrics",
        type=str,
        default="fidelity,spectral_entropy,structure_exponent",
        help="comma-separated list of metrics to collect",
    )
    parser.add_argument("--noise-alpha-std", type=float, default=0.0)
    parser.add_argument("--noise-phase-std", type=float, default=0.0)
    parser.add_argument(
        "--noise-phase-space",
        choices=["spatial", "fourier"],
        default="spatial",
    )
    parser.add_argument("--structure-scales", type=str, help="comma-separated scales")
    parser.add_argument("--store-stride", type=int, default=1)
    parser.add_argument("--no-store-states", action="store_true")
    parser.add_argument("--revival-threshold", type=float, default=0.95)
    parser.add_argument("--revival-gap", type=int, default=1)
    parser.add_argument("--save", type=Path, help="path to save an .npz archive")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="display plots (requires Matplotlib)",
    )
    parser.add_argument(
        "--plot-file",
        type=Path,
        help="save the figure as PNG (requires Matplotlib)",
    )
    parser.add_argument(
        "--dtype",
        choices=["complex64", "complex128"],
        default="complex128",
        help="working precision for the simulation",
    )
    args = parser.parse_args(argv)

    grid_shape = parse_grid(args.grid)
    dtype = np.complex64 if args.dtype == "complex64" else np.complex128

    alpha = resolve_alpha(args.alpha, args.alpha_ratio)

    psi0 = build_initial_state(args, grid_shape)

    simulator = DiscreteScaleRecurrenceSimulator(
        grid_shape=grid_shape,
        normalize_each_step=True,
        dtype=dtype,
    )

    noise = None
    if args.noise_alpha_std > 0.0 or args.noise_phase_std > 0.0:
        noise = NoiseModel(
            alpha_std=args.noise_alpha_std,
            phase_std=args.noise_phase_std,
            phase_space=args.noise_phase_space,
            seed=args.random_seed,
        )

    metrics_list = [tok.strip() for tok in args.metrics.split(",") if tok.strip()]
    structure_scales = parse_scales(args.structure_scales)

    result = simulator.run(
        psi0=psi0,
        alpha=alpha,
        num_steps=args.steps,
        noise=noise,
        metrics=metrics_list,
        structure_scales=structure_scales,
        store_states=not args.no_store_states,
        store_stride=max(1, args.store_stride),
        revival_threshold=args.revival_threshold,
        revival_min_separation=max(1, args.revival_gap),
    )

    summarize(
        result,
        alpha=alpha,
        grid_shape=grid_shape,
        save_path=args.save,
    )

    if args.plot or args.plot_file is not None:
        plot_results(
            result,
            grid_shape=grid_shape,
            plot_file=args.plot_file,
        )


if __name__ == "__main__":
    main()
