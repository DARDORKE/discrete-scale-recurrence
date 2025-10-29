#!/usr/bin/env python3
"""
Real-time animation of the discrete scale recurrence.

This script uses :class:`DiscreteScaleRecurrenceSimulator` to generate step-by-step
states and updates a Matplotlib figure (1D or 2D profile).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from scale_recurrence import (
    DiscreteScaleRecurrenceSimulator,
    NoiseModel,
    initial_states,
)
from scale_recurrence.simulator import StepState


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


class Visualizer:
    """Plot management for real-time animation."""

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        show_entropy: bool,
        show_structure: bool,
    ) -> None:
        self.grid_shape = grid_shape
        self.dim = len(grid_shape)
        if self.dim not in (1, 2):
            raise ValueError("real-time visualisation currently supports 1D or 2D grids")
        self.show_entropy = show_entropy
        self.show_structure = show_structure

        self.fig = None
        self.ax_state = None
        self.ax_fid = None
        self.state_line = None
        self.state_im = None
        self.fid_line = None
        self.info_text = None

        self.steps: List[int] = []
        self.fidelities: List[float] = []

    def setup(self):
        import matplotlib.pyplot as plt

        self.fig, (self.ax_state, self.ax_fid) = plt.subplots(
            2,
            1,
            figsize=(8, 6),
            sharex=False,
            gridspec_kw={"height_ratios": [3, 2]},
        )

        if self.dim == 1:
            x = np.arange(self.grid_shape[0])
            self.state_line, = self.ax_state.plot(x, np.zeros_like(x), color="tab:purple")
            self.ax_state.set_xlim(0, self.grid_shape[0] - 1)
        else:
            data = np.zeros(self.grid_shape)
            self.state_im = self.ax_state.imshow(data, origin="lower", cmap="viridis")
            self.fig.colorbar(self.state_im, ax=self.ax_state, fraction=0.046, pad=0.04)
        self.ax_state.set_ylabel("|psi|^2")
        self.ax_state.set_title("Spatial evolution")
        self.ax_state.grid(self.dim == 1, alpha=0.2)

        self.fid_line, = self.ax_fid.plot([], [], color="tab:blue")
        self.ax_fid.set_xlim(0, 1)
        self.ax_fid.set_ylim(0, 1.05)
        self.ax_fid.set_xlabel("Iteration")
        self.ax_fid.set_ylabel("Fidelity")
        self.ax_fid.grid(True, alpha=0.2)

        self.info_text = self.ax_fid.text(
            0.02,
            0.95,
            "",
            transform=self.ax_fid.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        return self.fig

    def init(self):
        artists = []
        if self.state_line is not None:
            artists.append(self.state_line)
        if self.state_im is not None:
            artists.append(self.state_im)
        if self.fid_line is not None:
            artists.append(self.fid_line)
        if self.info_text is not None:
            artists.append(self.info_text)
        return artists

    def update(self, snapshot: StepState):
        if snapshot.state is None:
            raise ValueError("step_stream must be called with return_state=True for visualization")

        prob = np.abs(snapshot.state) ** 2

        if self.dim == 1:
            x = np.arange(prob.size)
            self.state_line.set_data(x, prob.real)
            ymax = max(1e-9, float(prob.real.max()))
            self.ax_state.set_ylim(0.0, ymax * 1.05)
        else:
            self.state_im.set_data(prob.real)
            vmax = max(1e-9, float(prob.real.max()))
            self.state_im.set_clim(0.0, vmax)

        self.steps.append(snapshot.step)
        self.fidelities.append(snapshot.fidelity)
        self.fid_line.set_data(self.steps, self.fidelities)
        self.ax_fid.set_xlim(0, max(1, snapshot.step))
        ymax_fid = max(0.05, max(self.fidelities))
        self.ax_fid.set_ylim(0, min(1.05, ymax_fid * 1.05))

        info = f"step={snapshot.step}  fidelity={snapshot.fidelity:.6f}"
        if self.show_entropy and snapshot.spectral_entropy is not None:
            info += f"  entropy={snapshot.spectral_entropy:.4f}"
        if self.show_structure and snapshot.structure is not None:
            info += f"  slope={snapshot.structure.slope:.4f}"
        if snapshot.alpha is not None:
            info += f"  alpha_eff={snapshot.alpha:.6g}"
        self.info_text.set_text(info)

        artists = []
        if self.state_line is not None:
            artists.append(self.state_line)
        if self.state_im is not None:
            artists.append(self.state_im)
        artists.append(self.fid_line)
        artists.append(self.info_text)
        return artists


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Real-time animation of the U_alpha dynamics.",
    )
    parser.add_argument("--grid", default="256", help="grid, e.g. 256 or 128x128")
    parser.add_argument("--steps", type=int, default=256, help="number of iterations")
    parser.add_argument("--alpha", type=float, help="alpha in radians")
    parser.add_argument(
        "--alpha-ratio",
        type=str,
        help="fraction p/q ⇒ alpha = 2π * (p/q)",
    )
    parser.add_argument(
        "--initial",
        choices=["gaussian", "comb", "random"],
        default="gaussian",
        help="initial profile",
    )
    parser.add_argument("--width", type=float, default=0.08, help="gaussian width")
    parser.add_argument("--center", type=str, help="center (comma-separated)")
    parser.add_argument("--momentum", type=str, help="momentum tilt")
    parser.add_argument("--period", type=str, help="period for comb initial state")
    parser.add_argument("--random-seed", type=int, help="seed for random phases")
    parser.add_argument("--fps", type=float, default=30.0, help="frames per second (animation)")
    parser.add_argument("--structure-scales", type=str, help="scales for the structure function")
    parser.add_argument("--noise-alpha-std", type=float, default=0.0)
    parser.add_argument("--noise-phase-std", type=float, default=0.0)
    parser.add_argument(
        "--noise-phase-space",
        choices=["spatial", "fourier"],
        default="spatial",
    )
    parser.add_argument(
        "--compute-entropy",
        action="store_true",
        help="track spectral entropy over time",
    )
    parser.add_argument(
        "--compute-structure",
        action="store_true",
        help="track the structure function exponent (q = 2)",
    )
    parser.add_argument(
        "--dtype",
        choices=["complex64", "complex128"],
        default="complex128",
        help="numeric precision",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="headless mode (no window) — requires --save-frames",
    )
    parser.add_argument(
        "--save-frames",
        type=Path,
        help="output directory to write a PNG sequence (use with --no-show)",
    )
    args = parser.parse_args(argv)

    if args.no_show and args.save_frames is None:
        parser.error("--no-show requires --save-frames to preserve the simulation.")
    if not args.no_show and args.save_frames is not None:
        parser.error("--save-frames currently requires --no-show.")

    grid_shape = parse_grid(args.grid)
    if len(grid_shape) not in (1, 2):
        raise SystemExit("Only 1D or 2D grids are supported for the animation.")

    alpha = resolve_alpha(args.alpha, args.alpha_ratio)
    dtype = np.complex64 if args.dtype == "complex64" else np.complex128

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

    metrics = ["fidelity"]
    if args.compute_entropy:
        metrics.append("spectral_entropy")
    if args.compute_structure:
        metrics.append("structure_exponent")
    structure_scales = parse_scales(args.structure_scales)

    def stream():
        return simulator.step_stream(
            psi0=psi0,
            alpha=alpha,
            num_steps=args.steps,
            noise=noise,
            metrics=metrics,
            structure_scales=structure_scales,
            return_state=True,
        )

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        viz = Visualizer(
            grid_shape=grid_shape,
            show_entropy=args.compute_entropy,
            show_structure=args.compute_structure,
        )
        fig = viz.setup()
        save_dir = args.save_frames
        save_dir.mkdir(parents=True, exist_ok=True)
        for snapshot in stream():
            viz.update(snapshot)
            frame_path = save_dir / f"frame_{snapshot.step:04d}.png"
            fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        return

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    viz = Visualizer(
        grid_shape=grid_shape,
        show_entropy=args.compute_entropy,
        show_structure=args.compute_structure,
    )
    fig = viz.setup()

    frame_interval = max(10, int(1000.0 / max(args.fps, 1.0)))

    animation = FuncAnimation(
        fig,
        func=viz.update,
        frames=stream(),
        init_func=viz.init,
        interval=frame_interval,
        blit=False,
        repeat=False,
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
