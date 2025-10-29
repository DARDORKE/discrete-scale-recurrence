#!/usr/bin/env python3
"""
Minimal web server to visualize the discrete scale recurrence.

The app precomputes the simulation, serves JSON data and an HTML/JS page
to navigate through the evolution.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scale_recurrence import (
    DiscreteScaleRecurrenceSimulator,
    NoiseModel,
    initial_states,
)

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Discrete Scale Recurrence — Viewer</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }
    header { padding: 1rem; background: #222; }
    main { display: flex; flex-direction: column; gap: 1rem; padding: 1rem; }
    canvas { border: 1px solid #444; background: #000; }
    #stateCanvas { width: 100%; max-width: 960px; height: 320px; }
    #stateCanvas[data-dim="2"] { height: auto; }
    #fidelityCanvas { width: 100%; max-width: 960px; height: 140px; }
    #info { font-size: 0.95rem; line-height: 1.5; background: #1d1d1d; padding: 0.75rem; border-radius: 6px; max-width: 960px; }
    #controls { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
    button { background: #2a6bff; border: none; color: #fff; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: default; }
    input[type="range"] { width: 200px; }
  </style>
</head>
<body>
  <header>
    <h1>Discrete Scale Recurrence — Visualization</h1>
    <div id="meta"></div>
  </header>
  <main>
    <section id="controls">
      <button id="playBtn">Pause</button>
      <label>Speed: <input id="speedRange" type="range" min="5" max="120" value="30" /></label>
      <label>Iteration: <input id="stepRange" type="range" min="0" max="0" value="0" /></label>
      <span id="stepLabel">step 0</span>
    </section>
    <canvas id="stateCanvas"></canvas>
    <canvas id="fidelityCanvas"></canvas>
    <div id="info"></div>
  </main>
  <script src="/app.js" defer></script>
</body>
</html>
"""

APP_JS = """'use strict';

function createLineDrawer(canvas) {
  const ctx = canvas.getContext('2d');
  return (values) => {
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    if (!values || values.length === 0) return;
    const maxVal = Math.max(...values);
    const minVal = Math.min(...values);
    const range = Math.max(1e-12, maxVal - minVal);
    ctx.beginPath();
    values.forEach((val, idx) => {
      const x = (idx / (values.length - 1)) * width;
      const y = height - ((val - minVal) / range) * height;
      if (idx === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.strokeStyle = '#8be9fd';
    ctx.lineWidth = 2;
    ctx.stroke();
  };
}

function createHeatmapDrawer(canvas, width, height) {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(width, height);
  return (matrix) => {
    if (!matrix || matrix.length === 0) return;
    let maxVal = Number.NEGATIVE_INFINITY;
    let minVal = Number.POSITIVE_INFINITY;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const v = matrix[y][x];
        if (v > maxVal) maxVal = v;
        if (v < minVal) minVal = v;
      }
    }
    const range = Math.max(1e-12, maxVal - minVal);
    const data = imageData.data;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const value = (matrix[y][x] - minVal) / range;
        const r = Math.floor(255 * value);
        const g = Math.floor(255 * Math.pow(value, 0.5));
        const b = Math.floor(255 * (1 - value));
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  };
}

function drawFidelity(canvas, fidelities, currentStep) {
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  if (!fidelities || fidelities.length === 0) return;
  const visible = fidelities.slice(0, currentStep + 1);
  const maxVal = Math.max(...visible, 1e-12);
  ctx.beginPath();
  visible.forEach((val, idx) => {
    const x = (idx / Math.max(1, fidelities.length - 1)) * width;
    const y = height - (val / maxVal) * height;
    if (idx === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.strokeStyle = '#50fa7b';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.strokeStyle = '#444';
  ctx.beginPath();
  ctx.moveTo(0, height - 0.95 * height);
  ctx.lineTo(width, height - 0.95 * height);
  ctx.stroke();
}

async function main() {
  const response = await fetch('/data', { cache: 'no-store' });
  if (!response.ok) {
    document.body.innerHTML = '<p>Failed to load data.</p>';
    return;
  }
  const data = await response.json();

  const stateCanvas = document.getElementById('stateCanvas');
  const fidelityCanvas = document.getElementById('fidelityCanvas');
  fidelityCanvas.width = 960;
  fidelityCanvas.height = 140;

  const playBtn = document.getElementById('playBtn');
  const speedRange = document.getElementById('speedRange');
  const stepRange = document.getElementById('stepRange');
  const stepLabel = document.getElementById('stepLabel');
  const infoDiv = document.getElementById('info');
  const metaDiv = document.getElementById('meta');

  const dim = data.grid_shape.length;
  const stepsCount = data.states.length;
  stepRange.max = stepsCount - 1;

  let drawState;
  if (dim === 1) {
    stateCanvas.width = 960;
    stateCanvas.height = 320;
    stateCanvas.dataset.dim = '1';
    drawState = createLineDrawer(stateCanvas);
  } else if (dim === 2) {
    const [h, w] = data.grid_shape;
    stateCanvas.dataset.dim = '2';
    drawState = createHeatmapDrawer(stateCanvas, w, h);
  } else {
    throw new Error('Unsupported dimensionality');
  }

  metaDiv.textContent = `Grid: ${data.grid_shape.join('×')} | α = ${data.alpha.toFixed(6)} | steps: ${stepsCount - 1}`;

  let playing = true;
  let currentStep = 0;
  let timerId = null;

  function updateFrame(step) {
    const state = data.states[step];
    drawState(state);
    drawFidelity(fidelityCanvas, data.fidelities, step);
    stepLabel.textContent = `step ${data.steps[step]} / ${stepsCount - 1}`;
    stepRange.value = step;
    const entropy = data.entropies ? data.entropies[step] : null;
    const slope = data.structure_slopes ? data.structure_slopes[step] : null;
    const alphaEff = data.alphas ? data.alphas[Math.max(0, step - 1)] : null;
    const lines = [
      `Fidelity: ${data.fidelities[step].toFixed(6)}`,
    ];
    if (entropy !== null && !Number.isNaN(entropy)) {
      lines.push(`Spectral entropy: ${entropy.toFixed(4)}`);
    }
    if (slope !== null && !Number.isNaN(slope)) {
      lines.push(`Structure slope: ${slope.toFixed(4)}`);
    }
    if (alphaEff !== null) {
      lines.push(`α_eff (n): ${alphaEff.toFixed(6)}`);
    }
    infoDiv.textContent = lines.join(' | ');
  }

  function scheduleNextFrame() {
    const interval = 1000 / speedRange.value;
    timerId = setTimeout(() => {
      currentStep = (currentStep + 1) % stepsCount;
      updateFrame(currentStep);
      if (playing) {
        scheduleNextFrame();
      }
    }, interval);
  }

  playBtn.addEventListener('click', () => {
    playing = !playing;
    playBtn.textContent = playing ? 'Pause' : 'Play';
    if (playing) {
      scheduleNextFrame();
    } else if (timerId !== null) {
      clearTimeout(timerId);
    }
  });

  stepRange.addEventListener('input', (event) => {
    const value = Number(event.target.value);
    currentStep = value;
    updateFrame(currentStep);
  });

  speedRange.addEventListener('input', () => {
    if (timerId !== null) {
      clearTimeout(timerId);
      if (playing) {
        scheduleNextFrame();
      }
    }
  });

  updateFrame(0);
  scheduleNextFrame();
}

window.addEventListener('DOMContentLoaded', () => {
  main().catch((err) => {
    console.error(err);
    document.body.innerHTML = '<p>Error while fetching data.</p>';
  });
});
"""


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


def run_simulation(args: argparse.Namespace, grid_shape: Tuple[int, ...], alpha: float) -> Dict[str, object]:
    dtype = np.complex64 if args.dtype == "complex64" else np.complex128
    simulator = DiscreteScaleRecurrenceSimulator(
        grid_shape=grid_shape,
        normalize_each_step=True,
        dtype=dtype,
    )

    psi0 = build_initial_state(args, grid_shape)

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

    steps: List[int] = []
    states: List[object] = []
    fidelities: List[float] = []
    entropies: List[Optional[float]] = []
    structure_slopes: List[Optional[float]] = []
    alphas: List[float] = []

    stream = simulator.step_stream(
        psi0=psi0,
        alpha=alpha,
        num_steps=args.steps,
        noise=noise,
        metrics=metrics,
        structure_scales=structure_scales,
        return_state=True,
    )

    for snapshot in stream:
        prob = np.abs(snapshot.state) ** 2 if snapshot.state is not None else None
        if prob is None:
            raise RuntimeError("step_stream returned None state while return_state=True")
        if prob.ndim == 1:
            states.append(prob.real.tolist())
        else:
            states.append(prob.real.tolist())

        steps.append(snapshot.step)
        fidelities.append(float(snapshot.fidelity))
        entropies.append(float(snapshot.spectral_entropy) if snapshot.spectral_entropy is not None else None)
        structure_slopes.append(float(snapshot.structure.slope) if snapshot.structure is not None else None)
        if snapshot.alpha is not None:
            alphas.append(float(snapshot.alpha))
        else:
            alphas.append(float(alpha))

    dataset = {
        "grid_shape": list(grid_shape),
        "steps": steps,
        "states": states,
        "fidelities": fidelities,
        "entropies": entropies if args.compute_entropy else None,
        "structure_slopes": structure_slopes if args.compute_structure else None,
        "alphas": alphas if alphas else None,
        "alpha": alpha,
        "metadata": {
            "initial": args.initial,
            "width": args.width,
            "noise_alpha_std": args.noise_alpha_std,
            "noise_phase_std": args.noise_phase_std,
        },
    }
    return dataset


class VisualizationHandler(BaseHTTPRequestHandler):
    dataset_json: bytes = b""
    verbose: bool = False

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/app.js":
            body = APP_JS.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/data":
            body = self.dataset_json
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404, "Not Found")

    def log_message(self, format: str, *args):
        if self.verbose:
            super().log_message(format, *args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Web server to visualize the discrete scale recurrence.",
    )
    parser.add_argument("--grid", default="256", help="grid, e.g. 256 or 128x128")
    parser.add_argument("--steps", type=int, default=256, help="number of computed iterations")
    parser.add_argument("--alpha", type=float, help="alpha in radians")
    parser.add_argument("--alpha-ratio", type=str, help="fraction p/q ⇒ alpha = 2π * (p/q)")
    parser.add_argument("--initial", choices=["gaussian", "comb", "random"], default="gaussian")
    parser.add_argument("--width", type=float, default=0.08, help="gaussian width")
    parser.add_argument("--center", type=str, help="initial center")
    parser.add_argument("--momentum", type=str, help="initial momentum")
    parser.add_argument("--period", type=str, help="period for comb initial state")
    parser.add_argument("--random-seed", type=int, help="seed for random states/noise")
    parser.add_argument("--noise-alpha-std", type=float, default=0.0)
    parser.add_argument("--noise-phase-std", type=float, default=0.0)
    parser.add_argument("--noise-phase-space", choices=["spatial", "fourier"], default="spatial")
    parser.add_argument("--compute-entropy", action="store_true", help="include spectral entropy")
    parser.add_argument("--compute-structure", action="store_true", help="include structure-function slope")
    parser.add_argument("--structure-scales", type=str, help="scales for the structure function")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex128")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--host", default="127.0.0.1", help="listening host (default: 127.0.0.1)")
    parser.add_argument("--open-browser", action="store_true", help="automatically open the browser")
    parser.add_argument("--verbose", action="store_true", help="verbose HTTP logs")
    args = parser.parse_args(argv)

    grid_shape = parse_grid(args.grid)
    if len(grid_shape) not in (1, 2):
        raise SystemExit("This web visualization only supports 1D or 2D grids.")

    alpha = resolve_alpha(args.alpha, args.alpha_ratio)

    dataset = run_simulation(args, grid_shape, alpha)
    dataset_json = json.dumps(dataset).encode("utf-8")

    VisualizationHandler.dataset_json = dataset_json
    VisualizationHandler.verbose = args.verbose

    server = HTTPServer((args.host, args.port), VisualizationHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Simulation ready. Server available at {url}")

    if args.open_browser:
        try:
            import webbrowser

            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        except Exception as exc:  # noqa: BLE001
            print(f"Could not open the browser automatically: {exc}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
