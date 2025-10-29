# Discrete Scale Recurrence — simulator and tools

This repository provides a compact simulator to explore repeated applications of
U_α = exp(i α Δ) on a periodic grid. It includes:

- a Python library (`scale_recurrence`) with the simulator, metrics (fidelity, spectral entropy, structure‑function slope), and common initial states;
- a CLI (`simulate.py`) to run experiments, detect revivals, and save time series.

Dependencies: `numpy` (required). Install `matplotlib` to display or save figures.

---

## Quick start

```bash
python3 simulate.py \
  --grid 256 \
  --steps 256 \
  --alpha-ratio 1/4 \
  --initial gaussian \
  --width 0.06 \
  --revival-threshold 0.99 \
  --plot-file talbot.png
```

- `--alpha-ratio p/q` sets α = 2π p/q (expected period ≈ q).
- `--metrics` chooses diagnostics (`fidelity`, `spectral_entropy`, `structure_exponent`).
- `--save results.npz` exports arrays (fidelities, spectral entropy, etc.).
- `--plot` opens interactive plots; `--plot-file out.png` saves a PNG (good for servers).

### Real‑time animation

```bash
python3 realtime_simulation.py \
  --grid 128 \
  --steps 128 \
  --alpha-ratio 1/4 \
  --fps 30 \
  --compute-entropy
```

- Live display of |ψ_n|² (1D/2D) and cumulative fidelity.
- Headless mode: `--no-show --save-frames frames/` writes a PNG sequence for a video.
- Extra options: initial state (`--initial`), noise (`--noise-alpha-std`, `--noise-phase-std`), structure function (`--compute-structure`).

### Web visualization

```bash
python3 web_visualization.py \
  --grid 128 \
  --steps 256 \
  --alpha-ratio 1/4 \
  --compute-entropy \
  --open-browser
```

- Serves the full evolution at `http://127.0.0.1:8000/`: |ψ_n|², fidelity, optional entropy/structure.
- Built‑in controls: play/pause, time slider, speed.
- Headless server also works (omit `--open-browser`). Data is available as JSON at `/data`.

### Post‑analysis

```bash
python3 analyze_results.py outputs/*.npz \
  --csv-dir diagnostics/csv \
  --plot-dir diagnostics/figures
```

- Prints a summary (revivals, fidelity/entropy stats).
- Exports a CSV `step,fidelity,…` per file.
- Produces combined figures (curves + final density).

### Irrational example (textures)

```bash
python3 simulate.py \
  --grid 256 \
  --steps 512 \
  --alpha 1.0 \
  --initial gaussian \
  --noise-phase-std 0.01 \
  --metrics fidelity,spectral_entropy,structure_exponent
```

Expect no strict period while spectral entropy and the structure slope drift.

### Noise and robustness

- `--noise-alpha-std`: multiplicative jitter on α per step (`alpha_n = alpha * (1 + N(0,σ))`).
- `--noise-phase-std`: Gaussian phase noise (space chosen by `--noise-phase-space`).

---

## Programmatic usage

```python
import numpy as np
from scale_recurrence import (
    DiscreteScaleRecurrenceSimulator,
    NoiseModel,
    initial_states,
)

grid = (256,)
psi0 = initial_states.gaussian_wavepacket(grid, width=0.08)
sim = DiscreteScaleRecurrenceSimulator(grid)
result = sim.run(
    psi0=psi0,
    alpha=2*np.pi*(1/4),
    num_steps=256,
    noise=NoiseModel(alpha_std=1e-3),
)
print(result.revivals[:5])
```

- Stored states (`result.stored_states`) follow the indices in `result.stored_steps`.
- `result.structure_details` stores the structure‑function moments and fitted slope per step.

---

## Analysis tips

- Revival detection: `metrics.detect_revivals` scans the fidelity series and returns `(step, fidelity)` peaks above a threshold — useful to estimate minimal period and robustness under noise.
- Rational approximation: `metrics.rational_approximation(alpha / 2π)` provides the best fraction p/q (bounded denominator) to label quasi‑revivals.
- Texture/fractalization: the structure‑function slope (log‑log fit) computed from |ψ_n|² rises with roughness in irrational/noisy regimes.
