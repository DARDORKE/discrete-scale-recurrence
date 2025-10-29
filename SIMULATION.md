# Simulation de la récurrence d’échelle discrète

Ce dépôt contient maintenant un simulateur numérique générique pour explorer l’itération de l’opérateur
$$U\_\\alpha = e^{i\\alpha\\Delta}$$ sur un tore discret. Il fournit :

- un **module Python** (`scale_recurrence`) avec une classe de simulation, des métriques (fidélité, entropie spectrale, exposant de structure) et plusieurs états initiaux prédéfinis ;
- un **script CLI** (`simulate.py`) pour lancer des balayages rapides, détecter automatiquement les revivals et sauvegarder les séries temporelles.

Les dépendances se limitent à `numpy` (`pip install numpy`). Pour afficher/sauvegarder
des figures, installez aussi `matplotlib`.

---

## Lancer une expérience rapide

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

- `--alpha-ratio p/q` fixe $\\alpha = 2\\pi p/q$ (donc période $\\approx q$ attendue).
- `--metrics` permet de sélectionner les diagnostics à calculer (`fidelity`, `spectral_entropy`, `structure_exponent`).
- `--save results.npz` exporte les séries (`fidelities`, `spectral_entropy`, etc.) pour post-traitement.
- `--plot` ouvre les graphes interactifs; `--plot-file out.png` les enregistre en PNG sans display (idéal sur serveur).

### Animation temps réel

```bash
python3 realtime_simulation.py \
  --grid 128 \
  --steps 128 \
  --alpha-ratio 1/4 \
  --fps 30 \
  --compute-entropy
```

- Affiche en direct $|\\psi\_n|^2$ (1D ou 2D) et la fidélité cumulée.
- Mode headless : `--no-show --save-frames frames/` génère une séquence PNG pour montage vidéo.
- Options disponibles : états initiaux (`--initial`), bruit (`--noise-alpha-std`, `--noise-phase-std`), structure function (`--compute-structure`).

### Visualisation web (navigateur)

```bash
python3 web_visualization.py \
  --grid 128 \
  --steps 256 \
  --alpha-ratio 1/4 \
  --compute-entropy \
  --open-browser
```

- Le serveur (par défaut `http://127.0.0.1:8000/`) diffuse l’évolution complète : $|\\psi\_n|^2$, fidélité, entropie/structure optionnelles.
- Contrôles intégrés : lecture/pause, curseur d’itération, réglage de vitesse.
- Mode headless possible (`--open-browser` omis) avec journal console; les données sont servies via `/data` (JSON) pour une exploitation externe.

### Analyse automatique des résultats

```bash
python3 analyze_results.py outputs/*.npz \
  --csv-dir diagnostics/csv \
  --plot-dir diagnostics/figures
```

- Affiche un résumé (revivals détectés, statistiques de fidélité/entropie).
- Exporte un CSV `step,fidelity,…` par fichier dans `diagnostics/csv/`.
- Produit des figures PNG cumulant courbes et densité finale (1D/2D).

### Exemple irrationnel (textures fractales)

```bash
python3 simulate.py \
  --grid 256 \
  --steps 512 \
  --alpha 1.0 \
  --initial gaussian \
  --noise-phase-std 0.01 \
  --metrics fidelity,spectral_entropy,structure_exponent
```

Les revivals devraient disparaître tandis que l’entropie spectrale augmente et que l’exposant de structure dérive.

### Option bruit et robustesse

- `--noise-alpha-std` : jitter multiplicatif sur $\\alpha$ par pas (`alpha_n = alpha * (1 + \\mathcal{N}(0, \\sigma))`).
- `--noise-phase-std` : bruit gaussien sur la phase (dans l’espace choisi par `--noise-phase-space`).

---

## Utilisation programmatique

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

- Les états stockés (`result.stored_states`) suivent les indices `result.stored_steps`.
- `result.structure_details` contient, pour chaque pas, les moments du *structure function* et la pente ajustée.

---

## Points clés pour l’analyse

- **Détection de revivals** : `metrics.detect_revivals` scanne la série de fidélité et renvoie `(pas, fidélité)` pour tous les pics > seuil. Cela facilite l’estimation de la période minimale et l’étude de la robustesse sous bruit.
- **Approximation rationnelle** : `metrics.rational_approximation(alpha / 2π)` fournit la meilleure fraction $p/q$ (dénominateur limité) pour relier numériquement un quasi-revival observé à une valeur rationnelle voisine.
- **Texture/fractalisation** : l’exposant de structure (pente dans l’espace `log`‑`log`) est suivi à partir de $|\\psi\_n|^2$. Les dérives à la hausse signalent une rugosité croissante, typique des irrationnels.