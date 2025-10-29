#!/usr/bin/env python3
"""
Analyse des fichiers .npz produits par simulate.py.

Pour chaque archive, le script charge les séries temporelles, affiche un résumé,
détecte les revivals et, optionnellement, exporte un CSV et/ou une figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from scale_recurrence.metrics import detect_revivals


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load an .npz archive with pickle support for metadata."""
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def decode_metadata(meta_array: np.ndarray) -> Dict[str, object]:
    """Decode the JSON metadata stored in the npz archive."""
    if meta_array.size == 0:
        return {}
    raw = meta_array.flat[0]
    if isinstance(raw, bytes):
        raw_str = raw.decode("utf-8")
    else:
        raw_str = str(raw)
    try:
        return json.loads(raw_str)
    except json.JSONDecodeError:
        return {"raw": raw_str}


def summarize_dataset(
    path: Path,
    arrays: Dict[str, np.ndarray],
    revival_threshold: float,
    revival_gap: int,
) -> Dict[str, object]:
    fidelities = arrays.get("fidelities")
    if fidelities is None:
        raise ValueError(f"{path}: archive does not contain 'fidelities'")
    alphas = arrays.get("alphas")
    spectral_entropy = arrays.get("spectral_entropy")
    structure_slopes = arrays.get("structure_slopes")
    metadata = decode_metadata(arrays.get("metadata", np.array([])))

    steps = np.arange(fidelities.size)
    revivals = detect_revivals(
        fidelities,
        threshold=revival_threshold,
        min_separation=revival_gap,
    )

    summary = {
        "path": str(path),
        "steps": int(fidelities.size - 1),
        "max_fidelity": float(np.max(fidelities)),
        "revivals": revivals,
        "metadata": metadata,
        "alphas_mean": float(np.mean(alphas)) if alphas is not None else None,
        "alphas_std": float(np.std(alphas)) if alphas is not None else None,
    }
    if spectral_entropy is not None:
        summary["entropy_start"] = float(spectral_entropy[0])
        summary["entropy_end"] = float(spectral_entropy[-1])
    if structure_slopes is not None:
        summary["structure_start"] = float(structure_slopes[0])
        summary["structure_end"] = float(structure_slopes[-1])
    summary["has_states"] = "stored_states" in arrays
    summary["stored_steps"] = (
        arrays["stored_steps"].astype(int).tolist() if "stored_steps" in arrays else []
    )
    return summary


def print_summary(summary: Dict[str, object]) -> None:
    meta = summary["metadata"]
    grid = meta.get("grid_shape") if isinstance(meta, dict) else None
    print(f"\n=== {summary['path']} ===")
    if grid is not None:
        print(f"Grille: {tuple(grid)} | itérations: {summary['steps']}")
    else:
        print(f"Itérations: {summary['steps']}")
    print(f"Fidélité max: {summary['max_fidelity']:.6f}")
    if summary.get("alphas_mean") is not None:
        mean = summary["alphas_mean"]
        std = summary["alphas_std"]
        print(f"alpha moyen: {mean:.6g} ± {std:.3g}")
    if summary.get("entropy_start") is not None:
        print(
            "Entropie spectrale: "
            f"{summary['entropy_start']:.4f} → {summary['entropy_end']:.4f}"
        )
    if summary.get("structure_start") is not None:
        print(
            "Structure (pente q=2): "
            f"{summary['structure_start']:.4f} → {summary['structure_end']:.4f}"
        )
    revivals = summary["revivals"]
    if revivals:
        print("Revivals détectés (pas : fidélité) :")
        for step, fid in revivals[:10]:
            print(f"  {step:5d} : {fid:.6f}")
        if len(revivals) > 10:
            print(f"  ... ({len(revivals) - 10} supplémentaires)")
    else:
        print("Aucun revival au-dessus du seuil n'a été détecté.")
    if summary["has_states"]:
        stored_count = len(summary["stored_steps"])
        print(f"États stockés disponibles ({stored_count} instantanés).")


def export_csv(
    path: Path,
    arrays: Dict[str, np.ndarray],
    target_dir: Path,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / f"{path.stem}.csv"
    fidelities = arrays["fidelities"]
    spectral_entropy = arrays.get("spectral_entropy")
    structure_slopes = arrays.get("structure_slopes")

    headers = ["step", "fidelity"]
    if spectral_entropy is not None:
        headers.append("spectral_entropy")
    if structure_slopes is not None:
        headers.append("structure_slope")

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for idx, fidelity in enumerate(fidelities):
            row = [idx, float(fidelity)]
            if spectral_entropy is not None:
                row.append(float(spectral_entropy[idx]))
            if structure_slopes is not None:
                row.append(float(structure_slopes[idx]))
            writer.writerow(row)
    return csv_path


def project_final_state(state: np.ndarray) -> Tuple[np.ndarray, str]:
    prob = np.abs(state) ** 2
    if prob.ndim == 1:
        return prob, "|psi|^2 (final)"
    if prob.ndim == 2:
        return prob, "|psi|^2 (final)"
    collapsed = prob.copy()
    while collapsed.ndim > 2:
        collapsed = collapsed.sum(axis=-1)
    return collapsed, "|psi|^2 (sum over trailing axes)"


def plot_dataset(
    path: Path,
    arrays: Dict[str, np.ndarray],
    summary: Dict[str, object],
    show: bool,
    plot_dir: Optional[Path],
) -> Optional[Path]:
    if not show and plot_dir is None:
        return None
    try:
        if plot_dir is not None:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib est requis pour générer des graphiques. "
            "Installez-le via `python3 -m pip install matplotlib`."
        ) from exc

    fidelities = arrays["fidelities"]
    entropy = arrays.get("spectral_entropy")
    slopes = arrays.get("structure_slopes")
    stored = arrays.get("stored_states")

    sections = ["fidelity"]
    if entropy is not None:
        sections.append("entropy")
    if slopes is not None:
        sections.append("structure")
    if stored is not None:
        sections.append("state")

    steps = np.arange(fidelities.size)
    fig, axes = plt.subplots(len(sections), 1, figsize=(8, 2.6 * len(sections)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax_idx = 0
    ax = axes[ax_idx]
    ax.plot(steps, fidelities, color="tab:blue", label="Fidélité")
    for step, _ in summary["revivals"]:
        ax.axvline(step, color="tab:orange", linestyle="--", alpha=0.3)
    ax.set_ylabel("Fidélité")
    ax.set_title(path.name)
    ax.grid(True, alpha=0.2)
    ax.legend()

    if entropy is not None:
        ax_idx += 1
        ax = axes[ax_idx]
        ax.plot(steps, entropy, color="tab:green")
        ax.set_ylabel("Entropie")
        ax.grid(True, alpha=0.2)

    if slopes is not None:
        ax_idx += 1
        ax = axes[ax_idx]
        ax.plot(steps, slopes, color="tab:red")
        ax.set_ylabel("Pente")
        ax.grid(True, alpha=0.2)

    if stored is not None:
        ax_idx += 1
        ax = axes[ax_idx]
        final_state = stored[-1]
        data, label = project_final_state(final_state)
        if data.ndim == 1:
            ax.plot(np.arange(data.size), data, color="tab:purple")
        else:
            im = ax.imshow(data, origin="lower", cmap="viridis")
            fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_ylabel(label)
        ax.grid(False)

    axes[-1].set_xlabel("Itération")
    fig.tight_layout()

    output_path = None
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        output_path = plot_dir / f"{path.stem}_analysis.png"
        fig.savefig(output_path, dpi=160)
        print(f"Graphique sauvegardé dans {output_path}")
        plt.close(fig)
    if show:
        plt.show()
    return output_path


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyse des résultats .npz générés par simulate.py.",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="archives .npz à analyser")
    parser.add_argument(
        "--revival-threshold",
        type=float,
        default=0.95,
        help="seuil de fidélité pour la détection des revivals (défaut 0.95)",
    )
    parser.add_argument(
        "--revival-gap",
        type=int,
        default=1,
        help="séparation minimale entre deux revivals (défaut 1)",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        help="répertoire de sortie pour exporter un CSV par fichier (facultatif)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="afficher les graphes à l'écran (Matplotlib requis)",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="répertoire de sortie pour sauvegarder les graphes (PNG) sans affichage",
    )
    args = parser.parse_args(argv)

    for path in args.paths:
        if not path.exists():
            print(f"Fichier introuvable : {path}")
            continue
        arrays = load_npz(path)
        summary = summarize_dataset(
            path,
            arrays,
            revival_threshold=args.revival_threshold,
            revival_gap=max(1, args.revival_gap),
        )
        print_summary(summary)
        if args.csv_dir is not None:
            csv_path = export_csv(path, arrays, args.csv_dir)
            print(f"CSV exporté : {csv_path}")
        if args.plot or args.plot_dir is not None:
            plot_dataset(
                path,
                arrays,
                summary,
                show=args.plot,
                plot_dir=args.plot_dir,
            )


if __name__ == "__main__":
    main()
