"""Render pilot figures to base64 PNGs + a JSON bundle for the HTML report."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flare_forecaster.contracts import PHYSICS_CHANNELS  # noqa: E402

PLOT_STYLE = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "text.color": "#8b93a7",
    "axes.labelcolor": "#8b93a7",
    "xtick.color": "#8b93a7",
    "ytick.color": "#8b93a7",
    "axes.edgecolor": "#8b93a7",
    "font.size": 9,
}
SERIES = ["#4c8dff", "#ff8f4c", "#3fc98a", "#c96fe0", "#e0575b"]


def encode(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts_pilot")
    parser.add_argument("--cubes", default="flare_forecaster/cache/pilot")
    parser.add_argument("--out", default="artifacts_pilot/report_data.json")
    args = parser.parse_args()

    plt.rcParams.update(PLOT_STYLE)
    root = Path(args.artifacts)
    results = torch.load(root / "caches" / "pilot_results.pt", weights_only=False)
    noaa_list = results["noaa"]
    figures: dict[str, str] = {}

    # --- A. inputs -------------------------------------------------------
    for index, noaa in enumerate(noaa_list):
        cube = np.load(Path(args.cubes) / f"noaa{noaa}_cube.npz")["cube"]
        bz = np.nan_to_num(cube[len(cube) // 2, 0])
        readout = results["readout"][index].numpy()

        fig, axes = plt.subplots(3, 3, figsize=(9, 7.5))
        limit = np.percentile(np.abs(bz), 99) or 1.0
        axes[0, 0].imshow(bz, cmap="gray", vmin=-limit, vmax=limit, origin="lower")
        axes[0, 0].set_title("SHARP $B_r$ (context)", fontsize=9)
        axes[0, 0].axis("off")

        for j, name in enumerate(PHYSICS_CHANNELS):
            ax = axes[(j + 1) // 3, (j + 1) % 3]
            channel = readout[j]
            if name in ("trace_valid", "strong_field_mask"):
                ax.imshow(channel, cmap="magma", origin="lower", vmin=0, vmax=1)
            else:
                scale = np.percentile(np.abs(channel), 98) or 1.0
                ax.imshow(channel, cmap="RdBu_r", origin="lower", vmin=-scale, vmax=scale)
            ax.set_title(name, fontsize=8)
            ax.axis("off")
        fig.suptitle(
            f"NOAA {noaa} — label {int(results['labels'][noaa])}", fontsize=10, y=0.98
        )
        figures[f"inputs_{noaa}"] = encode(fig)

    # --- B. training curves ----------------------------------------------
    curves = results["curves"]
    stages = ["fusion", "deterministic", "flow", "flare_head"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.8))
    for ax, name, color in zip(axes, stages, SERIES):
        values = curves[name]
        ax.plot(values, color=color, linewidth=1.6)
        ax.set_yscale("log" if min(values) > 0 else "linear")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.15, linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("training loss")
    figures["curves"] = encode(fig)

    # --- C. PCA of fused latents -----------------------------------------
    fused = results["fused"]
    history = results["history"]
    ensemble = results["ensemble"]
    det_future = results["det_future"]
    groups = results["groups"]

    flat = fused.reshape(-1, fused.shape[-1])
    mean = flat.mean(0, keepdim=True)
    _, _, basis = torch.pca_lowrank(flat - mean, q=2)

    def project(x):
        return ((x.reshape(-1, x.shape[-1]) - mean) @ basis).reshape(*x.shape[:-1], 2)

    fig, axes = plt.subplots(1, len(noaa_list), figsize=(11, 4.6))
    for ax, noaa in zip(np.atleast_1d(axes), noaa_list):
        mask = torch.tensor([g == noaa for g in groups])
        idx = int(mask.nonzero()[0])
        obs = project(history[idx])
        ens = project(ensemble[:, idx])
        det = project(det_future[idx])

        for k in range(ens.shape[0]):
            ax.plot(ens[k, :, 0], ens[k, :, 1], color=SERIES[0], alpha=0.18, linewidth=0.8)
        ax.plot(ens.mean(0)[:, 0], ens.mean(0)[:, 1], color=SERIES[0],
                linewidth=2.2, label="flow ensemble mean")
        ax.plot(det[:, 0], det[:, 1], color=SERIES[1], linewidth=2.0,
                linestyle="--", label="deterministic")
        persistence = obs[-1:].repeat(det.shape[0], 1)
        ax.plot(persistence[:, 0], persistence[:, 1], color=SERIES[3],
                linewidth=1.6, linestyle=":", label="persistence")
        ax.plot(obs[:, 0], obs[:, 1], color=SERIES[2], linewidth=2.4, label="observed history")
        ax.scatter(obs[-1, 0], obs[-1, 1], color=SERIES[2], s=40, zorder=5)
        ax.set_title(f"NOAA {noaa} (label {int(results['labels'][noaa])})", fontsize=10)
        ax.set_xlabel("PC1")
        ax.grid(alpha=0.15, linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("PC2")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False)
    figures["pca"] = encode(fig)

    bundle = {
        "figures": figures,
        "probabilities": results["probabilities"],
        "diagnostics": results["diagnostics"],
        "timings": results["timings"],
        "labels": results["labels"],
        "noaa": noaa_list,
        "physics_diagnostics": results["physics_diagnostics"],
        "history_steps": results["history_steps"],
        "n_windows": len(groups),
    }
    Path(args.out).write_text(json.dumps(bundle))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")
    print("figures:", list(figures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
