import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def plot_font():
    for name in ("Times New Roman", "Arial", "DejaVu Serif"):
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=[name]),
                fallback_to_default=False,
            )
            return name
        except ValueError:
            continue
    return "DejaVu Serif"


def normalized_hundredths(counts):
    """Round row proportions to hundredths while preserving a total of 100.

    Allocate remaining hundredths to the largest exact remainders, breaking
    ties by column order. Empty rows remain undefined (NaN).
    Integer arithmetic avoids floating-point rounding errors in allocation.
    """
    rounded = np.full(counts.shape, np.nan, dtype=float)
    for i, row in enumerate(counts):
        total = sum(int(count) for count in row)
        if total == 0:
            continue
        quotients = [divmod(int(count) * 100, total) for count in row]
        hundredths = [quotient for quotient, _ in quotients]
        remaining = 100 - sum(hundredths)
        order = sorted(
            range(len(row)), key=lambda j: (-quotients[j][1], j)
        )
        for j in order[:remaining]:
            hundredths[j] += 1
        rounded[i] = np.asarray(hundredths, dtype=float) / 100
    return rounded


def evaluate(targets, probabilities, label_dim):
    labels = np.arange(label_dim)
    predictions = probabilities.argmax(1)

    counts = confusion_matrix(
        targets, predictions, labels=labels
    )
    support = counts.sum(axis=1)

    matrix = confusion_matrix(
        targets, predictions, labels=labels, normalize="true"
    )
    # A class with no observations has no defined conditional proportions.
    matrix[support == 0] = np.nan

    complete = bool(np.all(support > 0))
    metrics = {
        "accuracy": float(np.trace(counts) / counts.sum()),
        "balanced_accuracy": (
            float(np.diag(matrix).mean()) if complete else None
        ),
        "macro_f1": (
            float(f1_score(
                targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )) if complete else None
        ),
    }

    return counts, matrix, metrics


def save_comparison(real, synthetic, classes, out, metadata):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    real_counts, real_matrix, real_metrics = real
    gen_counts, gen_matrix, gen_metrics = synthetic
    support = real_counts.sum(axis=1)
    # Keep full precision for metrics; use sum-preserving rounding in plots.
    real_display = normalized_hundredths(real_counts)
    gen_display = normalized_hundredths(gen_counts)

    if not np.array_equal(support, gen_counts.sum(axis=1)):
        raise ValueError("Real and synthetic class counts must match.")

    # Mean total variation distance between corresponding matrix rows.
    distance = (
        float(np.abs(real_matrix - gen_matrix).sum() / (2 * len(classes)))
        if np.all(support > 0) else None
    )

    font = plot_font()
    with (out / "metrics.json").open("w") as f:
        json.dump({
            "classes": classes,
            "samples_per_class": support.tolist(),
            "real": real_metrics,
            "synthetic": gen_metrics,
            "confusion_distance": distance,
            "metadata": {
                **metadata,
                "plot_font": font,
                "plot_rounding": "largest_remainder_hundredths",
            },
        }, f, indent=4, allow_nan=False)

    np.savez_compressed(
        out / "confusion_matrices.npz",
        classes=np.asarray(classes),
        real_counts=real_counts,
        synthetic_counts=gen_counts,
        real_normalized=real_matrix,
        synthetic_normalized=gen_matrix,
        real_display_normalized=real_display,
        synthetic_display_normalized=gen_display,
    )

    style = {
        "font.family": font,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with plt.rc_context(style):
        cmap = plt.get_cmap("Blues").with_extremes(bad="#EEEEEE")
        fig, axes = plt.subplots(
            1, 2, figsize=(8.0, 4.3), constrained_layout=True
        )

        for ax, matrix, title, ylabel in zip(
            axes,
            (real_display, gen_display),
            ("(a) Real", "(b) Generated"),
            ("True class", "Requested class"),
        ):
            image = ax.imshow(
                np.ma.masked_invalid(matrix),
                cmap=cmap, vmin=0, vmax=1,
            )
            ax.set(
                title=f"{title} (N={int(support.sum())})",
                xlabel="Predicted class",
                ylabel=ylabel,
                xticks=np.arange(len(classes)),
                yticks=np.arange(len(classes)),
            )
            ax.set_xticklabels(
                classes, rotation=45, ha="right", rotation_mode="anchor"
            )
            ax.set_yticklabels(classes)
            ax.tick_params(which="both", length=0)
            ax.set_xticks(np.arange(len(classes) + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(classes) + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linewidth=0.6)

            for spine in ax.spines.values():
                spine.set_visible(False)

            for i in range(len(classes)):
                for j in range(len(classes)):
                    value = matrix[i, j]
                    ax.text(
                        j, i,
                        f"{value:.2f}" if np.isfinite(value) else "\u2014",
                        ha="center", va="center", fontsize=10,
                        color="white" if value > 0.6 else "black",
                    )

        colorbar_ax = axes[1].inset_axes([1.06, 0, 0.045, 1])
        fig.colorbar(
            image, cax=colorbar_ax,
            ticks=np.linspace(0, 1, 6),
            format=FormatStrFormatter("%.2f"),
            label="Row-normalised proportion",
        )
        fig.savefig(
            out / "confusion_matrices.png",
            dpi=600, bbox_inches="tight", pad_inches=0.06,
        )
        fig.savefig(
            out / "confusion_matrices.pdf",
            bbox_inches="tight", pad_inches=0.06,
        )
        plt.close(fig)

    return real_metrics, gen_metrics
