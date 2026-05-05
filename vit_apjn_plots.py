from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


def _alpha_colors(alphas, cmap_name="viridis"):
    alphas = np.asarray(alphas, dtype=float)
    cmap = plt.get_cmap(cmap_name)
    if alphas.size == 1:
        return {float(alphas[0]): cmap(0.6)}
    return {
        float(alpha): cmap(idx / max(1, alphas.size - 1))
        for idx, alpha in enumerate(alphas)
    }


def prettify_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.22)
    ax.grid(False, which="minor")


def prettify_log_axis(ax, axis="y"):
    locator_major = LogLocator(base=10.0, numticks=10)
    locator_minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    formatter = LogFormatterMathtext(base=10.0)
    if axis == "y":
        ax.yaxis.set_major_locator(locator_major)
        ax.yaxis.set_minor_locator(locator_minor)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_major_locator(locator_major)
        ax.xaxis.set_minor_locator(locator_minor)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(NullFormatter())


def plot_equangular_apjn_comparison(comparison_bundle, *, figsize=(14, 5)):
    inverse = comparison_bundle["inverse"]
    forward = comparison_bundle["forward"]
    alphas = np.asarray(sorted(float(alpha) for alpha in inverse["derf"].keys()), dtype=float)
    colors = _alpha_colors(alphas)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].plot(inverse["layers"], inverse["preln"]["theory"], color="black", lw=2.0)
    axes[0].scatter(inverse["layers"], inverse["preln"]["measured"], color="black", s=24, zorder=3)
    for alpha in alphas:
        axes[0].plot(inverse["layers"], inverse["derf"][float(alpha)]["theory"], color=colors[float(alpha)], lw=1.8)
        axes[0].scatter(
            inverse["layers"],
            inverse["derf"][float(alpha)]["measured"],
            color=colors[float(alpha)],
            edgecolors="black",
            linewidths=0.3,
            s=28,
            zorder=3,
        )
    axes[0].set_title(r"Backward APJN $\mathcal{J}^{\,B,b}$")
    axes[0].set_xlabel(r"$b$")
    axes[0].set_ylabel(r"$\mathcal{J}^{\,B,b}$")
    axes[0].set_yscale("log")
    prettify_log_axis(axes[0], "y")
    prettify_axes(axes[0])

    axes[1].plot(forward["layers"], forward["preln"]["theory"], color="black", lw=2.0)
    axes[1].scatter(forward["layers"], forward["preln"]["measured"], color="black", s=24, zorder=3)
    for alpha in alphas:
        axes[1].plot(forward["layers"], forward["derf"][float(alpha)]["theory"], color=colors[float(alpha)], lw=1.8)
        axes[1].scatter(
            forward["layers"],
            forward["derf"][float(alpha)]["measured"],
            color=colors[float(alpha)],
            edgecolors="black",
            linewidths=0.3,
            s=28,
            zorder=3,
        )
    axes[1].set_title(
        rf"Forward APJN $\mathcal{{J}}^{{\,b,{int(forward['source_block_index'])}}}$"
    )
    axes[1].set_xlabel(r"$b$")
    axes[1].set_ylabel(
        rf"$\mathcal{{J}}^{{\,b,{int(forward['source_block_index'])}}}$"
    )
    axes[1].set_yscale("log")
    prettify_log_axis(axes[1], "y")
    prettify_axes(axes[1])

    legend_handles = [
        Line2D([0], [0], color="black", lw=2.0, label="pre-LN theory"),
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=6, label="pre-LN ViT"),
    ]
    legend_handles.extend(
        Line2D([0], [0], color=colors[float(alpha)], lw=1.8, label=rf"Derf theory, $\alpha={float(alpha):g}$")
        for alpha in alphas
    )
    fig.legend(handles=legend_handles, loc="upper center", ncol=min(2 + len(alphas), 4), frameon=False)
    return fig, axes


def plot_theory_vs_experiment_curves(comparisons, *, direction, figsize=(8, 5)):
    if not isinstance(comparisons, (list, tuple)):
        comparisons = [comparisons]

    title = "Backward APJN" if direction == "inverse" else "Forward APJN"
    ylabel = r"$\mathcal{J}^{\,B,b}$" if direction == "inverse" else r"$\mathcal{J}$"
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    alphas = sorted(
        float(cmp["alpha"])
        for cmp in comparisons
        if cmp["model_variant"] == "derf" and cmp["alpha"] is not None
    )
    colors = _alpha_colors(alphas)

    for cmp in comparisons:
        if cmp["model_variant"] == "preln":
            color = "black"
            label = "pre-LN"
        else:
            color = colors[float(cmp["alpha"])]
            label = rf"Derf, $\alpha={float(cmp['alpha']):g}$"

        ax.plot(cmp["layers"], cmp["theory"], color=color, lw=1.9, label=f"{label} theory")
        ax.scatter(cmp["layers"], cmp["measured"], color=color, s=26, edgecolors="black", linewidths=0.3, zorder=3)

    ax.set_title(title)
    ax.set_xlabel(r"$b$")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    prettify_log_axis(ax, "y")
    prettify_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    ax.legend(dedup.values(), dedup.keys(), frameon=False)
    return fig, ax


def plot_depthwise_gmfe(records, *, metric_key="typ_mult_err", figsize=(10, 5)):
    region_names = ["Shallow", "Middle", "Deep"]
    region_centers = {"Shallow": 0.5, "Middle": 1.5, "Deep": 2.5}
    region_boundaries = [1.0, 2.0]
    alphas = sorted(float(row["alpha"]) for row in records["derf"])
    colors = _alpha_colors(alphas)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    rng = np.random.default_rng(0)

    alpha_offsets = np.linspace(-0.28, 0.18, max(1, len(alphas)))
    alpha_offset_map = {alpha: alpha_offsets[idx] for idx, alpha in enumerate(alphas)}
    preln_offset = 0.3

    for region_name in region_names:
        cx = region_centers[region_name]
        preln_subset = [
            row for row in records["preln"]
            if row["region"] == region_name and np.isfinite(row[metric_key])
        ]
        if preln_subset:
            x = np.full(len(preln_subset), cx + preln_offset) + rng.uniform(-0.04, 0.04, size=len(preln_subset))
            y = np.asarray([row[metric_key] for row in preln_subset], dtype=float)
            ax.scatter(x, y, color="black", s=24, alpha=0.8, zorder=3)

        for alpha in alphas:
            derf_subset = [
                row for row in records["derf"]
                if row["region"] == region_name
                and np.isfinite(row[metric_key])
                and float(row["alpha"]) == float(alpha)
            ]
            if not derf_subset:
                continue
            x = np.full(len(derf_subset), cx + alpha_offset_map[float(alpha)]) + rng.uniform(-0.04, 0.04, size=len(derf_subset))
            y = np.asarray([row[metric_key] for row in derf_subset], dtype=float)
            ax.scatter(x, y, color=colors[float(alpha)], s=24, alpha=0.8, zorder=3)

    for xline in region_boundaries:
        ax.axvline(xline, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    prettify_axes(ax)
    ax.set_xlim(0.0, 3.0)
    ax.set_xticks([region_centers[name] for name in region_names])
    ax.set_xticklabels(region_names)
    ax.set_ylabel("GMFE")
    ax.set_xlabel("Depth range")
    ax.grid(True, axis="y", alpha=0.25)

    handles = [Line2D([0], [0], marker="o", color="black", linestyle="None", label="pre-LN")]
    handles.extend(
        Line2D([0], [0], marker="o", color=colors[float(alpha)], linestyle="None", label=rf"Derf, $\alpha={float(alpha):g}$")
        for alpha in alphas
    )
    ax.legend(handles=handles, frameon=False)
    return fig, ax

