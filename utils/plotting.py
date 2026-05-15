import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import torch



def plot_single_roi_prediction(model, X_test, Y_test, device,
                                roi_idx=0, sample_idx=None,
                                align_mean=False):
    """
    Plot prediction vs ground truth for a single ROI.

    align_mean: if True, shifts predictions to match ground truth mean.
                WARNING: this artificially improves the visual appearance
                and should NOT be used when reporting results.
                Keep False for honest evaluation plots.
    """

    model.eval()

    if sample_idx is None:
        sample_idx = np.random.randint(len(X_test))

    with torch.no_grad():
        sample_x = torch.tensor(
            X_test[sample_idx:sample_idx + 1],
            dtype=torch.float32
        ).to(device)

        sample_y = Y_test[sample_idx]
        pred_y   = model(sample_x).cpu().numpy()[0]

        if align_mean:
            # Mean alignment: for exploration only, not for reporting
            print("WARNING: align_mean=True is enabled — plot is for exploration only.")
            pred_y = pred_y - pred_y.mean(axis=0) + sample_y.mean(axis=0)

    past_time   = np.arange(X_test.shape[1])
    future_time = np.arange(X_test.shape[1], X_test.shape[1] + Y_test.shape[1])

    plt.figure(figsize=(10, 5))

    plt.plot(past_time, X_test[sample_idx, :, roi_idx],
             label="Past Signal (Input)", linewidth=2, alpha=0.6)

    plt.plot(future_time, sample_y[:, roi_idx],
             "o-", label="Ground Truth", linewidth=2)

    plt.plot(future_time, pred_y[:, roi_idx],
             "o--", label="Prediction", linewidth=2)

    plt.axvline(x=X_test.shape[1] - 0.5, linestyle="--", label="Forecast Start")

    plt.title(f"ROI {roi_idx} | {Y_test.shape[1]}-step Forecast")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Signal")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.show()


def show_individual_roi_results(model, X_test, Y_test, device,
                                 roi_list=None, sample_idx=None):
    """
    Show separate prediction plots for each ROI in roi_list.
    Defaults to first 3 ROIs if roi_list is not provided.
    """

    if roi_list is None:
        n_roi = X_test.shape[2]
        roi_list = list(range(min(3, n_roi)))

    print("\nFINAL MODEL VISUALIZATION (INDIVIDUAL ROIs)")
    print("=" * 50)

    if sample_idx is None:
        sample_idx = np.random.randint(len(X_test))

    print(f"Using test sample index: {sample_idx}")

    for roi in roi_list:
        print(f"\nROI {roi}")
        plot_single_roi_prediction(
            model=model,
            X_test=X_test,
            Y_test=Y_test,
            device=device,
            roi_idx=roi,
            sample_idx=sample_idx,
            align_mean=False    # keep False for honest plots
        )


def plot_per_roi_bar(values, roi_labels=None, ylabel="Value", title="", figsize=(10, 4)):
    values = np.asarray(values)

    if roi_labels is None:
        roi_labels = [f"ROI {i}" for i in range(len(values))]

    plt.figure(figsize=figsize)
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), roi_labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_entropy_decomposition(
    H_marg,
    H_cond,
    roi_labels=None,
    figsize=(14, 8),
    cmap="flare_r",
    fontsize=14,
):
    """
    Plot:
        - marginal entropy H(Y) in background
        - conditional entropy H(Y|X) on top
        - eta (%) in lower panel

    Colors encode predictive information fraction eta.
    """

    H_marg = np.asarray(H_marg, dtype=np.float32)
    H_cond = np.asarray(H_cond, dtype=np.float32)

    eta = 1 - (H_cond / (H_marg + 1e-12))
    eta_percent = 100 * eta

    R = len(H_marg)

    if roi_labels is None:
        roi_labels = [f"ROI {i}" for i in range(1, R + 1)]

    # ----------------------------------------
    # Color mapping from eta
    # ----------------------------------------
    norm = Normalize(
        vmin=np.nanmin(eta_percent),
        vmax=np.nanmax(eta_percent)
    )

    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(eta_percent))

    # ----------------------------------------
    # Figure layout
    # ----------------------------------------
    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[3, 1],
        hspace=0.25,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    x = np.arange(R) + 1

    # ========================================
    # TOP PANEL
    # ========================================

    # Background entropy bars
    
    # Marginal entropy bars
    ax1.bar(
        x,
        H_marg,
        color=colors,
        edgecolor=None,
        linewidth=0.7,
        alpha=0.95,
        zorder=1,
    )

    # Conditional entropy overlay
    ax1.bar(
        x,
        H_cond,
        color="lightgray",
        edgecolor=None,
        linewidth=0.5,
        alpha=0.85,
        zorder=2,
    )

    ax1.set_ylabel("Entropy (nats)", fontsize=fontsize)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        roi_labels,
        rotation=45,
        ha="right"
    )
    ax1.tick_params(axis='x', labelbottom=True)

    #ax1.set_title(
    #    "Forecastable Information Across ROIs",
    #    fontsize=fontsize+4,
    #    pad=15,
    #)

    ax1.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1,
                          facecolor="lightgray",
                          alpha=0.85),

            plt.Rectangle((0, 0), 1, 1,
                          facecolor=cmap_obj(0.8),
                          alpha=0.95),
        ],
        labels=[
            r"$H(Y|X)$",
            r"$H(Y)$",
        ],
        fontsize=fontsize,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.05, 1)
    )

    # ========================================
    # BOTTOM PANEL
    # ========================================

    ax2.bar(
        x,
        eta_percent,
        color=colors,
        linewidth=0.5,
        alpha=0.95,
    )

    ax2.axhline(0, color="black", linewidth=1)

    ax2.set_ylabel(r"$\eta$ (%)", fontsize=fontsize)
    ax2.set_xlabel("ROIs", fontsize=fontsize)
    ax2.set_ylim(
        min(0, np.nanmin(eta_percent) * 1.1),
        np.nanmax(eta_percent) * 1.15
    )

    # ROI labels
    ax2.set_xticks([])
    

    # ========================================
    # Remove clutter
    # ========================================

    sns.despine(ax=ax1)
    sns.despine(ax=ax2)

    # ========================================
    # Shared colorbar
    # ========================================

    sm = ScalarMappable(
        norm=norm,
        cmap=cmap_obj,
    )

    sm.set_array([])

    # Leave room on right side
    fig.subplots_adjust(right=0.88)

    cax = fig.add_axes([0.90, 0.22, 0.02, 0.56])

    cbar = fig.colorbar(
        sm,
        cax=cax,
    )

    cbar.set_label(
        r"Predictive information fraction $\eta$ [%]",
        rotation=90,
        labelpad=15,
        fontsize=fontsize,
    )

    plt.show()

    return eta_percent

