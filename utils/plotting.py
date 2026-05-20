import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

import torch


def compute_residual_quantiles(
    y_true,
    y_pred,
    quantiles=(0.025, 0.16, 0.84, 0.975),
):
    """
    Compute per-ROI residual quantiles for uncertainty bands.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_samples, horizon, n_roi)
        Ground-truth and predicted forecast windows.
    quantiles : iterable of float
        Quantile levels to estimate from residuals ``y_true - y_pred``.

    Returns
    -------
    dict
        Mapping ``roi_idx -> {"q025": ..., "q16": ..., "q84": ..., "q975": ...}``
        for the default quantiles.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{y_true.shape} and {y_pred.shape}"
        )
    if y_true.ndim != 3:
        raise ValueError(
            "Expected y_true and y_pred with shape "
            "(n_samples, horizon, n_roi)."
        )

    labels = {
        0.025: "q025",
        0.16: "q16",
        0.84: "q84",
        0.975: "q975",
    }
    residuals = y_true - y_pred
    residual_quantiles = {}

    for roi_idx in range(residuals.shape[2]):
        vals = residuals[:, :, roi_idx].reshape(-1)
        residual_quantiles[int(roi_idx)] = {
            labels.get(float(q), f"q{int(q * 1000):03d}"): float(
                np.quantile(vals, q)
            )
            for q in quantiles
        }

    return residual_quantiles


def _as_numpy_forecast(preds, target_shape=None):
    preds = np.asarray(preds, dtype=np.float32)

    if target_shape is None or preds.shape == target_shape:
        return preds

    if preds.ndim == 2:
        return preds.reshape(target_shape)

    raise ValueError(
        f"Could not reshape predictions from {preds.shape} to {target_shape}"
    )


def predict_from_forecaster(model, X, batch_size=512, device=None, target_shape=None):
    """
    Predict forecast windows from either a PyTorch module or predict-style model.
    """
    X = np.asarray(X, dtype=np.float32)

    if isinstance(model, torch.nn.Module):
        if device is None:
            device = next(model.parameters()).device
        else:
            device = torch.device(device)

        model = model.to(device)
        model.eval()

        preds = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.as_tensor(
                    X[start:start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                preds.append(model(batch).detach().cpu().numpy())

        return _as_numpy_forecast(np.concatenate(preds, axis=0), target_shape)

    if hasattr(model, "predict"):
        try:
            preds = model.predict(X, batch_size=batch_size)
        except TypeError:
            preds = model.predict(X)
        return _as_numpy_forecast(preds, target_shape)

    raise TypeError(
        "model must be a torch.nn.Module or expose a predict(X) method."
    )


def make_proba_forecast(
    X,
    Y,
    Yhat,
    roi_idx,
    sample_idx,
    residual_quantiles,
):
    """
    Build the single-sample forecast traces and probabilistic uncertainty bands.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    Yhat = np.asarray(Yhat)

    if X.ndim != 3 or Y.ndim != 3 or Yhat.ndim != 3:
        raise ValueError("X, Y, and Yhat must all be 3D arrays.")
    if Y.shape != Yhat.shape:
        raise ValueError(f"Y and Yhat shape mismatch: {Y.shape} != {Yhat.shape}")

    roi_idx = int(roi_idx)
    sample_idx = int(sample_idx)
    q = residual_quantiles.get(roi_idx, residual_quantiles.get(str(roi_idx)))
    if q is None:
        raise KeyError(f"No residual quantiles found for ROI {roi_idx}.")

    past_time = np.arange(X.shape[1])
    future_time = np.arange(X.shape[1], X.shape[1] + Y.shape[1])
    mean = Yhat[sample_idx, :, roi_idx]

    return {
        "past_time": past_time,
        "future_time": future_time,
        "past": X[sample_idx, :, roi_idx],
        "truth": Y[sample_idx, :, roi_idx],
        "mean": mean,
        "lower_68": mean + q["q16"],
        "upper_68": mean + q["q84"],
        "lower_95": mean + q["q025"],
        "upper_95": mean + q["q975"],
    }


def plot_proba_forecast(
    X,
    Y,
    Yhat=None,
    *,
    model=None,
    roi_idx=0,
    sample_idx=None,
    residual_quantiles=None,
    batch_size=512,
    device=None,
    ax=None,
    figsize=(10, 5),
    title=None,
    save_path=None,
    show=True,
):
    """
    Plot a probabilistic forecast with residual uncertainty bands.

    ``Yhat`` can be supplied directly, or ``model`` can be a trained PyTorch
    module / predict-style forecaster. If ``residual_quantiles`` is omitted,
    they are estimated from ``Y - Yhat`` across all supplied samples.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if sample_idx is None:
        sample_idx = np.random.randint(len(X))

    if Yhat is None:
        if model is None:
            raise ValueError("Pass either Yhat predictions or a trained model.")
        Yhat = predict_from_forecaster(
            model,
            X,
            batch_size=batch_size,
            device=device,
            target_shape=Y.shape,
        )
    else:
        Yhat = _as_numpy_forecast(Yhat, target_shape=Y.shape)

    if residual_quantiles is None:
        residual_quantiles = compute_residual_quantiles(Y, Yhat)

    traces = make_proba_forecast(
        X=X,
        Y=Y,
        Yhat=Yhat,
        roi_idx=roi_idx,
        sample_idx=sample_idx,
        residual_quantiles=residual_quantiles,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Color palette
    colors = sns.color_palette("colorblind")

    past_c = colors[0]
    truth_c = colors[1]
    pred_c = colors[2]

    # Past signal
    ax.plot(
        traces["past_time"],
        traces["past"],
        color=past_c,
        label="Past Signal",
        alpha=0.75,
    )

    # Ground truth
    ax.plot(
        traces["future_time"],
        traces["truth"],
        "o-",
        color=truth_c,
        markersize=7,
        label="Ground Truth",
    )

    # Prediction mean
    ax.plot(
        traces["future_time"],
        traces["mean"],
        "o--",
        color=pred_c,
        markersize=7,
        label="Prediction",
    )

    # 68% band
    ax.fill_between(
        traces["future_time"],
        traces["lower_68"],
        traces["upper_68"],
        color=pred_c,
        alpha=0.25,
        linewidth=0,
        label="68% Interval",
    )

    # 95% band
    ax.fill_between(
        traces["future_time"],
        traces["lower_95"],
        traces["upper_95"],
        color=pred_c,
        alpha=0.12,
        linewidth=0,
        label="95% Interval",
    )

    ax.axvline(x=X.shape[1] - 0.5, linestyle="--", label="Forecast Start", alpha=0.7,)
    ax.set_xlabel("Time Step", fontsize=14)
    ax.set_ylabel("Normalized Signal", fontsize=14)
    ax.set_title(title or f"Probabilistic Forecast (ROI {roi_idx+1})", fontsize=16)
    # Cleaner legend
    ax.legend(
        loc="best",
        fontsize=12,
    )
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.margins(x=0.02)
    ax.figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return ax, residual_quantiles, Yhat



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

