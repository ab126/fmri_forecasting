# Module containing information theoretic measures
import numpy as np

# ============================================================
# Basic utilities
# ============================================================

def make_reduced_input(X, source_roi, mode="zero", rng=None):
    """
    Remove source ROI history from X.

    X: shape (N, M, R)
    source_roi: int
    mode:
        "zero"    -> replace source history by 0
        "mean"    -> replace by within-window mean
        "shuffle" -> shuffle source histories across samples
    """
    X_red = X.copy()

    if mode == "zero":
        X_red[:, :, source_roi] = 0.0

    elif mode == "mean":
        src_mean = X_red[:, :, source_roi].mean(axis=1, keepdims=True)
        X_red[:, :, source_roi] = src_mean

    elif mode == "shuffle":
        if rng is None:
            rng = np.random.default_rng(0)
        perm = rng.permutation(X.shape[0])
        X_red[:, :, source_roi] = X[perm, :, source_roi]

    else:
        raise ValueError(f"Unknown reduction mode: {mode}")

    return X_red


def safe_log(x, eps=1e-12):
    return np.log(np.maximum(x, eps))


# ============================================================
# Generic DI from log-probabilities
# ============================================================

def compute_di_from_log_prob_api(
    model,
    X,
    Y,
    horizon_idx=0,
    reduction_mode="zero",
    batch_size=512,
    rng=None,
):
    """
    Generic directed information estimator.

    Requires:
        model.predict_proba(X, batch_size=batch_size)

    The output of predict_proba must support one of:
        1. {"log_prob": array of shape (N, H, R)}
        2. {"log_prob_fn": callable(y, roi_idx, horizon_idx) -> logp}
        3. {"probs": array of shape (N, H, R, B), "bin_edges": ...}
        4. {"mean": ..., "std": ...}  fallback Gaussian

    X: shape (N, M, R)
    Y: shape (N, H, R)

    Returns:
        DI matrix of shape (R, R), where DI[i, j] = source i -> target j
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    N, M, R = X.shape
    DI = np.zeros((R, R), dtype=np.float32)

    full_pred = model.predict_proba(X, batch_size=batch_size)

    for source_roi in range(R):
        X_red = make_reduced_input(
            X,
            source_roi=source_roi,
            mode=reduction_mode,
            rng=rng,
        )

        red_pred = model.predict_proba(X_red, batch_size=batch_size)

        for target_roi in range(R):
            y = Y[:, horizon_idx, target_roi]

            logp_full = extract_log_prob(
                pred=full_pred,
                y=y,
                roi_idx=target_roi,
                horizon_idx=horizon_idx,
            )

            logp_red = extract_log_prob(
                pred=red_pred,
                y=y,
                roi_idx=target_roi,
                horizon_idx=horizon_idx,
            )

            sample_di = logp_full - logp_red
            DI[source_roi, target_roi] = np.nanmean(sample_di)

    return DI


def extract_log_prob(pred, y, roi_idx, horizon_idx=0):
    """
    Converts model.predict_proba output into log p(y | history).

    Supported prediction formats:

    Flow-style:
        pred["log_prob"] shape (N, H, R)

    Callable style:
        pred["log_prob_fn"](y, roi_idx, horizon_idx)

    Histogram style:
        pred["probs"] shape (N, H, R, B)
        pred["bin_edges"] either:
            - shape (B + 1,)
            - shape (R, B + 1)

    Gaussian fallback:
        pred["mean"], pred["std"]
    """
    y = np.asarray(y)

    if "log_prob" in pred:
        return np.asarray(pred["log_prob"])[:, horizon_idx, roi_idx]

    if "log_prob_fn" in pred:
        return pred["log_prob_fn"](y, roi_idx=roi_idx, horizon_idx=horizon_idx)

    if "probs" in pred and "bin_edges" in pred:
        probs = pred["probs"][:, horizon_idx, roi_idx, :]
        bin_edges = pred["bin_edges"]

        if np.asarray(bin_edges).ndim == 2:
            edges = np.asarray(bin_edges)[roi_idx]
        else:
            edges = np.asarray(bin_edges)

        bin_ids = np.digitize(y, edges) - 1
        bin_ids = np.clip(bin_ids, 0, probs.shape[1] - 1)

        p = probs[np.arange(len(y)), bin_ids]
        return safe_log(p)

    if "mean" in pred and "std" in pred:
        mu = pred["mean"][:, horizon_idx, roi_idx]
        std = pred["std"][:, horizon_idx, roi_idx]
        var = std ** 2 + 1e-8
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((y - mu) ** 2 / var)

    raise ValueError(
        "predict_proba output must contain one of: "
        "'log_prob', 'log_prob_fn', 'probs'+'bin_edges', or 'mean'+'std'."
    )

