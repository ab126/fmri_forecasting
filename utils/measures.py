# Module containing information theoretic measures
import numpy as np
from tqdm.auto import tqdm

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

    full_pred = model.predict_proba(X, Y=Y, batch_size=batch_size)

    with tqdm(total=R * R, desc="Computing DI") as pbar:
        for source_roi in range(R):
            X_red = make_reduced_input(
                X,
                source_roi=source_roi,
                mode=reduction_mode,
                rng=rng,
            )

            red_pred = model.predict_proba(X_red, Y=Y, batch_size=batch_size)

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
                pbar.update(1)
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

        mu = pred["mean"][:, horizon_idx, roi_idx]
        residual = y - mu

        if np.asarray(bin_edges).ndim == 2:
            edges = np.asarray(bin_edges)[roi_idx]
        else:
            edges = np.asarray(bin_edges)

        bin_ids = np.digitize(residual, edges) - 1
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


# ============================================================
# Model Wrapper for DI estimation
# ============================================================
class FlowPredictorAPI:
    """
    Wrapper for a neural forecaster with conditional normalizing flow head.

    Assumes the internal model can compute log p(Y | X).
    """

    def __init__(self, model_obj, device="cpu"):
        self.model_obj = model_obj
        self.device = device
        self.model_obj.to(device)
        self.model_obj.eval()

    def predict_proba(self, X, Y=None, batch_size=512):
        import torch

        if Y is None:
            raise ValueError("Flow likelihood requires Y to compute log p(Y | X).")

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        logps = []

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))

                xb = torch.tensor(X[start:end], dtype=torch.float32).to(self.device)
                yb = torch.tensor(Y[start:end], dtype=torch.float32).to(self.device)

                logp_batch = self.model_obj.log_prob_next(xb, yb)
                logps.append(logp_batch.cpu().numpy())

        return {
            "log_prob": np.concatenate(logps, axis=0)
        }

class HistogramProbaAdapter: # TODO: debug bins, think they are all zero
    """
    Adds predict_proba to any forecasting model using empirical residual histograms.

    Base model must implement:
        predict(X)

    Expected model prediction shape:
        either (N, H, R)
        or flattened (N, H * R)
    """

    def __init__(
        self,
        base_model,
        bin_edges,
        residual_hist,
        H,
        R,
        flatten_input=False,
        flatten_output=False,
    ):
        self.base_model = base_model
        self.bin_edges = bin_edges
        self.residual_hist = residual_hist
        self.H = int(H)
        self.R = int(R)
        self.flatten_input = flatten_input
        self.flatten_output = flatten_output

    def predict(self, X):
        X_in = X.reshape(X.shape[0], -1) if self.flatten_input else X
        pred = self.base_model.predict(X_in)

        pred = np.asarray(pred, dtype=np.float32)

        if pred.ndim == 2:
            pred = pred.reshape(X.shape[0], self.H, self.R)

        return pred

    def predict_proba(self, X, Y=None, batch_size=512): # keep Y and batch_size for API consistency, but this adapter does not use them
        """
        Returns a discrete predictive distribution over fixed bins.

        Output:
            probs shape = (N, H, R, B)
            bin_edges shape = (R, B + 1) or (B + 1,)
        """
        mu = self.predict(X)
        N, H, R = mu.shape
        B = self.residual_hist.shape[-1]

        probs = np.zeros((N, H, R, B), dtype=np.float32)

        # Residual histogram is indexed by horizon and ROI.
        # Shape: (H, R, B)
        for h in range(H):
            for r in range(R):
                probs[:, h, r, :] = self.residual_hist[h, r, :]

        return {
            "mean": mu,
            "probs": probs,
            "bin_edges": self.bin_edges,
        }

def fit_histogram_proba_adapter(
    base_model,
    X_calib,
    Y_calib,
    n_bins=50,
    flatten_input=False,
    flatten_output=False,
    eps=1e-6,
):
    """
    Calibrates a histogram likelihood adapter from residuals.

    X_calib: shape (N, M, R)
    Y_calib: shape (N, H, R)
    """
    X_calib = np.asarray(X_calib, dtype=np.float32)
    Y_calib = np.asarray(Y_calib, dtype=np.float32)

    N, H, R = Y_calib.shape

    X_in = X_calib.reshape(N, -1) if flatten_input else X_calib
    pred = base_model.predict(X_in)
    pred = np.asarray(pred, dtype=np.float32)

    if pred.ndim == 2:
        pred = pred.reshape(N, H, R)

    residuals = Y_calib - pred

    bin_edges = np.linspace(np.min(residuals), np.max(residuals), n_bins + 1)
    residual_hist = np.zeros((H, R, n_bins), dtype=np.float32)

    for h in range(H):
        for r in range(R):
            hist, _ = np.histogram(
                residuals[:, h, r],
                bins=bin_edges,
                density=False,
            )
            hist = hist.astype(np.float32) + eps
            hist = hist / hist.sum()
            residual_hist[h, r, :] = hist

    return HistogramProbaAdapter(
        base_model=base_model,
        bin_edges=bin_edges,
        residual_hist=residual_hist,
        H=H,
        R=R,
        flatten_input=flatten_input,
        flatten_output=flatten_output,
    )


