# Module containing information theoretic measures and error metrics
import numpy as np
from tqdm.auto import tqdm

# ============================================================
# Basic utilities
# ============================================================

def fit_value_bin_edges(Y_train, n_bins=100, mode="quantile", eps=1e-6):
    """
    Fit fixed target-value bin edges from training targets.

    This helper is intended for neural models whose final layer emits logits
    over value bins. The returned edges define the discrete support used both
    during binned-likelihood training and later when continuous observed
    targets are mapped back to bins for log-probability evaluation.

    Parameters
    ----------
    Y_train : array-like, shape (N, H, R)
        Training targets. Fit these edges on training/calibration data only, not
        held-out test targets.
    n_bins : int, default=100
        Number of bins per horizon and ROI.
    mode : {"quantile", "uniform"}, default="quantile"
        ``"quantile"`` creates approximately balanced bins. ``"uniform"``
        creates equal-width bins between the observed min and max.
    eps : float, default=1e-6
        Padding for outer edges and minimum increment used to break quantile
        ties.

    Returns
    -------
    np.ndarray, shape (H, R, n_bins + 1)
        Bin edges indexed by forecast horizon, ROI, then edge id.
    """
    Y_train = np.asarray(Y_train, dtype=np.float32)
    if Y_train.ndim != 3:
        raise ValueError(f"Y_train must have shape (N,H,R), got {Y_train.shape}")

    _, H, R = Y_train.shape
    edges = np.empty((H, R, n_bins + 1), dtype=np.float32)

    for h in range(H):
        for r in range(R):
            y = Y_train[:, h, r]

            if mode == "quantile":
                q = np.linspace(0.0, 1.0, n_bins + 1)
                e = np.quantile(y, q).astype(np.float32)
                for k in range(1, len(e)):
                    if e[k] <= e[k - 1]:
                        e[k] = e[k - 1] + eps
            elif mode == "uniform":
                y_min, y_max = float(np.min(y)), float(np.max(y))
                if y_max <= y_min:
                    y_max = y_min + eps
                e = np.linspace(y_min, y_max, n_bins + 1, dtype=np.float32)
            else:
                raise ValueError("mode must be 'quantile' or 'uniform'")

            e[0] -= eps
            e[-1] += eps
            edges[h, r, :] = e

    return edges


def targets_to_bin_ids(Y, bin_edges):
    """
    Map continuous multi-step targets to integer value-bin labels.

    Parameters
    ----------
    Y : array-like, shape (N, H, R)
        Continuous target values to discretize.
    bin_edges : array-like, shape (H, R, B + 1)
        Per-horizon/per-ROI bin edges, typically from ``fit_value_bin_edges``.

    Returns
    -------
    np.ndarray, shape (N, H, R)
        Integer bin ids in ``[0, B - 1]``. Values outside the fitted edge range
        are clipped to the closest edge bin.
    """
    Y = np.asarray(Y, dtype=np.float32)
    edges = np.asarray(bin_edges, dtype=np.float32)

    if Y.ndim != 3:
        raise ValueError(f"Y must have shape (N,H,R), got {Y.shape}")
    if edges.shape[:2] != Y.shape[1:]:
        raise ValueError(
            f"bin_edges first dims must match (H,R)={Y.shape[1:]}, got {edges.shape}"
        )

    N, H, R = Y.shape
    n_bins = edges.shape[-1] - 1
    ids = np.empty((N, H, R), dtype=np.int64)

    for h in range(H):
        for r in range(R):
            ids[:, h, r] = np.digitize(Y[:, h, r], edges[h, r]) - 1

    return np.clip(ids, 0, n_bins - 1)


def bin_centers(bin_edges):
    """
    Compute bin centers from per-horizon/per-ROI bin edges.

    Parameters
    ----------
    bin_edges : array-like, shape (..., B + 1)
        Bin edges along the final axis.

    Returns
    -------
    np.ndarray, shape (..., B)
        Midpoints between adjacent edges. For ``(H, R, B + 1)`` input this is
        shaped ``(H, R, B)``.
    """
    edges = np.asarray(bin_edges, dtype=np.float32)
    return 0.5 * (edges[..., :-1] + edges[..., 1:])


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
# Metrics and losses
# ============================================================

def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_rmsse(y_true, y_pred):
    """Root Mean Squared Scaled Error (RMSSE) for multi-step forecasts."""
    numerator = np.mean((y_true - y_pred) ** 2)
    denominator = np.mean((y_true[:, 1:, :] - y_true[:, :-1, :]) ** 2) + 1e-8
    return float(np.sqrt(numerator / denominator))

# ============================================================
# Entropy, Conditional Enttropy and Information Content
# ============================================================

def compute_eta_gauss(y_true, y_pred):
    """
    Information-theoretic information content (eta in the paper) with guassian assumption computed per ROI and averaged.
    """
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    n_roi = y_true.shape[1]
    etas = []

    for roi in range(n_roi):
        yt = y_true[:, roi]
        yp = y_pred[:, roi]

        if yt.std() < 1e-8 or yp.std() < 1e-8:
            continue

        r = np.corrcoef(yt, yp)[0, 1]
        r = np.clip(r, -1 + 1e-7, 1 - 1e-7)

        mi = -0.5 * np.log(1 - r ** 2)
        hy = 0.5 * np.log(2 * np.pi * np.e * (yt.var() + 1e-12))
        etas.append(mi / (hy + 1e-12))

    return float(np.nanmean(etas))

def compute_marginal_entropy_histogram(
    Y,
    horizon_idx=0,
    n_bins=50,
    eps=1e-12,
):
    """
    Non-Gaussian plug-in estimate of per-ROI marginal entropy H(Y_r).

    Returns entropy in nats.
    """
    Y = np.asarray(Y, dtype=np.float32)
    Yh = Y[:, horizon_idx, :]   # (N, R)

    R = Yh.shape[1]
    H = np.zeros(R, dtype=np.float32)

    for r in range(R):
        y = Yh[:, r]

        hist, _ = np.histogram(y, bins=n_bins, density=False)
        p = hist.astype(np.float64)
        p = p / (p.sum() + eps)

        p = p[p > 0]
        H[r] = -np.sum(p * np.log(p + eps))

    return H

def compute_conditional_entropy_from_log_prob_api(
    model,
    X,
    Y,
    horizon_idx=0,
    batch_size=512,
):
    """
    Estimate per-ROI conditional entropy H(Y_r | X)
    using the model predictive distribution.

    Returns entropy in nats.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    _, _, R = Y.shape

    pred = model.predict_proba(X, Y=Y, batch_size=batch_size)

    H_cond = np.zeros(R, dtype=np.float32)

    for r in range(R):
        y = Y[:, horizon_idx, r]

        logp = extract_log_prob(
            pred=pred,
            y=y,
            roi_idx=r,
            horizon_idx=horizon_idx,
        )

        H_cond[r] = -np.nanmean(logp)

    return H_cond


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

    Histogram/value-bin style:
        pred["probs"] shape (N, H, R, B)
        pred["bin_edges"] either:
            - shape (B + 1,)
            - shape (R, B + 1)
            - shape (H, R, B + 1)
        If pred also contains "mean", bin_edges are interpreted as residual
        edges around the mean. Otherwise they are interpreted as target-value
        edges.

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

        bin_edges = np.asarray(bin_edges)
        if bin_edges.ndim == 3:
            edges = bin_edges[horizon_idx, roi_idx]
        elif bin_edges.ndim == 2:
            edges = bin_edges[roi_idx]
        else:
            edges = bin_edges

        values_to_bin = y
        if "mean" in pred:
            values_to_bin = y - pred["mean"][:, horizon_idx, roi_idx]

        bin_ids = np.digitize(values_to_bin, edges) - 1
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

class DiscreteNeuralPredictorAPI:
    """
    DI-compatible API for neural forecast models.

    Preferred discrete mode:
        The wrapped torch model returns logits shaped ``(N, H, R, B)``, where
        ``B`` is the number of target-value bins. ``softmax(logits)`` is treated
        as ``p_theta(Y[h,r] in bin b | X)``, and observed continuous targets are
        mapped to bins using ``bin_edges``.

    Backward-compatible point-forecast mode:
        If the wrapped model returns deterministic forecasts shaped
        ``(N, H, R)``, this adapter falls back to the same calibrated Gaussian
        likelihood used by ``FlowPredictorAPI``. In that case call
        ``fit_residual_std(X_calib, Y_calib)`` before DI estimation, or pass
        ``residual_std`` to ``__init__``.
    """

    def __init__(self, model_obj, bin_edges=None, device="cpu", residual_std=None, eps=1e-12):
        """
        Wrap a trained torch forecasting model for ``compute_di_from_log_prob_api``.

        Parameters
        ----------
        model_obj : torch.nn.Module
            Trained model. It may emit either discrete logits ``(N,H,R,B)`` or
            point forecasts ``(N,H,R)``.
        bin_edges : array-like, shape (H, R, B + 1), optional
            Target-value bin edges required for discrete-logit models. Not used
            for point-forecast Gaussian fallback mode.
        device : str or torch.device, default="cpu"
            Device used for model inference.
        residual_std : array-like, optional
            Calibrated residual standard deviation for point-forecast fallback.
            Shape may be ``(H,R)`` or ``(R,)``.
        eps : float, default=1e-12
            Numerical floor added before taking logs and variances.
        """
        import torch

        self.model_obj = model_obj.to(device)
        self.model_obj.eval()
        self.device = torch.device(device)
        self.bin_edges = None if bin_edges is None else np.asarray(bin_edges, dtype=np.float32)
        self.residual_std = residual_std
        self.eps = float(eps)

    @property
    def n_bins(self):
        if self.bin_edges is None:
            raise ValueError("n_bins is only defined when bin_edges are provided.")
        return self.bin_edges.shape[-1] - 1

    def _predict_model_output(self, X, batch_size=512):
        """Run batched torch inference and return the raw model output as numpy."""
        import torch

        X = np.asarray(X, dtype=np.float32)
        outputs = []

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                xb = torch.tensor(X[start:end], dtype=torch.float32).to(self.device)
                out = self.model_obj(xb)
                outputs.append(out.detach().cpu().numpy())

        return np.concatenate(outputs, axis=0)

    def predict_logits(self, X, batch_size=512):
        """
        Return discrete predictive logits shaped ``(N,H,R,B)``.

        This method is only valid for models with a binned/discrete output head.
        For legacy deterministic models that return ``(N,H,R)``, use
        ``predict`` or ``predict_proba`` after calibrating residuals.
        """
        logits = self._predict_model_output(X, batch_size=batch_size)
        if logits.ndim != 4:
            raise ValueError(
                "predict_logits requires model output shaped (N,H,R,B). "
                f"Got {logits.shape}; this looks like a point-forecast model. "
                "Use predict()/fit_residual_std()/predict_proba() for Gaussian fallback."
            )
        if self.bin_edges is None:
            raise ValueError("bin_edges are required when model output is shaped (N,H,R,B).")
        if logits.shape[1:3] != self.bin_edges.shape[:2]:
            raise ValueError(
                "Model logits horizon/ROI dimensions must match bin_edges first "
                f"dimensions. Got logits {logits.shape} and bin_edges {self.bin_edges.shape}"
            )
        if logits.shape[-1] != self.n_bins:
            raise ValueError(
                f"Model emitted {logits.shape[-1]} bins, but bin_edges define {self.n_bins}"
            )

        return logits

    def fit_residual_std(self, X_calib, Y_calib, batch_size=512):
        """
        Calibrate Gaussian residual scale for point-forecast fallback mode.

        Use this only when the wrapped model emits deterministic predictions
        shaped ``(N,H,R)``. Discrete-logit models do not need residual
        calibration because their likelihood is read directly from their output
        probabilities.

        Parameters
        ----------
        X_calib : array-like, shape (N, M, R)
            Calibration input windows.
        Y_calib : array-like, shape (N, H, R)
            Calibration targets aligned with ``X_calib``.
        batch_size : int, default=512
            Inference batch size.

        Returns
        -------
        DiscreteNeuralPredictorAPI
            Returns ``self`` for notebook-style chaining.
        """
        mean = self.predict(X_calib, batch_size=batch_size)
        resid = np.asarray(Y_calib, dtype=np.float32) - mean
        self.residual_std = np.std(resid, axis=0) + self.eps
        return self

    def predict_proba(self, X, Y=None, batch_size=512):
        """
        Return predictive distribution information for DI estimation.

        For discrete-logit models this returns ``{"probs", "bin_edges"}``, plus
        ``"log_prob"`` when ``Y`` is provided. For point-forecast models this
        returns Gaussian ``"log_prob"`` using calibrated ``residual_std`` and
        therefore requires ``Y``.
        """
        import torch

        raw = self._predict_model_output(X, batch_size=batch_size)

        if raw.ndim == 3:
            if Y is None:
                raise ValueError("Y is required to evaluate log p(Y | X).")
            if self.residual_std is None:
                raise ValueError(
                    "Wrapped model returned point forecasts shaped (N,H,R), not "
                    "discrete logits. Call prob_model.fit_residual_std(X_calib, "
                    "Y_calib) first, pass residual_std, or use a model with a "
                    "discrete output head shaped (N,H,R,B)."
                )

            Y = np.asarray(Y, dtype=np.float32)
            std = np.asarray(self.residual_std, dtype=np.float32)
            if std.ndim == 1:
                std = std.reshape(1, -1)

            var = std[None, :, :] ** 2 + self.eps
            log_prob = (
                -0.5 * np.log(2 * np.pi * var)
                -0.5 * ((Y - raw) ** 2 / var)
            ).astype(np.float32)

            return {"log_prob": log_prob, "mean": raw, "std": np.sqrt(var).astype(np.float32)}

        if raw.ndim != 4:
            raise ValueError(
                "DiscreteNeuralPredictorAPI expects model output shaped either "
                f"(N,H,R,B) for discrete logits or (N,H,R) for point forecasts, got {raw.shape}"
            )
        if self.bin_edges is None:
            raise ValueError("bin_edges are required when model output is shaped (N,H,R,B).")

        logits = raw
        if logits.shape[1:3] != self.bin_edges.shape[:2]:
            raise ValueError(
                "Model logits horizon/ROI dimensions must match bin_edges first "
                f"dimensions. Got logits {logits.shape} and bin_edges {self.bin_edges.shape}"
            )
        if logits.shape[-1] != self.n_bins:
            raise ValueError(
                f"Model emitted {logits.shape[-1]} bins, but bin_edges define {self.n_bins}"
            )

        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy().astype(np.float32)

        out = {
            "probs": probs,
            "bin_edges": self.bin_edges,
        }

        if Y is not None:
            y_bins = targets_to_bin_ids(Y, self.bin_edges)
            log_prob = np.log(
                np.take_along_axis(probs, y_bins[..., None], axis=-1)[..., 0]
                + self.eps
            ).astype(np.float32)
            out["log_prob"] = log_prob

        return out

    def predict(self, X, batch_size=512):
        """
        Return deterministic forecasts.

        For point-forecast models this is the raw model output ``(N,H,R)``. For
        discrete-logit models this is the posterior mean under the learned
        binned predictive distribution.
        """
        raw = self._predict_model_output(X, batch_size=batch_size)
        if raw.ndim == 3:
            return raw.astype(np.float32)
        if raw.ndim != 4:
            raise ValueError(
                "Expected model output shaped either (N,H,R) or (N,H,R,B), "
                f"got {raw.shape}"
            )
        if self.bin_edges is None:
            raise ValueError("bin_edges are required to summarize discrete logits.")

        probs = torch.softmax(torch.tensor(raw), dim=-1).numpy().astype(np.float32)
        centers = bin_centers(self.bin_edges)
        return np.sum(probs * centers[None, ...], axis=-1).astype(np.float32)

    def log_prob_next(self, X, Y, batch_size=512):
        """Return ``log p(Y | X)`` shaped ``(N,H,R)`` for either adapter mode."""
        return self.predict_proba(X, Y=Y, batch_size=batch_size)["log_prob"]


class SecondOrderPredictorAPI:
    """
    Likelihood wrapper for torch forecasters.

    If model_obj has log_prob_next(X, Y), use it directly.
    Otherwise, use point forecasts + calibrated residual std to compute
    log p(Y | X). This fallback is Gaussian, not a true flow.
    """

    def __init__(self, model_obj, device="cpu", residual_std=None, eps=1e-8):
        import torch
        self.model_obj = model_obj.to(device)
        self.device = torch.device(device)
        self.model_obj.eval()
        self.residual_std = residual_std
        self.eps = eps


    def fit_residual_std(self, X_calib, Y_calib, batch_size=512):
        """
        Estimate residual std from calibration data.

        X_calib: (N, M, R)
        Y_calib: (N, H, R)
        """
        mean = self.predict(X_calib, batch_size=batch_size)
        resid = Y_calib - mean
        self.residual_std = np.std(resid, axis=0) + self.eps  # shape (H, R)
        return self
    
    def predict(self, X, batch_size=512):
        import torch
        X = np.asarray(X, dtype=np.float32)

        preds = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                xb = torch.tensor(X[start:end], dtype=torch.float32).to(self.device)
                out = self.model_obj(xb)
                preds.append(out.detach().cpu().numpy())

        return np.concatenate(preds, axis=0)

    def log_prob_next(self, X, Y, batch_size=512):
        """
        Returns log p(Y | X), shape (N, H, R).
        """
        import torch

        # If the torch model already has a real flow likelihood, use it.
        if hasattr(self.model_obj, "log_prob_next"):
            X = np.asarray(X, dtype=np.float32)
            Y = np.asarray(Y, dtype=np.float32)

            logps = []
            with torch.no_grad():
                for start in range(0, len(X), batch_size):
                    end = min(start + batch_size, len(X))
                    xb = torch.tensor(X[start:end], dtype=torch.float32).to(self.device)
                    yb = torch.tensor(Y[start:end], dtype=torch.float32).to(self.device)
                    lp = self.model_obj.log_prob_next(xb, yb)
                    logps.append(lp.detach().cpu().numpy())

            return np.concatenate(logps, axis=0)

        # Fallback: point forecast + calibrated residual likelihood
        if self.residual_std is None:
            raise ValueError(
                "No true log_prob_next found and residual_std is None. "
                "Call flow_api.fit_residual_std(X_val, Y_val) first."
            )

        mean = self.predict(X, batch_size=batch_size)
        Y = np.asarray(Y, dtype=np.float32)

        std = np.asarray(self.residual_std, dtype=np.float32)
        if std.ndim == 1:
            std = std.reshape(1, -1)  # (1, R)

        var = std[None, :, :] ** 2 + self.eps  # (1, H, R)

        logp = (
            -0.5 * np.log(2 * np.pi * var)
            -0.5 * ((Y - mean) ** 2 / var)
        )
        return logp.astype(np.float32)
    
    def predict_proba(self, X, Y=None, batch_size=512):
        if Y is None:
            raise ValueError("Y is required to evaluate log p(Y | X).")

        logp = self.log_prob_next(X, Y, batch_size=batch_size)

        return {
            "log_prob": logp
        }
  

class HistogramProbaAdapter: 
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
    n_bins=100,
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






