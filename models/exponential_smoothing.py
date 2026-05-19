"""
Per-window exponential-smoothing forecaster for fMRI ROI time series.

This module exposes a lightweight sklearn-style adapter used by the shared
LOSO cross-validation pipeline. Unlike the torch models, the forecaster does
not learn global parameters during ``fit``. Instead, each call to ``predict``
fits a separate statsmodels exponential-smoothing model for each ROI in each
input window, then forecasts ``H`` future time points.
"""

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing



def fit_exp_smoothing_and_forecast(
    train_series,
    forecast_steps,
    trend=None,
    seasonal=None,
    seasonal_periods=None,
):
    """
    Fit one statsmodels exponential-smoothing model and forecast ahead.

    Parameters
    ----------
    train_series:
        One ROI signal from a single window, shaped ``(M,)``.
    forecast_steps:
        Number of future samples to forecast.
    trend, seasonal, seasonal_periods:
        Passed through to ``statsmodels.tsa.holtwinters.ExponentialSmoothing``.

    Returns
    -------
    forecast:
        Forecast values as ``float32`` with shape ``(forecast_steps,)``.
    fit:
        The fitted statsmodels results object, useful for debugging notebooks.
    """
    model = ExponentialSmoothing(
        endog=train_series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )

    fit = model.fit(optimized=True)
    forecast = fit.forecast(forecast_steps)

    return np.asarray(forecast, dtype=np.float32), fit

# TODO: Come back after Transformer
class ExponentialSmoothingForecaster:
    """
    sklearn-style adapter for stateless exponential-smoothing forecasts.

    ``fit`` receives flattened windows and targets from
    ``utils.training.train_forecasting_model``. It only infers and stores the
    window length and ROI count. ``predict`` then reshapes each flattened input
    back to ``(M, ROI)``, fits one univariate exponential-smoothing model per
    ROI, and returns flattened predictions shaped ``(N, H * ROI)``.

    If statsmodels cannot fit a particular ROI/window, prediction falls back to
    repeating the last observed ROI value for that horizon.
    """

    def __init__(self, H=1, trend=None, seasonal=None, seasonal_periods=None):
        self.H = int(H)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.n_roi = None
        self.window_size = None

    def fit(self, X, y=None):
        """
        Keep sklearn-style compatibility with the shared training utility.

        X is expected as flattened windows with shape (N, M * ROI).
        y is expected as flattened targets with shape (N, H * ROI).
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D flattened X, got shape {X.shape}")

        if y is not None:
            y = np.asarray(y, dtype=np.float32)
            if y.ndim != 2:
                raise ValueError(f"Expected 2D flattened y, got shape {y.shape}")
            if y.shape[1] % self.H != 0:
                raise ValueError(
                    f"Target width {y.shape[1]} is not divisible by H={self.H}"
                )
            self.n_roi = y.shape[1] // self.H

        if self.n_roi is None:
            raise ValueError(
                "Could not infer ROI count. Pass y during fit so the flattened "
                "target dimension can be used to infer it."
            )

        if X.shape[1] % self.n_roi != 0:
            raise ValueError(
                f"Input width {X.shape[1]} is not divisible by inferred ROI count "
                f"{self.n_roi}."
            )

        self.window_size = X.shape[1] // self.n_roi
        return self

    def predict(self, X):
        """
        Predict flattened outputs so the shared training utility can reshape
        them back to (N, H, ROI).
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D flattened X, got shape {X.shape}")
        if self.n_roi is None or self.window_size is None:
            raise ValueError("Model must be fit before calling predict.")
        if X.shape[1] != self.window_size * self.n_roi:
            raise ValueError(
                f"Expected input width {self.window_size * self.n_roi}, got {X.shape[1]}"
            )

        X_windows = X.reshape(X.shape[0], self.window_size, self.n_roi)
        preds = np.empty((X.shape[0], self.H, self.n_roi), dtype=np.float32)

        for sample_idx, window in enumerate(X_windows):
            for roi_idx in range(self.n_roi):
                train_series = window[:, roi_idx]
                try:
                    forecast, _ = fit_exp_smoothing_and_forecast(
                        train_series=train_series,
                        forecast_steps=self.H,
                        trend=self.trend,
                        seasonal=self.seasonal,
                        seasonal_periods=self.seasonal_periods,
                    )
                except Exception:
                    forecast = np.repeat(train_series[-1], self.H).astype(np.float32)

                preds[sample_idx, :, roi_idx] = forecast

        return preds.reshape(X.shape[0], -1)


def exponential_smoothing_generator(H=1, trend=None, seasonal=None, seasonal_periods=None):
    """
    Create a fresh exponential-smoothing forecaster for one CV fold.

    The returned object follows the estimator interface expected by
    ``utils.training.run_loso_cv`` but performs per-window fitting at inference
    time rather than global model training.
    """
    return ExponentialSmoothingForecaster(
        H=H,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    )

