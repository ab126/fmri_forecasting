import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def zscore_per_roi(timeseries):
    mean = timeseries.mean(axis=1, keepdims=True)
    std = timeseries.std(axis=1, keepdims=True)
    return (timeseries - mean) / (std + 1e-8)


def compute_scaled_mse(train_series, y_true, y_pred):
    """
    M5-style scaled error idea:
    scale by the in-sample naive one-step squared error.

    train_series: 1D training signal
    y_true: 1D true future values
    y_pred: 1D predicted future values
    """
    if len(train_series) < 2:
        return np.nan

    naive_diffs = np.diff(train_series)
    denom = np.mean(naive_diffs ** 2)

    if denom < 1e-12:
        return np.nan

    mse = np.mean((y_true - y_pred) ** 2)
    return mse / denom


def fit_exp_smoothing_and_forecast(
    train_series,
    forecast_steps,
    trend=None,
    seasonal=None,
    seasonal_periods=None,
):
    """
    Fit exponential smoothing to one 1D train series and forecast future steps.
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


class ExponentialSmoothingForecaster:
    """
    Forecasting API compatible with the shared LOSO pipeline.

    The model is stateless across samples: each prediction fits an
    exponential-smoothing model per ROI on the provided input window
    and forecasts the next H steps.
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
    """Factory function matching the rest of the framework."""
    return ExponentialSmoothingForecaster(
        H=H,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    )


def evaluate_one_run(file_path, window_size=30, horizon=1,
                     trend=None, seasonal=None, seasonal_periods=None,
                     min_rois=18):
    """
    Legacy single-run evaluation helper kept for notebook compatibility.
    """
    data = np.load(file_path, allow_pickle=True)
    ts = data["timeseries"][:min_rois]

    ts = zscore_per_roi(ts)

    num_rois, T = ts.shape

    train_end = T - horizon
    if train_end <= 5:
        raise ValueError(f"Run too short for forecasting: {file_path}")

    all_true = []
    all_pred = []
    all_mse = []
    all_scaled_mse = []

    for roi_idx in range(num_rois):
        roi_series = ts[roi_idx]

        train_series = roi_series[:train_end]
        test_true = roi_series[train_end: train_end + horizon]

        try:
            test_pred, _ = fit_exp_smoothing_and_forecast(
                train_series=train_series,
                forecast_steps=horizon,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
            )
        except Exception:
            test_pred = np.repeat(train_series[-1], horizon).astype(np.float32)

        mse = np.mean((test_true - test_pred) ** 2)
        scaled_mse = compute_scaled_mse(train_series, test_true, test_pred)

        all_true.append(test_true)
        all_pred.append(test_pred)
        all_mse.append(mse)
        all_scaled_mse.append(scaled_mse)

    return {
        "y_true": np.array(all_true),
        "y_pred": np.array(all_pred),
        "roi_mse": np.array(all_mse),
        "roi_scaled_mse": np.array(all_scaled_mse),
    }
