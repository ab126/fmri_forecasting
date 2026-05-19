"""
Simple non-parametric forecasting baselines for fMRI ROI windows.

These estimators are compatible with the shared forecasting pipeline and expect
unflattened input windows shaped ``(N, M, ROI)``. They provide useful reference
models for interpreting whether learned forecasters beat basic persistence or
window-average behavior.
"""

import numpy as np


class LastValueForecaster:
    """
    Persistence baseline that repeats the final observed ROI frame.

    ``fit`` records the window and ROI dimensions but learns no parameters.
    ``predict`` returns an array shaped ``(N, H, ROI)`` where every forecast
    step equals ``X[:, -1, :]``.
    """

    expects_windowed_input = True
    
    def __init__(self, H=1):
        self.H = int(H)
        self.n_roi = None
        self.window_size = None

    def fit(self, X, y=None):
        """
        Validate and record input dimensions.

        Parameters
        ----------
        X:
            Input windows shaped ``(N, M, ROI)``.
        y:
            Optional targets shaped ``(N, H, ROI)``; accepted for estimator
            compatibility but not used.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"Expected X shape (N, M, ROI), got {X.shape}"
            )
        
        self.window_size = X.shape[1]
        self.n_roi = X.shape[-1]

        return self

    def predict(self, X):
        """Repeat the final frame of each input window for all ``H`` steps."""
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                f"Expected X shape (N, M, ROI), got {X.shape}"
            )

        last_frame = X[:, -1, :]  # (N, ROI)

        preds = np.repeat(
            last_frame[:, None, :],
            self.H,
            axis=1
        )

        return preds.astype(np.float32)

def last_value_model_generator(H=1):
    """Create a fresh last-value baseline for one evaluation or CV fold."""
    return LastValueForecaster(H=H)

class MeanValueForecaster:
    """
    Baseline that repeats the temporal mean of each input window.

    ``fit`` records dimensions but learns no parameters. ``predict`` computes
    the ROI-wise mean over the ``M`` input time points and repeats that frame
    for each of the ``H`` forecast steps.
    """
    expects_windowed_input = True

    def __init__(self, H=1):
        self.H = int(H)
        self.n_roi = None
        self.window_size = None

    def fit(self, X, y=None):
        """
        Validate and record input dimensions for estimator compatibility.

        ``y`` is accepted but unused because this baseline is fully determined
        by the input window at prediction time.
        """

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"Expected X shape (N, M, ROI), got {X.shape}"
            )

        self.window_size = X.shape[1]
        self.n_roi = X.shape[2]
        return self

    def predict(self, X):
        """Repeat each window's ROI-wise temporal mean for all forecast steps."""

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                f"Expected X shape (N, M, ROI), got {X.shape}"
            )

        mean_frame = X.mean(axis=1)  

        preds = np.repeat(
            mean_frame[:, None, :],
            self.H,
            axis=1
        )

        return preds.astype(np.float32)


def mean_value_model_generator(H=1):
    """Create a fresh mean-value baseline for one evaluation or CV fold."""
    return MeanValueForecaster(H=H)


