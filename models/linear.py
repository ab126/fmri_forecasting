"""
Ridge-regression baseline for fMRI ROI time-series forecasting.

The shared training utilities flatten each input window from ``(N, M, ROI)`` to
``(N, M * ROI)`` and each target from ``(N, H, ROI)`` to ``(N, H * ROI)`` before
fitting this estimator. Predictions are reshaped back to the forecasting tensor
shape by ``utils.training.predict_forecasting_model`` / ``run_loso_cv``.
"""

from sklearn.linear_model import Ridge


def linear_regression_generator(alpha: float = 1.0) -> Ridge:
    """
    Create a fresh multi-output Ridge estimator for one CV fold.

    Parameters
    ----------
    alpha:
        L2 regularization strength passed directly to
        ``sklearn.linear_model.Ridge``.
    """
    return Ridge(alpha=alpha)

