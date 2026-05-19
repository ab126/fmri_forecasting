"""
Linear regression model for fMRI ROI time-series forecasting.
"""

from sklearn.linear_model import Ridge


def linear_regression_generator(alpha: float = 1.0) -> Ridge:
    """Factory function to create a ridge regression model with the specified alpha."""
    return Ridge(alpha=alpha)

