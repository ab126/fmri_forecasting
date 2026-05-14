import numpy as np


class LastValueForecaster:
    """
    Predicts that all future values equal the final observed frame.

    Compatible with the shared forecasting pipeline.
    """

    expects_windowed_input = True
    
    def __init__(self, H=1):
        self.H = int(H)
        self.n_roi = None
        self.window_size = None

    def fit(self, X, y=None):
        """
        X expected shape:
            (N, M,  ROI)

        y expected shape:
            (N, H, ROI)
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
    return LastValueForecaster(H=H)

class MeanValueForecaster:
    """
    Predicts the within-window temporal mean for all future steps.
    """
    expects_windowed_input = True

    def __init__(self, H=1):
        self.H = int(H)
        self.n_roi = None
        self.window_size = None

    def fit(self, X, y=None):

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"Expected X shape (N, M, ROI), got {X.shape}"
            )

        self.window_size = X.shape[1]
        self.n_roi = X.shape[2]
        return self

    def predict(self, X):

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
    return MeanValueForecaster(H=H)


