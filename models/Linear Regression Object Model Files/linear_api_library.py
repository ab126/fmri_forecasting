import joblib
import numpy as np

class LinearPredictorAPI:
    def __init__(self, bundle):
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]

        self.window_size = bundle["window_size"]
        self.num_rois = bundle["num_rois"]
        self.input_dim = bundle["input_dim"]
        self.residual_std = np.asarray(bundle["residual_std"], dtype=np.float32)

    @classmethod
    def load(cls, path):
        bundle = joblib.load(path)
        return cls(bundle)

    def _process_input(self, X):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        return X

    def predict(self, X):
        X = self._process_input(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        mean = self.predict(X)
        std = np.tile(self.residual_std, (mean.shape[0], 1))
        return {"mean": mean, "std": std}
