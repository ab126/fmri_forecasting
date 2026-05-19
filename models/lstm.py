"""
LSTM-based neural forecaster and inference wrapper for fMRI ROI windows.

``AdvancedLSTM`` is the trainable model used by ``utils.training.run_loso_cv``.
It consumes normalized sliding windows shaped ``(N, M, ROI)`` and predicts the
next ``H`` ROI frames shaped ``(N, H, ROI)``. ``FmriPredictorAPI`` is a
sklearn-compatible inference wrapper for already-trained LSTM models.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader

from utils.training import FMRIWindowDataset

class AdvancedLSTM(nn.Module):
    """
    Multi-layer LSTM for multi-step fMRI BOLD forecasting.

    Parameters
    ----------
    input_size:
        Number of ROI features per time point.
    hidden_size:
        LSTM hidden-state width.
    num_layers:
        Number of stacked recurrent layers.
    output_horizon:
        Number of future time points to predict.
    dropout:
        Dropout used between LSTM layers.

    Notes
    -----
    The forward pass expects ``x`` shaped ``(batch, M, input_size)`` and returns
    ``(batch, output_horizon, input_size)``. The final hidden state from the top
    LSTM layer is projected to the full forecast window.
    """
    def __init__(self, input_size, hidden_size=512, num_layers=3, output_horizon=5, dropout=0.5):
        super().__init__()
        self.output_horizon = output_horizon
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Maps the last hidden state to the full forecast window (H * ROI)
        self.fc = nn.Linear(hidden_size, output_horizon * input_size)

    def forward(self, x):
        """Run one batched forward pass from input windows to forecast windows."""
        # x shape: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)

        # Use the hidden state from the last LSTM layer
        last_h = h_n[-1]
        out = self.fc(last_h)

        # Reshape to (Batch, Horizon, ROI)
        return out.view(-1, self.output_horizon, self.input_size)

# TODO: Make it compatible with DI computation api
class FmriPredictorAPI(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible inference API for a trained ``AdvancedLSTM``.

    This wrapper is intended for downstream tools that expect ``fit`` and
    ``predict`` methods, such as directed-information utilities or notebooks.
    It does not train the wrapped model; training is handled by
    ``utils.training.train_forecasting_model`` / ``run_loso_cv``.
    """
    def __init__(self, model_obj=None, M=50, H=3, device='cpu'):
        self.model_obj = model_obj
        self.M = M
        self.H = H
        self.device = device

        if self.model_obj is not None:
            self.model_obj.to(self.device)
            self.model_obj.eval()

    def fit(self, X, y=None):
        """Exists for Scikit-Learn compatibility. No training performed here."""
        return self

    def predict(self, X, batch_size=512):
        """
        Performs inference on provided fMRI windows in batches.
        Input X: Numpy array of shape (N, M, ROI)
        Output: Numpy array of shape (N, H, ROI)
        """
        if self.model_obj is None:
            raise ValueError("Model object is not initialized.")

        self.model_obj.eval()

        inference_dataset = FMRIWindowDataset(X)
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        all_predictions = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch.to(self.device)
                predictions_batch = self.model_obj(x_batch)
                all_predictions.append(predictions_batch.cpu().numpy())

        return np.concatenate(all_predictions, axis=0)

def alstm_model_generator(n_roi, H):
    """
    Create a fresh ``AdvancedLSTM`` for one training or CV fold.

    Parameters
    ----------
    n_roi:
        Number of ROI features in each time point.
    H:
        Number of forecast steps to emit.
    """

    return AdvancedLSTM(
            input_size=n_roi,
            output_horizon=H,
            hidden_size=512,
            dropout=0.5
        )


