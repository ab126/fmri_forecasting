import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import Dataset, DataLoader

from utils.training import FMRIWindowDataset

class AdvancedLSTM(nn.Module):
    """
    Multi-layer LSTM for multi-step fMRI BOLD forecasting.
    Takes a sequence of M past ROI vectors and predicts H future steps.
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
        # x shape: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)

        # Use the hidden state from the last LSTM layer
        last_h = h_n[-1]
        out = self.fc(last_h)

        # Reshape to (Batch, Horizon, ROI)
        return out.view(-1, self.output_horizon, self.input_size)

class FmriPredictorAPI(BaseEstimator, RegressorMixin):
    """
    Scikit-Learn compatible API for the fMRI LSTM model.
    Encapsulates preprocessing (tensor conversion) and inference.
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
    Caller for fresh model generator. Returns a new instance of the model.
    """

    return AdvancedLSTM(
            input_size=n_roi,
            output_horizon=H,
            hidden_size=512,
            dropout=0.5
        )
