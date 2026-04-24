import os
import pickle
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class TransformerForecastModel(nn.Module):
    def __init__(self, input_dim, window_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.d_model = int(d_model)

        self.input_proj = nn.Linear(self.input_dim, self.d_model)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.window_size, self.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dropout=float(dropout),
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
        )

        self.output_layer = nn.Linear(self.d_model, self.input_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x_last = x[:, -1, :]
        out = self.output_layer(x_last)
        return out


class TransformerPredictorAPI:
    def __init__(
        self,
        model: TransformerForecastModel,
        config: Dict[str, Any],
        stats: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.config = config
        self.stats = stats if stats is not None else {}

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        self.window_size = int(config["window_size"])
        self.predict_step = int(config.get("predict_step", 1))
        self.input_dim = int(config["input_dim"])

    @classmethod
    def load(cls, model_path, config_path, stats_path=None, device=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        stats = {}
        if stats_path is not None:
            if not os.path.exists(stats_path):
                raise FileNotFoundError(f"Stats file not found: {stats_path}")
            with open(stats_path, "rb") as f:
                stats = pickle.load(f)

        model = TransformerForecastModel(
            input_dim=config["input_dim"],
            window_size=config["window_size"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )

        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        return cls(model=model, config=config, stats=stats, device=device)

    @classmethod
    def from_export_dir(
        cls,
        export_dir,
        model_filename="best_transformer_api.pth",
        config_filename="transformer_api_config.pkl",
        stats_filename="transformer_api_stats.pkl",
        device=None,
    ):
        return cls.load(
            model_path=os.path.join(export_dir, model_filename),
            config_path=os.path.join(export_dir, config_filename),
            stats_path=os.path.join(export_dir, stats_filename),
            device=device,
        )

    def _validate_input(self, X):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                f"X must have shape (N, window_size, input_dim), got {X.shape}"
            )
        if X.shape[1] != self.window_size:
            raise ValueError(
                f"Expected window_size {self.window_size}, got {X.shape[1]}"
            )
        if X.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected input_dim {self.input_dim}, got {X.shape[2]}"
            )

        return X

    def predict(self, X, batch_size=256):
        X = self._validate_input(X)

        preds = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch = torch.from_numpy(X[start:end]).to(self.device)
                out = self.model(batch)
                preds.append(out.cpu().numpy())

        return np.vstack(preds)

    def predict_proba(self, X, batch_size=256):
        mean = self.predict(X, batch_size=batch_size)

        residual_std = self.stats.get("residual_std", None)
        if residual_std is None:
            residual_std = np.ones((self.input_dim,), dtype=np.float32)

        residual_std = np.asarray(residual_std, dtype=np.float32)

        if residual_std.ndim == 0:
            residual_std = np.full((self.input_dim,), float(residual_std), dtype=np.float32)

        if residual_std.shape != (self.input_dim,):
            residual_std = np.ones((self.input_dim,), dtype=np.float32)

        std = np.tile(residual_std, (mean.shape[0], 1))

        return {
            "mean": mean,
            "std": std,
        }

    def summary(self):
        return {
            "window_size": self.window_size,
            "predict_step": self.predict_step,
            "input_dim": self.input_dim,
            "device": str(self.device),
            "config_keys": list(self.config.keys()),
            "stats_keys": list(self.stats.keys()),
        }
