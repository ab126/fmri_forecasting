"""
Core transformer forecasting pipeline for ROI-based fMRI time-series prediction.

This module keeps only the pieces needed to:
1. unzip / locate the dataset
2. collect a run-safe train/test split
3. load and preprocess runs
4. build sliding-window datasets
5. define the transformer model
6. train the model
7. generate predictions on a dataset

It intentionally leaves out:
- debugging / inspection cells
- plots
- statistics / comparison code
"""

from __future__ import annotations

import glob
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class TransformerConfig:
    zip_path: str = "/content/pooled_stratified_share.zip"
    extract_path: str = "/content"
    data_root_name: str = "pooled_stratified_share"

    target_roi: int = 18
    window_size: int = 30
    predict_step: int = 1
    batch_size: int = 256

    seed: int = 42
    train_fraction: float = 0.8

    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    learning_rate: float = 1e-3
    epochs: int = 5


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------

def unzip_dataset(zip_path: str, extract_path: str) -> str:
    """
    Unzip the dataset if needed and return the root folder path.
    """
    data_root = os.path.join(extract_path, "pooled_stratified_share")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Could not find dataset zip at: {zip_path}")

    if not os.path.exists(data_root):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    return data_root


def find_npz_files(data_root: str) -> List[str]:
    """
    Recursively collect all .npz run files.
    """
    npz_files = sorted(glob.glob(os.path.join(data_root, "**", "*.npz"), recursive=True))

    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found under: {data_root}")

    return npz_files


def make_run_split(
    npz_files: Sequence[str],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Create a run-safe train/test split by shuffling file indices.
    """
    run_ids = np.arange(len(npz_files))
    np.random.seed(seed)
    np.random.shuffle(run_ids)

    split_idx = int(train_fraction * len(run_ids))
    train_run_ids = run_ids[:split_idx]
    test_run_ids = run_ids[split_idx:]

    train_files = [npz_files[i] for i in train_run_ids]
    test_files = [npz_files[i] for i in test_run_ids]

    return train_files, test_files


def load_runs_from_filelist(
    file_list: Sequence[str],
    target_roi: int = 18,
    zscore: bool = True,
) -> Tuple[List[np.ndarray], List[Dict[str, object]]]:
    """
    Load all runs from a file list.

    Each .npz file is expected to contain:
        timeseries: shape (ROI, time)

    Returns:
        runs: list of arrays with shape (time, ROI)
        run_info: lightweight metadata for tracking
    """
    runs: List[np.ndarray] = []
    run_info: List[Dict[str, object]] = []

    for fpath in file_list:
        data = np.load(fpath)
        ts = data["timeseries"][:target_roi, :]  # (ROI, time)
        ts = ts.T  # -> (time, ROI)

        if zscore:
            mean = ts.mean(axis=0, keepdims=True)
            std = ts.std(axis=0, keepdims=True)
            ts = (ts - mean) / (std + 1e-8)

        runs.append(ts.astype(np.float32))
        run_info.append({"file": fpath, "shape": ts.shape})

    return runs, run_info


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class FMRIDataset(Dataset):
    """
    Convert a list of runs into many sliding-window samples.

    Input:
        runs: list of arrays with shape (time, ROI)

    Output sample:
        x: shape (window_size, ROI)
        y: shape (ROI,)
    """

    def __init__(self, runs: Sequence[np.ndarray], window_size: int = 30, predict_step: int = 1):
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        for run in runs:
            T = run.shape[0]
            for t in range(T - window_size - predict_step + 1):
                x = run[t : t + window_size]
                y = run[t + window_size + predict_step - 1]
                self.samples.append((x.astype(np.float32), y.astype(np.float32)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def build_dataloaders(
    train_runs: Sequence[np.ndarray],
    test_runs: Sequence[np.ndarray],
    window_size: int = 30,
    predict_step: int = 1,
    batch_size: int = 256,
) -> Tuple[FMRIDataset, FMRIDataset, DataLoader, DataLoader]:
    """
    Build PyTorch datasets and dataloaders.
    """
    train_dataset = FMRIDataset(train_runs, window_size=window_size, predict_step=predict_step)
    test_dataset = FMRIDataset(test_runs, window_size=window_size, predict_step=predict_step)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class TransformerForecastModel(nn.Module):
    """
    Transformer-based model for one-step ROI forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch, window, ROI)
        """
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        x = self.output_proj(x)
        return x


def initialize_model(
    input_dim: int,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    device: str | None = None,
) -> Tuple[TransformerForecastModel, nn.Module, optim.Optimizer, torch.device]:
    """
    Initialize model, loss, optimizer, and device.
    """
    torch_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = TransformerForecastModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(torch_device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer, torch_device


# ---------------------------------------------------------------------
# Training / inference
# ---------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int = 5,
) -> List[float]:
    """
    Train the model and return average train loss per epoch.
    """
    epoch_losses: List[float] = []

    for _ in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_losses.append(running_loss / len(train_loader))

    return epoch_losses


def predict_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect predictions and ground truth from a loader.

    Returns:
        all_preds: shape (N_samples, ROI)
        all_truth: shape (N_samples, ROI)
    """
    model.eval()

    preds = []
    truth = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            preds.append(pred)
            truth.append(y.numpy())

    all_preds = np.concatenate(preds, axis=0)
    all_truth = np.concatenate(truth, axis=0)

    return all_preds, all_truth


def build_and_train_from_config(
    config: TransformerConfig,
) -> Dict[str, object]:
    """
    End-to-end helper for the core transformer pipeline.

    Returns a dictionary containing the main training objects and outputs.
    This is useful if you want a single entry point from a notebook.
    """
    data_root = unzip_dataset(config.zip_path, config.extract_path)
    npz_files = find_npz_files(data_root)
    train_files, test_files = make_run_split(
        npz_files=npz_files,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )

    train_runs, train_run_info = load_runs_from_filelist(
        train_files, target_roi=config.target_roi, zscore=True
    )
    test_runs, test_run_info = load_runs_from_filelist(
        test_files, target_roi=config.target_roi, zscore=True
    )

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(
        train_runs=train_runs,
        test_runs=test_runs,
        window_size=config.window_size,
        predict_step=config.predict_step,
        batch_size=config.batch_size,
    )

    input_dim = config.target_roi
    model, criterion, optimizer, device = initialize_model(
        input_dim=input_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
    )

    train_losses = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config.epochs,
    )

    return {
        "config": config,
        "data_root": data_root,
        "npz_files": npz_files,
        "train_files": train_files,
        "test_files": test_files,
        "train_runs": train_runs,
        "test_runs": test_runs,
        "train_run_info": train_run_info,
        "test_run_info": test_run_info,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "device": device,
        "train_losses": train_losses,
    }
