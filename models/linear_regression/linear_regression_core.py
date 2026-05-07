"""
Linear regression baseline for fMRI ROI time-series forecasting.

This script keeps only the core pipeline:
1. unzip / locate dataset
2. collect .npz runs
3. determine a consistent ROI count
4. build sliding-window samples
5. create a run-safe train/test split
6. scale features
7. train a ridge regression model

It intentionally leaves out plotting, statistics, and testing-only cells.
"""

from __future__ import annotations

import argparse
import glob
import os
import zipfile
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class ForecastingData:
    X_all: np.ndarray
    Y_all: np.ndarray
    run_ids: np.ndarray
    min_rois: int


@dataclass
class TrainArtifacts:
    model: Ridge
    scaler: StandardScaler
    window_size: int
    horizon: int
    min_rois: int
    train_runs: np.ndarray
    test_runs: np.ndarray


def unzip_dataset(zip_path: str, extract_path: str) -> str:
    """Unzip the dataset if needed and return the extracted folder path."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Could not find zip file at {zip_path}. Please provide the correct path."
        )

    os.makedirs(extract_path, exist_ok=True)

    # Only unzip if the folder does not already contain npz files.
    existing_npz = glob.glob(os.path.join(extract_path, "**", "*.npz"), recursive=True)
    if not existing_npz:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    return extract_path


def find_npz_files(dataset_root: str) -> list[str]:
    """Recursively collect all run files."""
    npz_files = sorted(glob.glob(os.path.join(dataset_root, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise ValueError(f"No .npz files found under {dataset_root}")
    return npz_files


def zscore_per_roi(timeseries: np.ndarray) -> np.ndarray:
    """Z-score each ROI across time. Input shape: (num_rois, num_timepoints)."""
    mean = timeseries.mean(axis=1, keepdims=True)
    std = timeseries.std(axis=1, keepdims=True)
    return (timeseries - mean) / (std + 1e-8)


def make_windows_from_timeseries(
    timeseries: np.ndarray,
    window_size: int = 30,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert one run into supervised forecasting samples.

    Input timeseries shape:
        (num_rois, num_timepoints)

    Returns
    -------
    X : np.ndarray
        Shape (num_samples, window_size * num_rois)
    Y : np.ndarray
        Shape (num_samples, num_rois)
    """
    num_rois, num_timepoints = timeseries.shape
    num_samples = num_timepoints - window_size - horizon + 1

    if num_samples <= 0:
        return (
            np.empty((0, window_size * num_rois), dtype=np.float32),
            np.empty((0, num_rois), dtype=np.float32),
        )

    X, Y = [], []

    for start_idx in range(num_samples):
        end_idx = start_idx + window_size
        target_idx = end_idx + horizon - 1

        x_window = timeseries[:, start_idx:end_idx]  # (num_rois, window_size)
        y_target = timeseries[:, target_idx]         # (num_rois,)

        X.append(x_window.reshape(-1))
        Y.append(y_target)

    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def determine_min_rois(npz_files: list[str]) -> int:
    """Find the minimum ROI count across all runs so feature dimensions stay consistent."""
    roi_counts = []
    for file_path in npz_files:
        data = np.load(file_path, allow_pickle=True)
        roi_counts.append(data["timeseries"].shape[0])
    return int(np.min(roi_counts))


def build_forecasting_dataset(
    npz_files: list[str],
    window_size: int = 30,
    horizon: int = 1,
) -> ForecastingData:
    """Load all runs, preprocess them, and build forecasting samples."""
    min_rois = determine_min_rois(npz_files)

    X_all, Y_all, run_ids = [], [], []

    for run_idx, file_path in enumerate(npz_files):
        data = np.load(file_path, allow_pickle=True)
        timeseries = data["timeseries"][:min_rois]
        timeseries = zscore_per_roi(timeseries)

        X_run, Y_run = make_windows_from_timeseries(
            timeseries=timeseries,
            window_size=window_size,
            horizon=horizon,
        )

        if len(X_run) == 0:
            continue

        X_all.append(X_run)
        Y_all.append(Y_run)
        run_ids.extend([run_idx] * len(X_run))

    if not X_all:
        raise ValueError("No forecasting samples were created. Check window size and data length.")

    return ForecastingData(
        X_all=np.vstack(X_all),
        Y_all=np.vstack(Y_all),
        run_ids=np.asarray(run_ids),
        min_rois=min_rois,
    )


def run_safe_train_test_split(
    X_all: np.ndarray,
    Y_all: np.ndarray,
    run_ids: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split at the run level so all samples from a run stay together.
    """
    unique_runs = np.unique(run_ids)
    rng = np.random.default_rng(seed)
    shuffled_runs = unique_runs.copy()
    rng.shuffle(shuffled_runs)

    split_idx = int((1.0 - test_fraction) * len(shuffled_runs))
    train_runs = shuffled_runs[:split_idx]
    test_runs = shuffled_runs[split_idx:]

    train_mask = np.isin(run_ids, train_runs)
    test_mask = np.isin(run_ids, test_runs)

    X_train = X_all[train_mask]
    Y_train = Y_all[train_mask]
    X_test = X_all[test_mask]
    Y_test = Y_all[test_mask]

    return X_train, Y_train, X_test, Y_test, train_runs, test_runs


def linear_regression_generator(alpha: float = 1.0) -> Ridge:
    """Factory function to create a ridge regression model with the specified alpha."""
    return Ridge(alpha=alpha)


def train_linear_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    alpha: float = 1.0,
) -> tuple[Ridge, StandardScaler]:
    """
    Scale features using training data only, then train a ridge regression model.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, Y_train)

    return model, scaler


def predict(model: Ridge, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Run inference with the fitted scaler + ridge model."""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def build_and_train_pipeline(
    zip_path: str,
    extract_path: str,
    window_size: int = 30,
    horizon: int = 1,
    alpha: float = 1.0,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[TrainArtifacts, tuple[np.ndarray, np.ndarray]]:
    """
    Full core pipeline. Returns trained artifacts and the held-out test arrays.
    """
    dataset_root = unzip_dataset(zip_path=zip_path, extract_path=extract_path)
    npz_files = find_npz_files(dataset_root)

    forecasting_data = build_forecasting_dataset(
        npz_files=npz_files,
        window_size=window_size,
        horizon=horizon,
    )

    X_train, Y_train, X_test, Y_test, train_runs, test_runs = run_safe_train_test_split(
        X_all=forecasting_data.X_all,
        Y_all=forecasting_data.Y_all,
        run_ids=forecasting_data.run_ids,
        test_fraction=test_fraction,
        seed=seed,
    )

    model, scaler = train_linear_model(
        X_train=X_train,
        Y_train=Y_train,
        alpha=alpha,
    )

    artifacts = TrainArtifacts(
        model=model,
        scaler=scaler,
        window_size=window_size,
        horizon=horizon,
        min_rois=forecasting_data.min_rois,
        train_runs=train_runs,
        test_runs=test_runs,
    )

    return artifacts, (X_test, Y_test)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ridge-regression baseline for fMRI forecasting."
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default="/content/pooled_stratified_share.zip",
        help="Path to the zipped dataset.",
    )
    parser.add_argument(
        "--extract_path",
        type=str,
        default="/content/pooled_stratified_share",
        help="Directory where the dataset should be extracted.",
    )
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifacts, (X_test, Y_test) = build_and_train_pipeline(
        zip_path=args.zip_path,
        extract_path=args.extract_path,
        window_size=args.window_size,
        horizon=args.horizon,
        alpha=args.alpha,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    print("Training complete.")
    print(f"Window size: {artifacts.window_size}")
    print(f"Horizon: {artifacts.horizon}")
    print(f"Number of ROIs used: {artifacts.min_rois}")
    print(f"Train runs: {len(artifacts.train_runs)}")
    print(f"Test runs: {len(artifacts.test_runs)}")
    print(f"Test sample matrix shape: {X_test.shape}")
    print(f"Test target matrix shape: {Y_test.shape}")


if __name__ == "__main__":
    main()
