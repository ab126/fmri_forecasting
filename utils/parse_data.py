import os, time
from pathlib import Path

import numpy as np
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm


def load_dataset(data_dir, TARGET_SCHEMA = None, TARGET_ROI_COUNT = None):
    """
    Load raw timeseries without normalization.
    Uses a fixed ROI schema to avoid repeated full-dataset schema scans.
        Example:
        TARGET_SCHEMA = ('ROI_1', 'ROI_2', 'ROI_3', ...)

        Fallback: use ROI count if full schema is not pasted yet.
        TARGET_ROI_COUNT = 19

        Set USE_FULL_SCHEMA to True when you paste the exact schema tuple above.

    Normalization will be applied after subject-level split.
    """
    dataset = []
    print("Scanning and loading dataset (optimized fixed schema mode)...")

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".npz"):
                continue

            file_path = os.path.join(root, f)

            try:
                relative_path = os.path.relpath(file_path, data_dir)
                subject_name = relative_path.split(os.sep)[0]

                with np.load(file_path, allow_pickle=True) as data:
                    if "roi_labels" not in data or "timeseries" not in data:
                        continue

                    roi_labels = tuple(map(str, data["roi_labels"].tolist()))

                    # Full schema match (recommended for research validity)
                    
                    if TARGET_SCHEMA is not None and TARGET_SCHEMA != roi_labels:
                        continue

                    # ROI-count fallback (faster, but less strict)
                    if TARGET_ROI_COUNT is not None and len(roi_labels) != TARGET_ROI_COUNT:
                        continue

                    ts = data["timeseries"].astype(np.float32)

                    if ts.ndim != 2:
                        continue

                    # Ensure shape is (T, ROI)
                    if ts.shape[0] < ts.shape[1]:
                        ts = ts.T
                    
                    # Additional check for ROI count if either schema or count is specified
                    if TARGET_SCHEMA is not None or TARGET_ROI_COUNT is not None:
                        expected_roi_count = len(TARGET_SCHEMA) if (TARGET_SCHEMA is not None) else TARGET_ROI_COUNT
                        if ts.shape[1] != expected_roi_count:
                            continue

                    dataset.append({
                        "timeseries": ts,
                        "subject": subject_name,
                        "roi_labels": roi_labels
                    })

            except Exception:
                continue

    subjects_found = sorted(set(d["subject"] for d in dataset))
    print(f"Loaded: {len(dataset)} runs from {len(subjects_found)} subjects")
    print(f"Subjects: {subjects_found}")

    return dataset


def normalize_items(items, eps=1e-8):
    """
    Apply run-level z-score normalization.

    Each run is normalized independently:
        - mean/std are computed per ROI within that run only
        - this avoids cross-subject distribution distortion
        - this is safer for BOLD forecasting under LOSO-CV
    """
    normalized = []

    for item in items:
        ts = item["timeseries"]

        mean = ts.mean(axis=0, keepdims=True).astype(np.float32)
        std = ts.std(axis=0, keepdims=True).astype(np.float32)
        std = np.maximum(std, eps)

        ts_norm = ((ts - mean) / std).astype(np.float32)

        normalized.append({
            **item,
            "timeseries": ts_norm,
            "run_mean": mean,
            "run_std": std
        })

    return normalized


def build_sliding_windows(data_list, M, H, stride=1):
    """
    Convert time series into supervised learning windows.

    Input:
        X -> (M, ROI)

    Target:
        Y -> (H, ROI)
    """
    X, Y = [], []

    for item in data_list:
        ts = item["timeseries"]

        if len(ts) < (M + H):
            continue

        for t in range(0, len(ts) - M - H + 1, stride):
            X.append(ts[t:t + M])
            Y.append(ts[t + M:t + M + H])

    if len(X) == 0:
        return np.empty((0, M, 0), dtype=np.float32), np.empty((0, H, 0), dtype=np.float32)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def split_by_subject(dataset, test_ratio=0.2, test_subjects=None, random_state=42, verbose=True):
    """
    Flexible subject split.

    Modes:
    1. LOSO mode:
        - provide test_subjects explicitly

    2. Random split mode:
        - use test_ratio
    """
    subjects = sorted(set(item["subject"] for item in dataset))

    if test_subjects is not None:
        train_subjects = [s for s in subjects if s not in test_subjects]
        test_subjects = list(test_subjects)
    else:
        rng = np.random.default_rng(random_state)
        rng.shuffle(subjects)

        split_idx = int(len(subjects) * (1 - test_ratio))
        train_subjects = subjects[:split_idx]
        test_subjects = subjects[split_idx:]

    train_items = [item for item in dataset if item["subject"] in train_subjects]
    test_items = [item for item in dataset if item["subject"] in test_subjects]
    
    if verbose:
        print(f"Train subjects: {train_subjects}")
        print(f"Test subjects : {test_subjects}")
        print(f"Train runs: {len(train_items)}")
        print(f"Test runs : {len(test_items)}")

    return train_items, test_items

# TODO: split_within_subjects() - within-subject forecasting


def load_dataset_main(root_dir=None):
    """
    Load the dataset given in session/run.npz organization.
    
    Scans the specified directory for .npz files containing ROI timeseries data,
    applies fixed ROI schema filtering, and returns the loaded dataset along with
    the appropriate computing device (CPU or GPU).
    
    Parameters
    ----------
    root_dir : str or Path, optional
        Path to the root directory containing the organized .npz files.
        If None, defaults to "train_data/pooled_stratified_share".
        Expected structure: root_dir/subject_name/run_name.npz
    
    Returns
    -------
    dataset : list of dict
        List of loaded timeseries runs. Each dict contains:
        - "timeseries": np.ndarray of shape (T, ROI), z-score normalized
        - "subject": str, subject identifier
        - "roi_labels": tuple, ROI names
    device : torch.device
        The appropriate device for model training (cuda if available, else cpu)
    
    Raises
    ------
    ValueError
        If the dataset folder does not exist.
    """

    if root_dir is None:
        root_dir = Path("data") / "train_pooled_stratified_share" 
    elif isinstance(root_dir, str):
        root_dir = Path(root_dir)

    if not os.path.exists(root_dir):
        raise ValueError(f"ERROR: Dataset folder not found: {root_dir}")

    print(f"Using dataset path: {root_dir}")

    # Load dataset directly using fixed schema settings from STEP 2

    print("Loading dataset with fixed schema settings...")
    start_load = time.time()
    dataset = load_dataset(root_dir)
    end_load = time.time()

    print(f"Successfully loaded runs: {len(dataset)}")
    print(f"Loading time: {end_load - start_load:.2f} seconds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    return dataset, device


def parse_dataset(root_dir=None, M=50, H=3, normalize=True, stride=1, test_ratio=0.2, test_subjects=None, random_state=42, verbose=True, phase_randomize=False, phase_randomize_seed=123):
    """
    Main function to load, normalize, window, and split the dataset for forecasting.
    
    This is the primary entry point for dataset preprocessing. It orchestrates loading,
    normalization, sliding window creation, and subject-based train/test splitting for
    BOLD signal forecasting models.
    
    Parameters
    ----------
    root_dir : str or Path, optional
        Path to the root directory containing .npz files. Defaults to
        "train_data/pooled_stratified_share" if None.
    M : int, default=50
        Context window size (number of past timepoints to use as input).
    H : int, default=3
        Horizon window size (number of future timepoints to predict).
    normalize : bool, default=True
        If True, apply run-level z-score normalization to each timeseries.
        Normalization is computed independently per run to avoid cross-subject
        distribution distortion.
    stride : int, default=1
        Stride for sliding window extraction. stride=1 creates overlapping windows.
    test_ratio : float, default=0.2
        Fraction of subjects to reserve for testing (0.0 to 1.0).
        Ignored if test_subjects is provided (LOSO mode).
    test_subjects : list or set, optional
        Explicit subject identifiers for testing (Leave-One-Subject-Out mode).
        If provided, overrides test_ratio. Remaining subjects used for training.
    random_state : int, default=42
        Random seed for reproducible subject splitting.
    verbose : bool, default=True
        If True, print detailed progress information during processing.
    phase_randomize : bool, default=False
        If True, apply phase randomization to the timeseries data for null model generation.
    phase_randomize_seed : int, default=123
        Random seed for phase randomization. Only used if phase_randomize is True.
        
    
    Returns
    -------
    X_train : np.ndarray
        Training input windows of shape (N_train, M, ROI).
    Y_train : np.ndarray
        Training target windows of shape (N_train, H, ROI).
    X_test : np.ndarray
        Testing input windows of shape (N_test, M, ROI).
    Y_test : np.ndarray
        Testing target windows of shape (N_test, H, ROI).
    device : torch.device
        The appropriate device for model training (cuda if available, else cpu).
    
    Notes
    -----
    - Normalization is applied per run (not globally), which is safer for LOSO-CV.
    - Subjects are split before windowing to prevent data leakage.
    - Windows with insufficient timepoints (len < M+H) are skipped.
    """
    dataset, device = load_dataset_main(root_dir=root_dir)

    if normalize:
        if verbose:
            print("Normalizing dataset...")
        normalized_data = normalize_items(dataset) # TODO: do normalization wihin train and test groups
    else:
        normalized_data = dataset

    if phase_randomize:
        if verbose:
            print("Applying phase randomization...")
        normalized_data = phase_randomize_dataset(
            normalized_data,
            seed=phase_randomize_seed,
            same_phase_across_rois=False
        )

    if verbose:
        print("Splitting by subject...")
    train_items, test_items = split_by_subject(normalized_data, test_ratio, test_subjects, random_state)

    if verbose:
        print("Building sliding windows...")
    X_train, Y_train = build_sliding_windows(train_items, M, H, stride)
    X_test, Y_test = build_sliding_windows(test_items, M, H, stride)

    if verbose:
        print(f"Final shapes - Train: {X_train.shape}, {Y_train.shape} | Test: {X_test.shape}, {Y_test.shape}")

    return X_train, Y_train, X_test, Y_test, device


# Phase Randomization
def phase_randomize_timeseries(ts, rng=None, same_phase_across_rois=False):
    """
    Phase-randomize a run-level ROI time series.

    Parameters
    ----------
    ts : array, shape (T, R)
        Time by ROI matrix.
    rng : np.random.Generator
    same_phase_across_rois : bool
        False: randomize each ROI independently.
        True : use the same random phase shifts across ROIs, preserving more cross-ROI phase structure.

    Returns
    -------
    ts_surr : array, shape (T, R)
        Phase-randomized surrogate with approximately preserved mean, variance,
        and power spectrum per ROI.
    """
    ts = np.asarray(ts, dtype=np.float64)

    if ts.ndim != 2:
        raise ValueError(f"Expected shape (T, R), got {ts.shape}")

    if rng is None:
        rng = np.random.default_rng(0)

    T, R = ts.shape

    # Remove mean before FFT, restore after inverse FFT
    mean = ts.mean(axis=0, keepdims=True)
    x = ts - mean

    # Real FFT along time
    Xf = np.fft.rfft(x, axis=0)
    amp = np.abs(Xf)

    n_freq = Xf.shape[0]

    # Random phases
    if same_phase_across_rois:
        phases = rng.uniform(0, 2 * np.pi, size=(n_freq, 1))
        phases = np.repeat(phases, R, axis=1)
    else:
        phases = rng.uniform(0, 2 * np.pi, size=(n_freq, R))

    # Preserve DC phase
    phases[0, :] = 0.0

    # Preserve Nyquist phase for even-length signals
    if T % 2 == 0:
        phases[-1, :] = 0.0

    Xf_surr = amp * np.exp(1j * phases)

    ts_surr = np.fft.irfft(Xf_surr, n=T, axis=0)
    ts_surr = ts_surr + mean

    return ts_surr.astype(np.float32)


def phase_randomize_dataset(dataset, seed=0, same_phase_across_rois=False):
    """
    Apply phase randomization to each run in the loaded dataset.
    Keeps subject labels and ROI labels unchanged.
    """
    rng = np.random.default_rng(seed)
    surrogate = []

    for item in dataset:
        new_item = copy.deepcopy(item)
        new_item["timeseries"] = phase_randomize_timeseries(
            item["timeseries"],
            rng=rng,
            same_phase_across_rois=same_phase_across_rois,
        )
        surrogate.append(new_item)

    return surrogate

