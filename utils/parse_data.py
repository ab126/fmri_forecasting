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


# =========================
# OPTIMIZED DATA LOADING FUNCTIONS
# (Fixed ROI schema, no repeated schema search)
# =========================

# Paste the full most common ROI schema tuple here once you have it.
# Example:
# TARGET_SCHEMA = ('ROI_1', 'ROI_2', 'ROI_3', ...)
TARGET_SCHEMA = None

# Fallback: use ROI count if full schema is not pasted yet.
TARGET_ROI_COUNT = 19

# Set this to True when you paste the exact schema tuple above.
USE_FULL_SCHEMA = False


def load_dataset(data_dir):
    """
    Load raw timeseries without normalization.
    Uses a fixed ROI schema to avoid repeated full-dataset schema scans.
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
                    if USE_FULL_SCHEMA:
                        if TARGET_SCHEMA is None:
                            raise ValueError("TARGET_SCHEMA is None. Paste the full ROI schema tuple first.")
                        if roi_labels != TARGET_SCHEMA:
                            continue

                    # ROI-count fallback (faster, but less strict)
                    else:
                        if len(roi_labels) != TARGET_ROI_COUNT:
                            continue

                    ts = data["timeseries"].astype(np.float32)

                    if ts.ndim != 2:
                        continue

                    # Ensure shape is (T, ROI)
                    if ts.shape[0] < ts.shape[1]:
                        ts = ts.T

                    expected_roi_count = len(TARGET_SCHEMA) if (USE_FULL_SCHEMA and TARGET_SCHEMA is not None) else TARGET_ROI_COUNT
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


def parse_dataset(
    root_dir=None,
    M=50,
    H=3,
    normalize=True,
    stride=1,
    test_ratio=0.2,
    test_subjects=None,
    random_state=42,
    verbose=True
):
    """
    Main function to load, split, normalize, window, and return train/test data
    for BOLD signal forecasting.

    Important order:
        1. Load raw runs
        2. Split by subject
        3. Normalize train/test groups separately
        4. Build sliding windows separately

    This order avoids train/test leakage and keeps the test subjects fully held out.
    """

    dataset, device = load_dataset_main(root_dir=root_dir)

    if verbose:
        print("Splitting raw dataset by subject...")

    train_items, test_items = split_by_subject(
        dataset,
        test_ratio=test_ratio,
        test_subjects=test_subjects,
        random_state=random_state,
        verbose=verbose
    )

    if normalize:
        if verbose:
            print("Normalizing train and test groups separately...")

        train_items = normalize_items(train_items)
        test_items = normalize_items(test_items)
    else:
        if verbose:
            print("Skipping normalization...")

    if verbose:
        print("Building train sliding windows...")

    X_train, Y_train = build_sliding_windows(
        train_items,
        M=M,
        H=H,
        stride=stride
    )

    if verbose:
        print("Building test sliding windows...")

    X_test, Y_test = build_sliding_windows(
        test_items,
        M=M,
        H=H,
        stride=stride
    )
    
    if verbose:
        print(
            f"Final shapes - "
            f"Train: X={X_train.shape}, Y={Y_train.shape} | "
            f"Test: X={X_test.shape}, Y={Y_test.shape}"
        )

        if len(X_train) == 0:
            print("WARNING: X_train is empty. Check M, H, stride, or train data length.")

        if len(X_test) == 0:
            print("WARNING: X_test is empty. Check M, H, stride, or test data length.")

    return X_train, Y_train, X_test, Y_test, device

    

