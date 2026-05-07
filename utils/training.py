import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.parse_data import split_by_subject, normalize_items, build_sliding_windows


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FMRIWindowDataset(Dataset):
    """PyTorch dataset for fMRI windowed sequences."""

    def __init__(self, X, Y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = None
        if Y is not None:
            self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.X[idx], self.Y[idx]
        return self.X[idx]


class DeltaAwareLoss(nn.Module):
    """
    Combined loss: HuberLoss + delta (change) penalty.

    HuberLoss is robust to outliers.
    Delta term penalizes wrong forecast dynamics between future steps.
    """

    def __init__(self, alpha=0.3, delta=0.5):
        super().__init__()
        self.alpha = alpha
        self.base = nn.HuberLoss(delta=delta)

    def forward(self, pred, target):
        base_loss = self.base(pred, target)

        pred_delta = pred[:, 1:, :] - pred[:, :-1, :]
        target_delta = target[:, 1:, :] - target[:, :-1, :]
        delta_loss = self.base(pred_delta, target_delta)

        return base_loss + self.alpha * delta_loss


def train_model(model, train_loader, val_loader=None, num_epochs=30, device=None, patience=5):
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    criterion = DeltaAwareLoss(alpha=0.3, delta=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for X_batch, Y_batch in loop:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), Y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += criterion(
                        model(xb.to(device)), yb.to(device)
                    ).item()
            val_loss /= len(val_loader)

            print(f"  Epoch {epoch+1:2d} | train: {avg_loss:.6f} | val: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"  Epoch {epoch+1:2d} | train loss: {avg_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Best val loss: {best_val_loss:.6f} | weights restored")

    return model


def get_predictions(model, loader, device):
    """Runs inference and returns concatenated predictions and targets."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb.to(device))
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_naive_rmse(X_test, Y_test):
    """Naive baseline: repeat last observed frame for all H future steps."""
    naive_preds = np.repeat(X_test[:, -1:, :], Y_test.shape[1], axis=1)
    return compute_rmse(Y_test, naive_preds)


def _is_torch_model(model):
    return isinstance(model, nn.Module)


def _flatten_model_inputs(X):
    """Convert windowed inputs (N, M, ROI) into tabular features for sklearn models."""
    return X.reshape(X.shape[0], -1)


def _flatten_model_targets(Y):
    """Convert forecasting targets into 2D multi-output targets for sklearn models."""
    return Y.reshape(Y.shape[0], -1)


def _reshape_predictions(preds, target_shape):
    """Restore flattened sklearn predictions back to forecasting shape."""
    preds = np.asarray(preds, dtype=np.float32)

    if preds.shape == target_shape:
        return preds

    if preds.ndim == 1:
        preds = preds[:, None]

    if preds.ndim == 2:
        return preds.reshape(target_shape)

    raise ValueError(
        f"Could not reshape predictions from {preds.shape} to {target_shape}"
    )


def _clone_model(model):
    """Best-effort clone for either PyTorch modules or sklearn estimators."""
    if _is_torch_model(model):
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return copy.deepcopy(model)


def train_forecasting_model(
    model,
    X_train,
    Y_train,
    X_val=None,
    Y_val=None,
    batch_size=512,
    num_epochs=30,
    device=None,
    patience=5,
):
    """
    Train either a PyTorch forecasting model or an sklearn-style estimator.
    """
    if _is_torch_model(model):
        train_loader = DataLoader(
            FMRIWindowDataset(X_train, Y_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        val_loader = None
        if X_val is not None and Y_val is not None and len(X_val) > 0:
            val_loader = DataLoader(
                FMRIWindowDataset(X_val, Y_val),
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
            )

        return train_model(
            model,
            train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            device=device,
            patience=patience,
        )

    if hasattr(model, "fit") and hasattr(model, "predict"):
        X_train_flat = _flatten_model_inputs(X_train)
        Y_train_flat = _flatten_model_targets(Y_train)
        model.fit(X_train_flat, Y_train_flat)
        return model

    raise TypeError(
        "Unsupported model type. Expected a torch.nn.Module or an estimator "
        "with fit/predict methods."
    )


def predict_forecasting_model(model, X, Y=None, batch_size=512, device=None):
    """Run inference for either a PyTorch forecasting model or an sklearn estimator."""
    if _is_torch_model(model):
        if Y is None:
            raise ValueError("Y is required for torch-model evaluation in this helper.")

        test_loader = DataLoader(
            FMRIWindowDataset(X, Y),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        return get_predictions(model, test_loader, device)

    if hasattr(model, "predict"):
        preds = model.predict(_flatten_model_inputs(X))
        if Y is not None:
            preds = _reshape_predictions(preds, Y.shape)
        return preds, Y

    raise TypeError(
        "Unsupported model type. Expected a torch.nn.Module or an estimator "
        "with a predict method."
    )


def compute_eta(y_true, y_pred):
    """
    Information-theoretic eta computed per ROI and averaged.
    """
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    n_roi = y_true.shape[1]
    etas = []

    for roi in range(n_roi):
        yt = y_true[:, roi]
        yp = y_pred[:, roi]

        if yt.std() < 1e-8 or yp.std() < 1e-8:
            continue

        r = np.corrcoef(yt, yp)[0, 1]
        r = np.clip(r, -1 + 1e-7, 1 - 1e-7)

        mi = -0.5 * np.log(1 - r ** 2)
        hy = 0.5 * np.log(2 * np.pi * np.e * (yt.var() + 1e-12))
        etas.append(mi / (hy + 1e-12))

    return float(np.nanmean(etas))


def horizon_rmse(y_true, y_pred):
    """Compute and print RMSE separately for each forecast step."""
    horizon_scores = []
    print("\nHorizon-wise RMSE:")
    for h in range(y_true.shape[1]):
        r = compute_rmse(y_true[:, h, :], y_pred[:, h, :])
        horizon_scores.append(r)
        print(f"  Step {h+1} RMSE: {r:.6f}")
    return horizon_scores


def run_loso_cv(dataset_raw, model_gen, M=20, H=3, stride=1,
                num_epochs=20, batch_size=512, device=device):
    """
    Leave-One-Subject-Out Cross Validation (LOSO-CV).

    Supports both:
    - PyTorch forecasting models with the existing training loop
    - sklearn-style estimators exposing fit(X, y) and predict(X)

    Returns:
        df                 - LOSO summary dataframe
        last_trained_model - model from the last fold
        last_X_test        - test windows from the last fold
        last_Y_test        - test targets from the last fold
        best_model         - model with highest eta across all folds
        best_X_test        - test windows of the best eta fold
        best_Y_test        - test targets of the best eta fold
    """

    subjects = sorted(set(d["subject"] for d in dataset_raw))
    n_subjects = len(subjects)

    print(f"\n{'='*60}")
    print(f"LOSO-CV | {n_subjects} subjects | M={M}, H={H}")
    print(f"{'='*60}")

    fold_results = []

    last_trained_model = None
    last_X_test = None
    last_Y_test = None

    best_eta_model = None
    best_eta_score = -float("inf")
    best_eta_subject = None
    best_X_test = None
    best_Y_test = None

    for fold_i, test_subj in enumerate(tqdm(subjects, desc="LOSO Folds")):
        print(f"\nFold {fold_i+1}/{n_subjects} | Test subject: {test_subj}")

        train_items, test_items = split_by_subject(
            dataset_raw,
            test_subjects=[test_subj]
        )

        train_norm = normalize_items(train_items)
        test_norm = normalize_items(test_items)

        X_tr, Y_tr = build_sliding_windows(train_norm, M, H, stride)
        X_te, Y_te = build_sliding_windows(test_norm, M, H, stride)

        if len(X_tr) == 0 or len(X_te) == 0:
            print("Skipping fold (no valid windows)")
            continue

        val_split = int(len(X_tr) * 0.9)
        X_val, Y_val = X_tr[val_split:], Y_tr[val_split:]
        X_tr, Y_tr = X_tr[:val_split], Y_tr[:val_split]

        n_roi = X_tr.shape[2]
        print(f"Train windows: {len(X_tr)} | Val windows: {len(X_val)} "
              f"| Test windows: {len(X_te)} | ROIs: {n_roi}")

        try:
            model = model_gen()
            if _is_torch_model(model):
                model = model.to(device)
        except AttributeError as e:
            print(f"Model is not CUDA compatible: {e}\nContinuing with CPU...")
            model = model_gen()

        print(f"Training (max {num_epochs} epochs, early stopping patience=5)...")
        model = train_forecasting_model(
            model,
            X_tr,
            Y_tr,
            X_val=X_val,
            Y_val=Y_val,
            batch_size=batch_size,
            num_epochs=num_epochs,
            device=device,
            patience=5,
        )

        all_preds, all_targets = predict_forecasting_model(
            model,
            X_te,
            Y=Y_te,
            batch_size=batch_size,
            device=device,
        )

        model_r = compute_rmse(all_targets, all_preds)
        naive_r = compute_naive_rmse(X_te, Y_te)
        eta = compute_eta(all_targets, all_preds)

        print(f"\nResults:")
        print(f"  MODEL RMSE  : {model_r:.6f}")
        print(f"  Naive RMSE : {naive_r:.6f}")
        print(f"  eta        : {eta:.4f}")
        print(f"  Beat naive : {'YES' if model_r < naive_r else 'NO'}")

        horizon_rmse(all_targets, all_preds)

        fold_results.append({
            "test_subject": test_subj,
            "Model_RMSE": round(model_r, 6),
            "Naive_RMSE": round(naive_r, 6),
            "eta": round(eta, 4),
            "beat_naive": model_r < naive_r,
        })

        last_trained_model = model
        last_X_test = X_te
        last_Y_test = Y_te

        if eta > best_eta_score:
            best_eta_score = eta
            best_eta_subject = test_subj
            best_eta_model = _clone_model(model)
            best_X_test = X_te.copy()
            best_Y_test = Y_te.copy()
            print(f"  New best eta model saved: {test_subj} (eta={eta:.4f})")

        del X_tr, Y_tr, X_val, Y_val
        del train_norm, test_norm
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("LOSO-CV SUMMARY")
    print(f"{'='*60}")

    df = pd.DataFrame(fold_results)
    print(df.to_string(index=False))
    print(f"\nMean Model RMSE  : {df['Model_RMSE'].mean():.6f}")
    print(f"Mean Naive RMSE : {df['Naive_RMSE'].mean():.6f}")
    print(f"Mean eta        : {df['eta'].mean():.4f}")
    print(f"Folds beat naive: {df['beat_naive'].sum()} / {len(df)}")

    df.to_csv("loso_results.csv", index=False)
    print("\nResults saved to loso_results.csv")

    print(f"\nBest eta fold : {best_eta_subject} (eta={best_eta_score:.4f})")

    best_model = model_gen()
    if _is_torch_model(best_model):
        best_model = best_model.to(device)
        if best_eta_model is not None:
            best_model.load_state_dict(best_eta_model)
        best_model.eval()
    elif best_eta_model is not None:
        best_model = best_eta_model

    return df, last_trained_model, last_X_test, last_Y_test, \
           best_model, best_X_test, best_Y_test
