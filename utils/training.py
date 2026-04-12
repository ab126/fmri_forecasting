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

    HuberLoss  — robust to outliers, less conservative than MSE on spikes.
    Delta term — penalizes wrong direction predictions, forces model
                 to learn sudden changes, not just mean-level signal.

    alpha controls how much the direction penalty contributes.
    delta controls the Huber threshold (below=MSE, above=MAE behavior).
    """
    def __init__(self, alpha=0.3, delta=0.5):
        super().__init__()
        self.alpha = alpha
        self.base  = nn.HuberLoss(delta=delta)

    def forward(self, pred, target):
        base_loss    = self.base(pred, target)

        # Change between consecutive forecast steps
        pred_delta   = pred[:, 1:, :]   - pred[:, :-1, :]
        target_delta = target[:, 1:, :] - target[:, :-1, :]
        delta_loss   = self.base(pred_delta, target_delta)

        return base_loss + self.alpha * delta_loss


def train_model(model, train_loader, val_loader=None, num_epochs=30, device=None, patience=5):
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    criterion = DeltaAwareLoss(alpha=0.3, delta=0.5)

    best_val_loss    = float('inf')
    patience_counter = 0
    best_state       = None

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",
                    leave=False)
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

        # --- Validation + Early Stopping ---
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += criterion(
                        model(xb.to(device)), yb.to(device)).item()
            val_loss /= len(val_loader)

            print(f"  Epoch {epoch+1:2d} | train: {avg_loss:.6f} "
                  f"| val: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_state       = {k: v.clone()
                                    for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"  Epoch {epoch+1:2d} | train loss: {avg_loss:.6f}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Best val loss: {best_val_loss:.6f} — weights restored")

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


def compute_eta(y_true, y_pred):
    """
    Information-theoretic eta: η = I(Y; Ŷ) / H(Y)  [Slide 11]

    Computed per ROI under Gaussian assumption, then averaged:
        I(Y; Ŷ) = -0.5 * log(1 - r²)       [nats]  — mutual information
        H(Y)    =  0.5 * log(2πe * σ²_Y)   [nats]  — signal entropy
        η       = I / H                              — fraction explained

    NOTE: This is NOT R² (explained variance = 1 - MSE/Var).
    R² can be negative; η is bounded and information-theoretically grounded.
    """
    # Flatten H and roi dims if 3D: (N, H, roi) → (N*H, roi)
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

        mi = -0.5 * np.log(1 - r ** 2)                        # I(Y; Ŷ)
        hy =  0.5 * np.log(2 * np.pi * np.e * (yt.var() + 1e-12))  # H(Y)
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


# Cross-validation
def run_loso_cv(dataset_raw, model_gen, M=20, H=3, stride=1,
                num_epochs=20, batch_size=512, device=device):
    """
    Leave-One-Subject-Out Cross Validation (LOSO-CV).

    Each run is normalized independently (run-level z-score).
    A fresh model is trained from scratch for every fold.
    10% of train windows used as validation set for early stopping.
    Tracks best eta model across all folds and returns it separately.

    Returns:
        df                — LOSO summary dataframe
        last_trained_model — model from the last fold
        last_X_test        — test windows from the last fold
        last_Y_test        — test targets from the last fold
        best_model         — model with highest eta across all folds
        best_X_test        — test windows of the best eta fold
        best_Y_test        — test targets of the best eta fold
    """

    subjects = sorted(set(d["subject"] for d in dataset_raw))
    n_subjects = len(subjects)

    print(f"\n{'='*60}")
    print(f"LOSO-CV | {n_subjects} subjects | M={M}, H={H}")
    print(f"{'='*60}")

    fold_results = []

    last_trained_model = None
    last_X_test        = None
    last_Y_test        = None

    best_eta_model     = None
    best_eta_score     = -float('inf')
    best_eta_subject   = None
    best_X_test        = None
    best_Y_test        = None

    for fold_i, test_subj in enumerate(tqdm(subjects, desc="LOSO Folds")):
        print(f"\nFold {fold_i+1}/{n_subjects} | Test subject: {test_subj}")

        # 1. Subject-level split
        train_items, test_items = split_by_subject(
            dataset_raw,
            test_subjects=[test_subj]
        )

        # 2. Run-level normalization (each run normalizes itself — no leakage)
        train_norm = normalize_items(train_items)
        test_norm  = normalize_items(test_items)

        # 3. Windowing
        X_tr, Y_tr = build_sliding_windows(train_norm, M, H, stride)
        X_te, Y_te = build_sliding_windows(test_norm,  M, H, stride)

        if len(X_tr) == 0 or len(X_te) == 0:
            print("Skipping fold (no valid windows)")
            continue

        # 4. Split train into train/val (90/10) for early stopping
        val_split    = int(len(X_tr) * 0.9)
        X_val, Y_val = X_tr[val_split:], Y_tr[val_split:]
        X_tr,  Y_tr  = X_tr[:val_split],  Y_tr[:val_split]

        n_roi = X_tr.shape[2]
        print(f"Train windows: {len(X_tr)} | Val windows: {len(X_val)} "
              f"| Test windows: {len(X_te)} | ROIs: {n_roi}")

        # 5. DataLoaders
        train_loader = DataLoader(
            FMRIWindowDataset(X_tr, Y_tr),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            FMRIWindowDataset(X_val, Y_val),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
        test_loader = DataLoader(
            FMRIWindowDataset(X_te, Y_te),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        # 6. Assign model
        model = model_gen().to(device)

        # 7. Train — delegates to train_model() from STEP 7
        print(f"Training (max {num_epochs} epochs, early stopping patience=5)...")
        model = train_model(
            model,
            train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            device=device,
            patience=5
        )

        # 8. Evaluate — delegates to get_predictions() from STEP 8
        all_preds, all_targets = get_predictions(model, test_loader, device)

        lstm_r  = compute_rmse(all_targets, all_preds)
        naive_r = compute_naive_rmse(X_te, Y_te)
        eta     = compute_eta(all_targets, all_preds)

        print(f"\nResults:")
        print(f"  LSTM RMSE  : {lstm_r:.6f}")
        print(f"  Naive RMSE : {naive_r:.6f}")
        print(f"  eta        : {eta:.4f}")
        print(f"  Beat naive : {'YES' if lstm_r < naive_r else 'NO'}")

        horizon_rmse(all_targets, all_preds)

        fold_results.append({
            "test_subject": test_subj,
            "LSTM_RMSE":  round(lstm_r,  6),
            "Naive_RMSE": round(naive_r, 6),
            "eta":        round(eta,     4),
            "beat_naive": lstm_r < naive_r,
        })

        # 9. Track last fold
        last_trained_model = model
        last_X_test        = X_te
        last_Y_test        = Y_te

        # 10. Track best eta fold
        if eta > best_eta_score:
            best_eta_score   = eta
            best_eta_subject = test_subj
            best_eta_model   = {k: v.clone()
                                for k, v in model.state_dict().items()}
            best_X_test      = X_te.copy()
            best_Y_test      = Y_te.copy()
            print(f"  New best eta model saved: {test_subj} (η={eta:.4f})")

        # 11. Memory cleanup
        del train_loader, val_loader, test_loader
        del X_tr, Y_tr, X_val, Y_val
        del train_norm, test_norm
        torch.cuda.empty_cache()

    # 12. Summary
    print(f"\n{'='*60}")
    print("LOSO-CV SUMMARY")
    print(f"{'='*60}")

    df = pd.DataFrame(fold_results)
    print(df.to_string(index=False))
    print(f"\nMean LSTM RMSE  : {df['LSTM_RMSE'].mean():.6f}")
    print(f"Mean Naive RMSE : {df['Naive_RMSE'].mean():.6f}")
    print(f"Mean eta        : {df['eta'].mean():.4f}")
    print(f"Folds beat naive: {df['beat_naive'].sum()} / {len(df)}")

    df.to_csv("loso_results.csv", index=False)
    print("\nResults saved to loso_results.csv")

    # 13. Restore best eta model
    print(f"\nBest eta fold : {best_eta_subject} (η={best_eta_score:.4f})")

    best_model = model_gen().to(device)
    if best_eta_model is not None:
        best_model.load_state_dict(best_eta_model)
    best_model.eval()

    return df, last_trained_model, last_X_test, last_Y_test, \
           best_model, best_X_test, best_Y_test

