import copy
import inspect
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .measures import compute_eta_gauss

from .parse_data import split_by_subject, normalize_items, build_sliding_windows


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


def train_model(
    model,
    train_loader,
    val_loader=None,
    num_epochs=30,
    device=None,
    patience=5,
    checkpoint_dir=None,
    checkpoint_prefix="forecast_model",
    checkpoint_every=None,
    save_best=True,
    save_last=False,
    verbose=True
):
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if type(checkpoint_dir) is str:
        checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience
    )

    criterion = DeltaAwareLoss(alpha=0.3, delta=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    def _save_checkpoint(path, epoch, train_loss, val_loss=None, is_best=False):
        if checkpoint_dir is None:
            return
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "is_best": is_best,
            },
            path,
        )

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

            if verbose:
                print(f"  Epoch {epoch+1:2d} | train: {avg_loss:.6f} | val: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0

                if save_best and checkpoint_dir is not None:
                    _save_checkpoint(
                        checkpoint_dir / f"{checkpoint_prefix}_best.pt",
                        epoch=epoch + 1,
                        train_loss=avg_loss,
                        val_loss=val_loss,
                        is_best=True,
                    )
            else:
                patience_counter += 1
                if verbose:
                    print(f"  No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
        else:
            if verbose:
                print(f"  Epoch {epoch+1:2d} | train loss: {avg_loss:.6f}")
            
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                if save_best and checkpoint_dir is not None:
                    _save_checkpoint(
                        checkpoint_dir / f"{checkpoint_prefix}_best.pt",
                        epoch=epoch + 1,
                        train_loss=avg_loss,
                        val_loss=None,
                        is_best=True,
                    )
        if (
            checkpoint_dir is not None
            and checkpoint_every is not None
            and checkpoint_every > 0
            and (epoch + 1) % checkpoint_every == 0
        ):
            _save_checkpoint(
                checkpoint_dir / f"{checkpoint_prefix}_epoch{epoch+1:03d}.pt",
                epoch=epoch + 1,
                train_loss=avg_loss,
                val_loss=val_loss if val_loader is not None else None,
                is_best=False,
            )
        
        if save_last and checkpoint_dir is not None:
            _save_checkpoint(
                checkpoint_dir / f"{checkpoint_prefix}_last.pt",
                epoch=epoch + 1,
                train_loss=avg_loss,
                val_loss=val_loss if val_loader is not None else None,
                is_best=False,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"  Best val loss: {best_val_loss:.6f} | weights restored")

    return model


def get_predictions(model, loader, device=None):
    """Runs inference and returns concatenated predictions."""
    
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                xb = batch[0]
            else:
                xb = batch

            xb = xb.to(device, non_blocking=True)

            preds = model(xb)
            all_preds.append(preds.detach().cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_naive_rmse(X_test, Y_test):
    """Naive baseline: repeat last observed frame for all H future steps."""
    naive_preds = np.repeat(X_test[:, -1:, :], Y_test.shape[1], axis=1)
    return compute_rmse(Y_test, naive_preds)


def compute_rmsse(y_true, y_pred): 
    """Root Mean Squared Scaled Error (RMSSE) for multi-step forecasts."""
    numerator = np.mean((y_true - y_pred) ** 2)
    denominator = np.mean((y_true[:, 1:, :] - y_true[:, :-1, :]) ** 2) + 1e-8
    return float(np.sqrt(numerator / denominator))


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


def _expects_windowed_input(model):
    return getattr(model, "expects_windowed_input", False)


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
    checkpoint_dir=None,
    checkpoint_prefix="forecast_model",
    checkpoint_every=None,
    save_best=True,
    save_last=False,
    verbose=True
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
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=checkpoint_prefix,
            checkpoint_every=checkpoint_every,
            save_best=save_best,
            save_last=save_last,
            verbose=verbose,
        )

    if hasattr(model, "fit") and hasattr(model, "predict"):
        if _expects_windowed_input(model):
            model.fit(X_train, Y_train)
        else:
            X_train_flat = _flatten_model_inputs(X_train)
            Y_train_flat = _flatten_model_targets(Y_train)
            model.fit(X_train_flat, Y_train_flat)
        return model

    raise TypeError(
        "Unsupported model type. Expected a torch.nn.Module or an estimator "
        "with fit/predict methods."
    )


def predict_forecasting_model(model, X, batch_size=512, device=None):
    """Run inference for either a PyTorch forecasting model or an sklearn estimator."""
    if _is_torch_model(model):
        test_loader = DataLoader(
            FMRIWindowDataset(X),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        return get_predictions(model, test_loader, device)

    if hasattr(model, "predict"):
        if _expects_windowed_input(model):
            preds = model.predict(X)
        else:
            preds = model.predict(_flatten_model_inputs(X))
        return preds

    raise TypeError(
        "Unsupported model type. Expected a torch.nn.Module or an estimator "
        "with a predict method."
    )


def _make_model(model_gen, n_roi=None, M=None, H=None):
    """
    Instantiate model_gen while preserving compatibility with zero-arg notebook
    lambdas and newer factories that accept fold dimensions.
    """
    try:
        signature = inspect.signature(model_gen)
    except (TypeError, ValueError):
        return model_gen()

    kwargs = {}
    for name in signature.parameters:
        if name in {"n_roi", "input_size", "input_dim"} and n_roi is not None:
            kwargs[name] = n_roi
        elif name in {"M", "window_size"} and M is not None:
            kwargs[name] = M
        elif name in {"H", "output_horizon", "horizon"} and H is not None:
            kwargs[name] = H

    if kwargs:
        return model_gen(**kwargs)
    return model_gen()


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
                num_epochs=20, batch_size=512, device=device,
                checkpoint_dir=None, checkpoint_prefix="forecast_model",
                checkpoint_every=None, save_best=True, save_last=False,
                results_path="loso_results.csv", patience=5):
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
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    best_n_roi = None

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
            model = _make_model(model_gen, n_roi=n_roi, M=M, H=H)
            if _is_torch_model(model):
                model = model.to(device)
        except AttributeError as e:
            print(f"Model is not CUDA compatible: {e}\nContinuing with CPU...")
            model = _make_model(model_gen, n_roi=n_roi, M=M, H=H)

        fold_checkpoint_dir = None
        if checkpoint_dir is not None and _is_torch_model(model):
            fold_checkpoint_dir = checkpoint_dir

        print(f"Training (max {num_epochs} epochs, early stopping patience={patience})...")
        model = train_forecasting_model(
            model,
            X_tr,
            Y_tr,
            X_val=X_val,
            Y_val=Y_val,
            batch_size=batch_size,
            num_epochs=num_epochs,
            device=device,
            patience=patience,
            checkpoint_dir=fold_checkpoint_dir,
            checkpoint_prefix=f"{checkpoint_prefix}_fold{fold_i+1:02d}_{test_subj}",
            checkpoint_every=checkpoint_every,
            save_best=save_best,
            save_last=save_last,
        )

        all_preds = predict_forecasting_model(
            model,
            X_te,
            batch_size=batch_size,
            device=device,
        )
        all_preds = _reshape_predictions(all_preds, Y_te.shape)
        all_targets = Y_te

        model_r = compute_rmse(all_targets, all_preds)
        naive_r = compute_naive_rmse(X_te, Y_te)
        eta = compute_eta_gauss(all_targets, all_preds)

        print(f"\nResults:")
        print(f"  MODEL RMSE  : {model_r:.6f}")
        print(f"  Naive RMSE : {naive_r:.6f}")
        print(f"  eta        : {eta:.4f}")
        print(f"  Beat naive : {'YES' if model_r < naive_r else 'NO'}")

        hor_rmse = horizon_rmse(all_targets, all_preds)

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
            best_n_roi = n_roi
            print(f"  New best eta model saved: {test_subj} (eta={eta:.4f})")

        del X_tr, Y_tr, X_val, Y_val
        del train_norm, test_norm
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("LOSO-CV SUMMARY")
    print(f"{'='*60}")

    df = pd.DataFrame(fold_results)
    if df.empty:
        raise ValueError(
            "LOSO-CV produced no valid folds. Check subject count and window "
            f"settings M={M}, H={H}, stride={stride}."
        )
    print(df.to_string(index=False))
    print(f"\nMean Model RMSE  : {df['Model_RMSE'].mean():.6f}")
    print(f"Mean Naive RMSE : {df['Naive_RMSE'].mean():.6f}")
    print(f"Mean eta        : {df['eta'].mean():.4f}")
    print(f"Folds beat naive: {df['beat_naive'].sum()} / {len(df)}")

    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")

    print(f"\nBest eta fold : {best_eta_subject} (eta={best_eta_score:.4f})")

    best_model = _make_model(model_gen, n_roi=best_n_roi, M=M, H=H)
    if _is_torch_model(best_model):
        best_model = best_model.to(device)
        if best_eta_model is not None:
            best_model.load_state_dict(best_eta_model)
        best_model.eval()
    elif best_eta_model is not None:
        best_model = best_eta_model

    return df, last_trained_model, last_X_test, last_Y_test, \
           best_model, best_X_test, best_Y_test


