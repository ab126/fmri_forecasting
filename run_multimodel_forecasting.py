#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.parse_data import (
    load_dataset,
    split_by_subject,
    normalize_items,
    build_sliding_windows,
)

from utils.training import (
    FMRIWindowDataset,
    DeltaAwareLoss,
    train_forecasting_model,
    predict_forecasting_model,
    compute_rmse,
    compute_naive_rmse,
    compute_rmsse,
    compute_eta,
    horizon_rmse,
)

from models.linear_regression.linear_regression_core import linear_regression_generator
from models.lstm.lstm_model_library import alstm_model_generator
from models.transformer.transformer_api_library import transformer_model_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Run fMRI forecasting models.")

    parser.add_argument("--root-dir", type=str, default="data/pooled_stratified_share_vc")
    parser.add_argument("--M", type=int, default=50)
    parser.add_argument("--H", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--results-dir", type=str, default="results_vc")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_vc")

    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear_regression", "lstm", "transformer"],
        choices=["linear_regression", "lstm", "transformer"],
    )

    return parser.parse_args()


def safe_name(name):
    return name.lower().replace(" ", "_").replace("/", "_")


def is_torch_model(model):
    return isinstance(model, nn.Module)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    return device


def load_and_prepare_data(args):
    root_dir = Path(args.root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root_dir}")

    print("=" * 80)
    print("Loading raw dataset")
    print("=" * 80)

    start = time.time()
    dataset = load_dataset(root_dir)
    print(f"Loaded {len(dataset)} runs in {time.time() - start:.2f} seconds")

    print("\nSplitting dataset by subject")
    train_items, test_items = split_by_subject(
        dataset,
        test_ratio=args.test_ratio,
        test_subjects=None,
        random_state=args.random_state,
        verbose=True,
    )

    print("\nNormalizing train and test groups separately")
    train_items = normalize_items(train_items)
    test_items = normalize_items(test_items)

    print("\nBuilding sliding windows")
    X_train, Y_train = build_sliding_windows(
        train_items,
        M=args.M,
        H=args.H,
        stride=args.stride,
    )

    X_test, Y_test = build_sliding_windows(
        test_items,
        M=args.M,
        H=args.H,
        stride=args.stride,
    )

    print("\nFinal data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"Y_train: {Y_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"Y_test : {Y_test.shape}")

    if len(X_train) == 0:
        raise ValueError("X_train is empty. Check M, H, stride, or dataset length.")

    if len(X_test) == 0:
        raise ValueError("X_test is empty. Check M, H, stride, or dataset length.")

    return X_train, Y_train, X_test, Y_test


def make_train_val_split(X_train, Y_train, val_ratio=0.1):
    split_idx = int(len(X_train) * (1 - val_ratio))

    X_tr = X_train[:split_idx]
    Y_tr = Y_train[:split_idx]
    X_val = X_train[split_idx:]
    Y_val = Y_train[split_idx:]

    print("\nTrain/validation split:")
    print(f"Train subset: {X_tr.shape}, {Y_tr.shape}")
    print(f"Val subset  : {X_val.shape}, {Y_val.shape}")

    return X_tr, Y_tr, X_val, Y_val


def save_model_checkpoint(
    model,
    optimizer,
    epoch,
    model_name,
    train_loss,
    val_loss,
    args,
    n_roi,
    checkpoint_dir,
):
    ckpt_path = checkpoint_dir / f"{safe_name(model_name)}_epoch_{epoch:03d}.pt"

    checkpoint = {
        "model_name": model_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "M": args.M,
        "H": args.H,
        "stride": args.stride,
        "n_roi": n_roi,
    }

    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")


def train_torch_model_with_checkpoints(
    model,
    model_name,
    X_tr,
    Y_tr,
    X_val,
    Y_val,
    args,
    device,
    n_roi,
    checkpoint_dir,
):
    model = model.to(device)

    train_loader = DataLoader(
        FMRIWindowDataset(X_tr, Y_tr),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        FMRIWindowDataset(X_val, Y_val),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    criterion = DeltaAwareLoss(alpha=0.3, delta=0.5)

    best_val_loss = float("inf")
    best_state = None

    print(f"\nTraining {model_name} for {args.epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, Y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                Y_batch = Y_batch.to(device, non_blocking=True)

                preds = model(X_batch)
                loss = criterion(preds, Y_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train: {avg_train_loss:.6f} | "
            f"val: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch % args.checkpoint_every == 0:
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                model_name=model_name,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                args=args,
                n_roi=n_roi,
                checkpoint_dir=checkpoint_dir,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best validation weights restored. Best val loss: {best_val_loss:.6f}")

    final_path = checkpoint_dir / f"{safe_name(model_name)}_final.pt"
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "M": args.M,
            "H": args.H,
            "stride": args.stride,
            "n_roi": n_roi,
            "best_val_loss": float(best_val_loss),
        },
        final_path,
    )
    print(f"Final PyTorch model saved: {final_path}")

    return model


def build_model_configs(args, n_roi):
    configs = []

    if "linear_regression" in args.models:
        configs.append(
            {
                "name": "Linear Regression",
                "model": linear_regression_generator(alpha=1.0),
            }
        )

    if "lstm" in args.models:
        configs.append(
            {
                "name": "LSTM",
                "model": alstm_model_generator(n_roi=n_roi, H=args.H),
            }
        )

    if "transformer" in args.models:
        configs.append(
            {
                "name": "Transformer",
                "model": transformer_model_generator(
                    n_roi=n_roi,
                    M=args.M,
                    H=args.H,
                    d_model=64,
                    nhead=4,
                    num_layers=2,
                    dropout=0.1,
                ),
            }
        )

    return configs


def evaluate_and_save_model(
    model_name,
    model,
    X_test,
    Y_test,
    X_train,
    args,
    device,
    results_dir,
    train_time_minutes,
):
    print(f"\nEvaluating {model_name}")

    preds, targets = predict_forecasting_model(
        model=model,
        X=X_test,
        Y=Y_test,
        batch_size=args.batch_size,
        device=device,
    )

    model_rmse = compute_rmse(targets, preds)
    naive_rmse = compute_naive_rmse(X_test, Y_test)
    model_rmsse = compute_rmsse(targets, preds, X_train)
    eta = compute_eta(targets, preds)
    horizon_scores = horizon_rmse(targets, preds)

    result = {
        "model": model_name,
        "M": args.M,
        "H": args.H,
        "stride": args.stride,
        "num_epochs": args.epochs if is_torch_model(model) else None,
        "model_rmse": float(model_rmse),
        "naive_rmse": float(naive_rmse),
        "rmsse": float(model_rmsse),
        "eta": float(eta),
        "beat_naive": bool(model_rmse < naive_rmse),
        "horizon_rmse": json.dumps([float(x) for x in horizon_scores]),
        "train_time_minutes": float(train_time_minutes),
    }

    result_path = results_dir / f"{safe_name(model_name)}_results.csv"
    pd.DataFrame([result]).to_csv(result_path, index=False)

    print(f"Saved result: {result_path}")
    print(f"Model RMSE : {model_rmse:.6f}")
    print(f"Naive RMSE : {naive_rmse:.6f}")
    print(f"RMSSE      : {model_rmsse:.6f}")
    print(f"Eta        : {eta:.6f}")
    print(f"Beat naive?: {'YES' if model_rmse < naive_rmse else 'NO'}")

    return result


def train_one_model(
    cfg,
    X_train,
    Y_train,
    X_tr,
    Y_tr,
    X_val,
    Y_val,
    X_test,
    Y_test,
    args,
    device,
    n_roi,
    checkpoint_dir,
    results_dir,
):
    model_name = cfg["name"]
    model = cfg["model"]

    print("\n" + "=" * 80)
    print(f"Starting model: {model_name}")
    print("=" * 80)

    start = time.time()

    if is_torch_model(model):
        model = train_torch_model_with_checkpoints(
            model=model,
            model_name=model_name,
            X_tr=X_tr,
            Y_tr=Y_tr,
            X_val=X_val,
            Y_val=Y_val,
            args=args,
            device=device,
            n_roi=n_roi,
            checkpoint_dir=checkpoint_dir,
        )
    else:
        model = train_forecasting_model(
            model=model,
            X_train=X_train,
            Y_train=Y_train,
            device=device,
        )

        model_path = checkpoint_dir / f"{safe_name(model_name)}_model.joblib"
        joblib.dump(model, model_path)
        print(f"Saved sklearn model: {model_path}")

    train_time_minutes = (time.time() - start) / 60
    print(f"Finished training {model_name} in {train_time_minutes:.2f} minutes")

    return evaluate_and_save_model(
        model_name=model_name,
        model=model,
        X_test=X_test,
        Y_test=Y_test,
        X_train=X_train,
        args=args,
        device=device,
        results_dir=results_dir,
        train_time_minutes=train_time_minutes,
    )


def merge_results(results_dir):
    result_files = sorted(results_dir.glob("*_results.csv"))

    if not result_files:
        print("No result files found to merge.")
        return None

    print("\nMerging result files:")
    for f in result_files:
        print(f" - {f}")

    merged = pd.concat([pd.read_csv(f) for f in result_files], ignore_index=True)
    merged_path = results_dir / "all_model_results_merged.csv"
    merged.to_csv(merged_path, index=False)

    print(f"\nMerged results saved to: {merged_path}")
    print(merged)

    return merged


def main():
    args = parse_args()

    print("=" * 80)
    print("Multi-model fMRI forecasting run")
    print("=" * 80)

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    results_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    device = get_device()

    X_train, Y_train, X_test, Y_test = load_and_prepare_data(args)
    n_roi = X_train.shape[2]

    X_tr, Y_tr, X_val, Y_val = make_train_val_split(X_train, Y_train, val_ratio=0.1)

    model_configs = build_model_configs(args, n_roi)

    for cfg in model_configs:
        train_one_model(
            cfg=cfg,
            X_train=X_train,
            Y_train=Y_train,
            X_tr=X_tr,
            Y_tr=Y_tr,
            X_val=X_val,
            Y_val=Y_val,
            X_test=X_test,
            Y_test=Y_test,
            args=args,
            device=device,
            n_roi=n_roi,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
        )

    merge_results(results_dir)

    print("\nRun complete.")


if __name__ == "__main__":
    main()
