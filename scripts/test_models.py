"""
Evaluate cross-validated forecasting models on the subject-level holdout set.

This script recreates the initial train/holdout split used by
``scripts.cross_validation`` and evaluates saved CV checkpoints on the holdout
subjects. It also reports simple persistence and window-mean baselines so the
held-out scores are interpretable without rerunning LOSO-CV.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved cross-validation checkpoints on holdout subjects."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "train_pooled_stratified_share_vc",
        help="Dataset root containing subject/run .npz files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "saves" / "checkpoints",
        help="Directory containing *_best.pt CV checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "saves" / "test-results",
        help="Directory for holdout CSV outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Models to evaluate. Defaults to all registered models plus any "
            "discovered checkpoint models."
        ),
    )
    parser.add_argument("--M", type=int, default=50, help="Input window length.")
    parser.add_argument("--H", type=int, default=3, help="Forecast horizon.")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-window stride.")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Subject-level holdout ratio; must match the CV run.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for subject split; must match the CV run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for checkpoint inference.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Max epochs when fitting models without checkpoints.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience when fitting torch models without checkpoints.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha.")
    parser.add_argument("--d-model", type=int, default=64,
                        help="Transformer hidden dimension.")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Transformer attention heads.")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Transformer encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer dropout.")
    return parser.parse_args()


def _safe_torch_load(path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint_model_name(path: Path) -> str:
    name = path.name
    match = re.match(r"(.+?)_fold\d+_.+?_best\.pt$", name)
    if match:
        return match.group(1)
    if name.endswith("_best.pt"):
        return name[:-len("_best.pt")]
    return path.stem


def discover_checkpoints(checkpoint_dir: Path):
    """Return ``{model_name: [checkpoint_paths...]}`` for saved best checkpoints."""
    discovered = {}
    if not checkpoint_dir.exists():
        return discovered

    for path in sorted(checkpoint_dir.glob("*_best.pt")):
        discovered.setdefault(_checkpoint_model_name(path), []).append(path)
    return discovered


def _reshape_predictions(preds, target_shape):
    import numpy as np

    preds = np.asarray(preds, dtype=np.float32)
    if preds.shape == target_shape:
        return preds
    return preds.reshape(target_shape)


def evaluate_predictions(model_name, y_true, y_pred, source="holdout"):
    from utils.measures import compute_eta_gauss, compute_rmse, compute_rmsse

    return {
        "model": model_name,
        "source": source,
        "Model_RMSE": round(compute_rmse(y_true, y_pred), 6),
        "Model_RMSSE": round(compute_rmsse(y_true, y_pred), 6),
        "eta": round(compute_eta_gauss(y_true, y_pred), 4),
    }


def evaluate_estimator(
    name,
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size,
    device,
    num_epochs=30,
    patience=5,
):
    from utils.training import predict_forecasting_model, train_forecasting_model

    if hasattr(model, "to"):
        model = model.to(device)

    model = train_forecasting_model(
        model,
        x_train,
        y_train,
        batch_size=batch_size,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        verbose=False,
    )
    preds = predict_forecasting_model(
        model,
        x_test,
        batch_size=batch_size,
        device=device,
    )
    preds = _reshape_predictions(preds, y_test.shape)
    return evaluate_predictions(name, y_test, preds)


def write_csv(path, rows):
    """Write rows of dictionaries without requiring pandas at runtime."""
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(rows):
    """Compute the small model-level summary table used by the CLI output."""
    import math
    from collections import defaultdict

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    summary = []
    for model_name in sorted(grouped):
        model_rows = grouped[model_name]
        rmses = [float(row["Model_RMSE"]) for row in model_rows]
        naive_rmses = [float(row["Naive_RMSE"]) for row in model_rows]
        rmsses = [float(row["Model_RMSSE"]) for row in model_rows]
        etas = [float(row["eta"]) for row in model_rows]

        rmse_mean = sum(rmses) / len(rmses)
        if len(rmses) > 1:
            variance = sum((value - rmse_mean) ** 2 for value in rmses) / (len(rmses) - 1)
            rmse_std = math.sqrt(variance)
        else:
            rmse_std = float("nan")

        summary.append({
            "model": model_name,
            "mean_model_rmse": round(rmse_mean, 6),
            "std_model_rmse": round(rmse_std, 6) if not math.isnan(rmse_std) else "",
            "mean_naive_rmse": round(sum(naive_rmses) / len(naive_rmses), 6),
            "mean_model_rmsse": round(sum(rmsses) / len(rmsses), 6),
            "mean_eta": round(sum(etas) / len(etas), 4),
            "evaluations_beat_naive": sum(bool(row["beat_naive"]) for row in model_rows),
            "n_evaluations": len(model_rows),
        })

    return summary


def print_table(rows):
    if not rows:
        return

    columns = list(rows[0])
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = " ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print(" ".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def evaluate_checkpoint(
    model_name,
    checkpoint_path,
    model_factory,
    x_test,
    y_test,
    batch_size,
    device,
):
    from utils.training import predict_forecasting_model

    model = model_factory(n_roi=x_test.shape[2], M=x_test.shape[1], H=y_test.shape[1])
    if hasattr(model, "to"):
        model = model.to(device)

    checkpoint = _safe_torch_load(checkpoint_path, device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    if hasattr(model, "eval"):
        model.eval()

    preds = predict_forecasting_model(
        model,
        x_test,
        batch_size=batch_size,
        device=device,
    )
    preds = _reshape_predictions(preds, y_test.shape)
    row = evaluate_predictions(model_name, y_test, preds, source=checkpoint_path.name)
    row["checkpoint"] = str(checkpoint_path)
    row["epoch"] = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    row["val_loss"] = checkpoint.get("val_loss") if isinstance(checkpoint, dict) else None
    return row


def main():
    args = parse_args()

    import torch

    from models.naive_models import last_value_model_generator, mean_value_model_generator
    from scripts.cross_validation import build_model_registry
    from utils.parse_data import parse_dataset

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset and creating train/holdout split...")

    x_train, y_train, x_test, y_test, detected_device = parse_dataset(
        root_dir=args.data_dir,
        M=args.M,
        H=args.H,
        normalize=True,
        stride=args.stride,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        verbose=True,
    )
    device = detected_device if torch.cuda.is_available() else torch.device("cpu")
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError(
            "No valid windows for train or holdout split. Check M, H, stride, and data."
        )
    print(
        f"Train windows: {len(x_train)} | Holdout windows: {len(x_test)} "
        f"| ROIs: {x_test.shape[2]}"
    )

    registry = build_model_registry(
        alpha=args.alpha,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    evaluation_registry = {
        "naive": lambda n_roi=None, M=None, H=None: last_value_model_generator(H=H),
        "mean": lambda n_roi=None, M=None, H=None: mean_value_model_generator(H=H),
        **registry,
    }

    selected = args.models or sorted(set(evaluation_registry) | set(checkpoints))
    selected = list(dict.fromkeys(selected))
    if "naive" not in selected:
        selected.insert(0, "naive")

    available = set(evaluation_registry) | set(checkpoints)
    unknown = sorted(set(selected) - available)
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {sorted(available)}")

    rows = []
    naive_rmse = None

    for model_name in selected:
        model_checkpoints = checkpoints.get(model_name, [])
        if model_checkpoints:
            if model_name not in registry:
                print(
                    f"\nSkipping {model_name}: checkpoint(s) found, but no "
                    "registered factory can rebuild the model."
                )
                continue

            print(f"\nEvaluating {len(model_checkpoints)} checkpoint(s) for {model_name}")
            for checkpoint_path in model_checkpoints:
                rows.append(
                    evaluate_checkpoint(
                        model_name=model_name,
                        checkpoint_path=checkpoint_path,
                        model_factory=registry[model_name],
                        x_test=x_test,
                        y_test=y_test,
                        batch_size=args.batch_size,
                        device=device,
                    )
                )
            continue

        model_factory = evaluation_registry.get(model_name)
        if model_factory is None:
            print(f"\nSkipping {model_name}: no checkpoint found in {args.checkpoint_dir}.")
            continue

        print(f"\nFitting {model_name} on train split for holdout evaluation")
        row = evaluate_estimator(
            model_name,
            model_factory(n_roi=x_train.shape[2], M=args.M, H=args.H),
            x_train,
            y_train,
            x_test,
            y_test,
            batch_size=args.batch_size,
            device=device,
            num_epochs=args.epochs,
            patience=args.patience,
        )
        rows.append(row)
        if model_name == "naive":
            naive_rmse = row["Model_RMSE"]

    if naive_rmse is None:
        naive_rows = [row for row in rows if row["model"] == "naive"]
        if not naive_rows:
            raise ValueError("Naive baseline evaluation did not produce a result.")
        naive_rmse = naive_rows[0]["Model_RMSE"]

    for row in rows:
        row["Naive_RMSE"] = naive_rmse
        row["beat_naive"] = row["Model_RMSE"] < row["Naive_RMSE"]

    results_path = args.output_dir / "holdout_model_results.csv"
    write_csv(results_path, rows)

    summary = summarize_results(rows)
    summary_path = args.output_dir / "holdout_model_summary.csv"
    write_csv(summary_path, summary)

    print("\nHoldout results saved to:")
    print(f"  {results_path}")
    print(f"  {summary_path}")
    print("\nSummary:")
    print_table(summary)


if __name__ == "__main__":
    main()
