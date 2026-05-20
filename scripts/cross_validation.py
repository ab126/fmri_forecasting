"""
Cross-validate all forecasting models on the training split.

The script first reserves a subject-level holdout split, then runs LOSO-CV only
over the train subjects. Torch model checkpoints are written under
``saves/checkpoints`` under the parent ``fmri_connectivity`` project root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_model_registry(alpha: float, d_model: int, nhead: int,
                         num_layers: int, dropout: float):
    """Return factories compatible with utils.training.run_loso_cv."""
    from models.exponential_smoothing import exponential_smoothing_generator
    from models.linear import linear_regression_generator
    from models.lstm import alstm_model_generator
    from models.transformer import transformer_model_generator

    return {
        "linear": lambda n_roi=None, M=None, H=None: linear_regression_generator(
            alpha=alpha
        ),
        "exponential_smoothing": (
            lambda n_roi=None, M=None, H=None: exponential_smoothing_generator(
                H=H,
                trend=None,
                seasonal=None,
                seasonal_periods=None,
            )
        ),
        "lstm": lambda n_roi=None, M=None, H=None: alstm_model_generator(
            n_roi=n_roi,
            H=H,
        ),
        "transformer": (
            lambda n_roi=None, M=None, H=None: transformer_model_generator(
                n_roi=n_roi,
                M=M,
                H=H,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
            )
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LOSO cross-validation for all forecasting models."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "train_pooled_stratified_share_vc",
        help="Dataset root containing subject/run .npz files.",
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models to run. Defaults to all four.")
    parser.add_argument("--M", type=int, default=50,
                        help="Input window length.")
    parser.add_argument("--H", type=int, default=3,
                        help="Forecast horizon.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Sliding-window stride.")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Subject-level holdout ratio before CV.")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Seed for the initial subject split.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max epochs for torch models.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for torch training/inference.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stopping patience for torch models.")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Optional periodic torch checkpoint interval.")
    parser.add_argument("--save-last", action="store_true",
                        help="Also save the last torch checkpoint per fold.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regression alpha.")
    parser.add_argument("--d-model", type=int, default=64,
                        help="Transformer hidden dimension.")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Transformer attention heads.")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Transformer encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer dropout.")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "saves" / "cv-results",
                        help="Directory for CV CSV outputs.")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "saves" / "checkpoints",
                        help="Directory for torch checkpoints.")
    return parser.parse_args()


def main():
    args = parse_args()

    import pandas as pd
    import torch

    from utils.parse_data import load_dataset_main, split_by_subject
    from utils.training import run_loso_cv
    from models.naive_models import last_value_model_generator

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset, detected_device = load_dataset_main(args.data_dir)
    device = detected_device if torch.cuda.is_available() else torch.device("cpu")

    train_items, holdout_items = split_by_subject( # 
        dataset,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        verbose=True,
    )
    holdout_subjects = sorted({item["subject"] for item in holdout_items})
    print(
        f"\nRunning LOSO-CV on {len(train_items)} train runs. "
        f"Held-out test subjects not used in CV: {holdout_subjects}"
    )

    registry = build_model_registry(
        alpha=args.alpha,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    selected_models = args.models or list(registry)
    unknown = sorted(set(selected_models) - set(registry) - {"naive"})
    if unknown:
        available = sorted(set(registry) | {"naive"})
        raise ValueError(f"Unknown model(s): {unknown}. Available: {available}")

    all_results = []

    print(f"\n{'#' * 72}")
    print("Cross-validating model: naive")
    print(f"{'#' * 72}")
    naive_df, *_ = run_loso_cv(
        dataset_raw=train_items,
        model_gen=lambda n_roi=None, M=None, H=None: last_value_model_generator(H=H),
        M=args.M,
        H=args.H,
        stride=args.stride,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        checkpoint_dir=None,
        checkpoint_prefix="naive",
        checkpoint_every=None,
        save_best=False,
        save_last=False,
        results_path=None,
        patience=args.patience,
        compute_naive_rmse=False,
    )
    naive_df["Naive_RMSE"] = naive_df["Model_RMSE"]
    naive_df["beat_naive"] = False
    naive_scores = naive_df[["test_subject", "Naive_RMSE"]].copy()
    naive_df.insert(0, "model", "naive")
    naive_results_path = args.output_dir / "naive_loso_results.csv"
    naive_df.to_csv(naive_results_path, index=False)
    all_results.append(naive_df)

    learned_models = [name for name in selected_models if name != "naive"]
    for model_name in learned_models:
        print(f"\n{'#' * 72}")
        print(f"Cross-validating model: {model_name}")
        print(f"{'#' * 72}")

        model_results_path = args.output_dir / f"{model_name}_loso_results.csv"
        df, *_ = run_loso_cv(
            dataset_raw=train_items,
            model_gen=registry[model_name],
            M=args.M,
            H=args.H,
            stride=args.stride,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_prefix=model_name,
            checkpoint_every=args.checkpoint_every,
            save_best=True,
            save_last=args.save_last,
            results_path=None,
            patience=args.patience,
            compute_naive_rmse=False,
        )
        df = df.drop(columns=["Naive_RMSE", "beat_naive"], errors="ignore")
        df = df.merge(naive_scores, on="test_subject", how="left")
        df["beat_naive"] = df["Model_RMSE"] < df["Naive_RMSE"]
        df.insert(0, "model", model_name)
        df.to_csv(model_results_path, index=False)
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined_path = args.output_dir / "all_models_loso_results.csv"
    combined.to_csv(combined_path, index=False)

    summary = combined.groupby("model", as_index=False).agg(
        mean_model_rmse=("Model_RMSE", "mean"),
        mean_naive_rmse=("Naive_RMSE", "mean"),
        mean_model_rmsse=("Model_RMSSE", "mean"),
        mean_eta=("eta", "mean"),
        folds_beat_naive=("beat_naive", "sum"),
        n_folds=("Model_RMSE", "count"),
    )
    summary_path = args.output_dir / "all_models_loso_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nCombined results saved to:")
    print(f"  {combined_path}")
    print(f"  {summary_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
