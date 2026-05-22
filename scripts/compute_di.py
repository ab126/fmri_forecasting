import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compute Directional Index (DI) from fMRI forecasting model')
parser.add_argument('--method', type=str, default='flow', choices=['flow', 'hist'],
                    help='DI computation method (default: flow)')
parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'lstm', 'linear', 'exp'],
                    help='Forecasting model type (default: transformer)')
parser.add_argument('--reduction-mode', type=str, default='zero', choices=['zero', 'shuffle'],
                    help='Reduction mode for DI computation (default: zero)')
parser.add_argument('--window-size', type=int, default=50,
                    help='Window size M for time series (default: 50)')
parser.add_argument('--predict-step', type=int, default=3,
                    help='Prediction step H (default: 3)')
parser.add_argument('--batch-size', type=int, default=512,
                    help='Batch size for training (default: 512)')
parser.add_argument('--num-epochs', type=int, default=50,
                    help='Number of epochs for training (default: 50)')
parser.add_argument('--data-path', type=str, default='data/pooled_stratified_share_vc',
                    help='Path to data directory (default: data/pooled_stratified_share_vc)')

args = parser.parse_args()

# Parameters
method = args.method
model = args.model
reduction_mode = args.reduction_mode

from fmri_forecasting.utils.parse_data import parse_dataset
from fmri_forecasting.utils.training import train_forecasting_model
from fmri_forecasting.utils.measures import SecondOrderPredictorAPI, fit_histogram_proba_adapter, compute_di_from_log_prob_api

from fmri_forecasting.models.transformer.transformer import transformer_model_generator, TransformerPredictorAPI


data_path = Path(args.data_path)
X_train, Y_train, X_test, Y_test, device = parse_dataset(root_dir=data_path, M=args.window_size, H=args.predict_step, normalize=True, stride=1, test_ratio=0.2,
                                                         test_subjects=None, random_state=42, verbose=True)

# TODO: Add LSTM, Linear, Exponential smoothing models
# Build transformer
model = transformer_model_generator(
    n_roi=X_train.shape[2],
    M=50,
    H=3,
    d_model=64,
    nhead=4,
    num_layers=2,
    dropout=0.1,
)

model = model.to(device)

# Train Model
model = train_forecasting_model(
    model,
    X_train,
    Y_train,
    X_val=X_test,
    Y_val=Y_test,
    batch_size=args.batch_size,
    num_epochs=args.num_epochs,
    device=device,
)

# Combine Data
X = np.concatenate([X_train, X_test], axis=0)
Y = np.concatenate([Y_train, Y_test], axis=0)

# Pick DI Computation Method
if method=="flow":
    di_model = SecondOrderPredictorAPI(
        model_obj=model,
        device=device,
    )

    di_model.fit_residual_std(
        X_calib=X,
        Y_calib=Y,
        batch_size=args.batch_size,
    )
else: # hist
    # Wrap with API
    config = {
        "window_size": args.window_size,
        "predict_step": args.predict_step,
        "output_horizon": args.predict_step,
        "input_dim": X_train.shape[2],
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
    }

    model_api = TransformerPredictorAPI(
        model=model,
        config=config,
        stats={},
        device=device,
    )

    # Build histogram likelihood adapter
    di_model = fit_histogram_proba_adapter(
        base_model=model_api,
        X_calib=X,
        Y_calib=Y,
        n_bins=50,
        flatten_input=False,
    )


DI_flow = compute_di_from_log_prob_api(
    model=di_model,
    X=X,
    Y=Y,
    horizon_idx=0,
    reduction_mode=args.reduction_mode,
)


plt.matshow(np.abs(DI_flow) )
plt.colorbar()
plt.title("DI Flow")
plt.xlabel("Target ROI")
plt.ylabel("Source ROI")

fig = plt.gcf()
Path("saves").mkdir(exist_ok=True)

plt.savefig(Path("saves/di_flow.png"))

