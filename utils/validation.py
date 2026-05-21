
import torch
from pathlib import Path
import joblib
import numpy as np
from tqdm.auto import tqdm

from utils.training import train_forecasting_model, predict_forecasting_model
from utils.measures import compute_eta_gauss

from models.transformer import transformer_model_generator
from models.lstm import alstm_model_generator
from models.linear import linear_regression_generator
from models.exponential_smoothing import exponential_smoothing_generator

model_path = Path("saves") / "trained_model_weights" / "transformer_final_vc.pt"


def load_model_from_weights_save(weights_path, model_str=None, device=None):
    """Loads the model weights from the specified state save path."""

    if model_str is None:
        model_str = weights_path.stem
        model_str = model_str.split('_')[0]

    obj = torch.load(weights_path, map_location="cpu")

    if model_str == "transformer":
        model = transformer_model_generator(
            n_roi=obj['n_roi'],
            M=obj['M'],
            H=obj['H'],
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.1,
        )

        if device is not None:
            model.to(device)

        model.load_state_dict(obj["model_state_dict"])
        model.eval()
        best_val_loss = obj["best_val_loss"]

    elif model_str == "lstm":
        model = alstm_model_generator(n_roi=obj['n_roi'], H=obj['H'])

        if device is not None:
            model.to(device)

        model.load_state_dict(obj["model_state_dict"])
        model.eval()
        best_val_loss = obj["best_val_loss"]

    elif model_str == "linear":

        model = joblib.load(weights_path)

    elif model_str == "exponential_smoothing":

        model = exponential_smoothing_generator( # problem with optimization step
            H=obj['H'],
            trend="add",
            seasonal=None,
            seasonal_periods=None,
        )

    else:
        raise ValueError(f"Unknown model type: {model_str}")
    
    return model
    

def permutation_forecast_test(
    model_gen,
    X_train,
    Y_train,
    X_test,
    Y_test,
    n_perm=100,
    metric_fn=None,
    rng=None,
    device=None,
):
    if rng is None:
        rng = np.random.default_rng(0)
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    if metric_fn is None:
        metric_fn = compute_eta_gauss

    null_scores = []

    for k in tqdm(range(n_perm)):

        perm = rng.permutation(len(Y_train))
        # X_perm = X_train[perm, ...]
        Y_perm = Y_train[perm, ...]

        model = model_gen()
        if device is not None and hasattr(model, "to"):
            model.to(device)

        model = train_forecasting_model(
            model,
            X_train,
            Y_perm,
            device=device,
            verbose=False
        )

        preds = predict_forecasting_model(
            model,
            X_test,
            device=device
        )

        score = metric_fn(Y_test, preds)
        null_scores.append(score)

    return np.array(null_scores)


