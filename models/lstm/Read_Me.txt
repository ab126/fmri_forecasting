BOLD FORWARD MODEL — USAGE GUIDE
=================================

REQUIREMENTS
------------
pip install torch numpy scikit-learn joblib


FILES
-----
You need 2 files:

    best_fmri_sklearn_api.joblib   — trained model weights
    LSTM_model_library.py          — model class definitions

Both files must be in the same directory.


BASIC USAGE
-----------

    import joblib

    # 1. Load the model
    model = joblib.load("best_fmri_sklearn_api.joblib")

    # 2. Run prediction
    Y_pred = model.predict(X)


INPUT / OUTPUT FORMAT
---------------------

    X : numpy.ndarray
        shape : (n_samples, 50, 19)
                 n_samples — number of windows to predict
                 50        — number of past TRs (lookback window)
                 19        — number of ROIs (BN19 atlas, same order)

    Y : numpy.ndarray
        shape : (n_samples, 5, 19)
                 n_samples — same as input
                 5         — number of future TRs predicted
                 19        — number of ROIs


FULL EXAMPLE
------------

    import joblib
    import numpy as np

    # Load model
    model = joblib.load("best_fmri_sklearn_api.joblib")

    # Example input — replace with your own data
    # shape: (n_samples, 50, 19)
    X = np.random.randn(10, 50, 19).astype(np.float32)

    # Predict
    Y_pred = model.predict(X)

    print(Y_pred.shape)   # (10, 5, 19)


SINGLE SAMPLE
-------------

    # Single window shape: (50, 19)
    x_single = X[0]                                  # (50, 19)
    y_single = model.predict(x_single[np.newaxis])   # (1, 5, 19)


DATA PREPARATION
----------------
The model was trained with run-level z-score normalization.
Apply the same normalization to your raw data before predicting:

    # ts shape: (T, 19) — raw timeseries of one run
    ts_norm = (ts - ts.mean(axis=0)) / (ts.std(axis=0) + 1e-8)

Then create 50-TR sliding windows:

    windows = []
    for t in range(0, len(ts_norm) - 50, 1):
        windows.append(ts_norm[t:t + 50])

    X = np.array(windows, dtype=np.float32)   # (n_windows, 50, 19)
    Y_pred = model.predict(X)


MODEL DETAILS
-------------
    Architecture : 3-layer LSTM
    Hidden size  : 512
    Lookback     : M = 50 TRs
    Horizon      : H = 5 TRs
    ROI atlas    : BN19 (19 ROIs)
    Training     : NSD dataset, 6 subjects, LOSO-CV
    Loss         : DeltaAwareLoss (HuberLoss + delta penalty)
    Best eta (η) : 0.1236 (best fold: subjxpYwO4azeZ)
    Mean eta (η) : 0.1108 across all 6 folds


TROUBLESHOOTING
---------------
AttributeError: Can't get attribute 'FmriPredictorAPI'
    → LSTM_model_library.py is not in the same directory as the .joblib file

ModuleNotFoundError: No module named 'torch'
    → pip install torch

ModuleNotFoundError: No module named 'joblib'
    → pip install joblib

ValueError: Model object is not initialized
    → joblib.load() failed — file may be corrupted or incomplete