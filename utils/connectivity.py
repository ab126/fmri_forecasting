import numpy as np
from sklearn.covariance import LedoitWolf


def compute_rs_graph(roi_ts, roi_names=None, method='signed', control_dict=None):
    """
    Compute the resting state graph from the BOLD timeseries.

    :param roi_ts: dictionary of roi_name: BOLD timeseries (T,)
    :param roi_names: ordered list of ROI names
    :param method:
        - 'rsfc' or 'signed': Pearson correlation
        - 'cov': covariance
        - 'pcov': partial covariance (requires control_dict)
    :param control_dict: dict of covariate_name: array (T,)
    :return: FC matrix (n_roi x n_roi)
    """

    # ---- 1. stack ROI time series in correct order ----
    ts_list = []
    for roi in roi_ts:
        if roi_names and roi not in roi_names:
            continue
        ts = np.asarray(roi_ts[roi]).squeeze()
        ts_list.append(ts)

    X = np.vstack(ts_list)  # shape: (n_roi, T)

    # ---- 2. optionally regress out covariates ----
    if method in ['pcov', 'plw']:
        if control_dict is None:
            raise ValueError("control_dict must be provided for partial methods")

        Z = np.column_stack([np.asarray(v).squeeze() for v in control_dict.values()])
        Z = np.column_stack([Z, np.ones(Z.shape[0])])  # intercept

        beta = np.linalg.lstsq(Z, X.T, rcond=None)[0]
        X = (X.T - Z @ beta).T  # residuals

    # ---- 3. compute connectivity ----
    if method in ['rsfc', 'signed']:
        X_centered = X - X.mean(axis=1, keepdims=True)
        std = X_centered.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        X_norm = X_centered / std
        fc = (X_norm @ X_norm.T) / (X.shape[1] - 1)

    elif method in ['cov', 'pcov']:
        X_centered = X - X.mean(axis=1, keepdims=True)
        fc = (X_centered @ X_centered.T) / (X.shape[1] - 1)

    elif method == 'lw':
        # Ledoit-Wolf shrinkage covariance
        lw = LedoitWolf(store_precision=False, assume_centered=False)
        lw.fit(X.T)  # shape must be (T, n_roi)
        fc = lw.covariance_

    elif method == 'plw':
        # Partial Ledoit-Wolf (regress first, then shrinkage)
        lw = LedoitWolf(store_precision=False, assume_centered=False)
        lw.fit(X.T)
        fc = lw.covariance_

    else:
        raise ValueError(f"Unknown method: {method}")

    # ---- 4. enforce symmetry ----
    fc = (fc + fc.T) / 2

    return fc