import numpy as np
import matplotlib.pyplot as plt

import torch



def plot_single_roi_prediction(model, X_test, Y_test, device,
                                roi_idx=0, sample_idx=None,
                                align_mean=False):
    """
    Plot prediction vs ground truth for a single ROI.

    align_mean: if True, shifts predictions to match ground truth mean.
                WARNING: this artificially improves the visual appearance
                and should NOT be used when reporting results.
                Keep False for honest evaluation plots.
    """

    model.eval()

    if sample_idx is None:
        sample_idx = np.random.randint(len(X_test))

    with torch.no_grad():
        sample_x = torch.tensor(
            X_test[sample_idx:sample_idx + 1],
            dtype=torch.float32
        ).to(device)

        sample_y = Y_test[sample_idx]
        pred_y   = model(sample_x).cpu().numpy()[0]

        if align_mean:
            # Mean alignment: for exploration only, not for reporting
            print("WARNING: align_mean=True is enabled — plot is for exploration only.")
            pred_y = pred_y - pred_y.mean(axis=0) + sample_y.mean(axis=0)

    past_time   = np.arange(X_test.shape[1])
    future_time = np.arange(X_test.shape[1], X_test.shape[1] + Y_test.shape[1])

    plt.figure(figsize=(10, 5))

    plt.plot(past_time, X_test[sample_idx, :, roi_idx],
             label="Past Signal (Input)", linewidth=2, alpha=0.6)

    plt.plot(future_time, sample_y[:, roi_idx],
             "o-", label="Ground Truth", linewidth=2)

    plt.plot(future_time, pred_y[:, roi_idx],
             "o--", label="Prediction", linewidth=2)

    plt.axvline(x=X_test.shape[1] - 0.5, linestyle="--", label="Forecast Start")

    plt.title(f"ROI {roi_idx} | {Y_test.shape[1]}-step Forecast")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Signal")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.show()


def show_individual_roi_results(model, X_test, Y_test, device,
                                 roi_list=None, sample_idx=None):
    """
    Show separate prediction plots for each ROI in roi_list.
    Defaults to first 3 ROIs if roi_list is not provided.
    """

    if roi_list is None:
        n_roi = X_test.shape[2]
        roi_list = list(range(min(3, n_roi)))

    print("\nFINAL MODEL VISUALIZATION (INDIVIDUAL ROIs)")
    print("=" * 50)

    if sample_idx is None:
        sample_idx = np.random.randint(len(X_test))

    print(f"Using test sample index: {sample_idx}")

    for roi in roi_list:
        print(f"\nROI {roi}")
        plot_single_roi_prediction(
            model=model,
            X_test=X_test,
            Y_test=Y_test,
            device=device,
            roi_idx=roi,
            sample_idx=sample_idx,
            align_mean=False    # keep False for honest plots
        )

