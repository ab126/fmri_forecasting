# fMRI Connectivity Forecasting

This repository contains code and notebooks for forecasting ROI-level fMRI time
series and derived functional connectivity measures from Natural Scenes Dataset
(NSD) scans.

The repository is intended to contain source code, notebooks used for the
reported analyses, and instructions needed to reproduce figures and tables. NSD
data are not included in this repository and must be downloaded separately under
the applicable NSD data-use terms.

## Repository Contents

- `models/`: sklearn-style forecasting model implementations.
- `utils/`: data loading, NSD processing, connectivity, training, validation,
  and plotting utilities.
- `scripts/`: command-line workflows for data fetching, cross-validation, and
  directed-information computation.
- `forecast_models.ipynb`: example forecasting workflow.
- `compute_di.ipynb`: directed-information analysis workflow.
- `quick_stats.ipynb` and `miscellaneous.ipynb`: supporting analysis notebooks.
- `test/`: basic tests for analysis utilities.

## Data Availability

This project uses the Natural Scenes Dataset (NSD). The large raw and derivative
NSD files are not archived in GitHub or Zenodo with this code release.

Before running the analysis, obtain NSD access and download the required data
separately. The helper utilities in `utils.nsd_utils` access the public NSD S3
layout using unsigned S3 requests where supported, and write local files under
`data/` by default.

Default local paths used by `NSDDataHandler`:

- downloads: `data/nsd_downloads/`
- derivatives: `data/nsd_derivatives/`
- templates and ROI atlases: `data/templates/`

## Setup

Create a Python environment and install the scientific Python dependencies used
by the codebase. Core packages used by the current utilities include `numpy`,
`pandas`, `scikit-learn`, `torch`, `matplotlib`, `tqdm`, `boto3`, `nibabel`,
`nilearn`, and `antspyx`/`ants`. Micromamba is recommended for a stable environment.

## Fetching and Preparing NSD Data

The basic entry point for NSD data access is `utils.nsd_utils.NSDDataHandler`.
It can list available runs, download functional BOLD runs, download the subject
anatomical scan, compute registration transforms, warp an ROI atlas, and extract
ROI-level time series.

Minimal example:

```python
from pathlib import Path

from utils.nsd_utils import NSDDataHandler

handler = NSDDataHandler()
subj = "subj01"

# Inspect available NSD time-series files for a subject.
runs = handler.list_runs(subj)
print(runs[:5])

# Download a small number of BOLD runs for testing the pipeline.
handler.download_runs(subj, n_runs=2, prefix="session", overwrite=False)

# Download the subject anatomical scan.
anat_path = handler.get_subject_anat_scan(subj)

# Compute a T1-to-EPI transform using one downloaded functional run.
handler.get_t1_to_epi_warp(subj)
```

To use a custom ROI atlas in MNI space, pass it at construction time and then
warp it into subject functional space before extracting ROI time series:

```python
from pathlib import Path

from utils.nsd_utils import NSDDataHandler

handler = NSDDataHandler()
atlas_path = Path(handler.template_dir) / "ROIs" / "visual_sphere_atlas_23.nii.gz"
handler = NSDDataHandler(roi_atlas_path=str(atlas_path))

subj = "subj01"

handler.download_runs(subj, n_runs=2, prefix="session", overwrite=False)
handler.get_t1_to_epi_warp(subj)
handler.get_warped_atlas(subj, warp_type="func")
handler.extract_all_roi_timeseries(subj, verbose=True)
```

The full scripted workflow in `scripts/fetch_data.py` loops over `subj01` to
`subj08`, downloads runs, computes transforms, warps the atlas, extracts ROI
time series, removes raw downloads, and writes encrypted train/test splits:

```bash
python scripts/fetch_data.py
```

Use `--overwrite` to re-download runs that already exist locally:

```bash
python scripts/fetch_data.py --overwrite
```

The resulting processed `.npz` files are expected by the training utilities.
For example, `utils.parse_data.load_dataset_main()` defaults to
`data/train_pooled_stratified_share`.

## Forecasting Usage

An example is given in `forecast_models.ipynb`. After building the sklearn
compatible model API, the typical steps are:

1. Load the dataset.

```python
from utils.parse_data import load_dataset_main

dataset, device = load_dataset_main()
```

2. Pick hyperparameters and set a model generator for cross-validation.

```python
n_roi, H = 19, 3

model_gen = lambda: alstm_model_generator(n_roi, H)
```

3. Train, cross-validate, or test using the `utils` modules.

```python
results_df, model, X_test, Y_test, \
best_model, best_X_test, best_Y_test = run_loso_cv(
    dataset_raw=dataset,
    model_gen=model_gen,
    M=50,
    H=3,
    stride=1,
    num_epochs=20,
    batch_size=512,
    device=device,
)
```

## Reproducing Figures and Tables

Use the notebooks as the top-level reproduction record:

- `forecast_models.ipynb` for model training, forecasting comparisons, and
  forecasting-related outputs.
- `compute_di.ipynb` for directed-information analyses.
- `quick_stats.ipynb` for supporting summary statistics.

Run notebooks from the repository root after preparing the processed NSD data in
the expected `data/` layout.

