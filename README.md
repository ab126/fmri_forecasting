



# Usage 

An exaple is given in [forecast_models.ipynb](forecast_models.ipynb). After building the sklearn compatible model API the steps are:



1. Load the dataset

```python
from utils.parse_data import load_dataset_main

dataset, device = load_dataset_main()
```

2. Pick hyper-parameters and set model generator (used in cross-validation etc.)

```python
n_roi, H = 19, 3

model_gen = lambda : alstm_model_generator(n_roi, H)
```

3. Train, cross-validate or test using [utils](utils) modules

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
    device=device
)
```

