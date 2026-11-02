# PROJECT PROPHECY

Predict Young's modulus (`E_GPa`) for titanium alloys with an Extra Trees
regression model.

## Project contents

- `train_models.py` — preprocessing, grouped cross-validation, training, and
  evaluation script.
- `train_all_models.py` — optional comparison of nine regression models.
- `train_random_forest.py` — train and evaluate a Random Forest model.
- `train_boosted_models.py` — compare LightGBM, XGBoost, and CatBoost.
- `train.csv` — training data.
- `test.csv` — held-out evaluation data, including measured `E_GPa` values.
- `best_model.joblib` — fitted Extra Trees pipeline trained on `train.csv`.
- `model_report.md` — model performance and methodology.
- `model_comparison.csv` — saved model metrics.
- `test_predictions.csv` — measured values, predictions, and residuals.

## Setup

Use Python 3.15 or a compatible Python version and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and evaluate

Place `train.csv` and `test.csv` in the project directory, then run:

```powershell
python train_models.py
```

The script uses `E_GPa` as the target and all remaining columns as input
features. It median-imputes numeric fields, imputes and one-hot encodes
categorical fields, then evaluates the Extra Trees pipeline with 5-fold
composition-grouped cross-validation. It subsequently fits the pipeline on
all training rows and evaluates against the labeled test set.

The run writes or updates `best_model.joblib`, `model_report.md`,
`model_comparison.csv`, and `test_predictions.csv`.

To compare the candidate models instead, run:

```powershell
python train_all_models.py
```

This writes separate `all_models_*` results and `best_model_all_models.joblib`
so it does not overwrite the Extra Trees-only run.

To train and evaluate a Random Forest model, run:

```powershell
python train_random_forest.py
```

It writes `random_forest_model.joblib`, `random_forest_metrics.csv`,
`random_forest_predictions.csv`, and `random_forest_report.md` without
overwriting the Extra Trees or all-model outputs.

To compare LightGBM, XGBoost, and CatBoost, use Python 3.14 (CatBoost does not
currently provide a wheel for the Python 3.15 environment used for the other
scripts):

```powershell
py -3.14 -m venv .venv-boost
.\.venv-boost\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-boosted.txt
python train_boosted_models.py
```

The three-model script writes `boosted_models.joblib`,
`boosted_models_comparison.csv`, `boosted_models_predictions.csv`, and
`boosted_models_report.md`, without replacing results from the other scripts.

## Predict with the saved model

```python
import joblib
import pandas as pd

artifact = joblib.load("best_model.joblib")
test = pd.read_csv("test.csv")
predictions = artifact["model"].predict(test[artifact["features"]])
print(predictions)
```

The reported test metrics apply to the provided test split and are not a
guarantee of performance for alloys outside the dataset's range. The test CSV
contains measured target values and is used for evaluation only.
