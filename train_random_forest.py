from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
TARGET = "E_GPa"
RANDOM_STATE = 42
MODEL_NAME = "Random Forest"


def build_pipeline(numeric_columns, categorical_columns):
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=400,
        max_features=0.8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("scaling", StandardScaler()),
            ("model", model),
        ]
    )


def main():
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    if TARGET not in train or TARGET not in test:
        raise ValueError(f"Both CSV files must contain the target column {TARGET!r}.")

    features = [column for column in train.columns if column != TARGET]
    if set(features) - set(test.columns):
        raise ValueError("The test CSV is missing one or more training feature columns.")
    x_train = train[features].copy()
    x_test = test[features].copy()
    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)

    numeric_columns = x_train.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in features if column not in numeric_columns
    ]
    composition_columns = [column for column in features if column.endswith("_wt")]
    if not composition_columns:
        raise ValueError("Could not identify composition columns ending in '_wt'.")
    groups = x_train[composition_columns].astype(str).agg("|".join, axis=1)
    if groups.nunique() < 5:
        raise ValueError("At least five unique alloy compositions are required for grouped CV.")

    pipeline = build_pipeline(numeric_columns, categorical_columns)
    cv_scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=GroupKFold(n_splits=5),
        groups=groups,
        scoring="r2",
        n_jobs=1,
    )

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "model": MODEL_NAME,
        "group_cv_r2_mean": float(cv_scores.mean()),
        "group_cv_r2_std": float(cv_scores.std()),
        "test_r2": float(r2_score(y_test, predictions)),
        "test_mae_GPa": float(mean_absolute_error(y_test, predictions)),
        "test_rmse_GPa": float(np.sqrt(mean_squared_error(y_test, predictions))),
    }

    pd.DataFrame([metrics]).to_csv(
        ROOT / "random_forest_metrics.csv", index=False, float_format="%.6f"
    )
    pd.DataFrame(
        {
            "actual_E_GPa": y_test,
            "predicted_E_GPa": predictions,
            "residual_E_GPa": y_test.to_numpy() - predictions,
        }
    ).to_csv(
        ROOT / "random_forest_predictions.csv",
        index=False,
        float_format="%.6f",
    )
    joblib.dump(
        {
            "model": pipeline,
            "model_name": MODEL_NAME,
            "target": TARGET,
            "features": features,
        },
        ROOT / "random_forest_model.joblib",
    )

    target_summary = y_train.describe()
    report = f"""# Random Forest Young's modulus model

## Performance

The target is `E_GPa` (Young's modulus in GPa). This script trains a Random Forest regression pipeline using all {len(features)} columns except the target as input features.

| 5-fold grouped CV R² (mean +/- SD) | Test R² | Test MAE (GPa) | Test RMSE (GPa) |
|---:|---:|---:|---:|
| {metrics['group_cv_r2_mean']:.3f} +/- {metrics['group_cv_r2_std']:.3f} | {metrics['test_r2']:.3f} | {metrics['test_mae_GPa']:.2f} | {metrics['test_rmse_GPa']:.2f} |

## Method

- Training data: {len(train)} rows; provided test set: {len(test)} rows.
- Numeric missing values are median-imputed; categorical fields are most-frequent-imputed and one-hot encoded. Preprocessing is fitted within each CV fold to avoid leakage.
- Evaluation uses 5-fold `GroupKFold`, grouping rows by the 19 elemental composition columns so repeated measurements of one composition do not cross CV folds.
- The Random Forest pipeline is fitted on all training rows and evaluated against the provided test CSV. The test target is used only for evaluation, not as an input feature.
- Forest settings: 400 trees, `max_features=0.8`, `min_samples_leaf=2`, and `random_state={RANDOM_STATE}`.
- Python {platform.python_version()}, scikit-learn {sklearn.__version__}.

## Training target context

Training `E_GPa`: n={len(y_train)}, mean={target_summary['mean']:.2f} GPa, standard deviation={target_summary['std']:.2f} GPa, range={target_summary['min']:.1f}–{target_summary['max']:.1f} GPa.

## Files

- `random_forest_model.joblib`: fitted preprocessing-and-model pipeline.
- `random_forest_metrics.csv`: grouped-CV and test metrics.
- `random_forest_predictions.csv`: measured values, predictions, and residuals for the test rows.
- `train_random_forest.py`: this training script.

Scores apply to these data and this split; they do not guarantee performance for alloys outside the dataset's range.
"""
    (ROOT / "random_forest_report.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame([metrics]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved Random Forest model and results in {ROOT}")


if __name__ == "__main__":
    main()
