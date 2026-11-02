from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parent
TARGET = "E_GPa"
RANDOM_STATE = 42


def build_pipeline(model, numeric_columns, categorical_columns):
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
    categorical_columns = [column for column in features if column not in numeric_columns]
    composition_columns = [column for column in features if column.endswith("_wt")]
    if not composition_columns:
        raise ValueError("Could not identify composition columns ending in '_wt'.")
    groups = x_train[composition_columns].astype(str).agg("|".join, axis=1)
    if groups.nunique() < 5:
        raise ValueError("At least five unique alloy compositions are required for grouped CV.")

    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Ridge regression": Ridge(alpha=10.0),
        "SVR (RBF)": SVR(C=30.0, epsilon=5.0),
        "K-nearest neighbors": KNeighborsRegressor(
            n_neighbors=7, weights="distance"
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=400,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra trees": ExtraTreesRegressor(
            n_estimators=400,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=3,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
        "Hist. gradient boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }

    cv = GroupKFold(n_splits=5)
    results = []
    fitted_models = {}
    for name, model in models.items():
        pipeline = build_pipeline(model, numeric_columns, categorical_columns)
        cv_scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            groups=groups,
            scoring="r2",
            n_jobs=1,
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        fitted_models[name] = pipeline
        results.append(
            {
                "model": name,
                "group_cv_r2_mean": float(cv_scores.mean()),
                "group_cv_r2_std": float(cv_scores.std()),
                "test_r2": float(r2_score(y_test, predictions)),
                "test_mae_GPa": float(mean_absolute_error(y_test, predictions)),
                "test_rmse_GPa": float(
                    np.sqrt(mean_squared_error(y_test, predictions))
                ),
            }
        )

    comparison = pd.DataFrame(results).sort_values(
        "group_cv_r2_mean", ascending=False
    )
    best_name = comparison.iloc[0]["model"]
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(x_test)
    comparison.to_csv(
        ROOT / "all_models_comparison.csv", index=False, float_format="%.6f"
    )
    pd.DataFrame(
        {
            "actual_E_GPa": y_test,
            "predicted_E_GPa": best_predictions,
            "residual_E_GPa": y_test.to_numpy() - best_predictions,
        }
    ).to_csv(
        ROOT / "all_models_test_predictions.csv",
        index=False,
        float_format="%.6f",
    )
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "target": TARGET,
            "features": features,
        },
        ROOT / "best_model_all_models.joblib",
    )

    rows = []
    for result in comparison.to_dict(orient="records"):
        rows.append(
            "| {model} | {group_cv_r2_mean:.3f} +/- {group_cv_r2_std:.3f} | "
            "{test_r2:.3f} | {test_mae_GPa:.2f} | {test_rmse_GPa:.2f} |".format(
                **result
            )
        )
    target_summary = y_train.describe()
    report = f"""# Young's modulus model comparison

## Summary

The target is `E_GPa` (Young's modulus in GPa). The selected model is **{best_name}**, chosen using the highest mean 5-fold composition-grouped cross-validation R² on `train.csv`. It achieved **R² = {comparison.iloc[0]['test_r2']:.3f}** on the provided `test.csv` holdout ({len(test)} rows), with MAE **{comparison.iloc[0]['test_mae_GPa']:.2f} GPa** and RMSE **{comparison.iloc[0]['test_rmse_GPa']:.2f} GPa**.

## Model scores

| Model | 5-fold grouped CV R² (mean +/- SD) | Test R² | Test MAE (GPa) | Test RMSE (GPa) |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Method

- Training data: {len(train)} rows; provided test set: {len(test)} rows.
- Predictors: all {len(features)} columns except `E_GPa` (composition, calculated descriptors, and phase fields).
- Numeric missing values are median-imputed; categorical fields are most-frequent-imputed and one-hot encoded. Preprocessing is fitted within each CV fold to avoid leakage.
- Model selection used 5-fold `GroupKFold`, grouping rows by the 19 elemental composition columns so repeated measurements of one composition do not cross CV folds.
- After model selection, each model was fit on the full training CSV and scored on the provided test CSV. The test target column is present in that file and was used only for evaluation, not as a predictor.
- Models are deterministic where applicable (`random_state={RANDOM_STATE}`). Python {platform.python_version()}, scikit-learn {sklearn.__version__}.

## Training target context

Training `E_GPa`: n={len(y_train)}, mean={target_summary['mean']:.2f} GPa, standard deviation={target_summary['std']:.2f} GPa, range={target_summary['min']:.1f}–{target_summary['max']:.1f} GPa.

## Files

- `best_model_all_models.joblib`: selected fitted preprocessing-and-model pipeline, trained on all training rows.
- `all_models_comparison.csv`: full-precision metrics for all nine models.
- `all_models_test_predictions.csv`: selected model's prediction, measured value, and residual per test row.
- `train_all_models.py`: reproducible comparison script.

Scores are specific to this dataset and split. The dataset is modest in size, so R² should not be interpreted as a guarantee for alloys outside its composition and measurement domain.
"""
    (ROOT / "all_models_report.md").write_text(report, encoding="utf-8")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected by grouped CV: {best_name}")
    print(f"Saved all-model comparison outputs in {ROOT}")


if __name__ == "__main__":
    main()
