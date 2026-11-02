from pathlib import Path
import platform
from importlib.metadata import version

import catboost
import joblib
import lightgbm
import numpy as np
import pandas as pd
import xgboost
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


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

    models = {
        "LightGBM": LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=15,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=400,
            depth=5,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        ),
    }

    cv = GroupKFold(n_splits=5)
    results = []
    fitted_models = {}
    predictions_by_model = {}
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
        predictions_by_model[name] = predictions
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
    comparison.to_csv(
        ROOT / "boosted_models_comparison.csv", index=False, float_format="%.6f"
    )

    predictions = pd.DataFrame({"actual_E_GPa": y_test.to_numpy()})
    for name, values in predictions_by_model.items():
        column = name.lower().replace(" ", "_")
        predictions[f"{column}_predicted_E_GPa"] = values
        predictions[f"{column}_residual_E_GPa"] = y_test.to_numpy() - values
    predictions.to_csv(
        ROOT / "boosted_models_predictions.csv", index=False, float_format="%.6f"
    )
    joblib.dump(
        {
            "models": fitted_models,
            "selected_by_grouped_cv": best_name,
            "target": TARGET,
            "features": features,
        },
        ROOT / "boosted_models.joblib",
    )

    report_rows = [
        "| {model} | {group_cv_r2_mean:.3f} +/- {group_cv_r2_std:.3f} | "
        "{test_r2:.3f} | {test_mae_GPa:.2f} | {test_rmse_GPa:.2f} |".format(
            **result
        )
        for result in comparison.to_dict(orient="records")
    ]
    target_summary = y_train.describe()
    sklearn_version = version("scikit-learn")
    report = f"""# LightGBM, XGBoost, and CatBoost comparison

## Summary

The target is `E_GPa` (Young's modulus in GPa). The three gradient-boosting models were compared using the same input features and grouped cross-validation. **{best_name}** had the highest mean grouped-CV R² and is the selected model by that criterion. The provided test set is reported separately.

## Model scores

| Model | 5-fold grouped CV R² (mean +/- SD) | Test R² | Test MAE (GPa) | Test RMSE (GPa) |
|---|---:|---:|---:|---:|
{chr(10).join(report_rows)}

## Method

- Training data: {len(train)} rows; provided test set: {len(test)} rows.
- Predictors: all {len(features)} columns except `E_GPa`.
- Numeric missing values are median-imputed; categorical fields are most-frequent-imputed and one-hot encoded. Preprocessing is fitted inside each CV fold to avoid leakage.
- Evaluation uses 5-fold `GroupKFold`, grouping rows by the 19 elemental composition columns so repeated measurements of one composition do not cross CV folds.
- All three model pipelines are fitted on the full training CSV and evaluated on the provided test CSV. The test target is used only for evaluation, not as an input feature.
- Random seed: {RANDOM_STATE}. Python {platform.python_version()}, scikit-learn {sklearn_version}, LightGBM {lightgbm.__version__}, XGBoost {xgboost.__version__}, CatBoost {catboost.__version__}.

## Training target context

Training `E_GPa`: n={len(y_train)}, mean={target_summary['mean']:.2f} GPa, standard deviation={target_summary['std']:.2f} GPa, range={target_summary['min']:.1f}–{target_summary['max']:.1f} GPa.

## Files

- `boosted_models.joblib`: fitted pipelines for all three models and the model selected by grouped CV.
- `boosted_models_comparison.csv`: full-precision validation and test metrics.
- `boosted_models_predictions.csv`: measured target and each model's predictions and residuals for all test rows.
- `train_boosted_models.py`: reproducible training and comparison script.

Scores apply to these data and this split; they do not guarantee performance for alloys outside the dataset's range.
"""
    (ROOT / "boosted_models_report.md").write_text(report, encoding="utf-8")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected by grouped CV: {best_name}")
    print(f"Saved three-model comparison outputs in {ROOT}")


if __name__ == "__main__":
    main()
