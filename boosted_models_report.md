# LightGBM, XGBoost, and CatBoost comparison

## Summary

The target is `E_GPa` (Young's modulus in GPa). The three gradient-boosting models were compared using the same input features and grouped cross-validation. **CatBoost** had the highest mean grouped-CV R² and is the selected model by that criterion. The provided test set is reported separately.

## Model scores

| Model | 5-fold grouped CV R² (mean +/- SD) | Test R² | Test MAE (GPa) | Test RMSE (GPa) |
|---|---:|---:|---:|---:|
| CatBoost | 0.558 +/- 0.074 | 0.526 | 11.39 | 15.49 |
| LightGBM | 0.534 +/- 0.054 | 0.562 | 11.22 | 14.90 |
| XGBoost | 0.533 +/- 0.095 | 0.513 | 11.65 | 15.70 |

## Method

- Training data: 775 rows; provided test set: 122 rows.
- Predictors: all 29 columns except `E_GPa`.
- Numeric missing values are median-imputed; categorical fields are most-frequent-imputed and one-hot encoded. Preprocessing is fitted inside each CV fold to avoid leakage.
- Evaluation uses 5-fold `GroupKFold`, grouping rows by the 19 elemental composition columns so repeated measurements of one composition do not cross CV folds.
- All three model pipelines are fitted on the full training CSV and evaluated on the provided test CSV. The test target is used only for evaluation, not as an input feature.
- Random seed: 42. Python 3.14.7, scikit-learn 1.9.1, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10.

## Training target context

Training `E_GPa`: n=775, mean=76.09 GPa, standard deviation=25.22 GPa, range=23.0–197.6 GPa.

## Files

- `boosted_models.joblib`: fitted pipelines for all three models and the model selected by grouped CV.
- `boosted_models_comparison.csv`: full-precision validation and test metrics.
- `boosted_models_predictions.csv`: measured target and each model's predictions and residuals for all test rows.
- `train_boosted_models.py`: reproducible training and comparison script.

Scores apply to these data and this split; they do not guarantee performance for alloys outside the dataset's range.
