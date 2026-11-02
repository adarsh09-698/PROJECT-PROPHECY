# Young's modulus model report

## Summary

The target is `E_GPa` (Young's modulus in GPa). This simplified script trains only **Extra Trees**, retained from the earlier model comparison because it had the highest mean 5-fold composition-grouped cross-validation R². It achieved **R² = 0.564** on the provided `test.csv` holdout (122 rows), with MAE **11.27 GPa** and RMSE **14.86 GPa**.

## Model performance

| Model | 5-fold grouped CV R² (mean +/- SD) | Test R² | Test MAE (GPa) | Test RMSE (GPa) |
|---|---:|---:|---:|---:|
| Extra Trees | 0.565 +/- 0.049 | 0.564 | 11.27 | 14.86 |

## Method

- Training data: 775 rows; provided test set: 122 rows.
- Predictors: all 29 columns except `E_GPa` (composition, calculated descriptors, and phase fields).
- Numeric missing values are median-imputed; categorical fields are most-frequent-imputed and one-hot encoded. Preprocessing is fitted within each CV fold to avoid leakage.
- Model selection used 5-fold `GroupKFold`, grouping rows by the 19 elemental composition columns so repeated measurements of one composition do not cross CV folds.
- The Extra Trees pipeline was fit on the full training CSV and scored on the provided test CSV. The test target column is present in that file and was used only for evaluation, not as a predictor.
- The model is deterministic (`random_state=42`). Python 3.15.2, scikit-learn 1.9.1.

## Training target context

Training `E_GPa`: n=775, mean=76.09 GPa, standard deviation=25.22 GPa, range=23.0–197.6 GPa.

## Files

- `best_model.joblib`: selected fitted preprocessing-and-model pipeline, trained on all training rows.
- `model_comparison.csv`: full-precision model metrics.
- `test_predictions.csv`: selected model's prediction, measured value, and residual per test row.
- `train_models.py`: reproducible training/evaluation script.

Scores are specific to this dataset and split. The dataset is modest in size, so R² should not be interpreted as a guarantee for alloys outside its composition and measurement domain.
