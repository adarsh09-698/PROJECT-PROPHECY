# Random Forest Young's modulus model

## Performance

The target is `E_GPa` (Young's modulus in GPa). This script trains a Random Forest regression pipeline using all 29 columns except the target as input features.

| 5-fold grouped CV R² (mean +/- SD) | Test R² | Test MAE (GPa) | Test RMSE (GPa) |
|---:|---:|---:|---:|
| 0.550 +/- 0.090 | 0.577 | 10.70 | 14.64 |

## Method

- Training data: 775 rows; provided test set: 122 rows.
- Numeric missing values are median-imputed; categorical fields are most-frequent-imputed and one-hot encoded. Preprocessing is fitted within each CV fold to avoid leakage.
- Evaluation uses 5-fold `GroupKFold`, grouping rows by the 19 elemental composition columns so repeated measurements of one composition do not cross CV folds.
- The Random Forest pipeline is fitted on all training rows and evaluated against the provided test CSV. The test target is used only for evaluation, not as an input feature.
- Forest settings: 400 trees, `max_features=0.8`, `min_samples_leaf=2`, and `random_state=42`.
- Python 3.15.2, scikit-learn 1.9.1.

## Training target context

Training `E_GPa`: n=775, mean=76.09 GPa, standard deviation=25.22 GPa, range=23.0–197.6 GPa.

## Files

- `random_forest_model.joblib`: fitted preprocessing-and-model pipeline.
- `random_forest_metrics.csv`: grouped-CV and test metrics.
- `random_forest_predictions.csv`: measured values, predictions, and residuals for the test rows.
- `train_random_forest.py`: this training script.

Scores apply to these data and this split; they do not guarantee performance for alloys outside the dataset's range.
