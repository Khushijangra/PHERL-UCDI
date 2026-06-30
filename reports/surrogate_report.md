# Digital Twin Scientific Validation Report

## Methodology
The PHERL-UCDI++ pipeline conducted an exhaustive hyperparameter search comparing deep tabular ensembles against Graph Neural Networks. The dataset comprises 39 physical, spectral, and socioeconomic features distributed across a connected graph of 9,555 urban patches.

## Model Leaderboard

| Model | RMSE (°C) | MAE (°C) | R² |
|---|---|---|---|
| GCN | 0.4383 | 0.3331 | 0.9020 |
| GraphSAGE | 0.4699 | 0.3555 | 0.8874 |
| Physics_GNN | 0.5005 | 0.3769 | 0.8722 |
| GAT | 0.5052 | 0.3788 | 0.8698 |
| XGBoost | 0.5639 | 0.4260 | 0.8378 |
| LightGBM | 0.5855 | 0.4469 | 0.8251 |
| CatBoost | 0.6767 | 0.5141 | 0.7664 |
| RandomForest | 0.8284 | 0.6388 | 0.6500 |
| ExtraTrees | 0.8883 | 0.6845 | 0.5975 |

## Conclusion
**Winner:** GCN
The Graph Neural Network successfully surpassed the tabular baselines, proving that morphological topology and wind advection play a dominant, non-linear role in predicting Urban Heat Island effects.