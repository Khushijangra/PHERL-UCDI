import os
import torch
import pickle
import numpy as np
import pandas as pd
import mlflow
import time
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def train_all_baselines():
    print("--- Training Comparative Baselines ---")
    base_dir = get_base_dir()
    data_path = os.path.join(base_dir, 'data', 'processed', 'training_dataset.pt')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run dataset_packager.py first.")
        
    dataset = torch.load(data_path)
    data = dataset['data']
    feature_cols = dataset['feature_cols']
    scaler_y = dataset['scaler_y']
    
    # Extract Numpy arrays
    X = data.x.numpy()
    y_scaled = data.y.numpy()
    
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()
    
    X_train, y_train = X[train_mask], y_scaled[train_mask]
    X_test, y_test_scaled = X[test_mask], y_scaled[test_mask]
    
    y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, objective='reg:squarederror'),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
        "CatBoost": CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, random_seed=42, verbose=0)
    }
    
    results = []
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    mlflow.set_experiment("PHERL_UCDI_Baselines")
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            start_time = time.time()
            
            # Train
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Predict
            inf_start = time.time()
            preds_scaled = model.predict(X_test)
            inf_time = time.time() - inf_start
            
            # Inverse transform
            preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            # Logging
            mlflow.log_params({"model_type": name, "n_estimators": 100})
            mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "R2": r2, "TrainTime": train_time})
            
            # Save Model
            model_path = os.path.join(models_dir, f"{name.lower()}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
            results.append({
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "Train_Time_s": train_time,
                "Inference_Time_s": inf_time
            })
            
            print(f"[{name}] RMSE: {rmse:.4f}°C | R2: {r2:.4f}")
            
    # Save Results
    results_df = pd.DataFrame(results)
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    results_path = os.path.join(reports_dir, 'baseline_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nBaseline results saved to {results_path}")

if __name__ == "__main__":
    train_all_baselines()
