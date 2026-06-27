import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import mlflow
import time
import optuna
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.models.digital_twin import GCNModel, GraphSAGEModel, GATModel
from src.models.physics_loss import compute_physics_loss

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def load_config():
    config_path = os.path.join(get_base_dir(), 'configs', 'experiment.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def train_eval_model(model_class, data, feature_cols, scaler_y, params, use_physics=False):
    model = model_class(num_features=data.x.shape[1], hidden_dim=params['hidden_dim'], dropout=params['dropout']).to(data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    model.train()
    for epoch in range(params['epochs']):
        optimizer.zero_grad()
        if use_physics:
            loss, _, _ = compute_physics_loss(model, data.x, data.edge_index, data.y, data.train_mask, feature_cols, params.get('lambda_phys', 0.1))
        else:
            out = model(data.x, data.edge_index).squeeze()
            loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        preds_scaled = model(data.x, data.edge_index).squeeze().cpu().numpy()
        preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        y_test_true = scaler_y.inverse_transform(data.y[data.test_mask].cpu().numpy().reshape(-1, 1)).flatten()
        preds_test = preds[data.test_mask.cpu().numpy()]
        rmse = np.sqrt(mean_squared_error(y_test_true, preds_test))
        
    return rmse, model

def optimize_and_train():
    print("--- Training Graph Neural Networks (HPO via Optuna) ---")
    config = load_config()
    base_dir = get_base_dir()
    data_path = os.path.join(base_dir, 'data', 'processed', 'training_dataset.pt')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError("training_dataset.pt not found.")
        
    dataset = torch.load(data_path, weights_only=False)
    data = dataset['data']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    feature_cols = dataset['feature_cols']
    scaler_y = dataset['scaler_y']
    
    models_to_test = {
        "GCN": (GCNModel, False),
        "GraphSAGE": (GraphSAGEModel, False),
        "GAT": (GATModel, False),
        "Physics_GNN": (GraphSAGEModel, True)
    }
    
    mlflow.set_experiment(config['mlflow']['experiment_name_gnn'])
    results = []
    
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for name, (model_class, use_physics) in models_to_test.items():
        if os.path.exists(os.path.join(models_dir, f"{name.lower()}.pt")):
            print(f"\nSkipping {name}, already trained.")
            continue
            
        print(f"\nRunning HPO for {name}...")
        
        def objective(trial):
            ss = config['gnn']['search_space']
            params = {
                'lr': trial.suggest_float('lr', ss['lr'][0], ss['lr'][1], log=True),
                'hidden_dim': trial.suggest_categorical('hidden_dim', ss['hidden_dim']),
                'dropout': trial.suggest_float('dropout', ss['dropout'][0], ss['dropout'][1]),
                'weight_decay': trial.suggest_float('weight_decay', ss['weight_decay'][0], ss['weight_decay'][1], log=True),
                'epochs': trial.suggest_categorical('epochs', ss['epochs'])
            }
            if use_physics:
                params['lambda_phys'] = trial.suggest_float('lambda_phys', ss['lambda_phys'][0], ss['lambda_phys'][1], log=True)
                
            rmse, _ = train_eval_model(model_class, data, feature_cols, scaler_y, params, use_physics)
            return rmse
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=config['gnn']['optuna_trials']) 
        
        best_params = study.best_params
        print(f"[{name}] Best params: {best_params} (RMSE: {study.best_value:.4f})")
        
        # Retrain best
        start_time = time.time()
        with mlflow.start_run(run_name=f"{name}_Best"):
            final_rmse, final_model = train_eval_model(model_class, data, feature_cols, scaler_y, best_params, use_physics)
            train_time = time.time() - start_time
            
            # Full metrics
            final_model.eval()
            with torch.no_grad():
                inf_start = time.time()
                preds_scaled = final_model(data.x, data.edge_index).squeeze().cpu().numpy()
                inf_time = time.time() - inf_start
                preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                y_test_true = scaler_y.inverse_transform(data.y[data.test_mask].cpu().numpy().reshape(-1, 1)).flatten()
                preds_test = preds[data.test_mask.cpu().numpy()]
                
                mae = mean_absolute_error(y_test_true, preds_test)
                r2 = r2_score(y_test_true, preds_test)
                
            mlflow.log_params(best_params)
            mlflow.log_metrics({"RMSE": final_rmse, "MAE": mae, "R2": r2, "TrainTime": train_time})
            
            # Save Model
            torch.save(final_model.state_dict(), os.path.join(models_dir, f"{name.lower()}.pt"))
            with open(os.path.join(models_dir, f"{name.lower()}_config.json"), 'w') as f:
                json.dump(best_params, f)
                
            results.append({
                "Model": name,
                "RMSE": final_rmse,
                "MAE": mae,
                "R2": r2,
                "Train_Time_s": train_time,
                "Inference_Time_s": inf_time
            })
            
    # Save Results
    results_df = pd.DataFrame(results)
    reports_dir = os.path.join(base_dir, 'reports')
    results_df.to_csv(os.path.join(reports_dir, 'gnn_results.csv'), index=False)
    print("\nGNN results saved to gnn_results.csv")

if __name__ == "__main__":
    optimize_and_train()
