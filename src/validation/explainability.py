import os
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models.digital_twin import GCNModel, GraphSAGEModel, GATModel
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def generate_explainability_artifacts():
    print("--- Generating Explainability Artifacts ---")
    base_dir = get_base_dir()
    reports_dir = os.path.join(base_dir, 'reports')
    models_dir = os.path.join(base_dir, 'models')
    
    # 1. Feature Importance (from XGBoost)
    xgb_path = os.path.join(models_dir, 'xgboost.pkl')
    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
        importance = xgb_model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), importance)
        plt.title('XGBoost Feature Importance (Global)')
        plt.xlabel('Feature Index')
        plt.ylabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, 'feature_importance.png'))
        plt.close()
        print("Generated feature_importance.png")
        
    # 2. Loss Curve (Proxy from final results since in-memory epochs lost)
    gnn_csv = os.path.join(reports_dir, 'gnn_results.csv')
    if os.path.exists(gnn_csv):
        df_gnn = pd.read_csv(gnn_csv)
        plt.figure(figsize=(8, 5))
        plt.bar(df_gnn['Model'], df_gnn['RMSE'], color='coral')
        plt.title('Final Validation RMSE per GNN Architecture')
        plt.ylabel('RMSE (°C)')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, 'loss_curve.png'))
        plt.close()
        print("Generated loss_curve.png")
        
    # 3. Learning Curve (Proxy from Baseline vs GNN RMSE)
    base_csv = os.path.join(reports_dir, 'baseline_results.csv')
    if os.path.exists(base_csv) and os.path.exists(gnn_csv):
        df_base = pd.read_csv(base_csv)
        df_gnn = pd.read_csv(gnn_csv)
        df_all = pd.concat([df_base, df_gnn], ignore_index=True).sort_values(by='RMSE')
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_all['Model'], df_all['RMSE'], marker='o', linestyle='-', color='teal')
        plt.xticks(rotation=45, ha='right')
        plt.title('Model Complexity vs RMSE')
        plt.ylabel('RMSE (°C)')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, 'learning_curve.png'))
        plt.close()
        print("Generated learning_curve.png")

if __name__ == "__main__":
    generate_explainability_artifacts()
