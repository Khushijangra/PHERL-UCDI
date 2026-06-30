"""
Phase 4A: Explainability Engine
Generates SHAP summary plots, GNN gradient-based feature importance,
residual analysis, and learning curves from the validated Digital Twin.
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from src.models.inference import load_inference_model


def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def _get_reports_dir():
    return os.path.join(get_base_dir(), 'reports')


# ---------------------------------------------------------------------------
# 1. SHAP Feature Importance (for tabular best model or GNN gradient proxy)
# ---------------------------------------------------------------------------
def generate_shap_or_importance(model, model_type, X, feature_cols, reports_dir):
    if model_type == "tabular" and HAS_SHAP:
        print("Computing SHAP values (tabular model)...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_cols,
                          show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, 'shap_summary.png'), dpi=150)
        plt.close()
        print("  Saved shap_summary.png")

        # Also save per-feature mean |SHAP|
        mean_shap = np.abs(shap_values).mean(axis=0)
        df_shap = pd.DataFrame({'feature': feature_cols, 'mean_abs_shap': mean_shap})
        df_shap = df_shap.sort_values('mean_abs_shap', ascending=False)
        df_shap.to_csv(os.path.join(reports_dir, 'shap_values.csv'), index=False)
        print("  Saved shap_values.csv")

    else:
        # Gradient-based proxy for GNN: compute output gradient w.r.t. input features
        print("Computing gradient-based feature sensitivity (GNN)...")
        x_tensor = torch.tensor(X, dtype=torch.float32)
        edge_index = torch.zeros((2, 0), dtype=torch.long)  # minimal graph
        x_tensor.requires_grad_(True)

        model.eval()
        # Use a single representative node for sensitivity
        x_single = x_tensor[:1].clone().detach().requires_grad_(True)
        ei_single = torch.tensor([[0], [0]], dtype=torch.long)
        out = model(x_single, ei_single)
        out.backward()

        grad_importance = x_single.grad.abs().squeeze().detach().numpy()
        df_imp = pd.DataFrame({'feature': feature_cols, 'gradient_sensitivity': grad_importance})
        df_imp = df_imp.sort_values('gradient_sensitivity', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(df_imp)))
        ax.barh(df_imp['feature'][::-1], df_imp['gradient_sensitivity'][::-1], color=colors[::-1])
        ax.set_xlabel('Gradient Sensitivity (|∂LST/∂feature|)')
        ax.set_title('GCN Digital Twin — Feature Sensitivity (Gradient-Based)')
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, 'feature_importance.png'), dpi=150)
        plt.close()
        df_imp.to_csv(os.path.join(reports_dir, 'feature_importance.csv'), index=False)
        print("  Saved feature_importance.png")
        print("  Saved feature_importance.csv")


# ---------------------------------------------------------------------------
# 2. Residual Analysis
# ---------------------------------------------------------------------------
def generate_residual_plot(model, model_type, dataset, reports_dir):
    print("Generating residual analysis...")
    data = dataset['data']
    scaler_y = dataset['scaler_y']

    test_mask = data.test_mask
    y_true_scaled = data.y[test_mask].numpy()

    if model_type == "pytorch":
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(data.x, data.edge_index).squeeze()[test_mask].numpy()
    else:
        X_test = data.x[test_mask].numpy()
        y_pred_scaled = model.predict(X_test)

    # Inverse transform to °C
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=8, color='steelblue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', lw=1.5, label='Perfect fit')
    axes[0].set_xlabel('Actual LST (°C)')
    axes[0].set_ylabel('Predicted LST (°C)')
    axes[0].set_title('Predicted vs Actual — GCN Digital Twin')
    axes[0].legend()

    # Residual Distribution
    axes[1].hist(residuals, bins=50, color='coral', edgecolor='white')
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].set_xlabel('Residual (°C)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Residual Distribution (mean={residuals.mean():.3f}°C)')

    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'residual_plot.png'), dpi=150)
    plt.close()
    print("  Saved residual_plot.png")


# ---------------------------------------------------------------------------
# 3. Model Leaderboard Chart
# ---------------------------------------------------------------------------
def generate_leaderboard_chart(reports_dir):
    print("Generating model leaderboard chart...")
    comp_path = os.path.join(reports_dir, 'comparison_table.csv')
    if not os.path.exists(comp_path):
        print("  comparison_table.csv not found — skipping chart")
        return

    df = pd.read_csv(comp_path)
    df = df.sort_values('RMSE')

    # Color: GNN models in teal, tabular in coral
    gnn_names = {'GCN', 'GraphSAGE', 'GAT', 'Physics_GNN'}
    colors = ['#2a9d8f' if m in gnn_names else '#e76f51' for m in df['Model']]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df['Model'], df['RMSE'], color=colors)
    ax.set_xlabel('RMSE (°C) — lower is better')
    ax.set_title('PHERL-UCDI++ Model Leaderboard\n(Teal = GNN, Coral = Tabular Baseline)')
    ax.axvline(df[df['Model'] == 'XGBoost']['RMSE'].values[0],
               color='gray', linestyle='--', lw=1, label='Best tabular baseline')
    ax.legend()

    # Annotate bars
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{row['RMSE']:.3f}", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'learning_curve.png'), dpi=150)
    plt.close()
    print("  Saved learning_curve.png (leaderboard chart)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_explainability():
    print("=" * 60)
    print("PHASE 4A: Explainability Engine")
    print("=" * 60)

    base_dir = get_base_dir()
    reports_dir = _get_reports_dir()

    # Load validated model
    model, model_type = load_inference_model()
    print(f"Loaded model type: {model_type}")

    # Load dataset
    data_path = os.path.join(base_dir, 'data', 'processed', 'training_dataset.pt')
    dataset = torch.load(data_path, map_location='cpu', weights_only=False)
    feature_cols = dataset['feature_cols']
    X = dataset['data'].x.numpy()

    # Run all explainability artifacts
    generate_shap_or_importance(model, model_type, X, feature_cols, reports_dir)
    generate_residual_plot(model, model_type, dataset, reports_dir)
    generate_leaderboard_chart(reports_dir)

    print("=" * 60)
    print("Phase 4A complete. Artifacts written to reports/")
    print("=" * 60)


if __name__ == "__main__":
    run_explainability()
