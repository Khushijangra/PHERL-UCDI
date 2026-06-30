"""
Phase 4B: Counterfactual Intervention Simulator
Simulates the temperature change when specific interventions are applied
to the top urban heat island hotspots, transforming the Digital Twin from
a prediction system into a decision-support system.
"""
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.inference import load_inference_model


def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Intervention definitions — scientifically grounded from literature
# Each intervention specifies which feature index to perturb and by how much
# (in units of standard deviations, since features are StandardScaler-normalized)
# ---------------------------------------------------------------------------
INTERVENTIONS = {
    "Tree Plantation (+20% NDVI)": {
        "feature": "ndvi",
        "delta_std": 0.5,   # +0.5σ ≈ +20% NDVI
        "description": "Planting trees / increasing urban canopy cover",
        "icon": "🌳"
    },
    "Cool Roofs (+30% Albedo)": {
        "feature": "albedo",
        "delta_std": 0.6,   # +0.6σ ≈ +30% albedo
        "description": "Reflective roof coatings and cool pavements",
        "icon": "🏠"
    },
    "Green Corridors (+15% Veg Fraction)": {
        "feature": "vegetation_fraction",
        "delta_std": 0.4,
        "description": "Connected green corridors along roads and water",
        "icon": "🌿"
    },
    "Reduced Impervious Surface (-20%)": {
        "feature": "impervious_fraction",
        "delta_std": -0.5,  # Decrease in impervious surfaces
        "description": "Replacing concrete with permeable materials",
        "icon": "🧱"
    },
    "Building Density Reduction (-10%)": {
        "feature": "building_density",
        "delta_std": -0.3,
        "description": "Urban zoning reforms reducing overcrowding",
        "icon": "🏙️"
    },
}


def simulate_interventions(model, model_type, dataset, top_k=10):
    """
    For the top-K hottest nodes (by predicted LST), simulate each intervention
    and compute the resulting ΔT (cooling in °C).
    """
    data = dataset['data']
    scaler_y = dataset['scaler_y']
    feature_cols = dataset['feature_cols']
    col_map = {c: i for i, c in enumerate(feature_cols)}

    # Predict baseline LST for all nodes
    if model_type == "pytorch":
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(data.x, data.edge_index).squeeze().numpy()
    else:
        y_pred_scaled = model.predict(data.x.numpy())

    # Find top-K hottest nodes
    hotspot_indices = np.argsort(y_pred_scaled)[-top_k:][::-1]

    results = []
    for node_idx in hotspot_indices:
        baseline_scaled = y_pred_scaled[node_idx]
        baseline_temp = scaler_y.inverse_transform([[baseline_scaled]])[0][0]

        node_results = {
            "node_id": int(node_idx),
            "baseline_lst_c": round(float(baseline_temp), 2)
        }

        for intervention_name, config in INTERVENTIONS.items():
            feature_name = config["feature"]
            if feature_name not in col_map:
                continue

            # Clone full graph features to preserve spatial context
            x_mod = data.x.clone()
            feat_idx = col_map[feature_name]
            x_mod[node_idx, feat_idx] += config["delta_std"]

            # Predict modified LST using the full graph
            if model_type == "pytorch":
                model.eval()
                with torch.no_grad():
                    out_mod = model(x_mod, data.edge_index)
                    y_mod_scaled = out_mod[node_idx].item()
            else:
                out_mod = model.predict(x_mod.numpy())
                y_mod_scaled = out_mod[node_idx]

            modified_temp = scaler_y.inverse_transform([[y_mod_scaled]])[0][0]
            delta_t = modified_temp - baseline_temp

            node_results[f"delta_t_{intervention_name}"] = round(float(delta_t), 3)

        results.append(node_results)

    return pd.DataFrame(results)


def generate_counterfactual_report(df_results, reports_dir):
    """Generate summary table and visualization."""
    intervention_cols = [c for c in df_results.columns if c.startswith("delta_t_")]

    # Summary: mean ΔT per intervention across all hotspots
    summary = {}
    for col in intervention_cols:
        name = col.replace("delta_t_", "")
        summary[name] = df_results[col].mean()

    df_summary = pd.DataFrame([
        {"Intervention": k, "Mean_Cooling_C": -v, "Direction": "Cooling" if v < 0 else "Heating"}
        for k, v in summary.items()
    ]).sort_values("Mean_Cooling_C", ascending=False)

    # Save CSV
    df_results.to_csv(os.path.join(reports_dir, 'counterfactual_results.csv'), index=False)
    df_summary.to_csv(os.path.join(reports_dir, 'intervention_summary.csv'), index=False)
    print("  → counterfactual_results.csv saved")
    print("  → intervention_summary.csv saved")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#2a9d8f' if v >= 0 else '#e63946' for v in df_summary['Mean_Cooling_C']]
    bars = ax.barh(df_summary['Intervention'], df_summary['Mean_Cooling_C'], color=colors)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Mean Temperature Reduction (°C) — positive = cooling')
    ax.set_title(
        'PHERL-UCDI++ Counterfactual Intervention Analysis\n'
        f'Effect on Top-{len(df_results)} Urban Heat Island Hotspots — Ahmedabad'
    )

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:+.3f}°C", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'counterfactual_analysis.png'), dpi=150)
    plt.close()
    print("  → counterfactual_analysis.png saved")

    return df_summary


def run_counterfactual():
    print("=" * 60)
    print("PHASE 4B: Counterfactual Intervention Simulator")
    print("=" * 60)

    base_dir = get_base_dir()
    reports_dir = os.path.join(base_dir, 'reports')

    model, model_type = load_inference_model()
    print(f"Loaded model type: {model_type}")

    data_path = os.path.join(base_dir, 'data', 'processed', 'training_dataset.pt')
    dataset = torch.load(data_path, map_location='cpu', weights_only=False)

    df_results = simulate_interventions(model, model_type, dataset, top_k=10)
    df_summary = generate_counterfactual_report(df_results, reports_dir)

    print("\n--- Intervention Summary (Top-10 Hotspots) ---")
    print(df_summary.to_string(index=False))
    print("=" * 60)
    print("Phase 4B complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_counterfactual()
