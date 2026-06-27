import os
import numpy as np
import shap

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# --- Mock Digital Twin for Scaffolding ---
class MockDigitalTwin:
    def predict(self, x_tensor):
        # Flatten input if necessary
        x = np.array(x_tensor)
        lst = np.zeros(x.shape[0] if x.ndim > 1 else 1)
        lst += 40.0
        # Dummy learned weights
        if x.ndim > 1:
            lst -= 8.0 * x[:, 0] # NDVI
            lst += 6.0 * x[:, 1] # NDBI
            lst -= 3.0 * x[:, 2] # Albedo
            lst += 2.0 * x[:, 4] # Anthropogenic proxy (Pop)
            lst -= 1.0 * x[:, 3] # Wind
        else:
            lst -= 8.0 * x[0]
            lst += 6.0 * x[1]
            lst -= 3.0 * x[2]
            lst += 2.0 * x[4]
            lst -= 1.0 * x[3]
        return lst

class HeatCausalityEngine:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def generate_attribution(self, background_data, instance_data):
        """Layer 1: SHAP Feature Attribution"""
        # Using SHAP KernelExplainer for a generic model
        explainer = shap.KernelExplainer(self.model.predict, background_data)
        shap_values = explainer.shap_values(instance_data)
        
        # Convert to percentages for human readability (ISRO Dashboard requirement)
        abs_vals = np.abs(shap_values)
        total_impact = np.sum(abs_vals)
        percentages = (abs_vals / total_impact) * 100
        
        attribution = {self.feature_names[i]: {"shap": shap_values[i], "percentage": percentages[i]} for i in range(len(self.feature_names))}
        return attribution
        
    def generate_counterfactuals(self, instance_data, interventions):
        """Layer 2: Counterfactual Engine"""
        base_lst = self.model.predict(instance_data)
        
        counterfactuals = {}
        for name, mod_idx, mod_val in interventions:
            cf_data = instance_data.copy()
            cf_data[mod_idx] += mod_val
            cf_lst = self.model.predict(cf_data)
            counterfactuals[name] = cf_lst - base_lst
            
        return counterfactuals
        
    def evaluate_physical_consistency(self, attribution):
        """Layer 3: Physical Consistency Verification"""
        # We know physics rules: low vegetation -> heating (positive SHAP). 
        # If model says high vegetation -> heating, it's inconsistent.
        # This is a proxy scoring function.
        
        consistency_score = 100.0
        
        # Example: NDBI (Urban Canyon/Impervious) should have positive SHAP for hotspots
        if attribution.get('NDBI', {}).get('shap', 0) < 0:
            consistency_score -= 20.0
            
        # Example: Albedo should have negative SHAP (high albedo cools, low albedo heats)
        if attribution.get('Albedo', {}).get('shap', 0) > 0:
            consistency_score -= 20.0
            
        return max(0.0, consistency_score)

def run_causality_pipeline():
    print("PHASE 4: Heat Causality Engine (Attribution & Counterfactuals)")
    
    # 1. Setup Model and Data
    model = MockDigitalTwin()
    feature_names = ['NDVI', 'NDBI', 'Albedo', 'Wind', 'Anthropogenic Heat']
    
    # Dummy background data for SHAP (e.g., 50 random cells)
    background_data = np.random.rand(50, 5)
    
    # A specific "Hotspot" cell to analyze
    # [Low NDVI, High NDBI, Low Albedo, Low Wind, High Pop]
    hotspot = np.array([0.1, 0.8, 0.1, 0.2, 0.9])
    
    engine = HeatCausalityEngine(model, feature_names)
    
    # 2. Run Layer 1: Attribution
    print("\n--- Layer 1: Hotspot Attribution (Why is it hot?) ---")
    attribution = engine.generate_attribution(background_data, hotspot)
    for feat, data in attribution.items():
        print(f"{feat:18s}: {data['percentage']:>5.1f}% contribution (SHAP: {data['shap']:+.2f})")
        
    # 3. Run Layer 2: Counterfactuals
    print("\n--- Layer 2: Counterfactual Simulation (What would cool it?) ---")
    interventions = [
        ("Urban Greening (NDVI +0.3)", 0, 0.3),
        ("Cool Roof (Albedo +0.4)", 2, 0.4)
    ]
    cf_results = engine.generate_counterfactuals(hotspot, interventions)
    for name, delta in cf_results.items():
        print(f"{name:30s}: Predicted ΔT = {delta:+.2f}°C")
        
    # 4. Run Layer 3: Physical Consistency
    print("\n--- Layer 3: Physical Consistency Verification ---")
    score = engine.evaluate_physical_consistency(attribution)
    print(f"Physics Consistency Score: {score:.1f}%")
    
    print("\nCausality Engine Ready for Dashboard Integration.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore") # Suppress SHAP runtime warnings for clean output
    run_causality_pipeline()
