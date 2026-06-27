import os
import torch
import numpy as np
from src.models.inference import load_inference_model

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def run_sanity_tests():
    print("PHASE 3.5: Urban Climate Sanity Verification Engine")
    
    try:
        model, model_type = load_inference_model()
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}")
        return
        
    print(f"Loaded {model_type} model for sanity verification.")
    
    # We will test on a dummy single-node graph (since we just want to test feature sensitivity)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    
    # Load dataset package to get the correct number of features
    base_dir = get_base_dir()
    data_path = os.path.join(base_dir, 'data', 'processed', 'training_dataset.pt')
    dataset = torch.load(data_path, map_location='cpu')
    feature_cols = dataset['feature_cols']
    num_features = len(feature_cols)
    col_map = {col: i for i, col in enumerate(feature_cols)}
    
    # Baseline scaled features (average urban environment)
    base_x = torch.zeros((1, num_features), dtype=torch.float)
    
    def predict(x_tensor):
        if model_type == 'pytorch':
            with torch.no_grad():
                return model(x_tensor, edge_index).item()
        else:
            return model.predict(x_tensor.numpy())[0]
            
    base_lst = predict(base_x)
    results = []
    
    # Test 1: Vegetation Cooling Test (NDVI +1 StdDev)
    if 'ndvi' in col_map:
        x_test1 = base_x.clone()
        x_test1[0, col_map['ndvi']] += 1.0
        dt_1 = predict(x_test1) - base_lst
        pass_1 = dt_1 < 0
        results.append(("NDVI (+1σ)", "Cooling (ΔT < 0)", f"{dt_1:+.2f}°C", "✅" if pass_1 else "❌"))
    
    # Test 2: Cool Roof Test (Albedo +1 StdDev)
    if 'albedo' in col_map:
        x_test2 = base_x.clone()
        x_test2[0, col_map['albedo']] += 1.0
        dt_2 = predict(x_test2) - base_lst
        pass_2 = dt_2 < 0
        results.append(("Albedo (+1σ)", "Cooling (ΔT < 0)", f"{dt_2:+.2f}°C", "✅" if pass_2 else "❌"))
    
    # Test 3: Impervious Surface Test (NDBI +1 StdDev)
    if 'ndbi' in col_map:
        x_test3 = base_x.clone()
        x_test3[0, col_map['ndbi']] += 1.0
        dt_3 = predict(x_test3) - base_lst
        pass_3 = dt_3 > 0
        results.append(("NDBI (+1σ)", "Heating (ΔT > 0)", f"{dt_3:+.2f}°C", "✅" if pass_3 else "❌"))
    
    # Test 4: Wind Advection Test (Wind +1 StdDev)
    if 'wind_speed' in col_map:
        x_test4 = base_x.clone()
        x_test4[0, col_map['wind_speed']] += 1.0
        dt_4 = predict(x_test4) - base_lst
        pass_4 = dt_4 < 0
        results.append(("Wind (+1σ)", "Cooling (ΔT < 0)", f"{dt_4:+.2f}°C", "✅" if pass_4 else "❌"))
    
    # Generate Report
    report = "# Digital Twin Sanity Verification Report\n\n"
    report += "This report validates that the underlying optimized model correctly models known urban climate thermodynamics. "
    report += "Only if all tests pass can the model be promoted to the 'Urban Digital Twin' status and utilized for SHAP causality and RL policy optimization.\n\n"
    report += "| Test | Expected | Actual (Scaled ΔT) | Pass |\n"
    report += "|---|---|---|---|\n"
    for r in results:
        report += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |\n"
        
    out_dir = os.path.join(get_base_dir(), 'data', 'processed')
    report_path = os.path.join(out_dir, 'digital_twin_sanity_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"Sanity verification complete. Report saved to {report_path}")
    
    if len(results) == 0:
        print("\nWARNING: No physical features (ndvi, ndbi, albedo, wind) found in dataset to test.")
    elif any(r[3] == "❌" for r in results):
        print("\nCRITICAL FAILURE: The surrogate model violated physical laws. Do NOT proceed to Explainability/RL.")
    else:
        print("\nSUCCESS: Model obeys urban climate physics. Promoted to Urban Digital Twin.")

if __name__ == "__main__":
    run_sanity_tests()
