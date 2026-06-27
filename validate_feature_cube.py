import pandas as pd
import numpy as np
import os
import json

def validate_cube():
    base_dir = os.path.dirname(__file__)
    cube_path = os.path.join(base_dir, 'data', 'processed', 'master_feature_cube.parquet')
    
    if not os.path.exists(cube_path):
        print(f"Error: {cube_path} not found.")
        return
        
    df = pd.read_parquet(cube_path)
    
    report = ["# Feature Cube Validation Report", ""]
    report.append(f"**Shape**: {df.shape[0]} rows, {df.shape[1]} columns")
    report.append("")
    
    missing_vals = df.isnull().sum().sum()
    report.append(f"**Missing values**: {missing_vals}")
    
    dupe_rows = df.duplicated().sum()
    report.append(f"**Duplicate rows**: {dupe_rows}")
    
    if 'cell_id' in df.columns:
        dupe_cells = df['cell_id'].duplicated().sum()
        report.append(f"**Duplicate cell IDs**: {dupe_cells}")
    else:
        report.append("**Duplicate cell IDs**: cell_id not found")
        
    report.append(f"**Data Types**: {df.dtypes.value_counts().to_dict()}")
    
    # Save report
    reports_dir = os.path.join(base_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, 'feature_cube_validation_report.md')
    
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
        
    print(f"Validation report saved to {report_path}")
    print("Missing values:", missing_vals)
    print("Duplicate rows:", dupe_rows)

if __name__ == '__main__':
    validate_cube()
