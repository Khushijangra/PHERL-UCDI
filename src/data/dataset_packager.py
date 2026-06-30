import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def package_dataset():
    print("Packaging Urban Climate Dataset for GPU Training...")
    base_dir = get_base_dir()
    
    # 1. Load Feature Cube
    cube_path = os.path.join(base_dir, 'data', 'processed', 'master_feature_cube.parquet')
    if not os.path.exists(cube_path):
        raise FileNotFoundError(f"Feature cube not found: {cube_path}")
        
    df = pd.read_parquet(cube_path)
    print(f"Loaded feature cube: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Extract Features and Target
    y_raw = df['lst_mean'].fillna(df['lst_mean'].mean()).values
    
    FEATURE_COLUMNS = [
        "ndvi",
        "ndbi",
        "ndwi",
        "savi",
        "evi",
        "albedo",
        "vegetation_fraction",
        "impervious_fraction",
        "air_temperature",
        "dewpoint",
        "wind_u",
        "wind_v",
        "solar_radiation",
        "pressure",
        "population_density",
        "building_density",
        "road_density",
        "latitude",
        "longitude"
    ]
    
    # Only include features that actually exist in the dataframe (in case some were dropped)
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    
    X_raw = df[feature_cols].fillna(0).values
    
    # 3. Scale Features and Target
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
    
    # 4. Load Graph Topology
    graph_path = os.path.join(base_dir, 'data', 'processed', 'graph.pkl')
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph topology not found: {graph_path}")
        
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
        
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Convert to undirected edge_index
    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    
    edges = list(set(tuple(e) for e in edges)) # Remove duplicates
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # 5. Create Train/Val/Test Masks (70/15/15)
    num_nodes = len(X_raw)
    indices = np.random.permutation(num_nodes)
    train_end = int(0.7 * num_nodes)
    val_end = int(0.85 * num_nodes)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # 5.5 Mask Permanent Water Bodies
    if 'ndwi' in df.columns:
        water_mask = df['ndwi'] > -0.10
        water_indices = np.where(water_mask)[0]
        train_mask[water_indices] = False
        val_mask[water_indices] = False
        test_mask[water_indices] = False
        print(f"Masked {len(water_indices)} permanent water body nodes from loss evaluation.")
    
    # 6. Build PyG Data Object
    x_tensor = torch.tensor(X_scaled, dtype=torch.float)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float)
    
    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    # 7. Package and Export
    package = {
        'data': data,
        'feature_cols': feature_cols,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    
    out_path = os.path.join(base_dir, 'data', 'processed', 'training_dataset.pt')
    torch.save(package, out_path)
    
    print(f"Dataset packaged successfully at {out_path}")
    print(f"Features: {len(feature_cols)}")
    print(f"Train/Val/Test: {train_mask.sum().item()} / {val_mask.sum().item()} / {test_mask.sum().item()}")

if __name__ == "__main__":
    package_dataset()
