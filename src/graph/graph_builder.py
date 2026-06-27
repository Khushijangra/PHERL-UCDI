import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import libpysal
from sklearn.metrics.pairwise import cosine_similarity
import math

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def load_feature_cube():
    cube_path = os.path.join(get_base_dir(), 'data', 'processed', 'master_feature_cube.parquet')
    if not os.path.exists(cube_path):
        # Return a dummy dataframe for code scaffolding if cube doesn't exist yet
        print("Warning: master_feature_cube.parquet not found. Generating dummy 100x100 grid for testing.")
        dummy_data = {
            'cell_id': range(10000),
            'latitude': np.random.uniform(23.01, 23.05, 10000),
            'longitude': np.random.uniform(72.52, 72.57, 10000),
            'wind_u': np.random.normal(2, 1, 10000),
            'wind_v': np.random.normal(2, 1, 10000),
            'building_density': np.random.uniform(0, 1, 10000),
            'ndvi': np.random.uniform(0, 0.8, 10000)
        }
        return pd.DataFrame(dummy_data)
    return pd.read_parquet(cube_path)

def build_spatial_edges(df):
    """
    Edge Type 1: Queen Adjacency
    Weight: 1 / Euclidean distance (simplified to 1 for adjacent 50m cells)
    """
    print("Computing Spatial Adjacency Edges...")
    # Using libpysal KNN or Queen if geometries exist. 
    # For robust simulation on lat/lon grids, Delaunay/KNN is fast.
    coordinates = df[['longitude', 'latitude']].values
    kd = libpysal.cg.KDTree(coordinates)
    # 8 nearest neighbors approx Queen contiguity for a regular grid
    wnn = libpysal.weights.KNN(kd, k=8)
    
    edges = []
    for i, neighbors in wnn.neighbors.items():
        for j in neighbors:
            edges.append((i, j, {'type': 'spatial', 'weight': 1.0}))
    return edges

def build_wind_edges(df):
    """
    Edge Type 2: Wind Advection
    Weight: wind_speed * alignment (cosine of angle between wind vector and edge vector)
    """
    print("Computing Wind Advection Edges...")
    edges = []
    # To avoid O(N^2), we only compute advection between spatial neighbors
    coords = df[['longitude', 'latitude']].values
    wind_u = df['wind_u'].values
    wind_v = df['wind_v'].values
    
    # We'll use the spatial neighbors to define potential advection pathways
    kd = libpysal.cg.KDTree(coords)
    wnn = libpysal.weights.KNN(kd, k=8)
    
    for i, neighbors in wnn.neighbors.items():
        wu, wv = wind_u[i], wind_v[i]
        wind_speed = math.sqrt(wu**2 + wv**2)
        if wind_speed < 0.1: continue # Negligible wind
        
        for j in neighbors:
            # Vector from i to j
            dx = coords[j][0] - coords[i][0]
            dy = coords[j][1] - coords[i][1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist == 0: continue
            
            # Dot product for alignment
            alignment = (wu*dx + wv*dy) / (wind_speed * dist)
            if alignment > 0: # Wind is blowing from i towards j
                edges.append((i, j, {'type': 'wind', 'weight': wind_speed * alignment}))
                
    return edges

def build_morphology_edges(df):
    """
    Edge Type 3: Morphological Similarity
    Weight: Cosine similarity of urban morphology vectors
    """
    print("Computing Morphological Similarity Edges...")
    # Define morphological features
    morph_features = ['building_density', 'ndvi']
    X = df[morph_features].fillna(0).values
    
    # Cosine similarity is O(N^2). For 10,000 nodes, it's 100M pairs.
    # We will limit edges to high similarity pairs (similarity > 0.95) within a localized radius (K=20)
    coords = df[['longitude', 'latitude']].values
    kd = libpysal.cg.KDTree(coords)
    wnn = libpysal.weights.KNN(kd, k=20)
    
    edges = []
    for i, neighbors in wnn.neighbors.items():
        v_i = X[i].reshape(1, -1)
        v_j = X[neighbors]
        # Avoid zero-vector warnings
        if np.linalg.norm(v_i) == 0: continue
        
        sims = cosine_similarity(v_i, v_j)[0]
        for idx, j in enumerate(neighbors):
            if sims[idx] > 0.95 and i != j:
                edges.append((i, j, {'type': 'morphology', 'weight': sims[idx]}))
                
    return edges

def construct_graph_and_validate():
    print("PHASE 2: Urban Climate Graph Construction...")
    df = load_feature_cube()
    N = len(df)
    print(f"Loaded {N} nodes.")
    
    G = nx.MultiDiGraph() # MultiDiGraph allows multiple edge types between same nodes
    G.add_nodes_from(range(N))
    
    # Build Edges
    spatial_edges = build_spatial_edges(df)
    wind_edges = build_wind_edges(df)
    morph_edges = build_morphology_edges(df)
    
    G.add_edges_from(spatial_edges)
    G.add_edges_from(wind_edges)
    G.add_edges_from(morph_edges)
    
    # Calculate Statistics
    e_spatial = len(spatial_edges)
    e_wind = len(wind_edges)
    e_morph = len(morph_edges)
    total_edges = G.number_of_edges()
    
    # Graph Density = E / (N * (N-1))
    density = total_edges / (N * (N - 1)) if N > 1 else 0
    avg_degree = total_edges / N
    
    # Check Connectivity (using underlying simple graph)
    G_simple = nx.Graph(G) 
    connected_components = nx.number_connected_components(G_simple)
    
    # Output Metadata
    metadata = {
      "city": "Ahmedabad",
      "resolution": "50m",
      "nodes": N,
      "features": len(df.columns),
      "edge_types": ["spatial", "wind", "morphological"],
      "actions": ["trees", "coolroof", "reflective_pavement", "water", "ventilation"]
    }
    
    meta_path = os.path.join(get_base_dir(), 'data', 'processed', 'graph_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    import pickle
    graph_path = os.path.join(get_base_dir(), 'data', 'processed', 'graph.pkl')
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph object saved to {graph_path}")
        
    # Output Feasibility Report
    report = f"""# Urban Climate Graph Feasibility Report

## Topology Metrics
* **Total Nodes (N)**: {N} (Target: ~10,000)
* **Spatial Edges**: {e_spatial}
* **Wind Advection Edges**: {e_wind}
* **Morphology Edges**: {e_morph}
* **Total Edges**: {total_edges}
* **Average Degree**: {avg_degree:.2f}
* **Graph Density**: {density:.6f}
* **Connected Components**: {connected_components} (Ideal: 1)

## Architecture Assessment
The graph utilizes a MultiDiGraph topology ensuring that purely physical connections (Queen contiguity) are isolated from fluid dynamics (Wind Advection) and semantic representations (Morphology).

*If Connected Components > 1, the Digital Twin will suffer from isolated thermal islands unable to propagate advection.*
"""
    report_path = os.path.join(get_base_dir(), 'data', 'processed', 'graph_feasibility_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
        
    print(f"Graph Metadata saved to {meta_path}")
    print(f"Graph Feasibility Report saved to {report_path}")

if __name__ == "__main__":
    construct_graph_and_validate()
