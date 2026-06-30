# PHERL-UCDI++ System Architecture

This document maps the flow of data, computation, and decision logic throughout the PHERL-UCDI++ platform.

## High-Level Architecture Flow

```text
Satellite Data & Geodatabases
        │   (Landsat, Sentinel, ERA5, OSM, WorldPop)
        ▼
Feature Engineering Engine
        │   (39 features: thermal, optical, meteorological, socioeconomic)
        ▼
Master Feature Cube
        │   (Parquet format, rigorously masked for water and outliers)
        ▼
Spatial Graph Construction
        │   (PyTorch Geometric: 9,555 nodes connected by k-NN / distance)
        ▼
Digital Twin (Graph Convolutional Network)
        │   (Validates thermodynamic physics, outperforms tabular models)
        │
        ├────────► Explainability Engine
        │          (Gradient-based feature sensitivity, residual analysis)
        │
        ├────────► Counterfactual Simulator
        │          (In-graph simulation of targeted interventions on hotspots)
        │
        ├────────► NSGA-II Optimization Engine
        │          (Pymoo: Generates Pareto optimal intervention portfolios)
        │
        ▼
Decision Support Dashboard
        (Streamlit frontend querying pre-computed scientific artifacts)
```

## Component Details

### 1. Data Acquisition & Processing (`src/data/`)
Multi-modal satellite imagery is extracted using Earth Engine (`gee_extractor.py`), while vector geometry is acquired from OpenStreetMap (`osm_extractor.py`). The inputs are co-registered and resampled to a consistent 50m spatial resolution.

### 2. Feature Engineering & Validation (`src/features/`)
Derives 39 complex geospatial indices (NDVI, NDBI, Albedo proxy, morphological building density). The `validate_feature_cube.py` layer enforces rigorous scientific constraints, ensuring no target leakage (LST variance) propagates downstream, and properly masks non-relevant nodes (e.g., river pixels).

### 3. Spatial Graph Construction (`src/graph/`)
Converts the pixel-wise feature cube into a topological graph. Nodes represent 50m urban patches. Edges represent spatial adjacency, allowing the downstream models to learn advection and neighborhood effects (e.g., cooling breeze from a nearby park).

### 4. Core Modeling & Digital Twin (`src/models/`)
Trains a suite of models including XGBoost, LightGBM, Random Forest, GraphSAGE, GAT, and GCN. Dynamic architecture resolution ensures the best-performing model is loaded as the definitive Digital Twin, provided it passes thermodynamic physics constraints (`sanity_verifier.py`).

### 5. Validation & Release Pipeline (`src/validation/`)
- **Explainability:** Calculates spatial SHAP / gradient sensitivities.
- **Counterfactuals:** Modifies specific node features within the full graph context to simulate urban interventions.
- **Artifact Freeze:** Cryptographically hashes the complete environment to guarantee deterministic reproducibility.

### 6. Decision Support (`dashboard/`)
A lightweight, instant-response interface that queries the frozen artifacts. It translates deep learning weights into human-readable policy interventions, allowing decision-makers to explore the Pareto frontier of budget versus urban cooling.
