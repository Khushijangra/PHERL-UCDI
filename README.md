# PHERL-UCDI++
**Physics-informed Hybrid Explainable Urban Cooling Decision Intelligence Platform**

*Official Submission for the ISRO Bharatiya Antariksh Hackathon 2026*

---

## 🌍 Problem Statement
Optimizing Urban Heat Mitigation and Cooling Strategies via AIML.

Current approaches to Urban Heat Island (UHI) mapping rely heavily on predicting temperature for isolated pixels (tabular machine learning). PHERL-UCDI++ challenges this approach with a novel scientific thesis: **Urban heat is a topological graph phenomenon governed by spatial advection and morphological networks, not just isolated pixel characteristics.** 

This platform does not merely ask *where* cities are hot. It asks *how India should cool its cities scientifically, equitably, economically, and resiliently*.

## 📡 Dataset
The dataset represents a 5km × 5km prototype grid over Ahmedabad, generated at a 50m spatial resolution. It fuses multiple multimodal datasets:
- **Thermal:** Landsat 8/9 LST
- **Optical:** Sentinel 2 (NDVI, NDBI, NDWI, Albedo proxy)
- **Meteorology:** ERA5 (Temperature, Humidity, Wind U/V, Pressure, Solar Radiation)
- **Urban:** OpenStreetMap, GHSL
- **Socioeconomic:** WorldPop, VIIRS Night Lights

## 🔬 Methodology & Architecture
PHERL-UCDI++ extracts raw satellite data, engineers 39 physical and socioeconomic features, and constructs a spatial graph representation of the city (9,555 nodes).

The core **Digital Twin** leverages a **Graph Convolutional Network (GCN)**, which mathematically proved its superiority by outperforming strong tabular baselines (XGBoost, LightGBM, Random Forest) through its ability to capture neighborhood thermodynamic interactions. 

The platform then utilizes gradient-based explainability, counterfactual simulations, and the NSGA-II genetic algorithm to synthesize Pareto-optimal cooling strategies.

## 📊 Results & Scientific Validation
1. **Model Leaderboard:** GCN (RMSE: 0.438°C, R²: 0.902) outperformed the best tabular baseline, XGBoost (RMSE: 0.563°C, R²: 0.837).
2. **Physics Sanity Checks (Passed):** 
   - NDVI increase (+1σ) → Cooling (-0.10°C)
   - Albedo increase (+1σ) → Cooling (-0.03°C)
   - NDBI increase (+1σ) → Heating (+0.16°C)
3. **Counterfactual Interventions:** Demonstrated measurable cooling yields from Tree Plantation and Cool Roofs deployments on the top 10 urban hotspots.

## 💻 Decision Support Dashboard
An interactive Streamlit dashboard serves as the platform's frontend, providing policymakers with an executive overview, interactive urban heat maps, model explainability interfaces, and an NSGA-II portfolio optimization engine with adjustable budget constraints.

## 🛠️ Installation & Reproducibility
```bash
# Clone repository
git clone https://github.com/Khushijangra/PHERL-UCDI.git
cd PHERL-UCDI

# Install dependencies
pip install -r requirements.txt

# Run the decision-support dashboard
streamlit run dashboard/app.py
```

### Reproducibility
The `v1.0-isro-final` release ensures absolute reproducibility. All critical data, models, and reports are hashed in `artifact_manifest.json`. To verify the cryptographic integrity of the environment:
```bash
python -m src.validation.freeze_artifacts
```

## 📁 Repository Structure
```text
PHERL-UCDI/
├── data/
│   ├── raw/                 # Earth Engine extractions
│   └── processed/           # Master Feature Cube & Graph Datasets
├── src/
│   ├── data/                # Downloaders and packagers
│   ├── features/            # Feature engineering
│   ├── graph/               # PyTorch Geometric construction
│   ├── models/              # GCN, baselines, and NSGA-II engine
│   └── validation/          # Explainability & Counterfactual engines
├── models/                  # Pre-trained weights and metadata
├── reports/                 # Results, CSVs, and PNG plots
├── dashboard/               # Streamlit application
├── configs/                 # Experiment hyperparameter JSONs
└── requirements.txt
```

## 📜 Citation & License
This project was developed for the ISRO Bharatiya Antariksh Hackathon 2026. Code released under the MIT License.
