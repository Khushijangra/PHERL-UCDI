# Digital Twin Sanity Verification Report

This report validates that the underlying optimized model correctly models known urban climate thermodynamics. Only if all tests pass can the model be promoted to the 'Urban Digital Twin' status and utilized for SHAP causality and RL policy optimization.

| Test | Expected | Actual (Scaled ΔT) | Pass |
|---|---|---|---|
| NDVI (+1σ) | Cooling (ΔT < 0) | -0.10°C | ✅ |
| Albedo (+1σ) | Cooling (ΔT < 0) | -0.03°C | ✅ |
| NDBI (+1σ) | Heating (ΔT > 0) | +0.16°C | ✅ |
