# Urban Heat Mitigation: NSGA-II Optimization Report

## Problem Definition
The optimization engine uses the NSGA-II genetic algorithm to identify the Pareto-optimal frontier balancing **Implementation Cost** and **Urban Cooling**. The decision variables map directly to the interventions validated by the GCN Digital Twin.

## Decision Variables
- $x_1$: Tree Plantation (%)
- $x_2$: Cool Roofs (%)
- $x_3$: Reflective Pavements (%)
- $x_4$: Green Corridors (%)

## Optimal Portfolio Extracts

| Budget (₹ Cr) | Cooling (°C) | Trees | Cool Roofs | Pavements | Corridors |
|---|---|---|---|---|---|
| Rs.0.00 | 0.00°C | 0.0% | 0.0% | 0.0% | 0.0% |
| Rs.12.92 | 0.55°C | 34.0% | 97.5% | 0.1% | 0.0% |
| Rs.29.63 | 1.01°C | 100.0% | 100.0% | 9.1% | 14.5% |
| Rs.52.82 | 1.18°C | 100.0% | 100.0% | 0.2% | 99.2% |
| Rs.78.00 | 1.34°C | 100.0% | 100.0% | 100.0% | 100.0% |
