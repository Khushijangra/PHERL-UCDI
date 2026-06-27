import os
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class PortfolioOptimizationProblem(ElementwiseProblem):
    """
    7D Objective Space Optimization for PHERL-UCDI++
    """
    def __init__(self, primitives, synergy_matrix, context):
        # 5 Variables: [Trees, Cool Roofs, Pavements, Water, Ventilation]
        # 7 Objectives: Cooling, Pop, Equity, Energy, Resilience, Cost, Water
        # 1 Constraint: Sum of allocations must equal 1.0 (handled via normalization)
        super().__init__(n_var=5, n_obj=7, n_ieq_constr=0, xl=0.0, xu=1.0)
        
        self.primitives = primitives
        self.synergy_matrix = synergy_matrix
        self.context = context
        self.interventions = ['Trees', 'Cool Roofs', 'Pavements', 'Water', 'Ventilation']

    def _evaluate(self, x, out, *args, **kwargs):
        # Normalize allocations so they sum to 1.0
        alloc_sum = np.sum(x)
        if alloc_sum == 0:
            alloc = np.ones(5) / 5.0
        else:
            alloc = x / alloc_sum
            
        # Base attributes
        costs = np.array([self.primitives[i]['cost'] for i in self.interventions])
        waters = np.array([self.primitives[i]['water'] for i in self.interventions])
        resiliences = np.array([self.primitives[i]['resilience'] for i in self.interventions])
        base_cooling = np.array([self.primitives[i]['cooling'] for i in self.interventions])
        energies = np.array([self.primitives[i]['energy'] for i in self.interventions])
        
        # Linear combinations
        raw_cooling = np.sum(alloc * base_cooling)
        total_cost = np.sum(alloc * costs)
        total_water = np.sum(alloc * waters)
        total_res = np.sum(alloc * resiliences)
        total_energy = np.sum(alloc * energies)
        
        # Synergy effect on cooling
        synergy_effect = np.dot(alloc.T, np.dot(self.synergy_matrix, alloc))
        total_cooling = min(5.0, raw_cooling * (0.5 + 0.5 * synergy_effect))
        
        # Contextual impacts
        pop_protected = self.context['population'] * (total_cooling / 5.0)
        equity_imp = self.context['equity_weight'] * total_cooling
        
        # Objectives (pymoo minimizes by default, so we negate what we want to maximize)
        f1 = -total_cooling     # Maximize Cooling
        f2 = -pop_protected     # Maximize Pop Protected
        f3 = -equity_imp        # Maximize Equity
        f4 = -total_energy      # Maximize Energy Savings
        f5 = -total_res         # Maximize Resilience
        f6 = total_cost         # Minimize Cost
        f7 = total_water        # Minimize Water
        
        out["F"] = [f1, f2, f3, f4, f5, f6, f7]

def run_nsga_optimizer():
    print("PHASE 6: NSGA-II Pareto Climate Optimizer (7D Space)")
    
    # Extended Primitives for 7D Space
    primitives = {
        'Trees': {'cooling': 3.0, 'cost': 5.0, 'water': 8.0, 'resilience': 7.0, 'energy': 4.0},
        'Cool Roofs': {'cooling': 2.0, 'cost': 3.0, 'water': 1.0, 'resilience': 5.0, 'energy': 8.0},
        'Pavements': {'cooling': 1.5, 'cost': 4.0, 'water': 0.0, 'resilience': 6.0, 'energy': 1.0},
        'Water': {'cooling': 4.0, 'cost': 9.0, 'water': 10.0, 'resilience': 8.0, 'energy': 2.0},
        'Ventilation': {'cooling': 3.0, 'cost': 10.0, 'water': 0.0, 'resilience': 9.0, 'energy': 5.0}
    }
    
    synergy_matrix = np.array([
        [ 1.00,  0.35,  0.12,  0.60, -0.12], # Trees
        [ 0.35,  1.00,  0.25,  0.10,  0.05], # Cool Roofs
        [ 0.12,  0.25,  1.00,  0.08,  0.15], # Pavements
        [ 0.60,  0.10,  0.08,  1.00,  0.20], # Water
        [-0.12,  0.05,  0.15,  0.20,  1.00]  # Ventilation
    ])
    
    context = {
        "population": 126000, # Western Ahmedabad Context
        "equity_weight": 0.85
    }
    
    problem = PortfolioOptimizationProblem(primitives, synergy_matrix, context)
    
    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=50,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    print("Running multi-objective evolutionary search...")
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 50),
                   seed=42,
                   verbose=False)
                   
    print(f"Discovered {len(res.X)} Pareto-optimal portfolios.")
    
    # Extracting some interesting points from the Pareto front
    # E.g., Max Cooling, Min Cost, Min Water
    f_vals = res.F
    x_vals = res.X
    
    # Normalize X to percentages
    row_sums = x_vals.sum(axis=1)
    x_normalized = x_vals / row_sums[:, np.newaxis]
    
    max_cooling_idx = np.argmin(f_vals[:, 0]) # argmin because we negated cooling
    min_cost_idx = np.argmin(f_vals[:, 5])
    min_water_idx = np.argmin(f_vals[:, 6])
    
    print("\n--- Pareto Frontier Examples ---")
    
    def print_portfolio(name, idx):
        alloc = x_normalized[idx]
        alloc_str = ", ".join([f"{int(alloc[i]*100)}% {problem.interventions[i]}" for i in range(5) if alloc[i] >= 0.01])
        print(f"\n{name} Strategy:")
        print(f"Portfolio: [{alloc_str}]")
        print(f"Cooling: {-f_vals[idx, 0]:.2f}°C")
        print(f"Cost Index: {f_vals[idx, 5]:.2f}/10")
        print(f"Water Index: {f_vals[idx, 6]:.2f}/10")
        
    print_portfolio("Maximum Cooling", max_cooling_idx)
    print_portfolio("Budget Efficient", min_cost_idx)
    print_portfolio("Water Efficient", min_water_idx)
    
    print("\nPhase 6 Complete. Pareto Engine ready for SCRE.")

if __name__ == "__main__":
    run_nsga_optimizer()
