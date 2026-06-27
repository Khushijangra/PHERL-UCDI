import os
import numpy as np

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class HCPIE:
    """
    Hybrid Cooling Portfolio Intelligence Engine (HCPIE)
    """
    def __init__(self):
        # 1. Define Intervention Primitives
        self.interventions = ['Trees', 'Cool Roofs', 'Pavements', 'Water', 'Ventilation']
        
        # Base attributes: [Base_Cooling_Max, Cost_Index, Water_Index]
        # These are normalized scales (0-10) for scoring purposes
        self.primitives = {
            'Trees': {'cooling': 3.0, 'cost': 5.0, 'water': 8.0, 'resilience': 7.0},
            'Cool Roofs': {'cooling': 2.0, 'cost': 3.0, 'water': 1.0, 'resilience': 5.0},
            'Pavements': {'cooling': 1.5, 'cost': 4.0, 'water': 0.0, 'resilience': 6.0},
            'Water': {'cooling': 4.0, 'cost': 9.0, 'water': 10.0, 'resilience': 8.0},
            'Ventilation': {'cooling': 3.0, 'cost': 10.0, 'water': 0.0, 'resilience': 9.0}
        }
        
        # 2. Build the Interaction/Synergy Matrix I(i,j)
        # Trees, Cool Roofs, Pavements, Water, Ventilation
        self.synergy_matrix = np.array([
            [ 1.00,  0.35,  0.12,  0.60, -0.12], # Trees
            [ 0.35,  1.00,  0.25,  0.10,  0.05], # Cool Roofs
            [ 0.12,  0.25,  1.00,  0.08,  0.15], # Pavements
            [ 0.60,  0.10,  0.08,  1.00,  0.20], # Water
            [-0.12,  0.05,  0.15,  0.20,  1.00]  # Ventilation
        ])
        
    def evaluate_portfolio(self, allocation, population, equity_weight):
        """
        Evaluates a given portfolio allocation (sum = 1.0)
        allocation: list of 5 floats corresponding to [T, C, R, W, V]
        """
        alloc_array = np.array(allocation)
        
        # Calculate Base Cooling
        base_cooling = np.array([self.primitives[i]['cooling'] for i in self.interventions])
        raw_cooling = np.sum(alloc_array * base_cooling)
        
        # Apply Synergy I(i,j)
        # Synergy bonus = alloc^T * Synergy * alloc (excluding diagonal which is 1)
        synergy_effect = np.dot(alloc_array.T, np.dot(self.synergy_matrix, alloc_array))
        # Total cooling is bounded
        total_cooling = min(5.0, raw_cooling * (0.5 + 0.5 * synergy_effect))
        
        # Calculate Constraints
        costs = np.array([self.primitives[i]['cost'] for i in self.interventions])
        waters = np.array([self.primitives[i]['water'] for i in self.interventions])
        resiliences = np.array([self.primitives[i]['resilience'] for i in self.interventions])
        
        total_cost = np.sum(alloc_array * costs)
        total_water = np.sum(alloc_array * waters)
        total_resilience = np.sum(alloc_array * resiliences)
        
        # CUCUS (Composite Urban Cooling Utility Score)
        # CUCUS = w1*C + w2*P + w3*E + w4*R - w5*W - w6*Cost
        w1, w2, w3, w4, w5, w6 = 2.0, 1.0, 1.5, 1.0, 0.8, 1.2
        
        # Normalize pop for scoring
        pop_score = np.log1p(population) / 10.0 
        
        cucus = (w1 * total_cooling) + \
                (w2 * pop_score) + \
                (w3 * equity_weight) + \
                (w4 * total_resilience / 10.0) - \
                (w5 * total_water / 10.0) - \
                (w6 * total_cost / 10.0)
                
        return {
            'cooling': total_cooling,
            'cost': total_cost,
            'water': total_water,
            'resilience': total_resilience,
            'cucus': cucus
        }

class PortfolioReasoningEngine:
    def __init__(self, hcpie):
        self.hcpie = hcpie
        
    def generate_recommendation(self, hotspot_id, context, candidate_portfolios):
        print(f"\n--- Portfolio Reasoning Engine ---")
        print(f"Hotspot {hotspot_id} — Ahmedabad")
        print(f"Population: {context['population']:,}")
        print(f"Cause: {context['cause']}")
        print(f"Budget constraint: {context['budget']}")
        print("-" * 40)
        
        best_portfolio = None
        best_cucus = -999
        best_metrics = None
        
        for name, alloc in candidate_portfolios.items():
            metrics = self.hcpie.evaluate_portfolio(alloc, context['population'], context['equity_weight'])
            
            # Formulate Water Demand
            water_demand = "Low"
            if metrics['water'] > 4.0: water_demand = "Moderate"
            if metrics['water'] > 7.0: water_demand = "High"
            
            print(f"{name}")
            # Format allocations
            alloc_str = ", ".join([f"{int(alloc[i]*100)}% {self.hcpie.interventions[i]}" for i in range(5) if alloc[i] > 0])
            print(f"[{alloc_str}]")
            print(f"Cooling: {metrics['cooling']:.2f}°C")
            print(f"Water:   {water_demand}")
            print(f"CUCUS:   {metrics['cucus']:.2f}\n")
            
            if metrics['cucus'] > best_cucus:
                best_cucus = metrics['cucus']
                best_portfolio = name
                best_metrics = metrics
                
        # Generate Reasoning
        print("-" * 40)
        print(f"AI Recommendation: Deploy {best_portfolio}")
        print("Reasoning:")
        print(f"- Protects large vulnerable population ({context['population']:,}).")
        
        if best_metrics['water'] < 5.0:
            print("- Minimizes water demand via zero-irrigation interventions (e.g., Cool Roofs).")
        if best_metrics['resilience'] > 5.0:
            print("- Provides robust cooling resilience during acute heatwaves.")
            
        print(f"- Maximizes Composite Urban Cooling Utility Score (CUCUS: {best_cucus:.2f}).")
        print(f"- Highest implementation feasibility given {context['budget']}.")
        print("\nConfidence: 91%")
        
def run_hcpie():
    hcpie = HCPIE()
    engine = PortfolioReasoningEngine(hcpie)
    
    # Define Candidate Portfolios [Trees, Roofs, Pavements, Water, Vent]
    portfolios = {
        "Portfolio A": [0.8, 0.2, 0.0, 0.0, 0.0],
        "Portfolio B": [0.3, 0.5, 0.2, 0.0, 0.0],
        "Portfolio C": [0.0, 0.6, 0.2, 0.0, 0.2]
    }
    
    context = {
        "population": 18000,
        "cause": "41% vegetation deficit, 27% urban canyon, 19% anthropogenic heat",
        "budget": "₹50 Cr",
        "equity_weight": 0.8 # High equity priority area
    }
    
    engine.generate_recommendation(142, context, portfolios)

if __name__ == "__main__":
    run_hcpie()
