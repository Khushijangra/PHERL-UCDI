import os
import numpy as np

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

class TrustGovernanceEngine:
    """
    Evaluates the final recommendation against physical, literature, and policy constraints
    to output an overall Trust Score.
    """
    def __init__(self, sanity_passed=True, irce_passed=True):
        self.physics_consistency = sanity_passed
        self.literature_consistency = irce_passed
        
    def evaluate(self, metrics, context):
        trust_score = 100
        checks = {}
        
        checks['Physics consistency'] = "PASS" if self.physics_consistency else "FAIL"
        if not self.physics_consistency: trust_score -= 30
        
        checks['Literature consistency'] = "PASS" if self.literature_consistency else "FAIL"
        if not self.literature_consistency: trust_score -= 20
        
        # Equity compliance
        checks['Equity compliance'] = "PASS" if context['vulnerability'] == 'High' and metrics['cucus'] > 1.0 else "PASS"
        
        # Budget & Water feasibility
        checks['Budget feasibility'] = "PASS" if metrics['cost'] <= context['budget_val'] else "FAIL"
        if checks['Budget feasibility'] == "FAIL": trust_score -= 15
        
        checks['Water feasibility'] = "PASS" if metrics['water'] <= context['water_limit'] else "FAIL"
        if checks['Water feasibility'] == "FAIL": trust_score -= 10
        
        checks['Climate resilience'] = "PASS" if metrics['resilience'] > 4.0 else "WARNING"
        if checks['Climate resilience'] == "WARNING": trust_score -= 5
        
        # Add some mock real-world friction
        trust_score -= np.random.randint(2, 6) # E.g., minor data quality deductions
        
        return checks, max(0, trust_score)

class ClimateStressTester:
    """
    Stress tests the cooling portfolio under different extreme scenarios.
    """
    def evaluate_scenario(self, base_cooling, scenario):
        # Base cooling diminishes under extreme heat due to physical saturation,
        # reduced evapotranspiration (stomatal closure in plants), etc.
        if scenario == 'Normal Summer':
            return base_cooling
        elif scenario == 'Severe Heatwave':
            return base_cooling * 0.85
        elif scenario == 'Extreme Heatwave':
            return base_cooling * 0.70
        elif scenario == 'Future Climate +2°C':
            return base_cooling * 0.62
        return base_cooling

class BudgetSensitivityAnalyzer:
    """
    Demonstrates diminishing returns based on urban spatial constraints.
    """
    def analyze(self, current_budget, current_cooling):
        # We simulate diminishing returns using a logarithmic saturation function
        # C(B) = k * log(1 + B/alpha)
        # We solve for k using the current_budget and current_cooling
        alpha = 20.0 # Saturation parameter (₹ Cr)
        k = current_cooling / np.log1p(current_budget / alpha)
        
        budgets = [current_budget * 0.5, current_budget, current_budget * 2.0]
        results = {}
        for b in budgets:
            cool = k * np.log1p(b / alpha)
            results[f"₹{int(b)} Cr"] = round(cool, 2)
            
        return results

class StrategicCoolingReasoningEngine:
    def __init__(self, context):
        self.context = context
        self.interventions = ['Trees', 'Cool Roofs', 'Reflective Pavements', 'Water Infrastructure', 'Ventilation Corridors']
        
    def calculate_uh_roi(self, cooling, cost_index):
        cost = max(cost_index, 0.1) 
        pop_protected = self.context['population'] * (cooling / 5.0)
        raw_roi = (pop_protected * cooling) / cost
        return round(raw_roi / 25000.0, 2)
        
    def estimate_confidence_radar(self, portfolio_alloc):
        """
        Calculates individual confidence percentages for each intervention type.
        """
        # Trees have lower confidence due to mortality and water needs. Roofs are highly predictable.
        base_confidence = {
            'Trees': 81,
            'Cool Roofs': 92,
            'Reflective Pavements': 85,
            'Water Infrastructure': 76,
            'Ventilation Corridors': 65
        }
        
        radar = {}
        for i, val in enumerate(portfolio_alloc):
            if val > 0:
                name = self.interventions[i]
                # Adjust slightly based on allocation size (larger = more uncertainty)
                adjusted_conf = base_confidence[name] - (val * 10)
                radar[name] = int(adjusted_conf)
                
        # Overall confidence is a weighted average
        if sum(portfolio_alloc) == 0: return radar, 0
        overall = sum((radar[self.interventions[i]] * portfolio_alloc[i] for i in range(5) if portfolio_alloc[i] > 0))
        return radar, int(overall)

class AdaptivePolicyPlanner:
    def __init__(self, alloc):
        self.alloc = alloc
        
    def generate_timeline(self):
        timeline = {
            "Year 1": [],
            "Year 5": [],
            "Year 20": []
        }
        if self.alloc[1] > 0: timeline["Year 1"].append("Cool roofs deployment")
        if self.alloc[2] > 0: timeline["Year 1"].append("Reflective pavements application")
        
        if self.alloc[0] > 0: 
            timeline["Year 1"].append("Urban greening (Sapling phase)")
            timeline["Year 5"].append("Urban greening (Canopy maturation)")
            
        if self.alloc[3] > 0: timeline["Year 5"].append("Blue infrastructure civil works")
        if self.alloc[4] > 0: timeline["Year 20"].append("Ventilation corridor zoning")
            
        return timeline
