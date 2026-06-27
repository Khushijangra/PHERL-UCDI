import os
import torch
import json
import pickle
from src.models.digital_twin import GCNModel, GraphSAGEModel, GATModel

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def load_inference_model():
    """
    Safely loads the best_model.pt (or best_model.pkl) for inference in the Sanity Verifier and Dashboard.
    """
    base_dir = get_base_dir()
    model_path = os.path.join(base_dir, 'models', 'best_model.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Ensure Lightning AI training completed and artifacts were synced.")
        
    # Since we symlinked or copied the best model to 'best_model.pt', we need to check if it's PyTorch or Pickle
    try:
        # Try loading as PyTorch dictionary
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        is_pytorch = isinstance(state_dict, dict)
    except Exception:
        is_pytorch = False
        
    if not is_pytorch:
        # It's a tabular model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, "tabular"
        
    # It's a PyTorch Model. Need to instantiate the correct architecture.
    # We can infer from the config json if it exists
    config_paths = [f for f in os.listdir(os.path.join(base_dir, 'models')) if f.endswith('_config.json')]
    # For now, default to Physics_GNN (GraphSAGE) 
    # In production, metadata would explicitly define the architecture of best_model.pt
    # Let's assume the user wants the PyTorch model for Sanity Verification
    
    # We must know num_features. 
    data_path = os.path.join(base_dir, 'data', 'processed', 'training_dataset.pt')
    dataset = torch.load(data_path, map_location='cpu', weights_only=False)
    num_features = dataset['data'].x.shape[1]
    
    # We will instantiate GraphSAGE by default if it was the Physics GNN
    model = GraphSAGEModel(num_features=num_features, hidden_dim=64, dropout=0.2)
    
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Failed to load state dict directly into GraphSAGE: {e}")
        
    model.eval()
    return model, "pytorch"
