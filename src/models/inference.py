import os
import json
import torch
import pickle

from src.models.digital_twin import GCNModel, GraphSAGEModel, GATModel

# Registry mapping config/metadata names to their classes
ARCHITECTURE_REGISTRY = {
    "GCN":          GCNModel,
    "GraphSAGE":    GraphSAGEModel,
    "GAT":          GATModel,
    "Physics_GNN":  GCNModel,  # Physics GNN uses GCN backbone with a physics loss
}

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def load_inference_model():
    """
    Dynamically loads the correct best model for inference.
    
    Resolution order:
    1. Read models/best_model_metadata.json to find the winning model name.
    2. Read models/<winning_model>_config.json for the exact hyperparameters.
    3. Instantiate the correct architecture class with those hyperparameters.
    4. Load the saved state_dict into the instantiated model.
    
    Returns:
        model: The loaded model (sklearn or PyTorch)
        model_type: "tabular" | "pytorch"
    """
    base_dir = get_base_dir()
    models_dir = os.path.join(base_dir, 'models')
    
    # --- Step 1: Resolve winning model from metadata ---
    metadata_path = os.path.join(models_dir, 'best_model_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"{metadata_path} not found. Run compare_models.py first to generate it."
        )
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    best_model_name = metadata['best_model_name']
    is_tabular = metadata['is_tabular']
    print(f"Resolved best model: {best_model_name} (tabular={is_tabular})")
    
    model_path = os.path.join(models_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run compare_models.py first."
        )
    
    # --- Step 2: Handle tabular models ---
    if is_tabular:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, "tabular"
    
    # --- Step 3: Read architecture config for exact hyperparameters ---
    config_path = os.path.join(models_dir, f"{best_model_name.lower()}_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Architecture config not found: {config_path}. "
            f"Ensure train_gnn.py writes a config for every model."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    hidden_dim = config.get('hidden_dim', 64)
    dropout    = config.get('dropout', 0.2)
    
    # --- Step 4: Resolve num_features from training dataset ---
    data_path = os.path.join(base_dir, 'data', 'processed', 'training_dataset.pt')
    dataset = torch.load(data_path, map_location='cpu', weights_only=False)
    num_features = dataset['data'].x.shape[1]
    
    # --- Step 5: Instantiate the correct architecture ---
    architecture_class = ARCHITECTURE_REGISTRY.get(best_model_name)
    if architecture_class is None:
        raise ValueError(
            f"Unknown architecture '{best_model_name}'. "
            f"Add it to ARCHITECTURE_REGISTRY in inference.py."
        )
    
    model = architecture_class(
        num_features=num_features,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    # --- Step 6: Load state dict ---
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Successfully loaded {best_model_name} "
          f"(features={num_features}, hidden_dim={hidden_dim}, dropout={dropout})")
    return model, "pytorch"


if __name__ == "__main__":
    model, model_type = load_inference_model()
    print(f"Model type: {model_type}")
    print(f"Model: {model}")
