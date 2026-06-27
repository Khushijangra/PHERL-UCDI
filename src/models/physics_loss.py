import torch
import torch.nn.functional as F

def compute_physics_loss(model, x, edge_index, y_true, mask, feature_cols, lambda_phys=0.1):
    """
    Computes standard MSE Loss + Physics-Informed Monotonic Penalty Loss.
    Ensures that the Digital Twin obeys fundamental urban climate laws.
    """
    # Enable gradient tracking on input features to compute sensitivity
    x.requires_grad_(True)
    out = model(x, edge_index).squeeze()
    
    # Base prediction loss (MSE)
    mse_loss = F.mse_loss(out[mask], y_true[mask])
    
    # Compute Jacobian-vector product (Sensitivities of LST to Inputs)
    gradients = torch.autograd.grad(outputs=out, inputs=x, 
                                    grad_outputs=torch.ones_like(out), 
                                    create_graph=True)[0]
                                    
    loss_phys = 0.0
    col_map = {col: i for i, col in enumerate(feature_cols)}
    
    # 1. Vegetation cools (dLST/dNDVI < 0) -> Penalize gradient > 0
    if 'ndvi' in col_map:
        idx = col_map['ndvi']
        loss_phys += F.relu(gradients[:, idx]).mean()
        
    # 2. Impervious warms (dLST/dNDBI > 0) -> Penalize gradient < 0
    if 'ndbi' in col_map:
        idx = col_map['ndbi']
        loss_phys += F.relu(-gradients[:, idx]).mean()
        
    # 3. Albedo cools (dLST/dAlbedo < 0) -> Penalize gradient > 0
    if 'albedo' in col_map:
        idx = col_map['albedo']
        loss_phys += F.relu(gradients[:, idx]).mean()
        
    # 4. Wind cools (dLST/dWindSpeed < 0) -> Penalize gradient > 0
    # Assuming wind_speed exists, or wind_u/wind_v magnitude
    if 'wind_speed' in col_map:
        idx = col_map['wind_speed']
        loss_phys += F.relu(gradients[:, idx]).mean()
        
    total_loss = mse_loss + (lambda_phys * loss_phys)
    
    # Return as tensors for backprop, plus item() for logging
    return total_loss, mse_loss.item(), (loss_phys.item() if isinstance(loss_phys, torch.Tensor) else 0.0)
