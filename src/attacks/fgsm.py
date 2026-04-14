import torch
import torch.nn as nn

def fgsm_attack(model, device, x, y, epsilon, constraint_fn=None):
    """
    Fast Gradient Sign Method (FGSM) attack with optional domain constraints.
    """
    x = x.to(device)
    y = y.to(device)
    x.requires_grad = True
    
    # Forward pass
    output = model(x)
    loss = nn.BCELoss()(output, y.view_as(output))
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Get gradients
    grad = x.grad.data
    
    # Apply immutable feature mask
    if constraint_fn is not None:
        grad = constraint_fn.apply_gradient_mask(grad)
    
    # Generate perturbation
    x_adv = x - epsilon * grad.sign()
    
    # Apply post-perturbation constraints (bounds, rounding, consistency)
    if constraint_fn is not None:
        x_adv = constraint_fn.project(x_adv, x)
    
    return x_adv.detach()
