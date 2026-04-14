import torch
import torch.nn as nn

def pgd_attack(model, device, x, y, epsilon, alpha, num_iter, constraint_fn=None):
    """
    Projected Gradient Descent (PGD) attack with optional domain constraints.
    """
    x = x.to(device)
    y = y.to(device)
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()
    
    for i in range(num_iter):
        x_adv.requires_grad = True
        
        # Forward pass
        output = model(x_adv)
        loss = nn.BCELoss()(output, y.view_as(output))
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Get and mask gradients
        grad = x_adv.grad.data
        if constraint_fn is not None:
            grad = constraint_fn.apply_gradient_mask(grad)
            
        # Update adversarial example
        x_adv = x_adv.detach() - alpha * grad.sign()
        
        # Project back to epsilon ball
        delta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = x_orig + delta
        
        # Apply domain constraints
        if constraint_fn is not None:
            x_adv = constraint_fn.project(x_adv, x_orig)
            
        x_adv = x_adv.detach()
        
    return x_adv
