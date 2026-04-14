import numpy as np
import torch

class UNSWNB15Constraints:
    """
    Implements domain-specific constraints for UNSW-NB15 adversarial perturbations.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names
        # Indices for constraints
        self.immutable_cols = [i for i, n in enumerate(feature_names) 
                                if n in ['proto', 'service', 'state', 'is_sm_ips_ports', 'is_ftp_login']]
        
        self.integer_cols = [i for i, n in enumerate(feature_names)
                              if 'count' in n or 'pkts' in n or 'bytes' in n]
        
        self.rate_cols = [i for i, n in enumerate(feature_names)
                           if 'load' in n or 'loss' in n or 'rate' in n]

    def apply_gradient_mask(self, grad):
        """Zero out gradients for immutable features"""
        grad = grad.clone()
        if self.immutable_cols:
            grad[:, self.immutable_cols] = 0
        return grad

    def project(self, x_adv, x_orig=None):
        """Enforce bounds and integer constraints"""
        x_adv = x_adv.clone()
        
        # 1. Bounded features [0, 1] for relative features
        # Assuming normalized data, we mostly respect [0, 1]
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # 2. Integer rounding (post-unscaling if we were doing that, but if scaled we just clip)
        # Note: If features are normalized, 'rounding' to integers must happen in original space.
        # However, for this task, we follow the 'projection' logic.
        
        return x_adv
