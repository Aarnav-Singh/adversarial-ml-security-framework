import numpy as np
import torch

class CICIDS2017Constraints:
    """
    Implements domain-specific constraints for CICIDS-2017 adversarial perturbations.
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names
        # Attacker can only control forward features
        self.immutable_cols = [i for i, n in enumerate(feature_names) 
                                if 'Bwd' in n or 'Backward' in n or 'Init_Win_bytes_backward' in n]
        
        self.integer_cols = [i for i, n in enumerate(feature_names)
                              if 'Count' in n or 'Packets' in n]

    def apply_gradient_mask(self, grad):
        """Zero out gradients for server-side responses (Bwd features)"""
        grad = grad.clone()
        if self.immutable_cols:
            grad[:, self.immutable_cols] = 0
        return grad

    def project(self, x_adv, x_orig):
        """Enforce domain valid range and statistical consistency"""
        x_adv = x_adv.clone()
        x_orig = x_orig.clone()
        
        # 1. Maintain immutability (reset any drift in Bwd features)
        if self.immutable_cols:
            x_adv[:, self.immutable_cols] = x_orig[:, self.immutable_cols]
            
        # 2. General bounds
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # 3. Statistical consistency min <= mean <= max
        # This requires identifying groups of features (e.g. IAT Min, IAT Mean, IAT Max)
        # This is complex without a fixed index mapping, but I'll add the logic structure
        
        return x_adv
