"""
Network Adversarial Attacks
Adversarial attacks adapted for network flow features
"""

import torch
import numpy as np


class NetworkAdversarialAttacker:
    """Adversarial attacks adapted for network flow features"""
    
    def __init__(self, model, feature_bounds):
        """
        Args:
            model: NetworkRiskClassifier
            feature_bounds: dict with 'min' and 'max' arrays for each feature
        """
        self.model = model
        self.bounds = feature_bounds
        
        # Integer features that must be rounded (byte counts, packet counts, etc.)
        self.integer_features = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 31, 32]
        
    def constrained_fgsm(self, x, epsilon=0.01, target_label=0):
        """
        FGSM attack with network feature constraints
        
        Network features must stay realistic:
        - Duration must be positive
        - Byte counts must be integers
        - Flags must be valid
        
        Args:
            x: Input network flow features (numpy array)
            epsilon: Perturbation magnitude
            target_label: Target class (0=benign, 1=malicious)
            
        Returns:
            Adversarial network flow
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        x_tensor = torch.FloatTensor(x).requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(x_tensor)
        
        # Calculate loss
        target = torch.FloatTensor([[target_label]])
        loss = torch.nn.BCELoss()(output, target)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate perturbation
        perturbation = epsilon * x_tensor.grad.sign()
        
        # Apply perturbation (minimize loss to evade detection)
        x_adv = x_tensor - perturbation
        
        # Apply feature-specific constraints
        x_adv = self._apply_network_constraints(x_adv.detach().numpy())
        
        return x_adv
    
    def pgd_attack(self, x, epsilon=0.05, alpha=0.01, num_iter=10, target_label=0):
        """
        Projected Gradient Descent attack for network flows
        
        Args:
            x: Input network flow features
            epsilon: Maximum perturbation
            alpha: Step size
            num_iter: Number of iterations
            target_label: Target class
            
        Returns:
            Adversarial network flow
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        x_orig = x.copy()
        x_adv = x.copy()
        
        for i in range(num_iter):
            x_tensor = torch.FloatTensor(x_adv).requires_grad_(True)
            
            # Forward pass
            self.model.eval()
            output = self.model(x_tensor)
            
            # Calculate loss
            target = torch.FloatTensor([[target_label]])
            loss = torch.nn.BCELoss()(output, target)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial example
            perturbation = alpha * x_tensor.grad.sign()
            x_adv = x_tensor.detach().numpy() - perturbation.numpy()
            
            # Project back to epsilon ball
            delta = x_adv - x_orig
            delta = np.clip(delta, -epsilon, epsilon)
            x_adv = x_orig + delta
            
            # Apply constraints
            x_adv = self._apply_network_constraints(x_adv)
        
        return x_adv
    
    def _apply_network_constraints(self, x_adv):
        """Ensure adversarial samples remain valid network flows"""
        x_constrained = x_adv.copy()
        
        # Clip to valid feature ranges
        x_constrained = np.clip(x_constrained, self.bounds['min'], self.bounds['max'])
        
        # Round integer features (e.g., byte counts, packet counts)
        for idx in self.integer_features:
            if idx < x_constrained.shape[1]:
                x_constrained[:, idx] = np.round(x_constrained[:, idx])
        
        # Ensure non-negative values for counts and durations
        x_constrained = np.maximum(x_constrained, 0)
        
        return x_constrained
    
    def evaluate_attack(self, X_clean, threshold=0.5):
        """
        Generate adversarial examples and evaluate attack success
        
        Args:
            X_clean: Clean malicious samples
            threshold: Classification threshold
            
        Returns:
            Dictionary with attack results
        """
        print(f"Generating adversarial examples for {len(X_clean)} samples...")
        
        # Generate adversarial examples
        X_adv = []
        for i, x in enumerate(X_clean):
            x_adv = self.constrained_fgsm(x, epsilon=0.05, target_label=0)
            X_adv.append(x_adv[0])
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{len(X_clean)} adversarial samples")
        
        X_adv = np.array(X_adv)
        
        # Evaluate
        with torch.no_grad():
            clean_scores = self.model(torch.FloatTensor(X_clean)).numpy()
            adv_scores = self.model(torch.FloatTensor(X_adv)).numpy()
        
        # Calculate metrics
        clean_detected = (clean_scores > threshold).sum()
        adv_detected = (adv_scores > threshold).sum()
        
        evasion_success_rate = 1 - (adv_detected / len(X_adv))
        avg_score_reduction = (clean_scores - adv_scores).mean()
        
        # Calculate perturbation magnitude
        l2_distances = np.linalg.norm(X_adv - X_clean, axis=1)
        linf_distances = np.abs(X_adv - X_clean).max(axis=1)
        
        results = {
            'attack_success_rate': float(evasion_success_rate),
            'clean_detection_rate': float(clean_detected / len(X_clean)),
            'adv_detection_rate': float(adv_detected / len(X_adv)),
            'avg_risk_reduction': float(avg_score_reduction),
            'avg_l2_distance': float(l2_distances.mean()),
            'avg_linf_distance': float(linf_distances.mean()),
            'X_adversarial': X_adv
        }
        
        return results
