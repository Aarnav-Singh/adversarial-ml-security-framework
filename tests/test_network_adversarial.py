import unittest
import numpy as np
import torch
import torch.nn as nn
from src.attacks.network_adversarial import NetworkAdversarialAttacker


class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.fc = nn.Linear(41, 1)
        # Set weights to be sensitive to the 21st feature (index 20, non-integer)
        with torch.no_grad():
            self.fc.weight.fill_(0.0)
            self.fc.weight[0, 20] = 1.0
            self.fc.bias.fill_(0.0)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class TestNetworkAdversarialAttacker(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.bounds = [(0.0, 1.0)] * 41
        self.attacker = NetworkAdversarialAttacker(self.model, self.bounds)
        self.x_clean = np.zeros(41)
        self.x_clean[20] = 0.8  # High value -> high risk in MockModel

    def test_fgsm_attack_generation(self):
        """Test that FGSM generates a perturbed sample"""
        # epsilon=0.1, we want to decrease risk (target_label=0)
        # gradient of risk w.r.t x[20] is positive, so we subtract epsilon * sign(grad)
        x_adv = self.attacker.constrained_fgsm(self.x_clean, epsilon=0.1, target_label=0)
        
        self.assertEqual(x_adv.shape, (1, 41))
        # Feature should have decreased
        self.assertLess(x_adv[0, 20], self.x_clean[20])
        self.assertAlmostEqual(x_adv[0, 20], 0.7, places=5)

    def test_attack_constraints(self):
        """Test that attacks respect feature bounds"""
        # Set feature near lower bound
        self.x_clean[20] = 0.05
        x_adv = self.attacker.constrained_fgsm(self.x_clean, epsilon=0.1, target_label=0)
        
        # Should be clipped to 0.0, not -0.05
        self.assertEqual(x_adv[0, 20], 0.0)
        self.assertTrue(np.all(x_adv >= 0.0))
        self.assertTrue(np.all(x_adv <= 1.0))

    def test_attack_success_evaluation(self):
        """Test attack success rate calculation"""
        X = np.array([self.x_clean])
        results = self.attacker.evaluate_attack(X, epsilon=0.1, threshold=0.5)
        
        self.assertIn('clean_detection_rate', results)
        self.assertIn('adv_detection_rate', results)
        self.assertIn('attack_success_rate', results)


if __name__ == '__main__':
    unittest.main()
