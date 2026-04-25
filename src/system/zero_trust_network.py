"""
Zero-Trust Network System
Complete integration of all components
"""

import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.risk_engine.network_classifier import NetworkRiskClassifier
from src.policy.network_context import NetworkContextBuilder
from src.policy.zero_trust_engine import ZeroTrustEngine, AccessDecision


class ZeroTrustNetworkSystem:
    """
    Complete Zero-Trust network access control system

    Pipeline:
    Network Flow → Feature Extraction → ML Risk Scoring →
    Context Enrichment → Policy Evaluation → Access Decision
    """

    def __init__(self, model_path=None):
        """
        Initialize Zero-Trust network system

        Args:
            model_path: Path to trained NetworkRiskClassifier model (.pth file).
                        If None or not found, an untrained model is used.
        """
        # Lazy-import torch to avoid breaking non-torch environments
        try:
            import torch
            self._torch = torch
            self._torch_available = True
        except ImportError:
            self._torch = None
            self._torch_available = False

        # Load ML risk classifier
        self.risk_model = NetworkRiskClassifier()

        if model_path and os.path.exists(model_path):
            if self._torch_available:
                self.risk_model.load_state_dict(
                    self._torch.load(model_path, map_location='cpu', weights_only=True)
                )
                print(f"Loaded risk model from {model_path}")
            else:
                print("Warning: torch not available; model weights not loaded.")
        else:
            if model_path:
                print(f"Warning: Model not found at {model_path}, using untrained model")

        if self._torch_available:
            self.risk_model.eval()

        # Initialize components
        self.context_builder = NetworkContextBuilder()
        self.policy_engine = ZeroTrustEngine()

        # Telemetry
        self.access_log = []

    def process_network_request(self, flow_features, flow_index):
        """
        Process a network access request through Zero-Trust pipeline

        Args:
            flow_features: Network flow feature vector
            flow_index: Index of the flow

        Returns:
            Dictionary with decision, reason, scores, and context
        """
        if not self._torch_available:
            raise RuntimeError(
                "torch is required for process_network_request. "
                "Install it with: pip install torch"
            )

        torch = self._torch

        # Step 1: ML Risk Scoring
        with torch.no_grad():
            if isinstance(flow_features, np.ndarray):
                flow_tensor = torch.FloatTensor(flow_features)
            else:
                flow_tensor = flow_features

            if len(flow_tensor.shape) == 1:
                flow_tensor = flow_tensor.unsqueeze(0)

            risk_score = self.risk_model(flow_tensor).item()

        # Step 2: Build Zero-Trust context
        context = self.context_builder.build_context(flow_features, flow_index)

        # Step 3: Policy evaluation — convert NetworkRequestContext to dict for ZeroTrustEngine
        ctx_dict = {
            'device_trust': context.device_trust_score,
            'geo_risk': context.geo_risk_score,
            'time_of_day': int(context.time_of_day_risk * 23),  # map 0-1 float → 0-23 hour
            'identity_verified': True,
            'resource_sensitivity': 0.5,
        }
        decision_obj = self.policy_engine.evaluate(
            ml_risk_score=risk_score,
            context=ctx_dict
        )

        # Step 4: Log decision
        log_entry = {
            'flow_id': context.flow_id,
            'user': context.user_identity,
            'segment': context.requested_segment,
            'ml_risk_score': float(risk_score),
            'device_trust': context.device_trust_score,
            'geo_risk': context.geo_risk_score,
            'decision': decision_obj.decision,
            'rule_fired': decision_obj.rule_fired,
        }
        self.access_log.append(log_entry)

        return {
            'decision': decision_obj.decision,
            'reason': decision_obj.rule_fired,
            'ml_risk_score': float(risk_score),
            'context': context,
            'log_entry': log_entry,
        }

    def evaluate_adversarial_evasion(self, X_clean, X_adv, flow_indices):
        """
        Test if adversarial attacks can bypass Zero-Trust controls

        Args:
            X_clean: Clean malicious samples
            X_adv: Adversarial samples
            flow_indices: Indices for flows

        Returns:
            Dictionary with evasion metrics
        """
        results = {
            'clean': [],
            'adversarial': []
        }

        print(f"Evaluating {len(X_clean)} clean samples...")
        for i, (x, idx) in enumerate(zip(X_clean, flow_indices)):
            result = self.process_network_request(x, idx)
            results['clean'].append(result)

        print(f"Evaluating {len(X_adv)} adversarial samples...")
        for i, (x, idx) in enumerate(zip(X_adv, flow_indices)):
            result = self.process_network_request(x, idx)
            results['adversarial'].append(result)

        # Calculate evasion metrics
        clean_denies = sum(1 for r in results['clean'] if r['decision'] == "DENY")
        adv_allows = sum(1 for r in results['adversarial'] if r['decision'] == "ALLOW")
        adv_denies = sum(1 for r in results['adversarial'] if r['decision'] == "DENY")

        evasion_rate = adv_allows / len(X_adv) if len(X_adv) > 0 else 0

        clean_risks = [r['ml_risk_score'] for r in results['clean']]
        adv_risks = [r['ml_risk_score'] for r in results['adversarial']]

        return {
            'evasion_success_rate': evasion_rate,
            'clean_deny_rate': clean_denies / len(X_clean),
            'adv_allow_rate': adv_allows / len(X_adv),
            'adv_deny_rate': adv_denies / len(X_adv),
            'avg_clean_risk': np.mean(clean_risks),
            'avg_adv_risk': np.mean(adv_risks),
            'risk_reduction': np.mean(clean_risks) - np.mean(adv_risks),
            'detailed_results': results
        }

    def get_decision_summary(self):
        """Get summary of all access decisions"""
        if not self.access_log:
            return "No decisions logged yet"

        decisions = [log['decision'] for log in self.access_log]
        summary = {
            'total': len(decisions),
            'ALLOW': decisions.count('ALLOW'),
            'DENY': decisions.count('DENY'),
        }
        return summary

    def export_telemetry(self, filepath):
        """Export access logs to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.access_log, f, indent=2)
        print(f"Exported {len(self.access_log)} log entries to {filepath}")
