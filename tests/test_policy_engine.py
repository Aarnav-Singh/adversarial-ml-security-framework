"""
Tests for the Zero-Trust policy engine.

Fixed: was importing ZeroTrustPolicyEngine (never existed) and using a
NetworkRequestContext-based API. Now uses the actual ZeroTrustEngine with its
evaluate(ml_risk_score, context) -> AccessDecision API.
"""

import unittest
import numpy as np
from src.policy.zero_trust_engine import ZeroTrustEngine, AccessDecision


class TestZeroTrustPolicyEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ZeroTrustEngine()
        # Default context: trusted device, low geo risk, business hours,
        # verified identity, low resource sensitivity -> should ALLOW on low ML risk
        self.default_context = {
            "device_trust": 0.9,
            "geo_risk": 0.1,
            "time_of_day": 10,
            "identity_verified": True,
            "resource_sensitivity": 0.5,
        }

    def test_allow_decision(self):
        """Test standard ALLOW case: low ML risk with all-good context."""
        result = self.engine.evaluate(ml_risk_score=0.1, context=self.default_context)
        self.assertEqual(result.decision, "ALLOW")
        self.assertEqual(result.rule_fired, "Default Allow")

    def test_high_ml_risk_deny(self):
        """Test DENY due to very high ML risk (Rule 2 fires)."""
        result = self.engine.evaluate(ml_risk_score=0.9, context=self.default_context)
        self.assertEqual(result.decision, "DENY")
        self.assertEqual(result.rule_fired, "High-Risk ML Score")

    def test_low_device_trust_deny(self):
        """Test DENY due to low device trust (Rule 3 fires)."""
        ctx = dict(self.default_context)
        ctx["device_trust"] = 0.2
        result = self.engine.evaluate(ml_risk_score=0.2, context=ctx)
        self.assertEqual(result.decision, "DENY")
        self.assertEqual(result.rule_fired, "Device Trust Threshold")

    def test_high_geo_risk_deny(self):
        """Test DENY due to high geo risk (Rule 4 fires)."""
        ctx = dict(self.default_context)
        ctx["geo_risk"] = 0.85
        result = self.engine.evaluate(ml_risk_score=0.2, context=ctx)
        self.assertEqual(result.decision, "DENY")
        self.assertEqual(result.rule_fired, "Geo-Risk Threshold")

    def test_compound_risk_deny(self):
        """Test DENY due to moderate ML + elevated geo risk (Rule 5)."""
        ctx = dict(self.default_context)
        ctx["geo_risk"] = 0.65
        result = self.engine.evaluate(ml_risk_score=0.6, context=ctx)
        self.assertEqual(result.decision, "DENY")
        self.assertEqual(result.rule_fired, "Compound Risk (ML + Geo)")

    def test_microsegment_deny(self):
        """Test DENY due to critical resource + low device trust (Rule 1)."""
        ctx = dict(self.default_context)
        ctx["resource_sensitivity"] = 0.95
        ctx["device_trust"] = 0.4
        result = self.engine.evaluate(ml_risk_score=0.2, context=ctx)
        self.assertEqual(result.decision, "DENY")
        self.assertEqual(result.rule_fired, "Critical Resource Microsegment")
        self.assertEqual(result.priority, 1)

    def test_identity_failure_deny(self):
        """Test DENY due to unverified identity + moderate ML risk (Rule 7)."""
        ctx = dict(self.default_context)
        ctx["identity_verified"] = False
        result = self.engine.evaluate(ml_risk_score=0.5, context=ctx)
        self.assertEqual(result.decision, "DENY")
        self.assertEqual(result.rule_fired, "Identity Verification Failure")

    def test_priority_ordering(self):
        """Rule 1 (microsegment) fires before Rule 2 (high ML) when both match."""
        ctx = dict(self.default_context)
        ctx["resource_sensitivity"] = 0.95
        ctx["device_trust"] = 0.3
        result = self.engine.evaluate(ml_risk_score=0.9, context=ctx)
        self.assertEqual(result.priority, 1)
        self.assertEqual(result.rule_fired, "Critical Resource Microsegment")

    def test_ml_only_mode_allows_bad_context(self):
        """ML-only mode: contextual rules disabled; low device trust does NOT trigger deny."""
        engine = ZeroTrustEngine(enabled_factors=set())
        ctx = {"device_trust": 0.1, "geo_risk": 0.9}
        result = engine.evaluate(ml_risk_score=0.3, context=ctx)
        self.assertEqual(result.decision, "ALLOW")

    def test_batch_evaluation(self):
        """Batch evaluation returns one AccessDecision per sample."""
        scores = np.array([0.1, 0.9, 0.3])
        contexts = [
            {"device_trust": 0.9},
            {"device_trust": 0.9},
            {"device_trust": 0.9},
        ]
        results = self.engine.evaluate_batch(scores, contexts)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(isinstance(r, AccessDecision) for r in results))
        self.assertEqual(results[0].decision, "ALLOW")
        self.assertEqual(results[1].decision, "DENY")


if __name__ == "__main__":
    unittest.main()
