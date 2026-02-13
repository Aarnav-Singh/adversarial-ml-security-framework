"""
Zero-Trust Policy Engine
Makes access control decisions based on ML risk + context
"""

from enum import Enum
from typing import Tuple
import numpy as np


class AccessDecision(Enum):
    """Possible access control decisions"""
    ALLOW = "ALLOW"
    DENY = "DENY"
    STEP_UP_AUTH = "STEP_UP_AUTH"  # Require MFA
    RATE_LIMIT = "RATE_LIMIT"
    ISOLATE = "ISOLATE"  # Quarantine to isolated network segment


# Resource-specific risk thresholds (Micro-segmentation modeling)
# Different resources require different security postures
RESOURCE_RISK_THRESHOLDS = {
    "web": 0.8,
    "internal": 0.7,
    "api": 0.6,
    "database": 0.5,
    "admin": 0.4
}

class ZeroTrustPolicyEngine:
    """
    Zero-Trust policy enforcement engine
    
    Makes final access decisions based on:
    - ML risk score
    - User trust
    - Device posture
    - Contextual factors
    - Resource sensitivity (Micro-segmentation)
    """
    
    def __init__(self, config=None):
        # Default policy thresholds
        self.config = config or {
            'ml_risk_deny_threshold': 0.8,
            'ml_risk_mfa_threshold': 0.6,
            'device_trust_minimum': 0.5,
            'geo_risk_maximum': 0.7,
            'high_privilege_segments': ['admin', 'database']
        }
        self.resource_thresholds = RESOURCE_RISK_THRESHOLDS
        
    def evaluate_access(self, ml_risk_score, context) -> Tuple[AccessDecision, str]:
        """
        Core Zero-Trust logic with Resource-Aware Micro-Segmentation
        
        Args:
            ml_risk_score: Risk score from ML model (0-1)
            context: NetworkRequestContext object
            
        Returns:
            (decision, reason) tuple
        """
        # Rule 0: Resource-specific risk threshold (Micro-segmentation)
        resource = context.requested_segment
        resource_threshold = self.resource_thresholds.get(resource, self.config['ml_risk_deny_threshold'])
        
        if ml_risk_score > resource_threshold:
            return AccessDecision.DENY, f"ML risk ({ml_risk_score:.3f}) exceeds threshold for sensitive resource '{resource}' ({resource_threshold})"
        
        # Rule 1: Global High ML risk → DENY
        if ml_risk_score > self.config['ml_risk_deny_threshold']:
            return AccessDecision.DENY, f"Global ML risk score too high: {ml_risk_score:.3f}"
        
        # Rule 2: Low device trust → DENY or MFA
        if context.device_trust_score < self.config['device_trust_minimum']:
            if ml_risk_score > 0.4:
                return AccessDecision.DENY, "Untrusted device + elevated ML risk"
            else:
                return AccessDecision.STEP_UP_AUTH, "Untrusted device requires MFA"
        
        # Rule 3: High geo risk → MFA required
        if context.geo_risk_score > self.config['geo_risk_maximum']:
            return AccessDecision.STEP_UP_AUTH, f"High geo risk: {context.geo_risk_score:.2f}"
        
        # Rule 4: Sensitive segments require extra scrutiny
        if context.requested_segment in self.config['high_privilege_segments']:
            if ml_risk_score > self.config['ml_risk_mfa_threshold']:
                return AccessDecision.STEP_UP_AUTH, "Sensitive segment access requires verification"
            if context.device_trust_score < 0.7:
                return AccessDecision.DENY, "Insufficient device trust for sensitive segment"
        
        # Rule 5: Medium ML risk → ALLOW but with rate limiting
        if ml_risk_score > self.config['ml_risk_mfa_threshold']:
            return AccessDecision.RATE_LIMIT, "Moderate risk - rate limited access"
        
        # Default: ALLOW
        return AccessDecision.ALLOW, "All checks passed"
    
    def log_decision(self, decision, reason, context, ml_risk_score):
        """
        Zero-Trust telemetry logging
        
        Args:
            decision: AccessDecision enum
            reason: String reason for decision
            context: NetworkRequestContext
            ml_risk_score: ML risk score
            
        Returns:
            Log entry dictionary
        """
        log_entry = {
            'timestamp': str(np.datetime64('now')),
            'flow_id': context.flow_id,
            'user': context.user_identity,
            'source_ip': context.source_ip,
            'dest_ip': context.dest_ip,
            'segment': context.requested_segment,
            'ml_risk_score': float(ml_risk_score),
            'device_trust': context.device_trust_score,
            'geo_risk': context.geo_risk_score,
            'decision': decision.value,
            'reason': reason
        }
        return log_entry
