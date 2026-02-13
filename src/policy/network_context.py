"""
Network Context Builder
Generates Zero-Trust context for network flows
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import hashlib


@dataclass
class NetworkRequestContext:
    """Zero-Trust context for each network flow"""
    flow_id: str
    user_identity: str
    device_trust_score: float  # 0-1
    geo_risk_score: float      # 0-1  
    time_of_day_risk: float    # 0-1
    source_ip: str
    dest_ip: str
    requested_segment: str     # e.g., "database", "web", "admin"
    flow_features: np.ndarray


class NetworkContextBuilder:
    """Generates realistic Zero-Trust context for network flows"""
    
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        
    def build_context(self, flow_features, flow_index):
        """
        Simulate Zero-Trust metadata for a network flow
        
        In real systems, this would come from:
        - Identity Provider (Okta, Azure AD)
        - Device Management (Intune, Jamf)
        - Threat Intelligence feeds
        - Geo-IP databases
        
        Args:
            flow_features: Network flow feature vector
            flow_index: Index of the flow (for deterministic generation)
            
        Returns:
            NetworkRequestContext object
        """
        # Generate deterministic user ID from flow
        user_id = f"user_{hashlib.md5(str(flow_index).encode()).hexdigest()[:8]}"
        
        # Simulate device trust (lower for suspicious flows)
        device_trust = self.rng.uniform(0.4, 0.95)
        
        # Simulate geo-risk (some IPs from risky regions)
        geo_risk = self.rng.choice([0.1, 0.3, 0.7], p=[0.7, 0.2, 0.1])
        
        # Time-based risk (higher risk at unusual hours)
        time_risk = self.rng.uniform(0.1, 0.5)
        
        # Simulate network segment access
        segments = ["web", "database", "admin", "api", "internal"]
        segment = self.rng.choice(segments)
        
        # Generate IPs
        src_ip = f"{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}"
        dst_ip = f"10.0.{self.rng.randint(1, 255)}.{self.rng.randint(1, 255)}"
        
        return NetworkRequestContext(
            flow_id=f"flow_{flow_index}",
            user_identity=user_id,
            device_trust_score=device_trust,
            geo_risk_score=geo_risk,
            time_of_day_risk=time_risk,
            source_ip=src_ip,
            dest_ip=dst_ip,
            requested_segment=segment,
            flow_features=flow_features
        )
