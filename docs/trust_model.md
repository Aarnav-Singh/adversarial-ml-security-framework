# Device Trust Model Documentation

## Overview

In a **Zero-Trust Architecture (ZTA)**, trust is never implicit based on network location. Instead, every access request must be verified based on user identity, device posture, and environmental context.

This project implements a weighted device trust model that substitutes random generation with measurable security attributes.

## Trust Calculation Logic

The `device_trust_score` (0.0 to 1.0) is computed using the following formula:

```python
Trust = (0.2 * AgeFactor) + (0.4 * ComplianceFactor) + (0.4 * BehaviorFactor)
```

### 1. Age Factor (20%)

Trust decays as the time since the last hardware/OS verification increases.

- **Goal**: Encourage regular re-authentication and health checks.
- **Slope**: Linear decay over 365 days, floor at 0.4.

### 2. Patch Compliance (40%)

Evaluates if the device is running the latest security patches and OS versions.

- **Goal**: Prevent compromised or vulnerable devices from accessing sensitive resources.
- **Scale**: 0.0 (Unpatched/Vulnerable) to 1.0 (Fully Compliant).

### 3. Behavior Factor (40%)

Integrates with anomaly detection systems to identify suspicious patterns.

- **Goal**: Detect compromised credentials or "living off the land" attacks.
- **Inverse**: `1.0 - AnomalyScore`.

## Thresholds in Policy Engine

- **Score > 0.8**: High Trust. Eligible for high-privilege resource access.
- **0.5 < Score < 0.8**: Medium Trust. May require **Step-Up Authentication (MFA)**.
- **Score < 0.5**: Low Trust. Usually results in **DENY** for sensitive resources, regardless of ML risk score.

## Real-World Mapping

In a production environment, these inputs would be sourced from:

- **Unified Endpoint Management (UEM)**: Microsoft Intune, Jamf, VMware Workspace ONE.
- **Endpoint Detection & Response (EDR)**: CrowdStrike, SentinelOne.
- **Hardware Roots of Trust**: TPM status, Secure Boot verification.
