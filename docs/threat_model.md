# Threat Model: Adversarial Network Defense

This document formalizes the attacker capabilities, system constraints, and security assumptions used to evaluate the **ZT-Shield** research framework.

## 1. Attacker Profile & Goals

The adversary is categorized as an **External Threat Actor** targeting a Zero-Trust environment protected by a Machine Learning (ML) gateway.

- **Primary Goal**: Evade detection by the Random Forest classifier to inject malicious traffic or exfiltrate data.
- **Secondary Goal**: Degrade system availability by forcing high false-positive rates (DoS via legitimate traffic spoofing).

## 2. Capability Matrix

| Capability | Level | Research Context |
| :--- | :--- | :--- |
| **Knowledge** | Black-Box | No access to training data, architecture, or weights. |
| **Access** | Oracle-Only | Can query the production API and observe the Allow/Block decision. |
| **Cost Constraint** | Query Limited | Attacker goal is to find evasive samples within < 500 queries. |
| **Perturbation** | ϵ-Bounded | Changes to packet sizes or timing must be subtle (L-inf < 0.3) to remain protocol-compliant. |

## 3. Attack Methodology

The framework evaluates the following attack vectors:

1. **HopSkipJump (HSJ)**: A query-efficient decision-boundary attack representing a highly capable black-box adversary.
2. **Fast Gradient Method (FGM)**: Representing "script kiddie" style perturbations that are transfer-based.
3. **Distribution Drift**: Representing environmental "noisy" adversaries that degrade performance through non-malicious variance.

## 4. Defense Assumptions

1. **Model Hardening**: The system uses **Adversarial Training** (Retraining on robust samples) as the primary defense.
2. **Zero-Trust Baseline**: The system interprets any "Unknown" or "Anomalous" traffic (detected by Isolation Forest) as a potential threat, regardless of the classification score.

## 5. Risk Assessment Matrix

| Threat Vector | Impact | Likelihood | Mitigation |
| :--- | :--- | :--- | :--- |
| **Evasion (HSJ)** | Critical | High | Bound Decision Boundaries via Retraining |
| **Data Drift** | High | High | Regular Threshold Recalibration |
| **Model Inversion** | Low | Low | Boundary Randomization (Future Work) |

---
Created for the ZT-Shield Research Publication. (Version 1.0)
