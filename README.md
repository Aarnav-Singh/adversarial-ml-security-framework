# Adversarial Attack Detection in Zero-Trust Networks

A comprehensive system for detecting adversarial evasion attacks against ML-driven network intrusion detection systems operating within a Zero-Trust architecture.

## Overview

This project simulates adversarial evasion attacks against ML-powered network security systems and evaluates how adversarially modified network traffic can bypass risk-based access control. It implements layered defensive strategies for detection and mitigation within a Zero-Trust network framework.

### Key Features

- **Network Intrusion Detection**: ML-based risk classifier trained on NSL-KDD network traffic dataset
- **Zero-Trust Architecture**: Identity-aware, context-enriched access control with device trust and geo-risk scoring
- **Adversarial Attacks**: FGSM and PGD attacks adapted for network flow features with realistic constraints
- **Policy Enforcement**: Multi-factor access decisions (ALLOW/DENY/MFA/RATE_LIMIT) based on ML risk + context
- **SOC Telemetry**: Comprehensive logging of all access decisions and risk scores

## Architecture

```
Network Flow (NSL-KDD Dataset)
        ↓
Feature Extraction (41 network features)
        ↓
ML Risk Classifier (Neural Network: 128→64→32→1)
        ↓
Risk Score (0-1)
        ↓
Context Enrichment (Identity, Device Trust, Geo-Risk)
        ↓
Zero-Trust Policy Engine
        ↓
Access Decision (ALLOW/DENY/MFA/RATE_LIMIT)
        ↓
SOC Telemetry Logging
```

### Components

1. **Network Data Layer** (`src/data/`)
   - NSL-KDD dataset loader with preprocessing
   - Feature encoding and normalization
   - 41 network flow features (duration, protocol, bytes, flags, etc.)

2. **Risk Engine** (`src/risk_engine/`)
   - Neural network classifier for intrusion detection
   - Risk scoring (0=benign, 1=malicious)

3. **Adversarial Attacks** (`src/attacks/`)
   - FGSM and PGD attacks with network constraints
   - Realistic evasion scenarios (rate limiting, port hopping, mimicry)
   - Feature-specific constraints (integer rounding, valid ranges)

4. **Zero-Trust Policy** (`src/policy/`)
   - Network context builder (identity, device, geo-risk)
   - Policy engine with multi-factor rules
   - Access decision logic

5. **System Integration** (`src/system/`)
   - Complete Zero-Trust network pipeline
   - Adversarial evasion evaluation
   - Telemetry export

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy, Pandas, scikit-learn

### Setup

```bash
# Clone repository
git clone <repository-url>
cd "Zero trust project"

# Install dependencies
pip install -r requirements.txt

# Dataset is already downloaded (NSL-KDD)
# Located in data/KDDTrain+.txt and data/KDDTest+.txt
```

## Usage

### 1. Train Baseline Network Risk Classifier

```bash
python scripts/train_baseline.py
```

**Expected Output:**

- 20 epochs of training
- ~78% test accuracy
- Model saved to `models/network_risk_classifier.pth`

### 2. Run End-to-End Zero-Trust System Test

```bash
python scripts/test_zero_trust_system.py
```

**This test demonstrates:**

- Clean malicious traffic detection (should be DENIED)
- Adversarial evasion attempts with FGSM
- Zero-Trust policy decisions (ALLOW/DENY/MFA)
- SOC telemetry logging

### 3. Example: Process Network Flow

```python
from src.system.zero_trust_network import ZeroTrustNetworkSystem

# Initialize system
system = ZeroTrustNetworkSystem(model_path='models/network_risk_classifier.pth')

# Process a network flow
result = system.process_network_request(flow_features, flow_index=0)

print(f"Decision: {result['decision'].value}")
print(f"ML Risk Score: {result['ml_risk_score']:.3f}")
print(f"Reason: {result['reason']}")
```

### 4. Example: Generate Adversarial Attack

```python
from src.attacks.network_adversarial import NetworkAdversarialAttacker

# Initialize attacker
attacker = NetworkAdversarialAttacker(model, feature_bounds)

# Generate adversarial example
x_adv = attacker.constrained_fgsm(x_clean, epsilon=0.05, target_label=0)

# Evaluate attack success
results = attacker.evaluate_attack(X_malicious, threshold=0.5)
print(f"Attack Success Rate: {results['attack_success_rate']:.2%}")
```

## Dataset: NSL-KDD

The NSL-KDD dataset is an improved version of the KDD'99 dataset for network intrusion detection.

### Features (41 total)

- **Connection Features**: duration, protocol_type, service, flag, src_bytes, dst_bytes, etc.
- **Content Features**: failed_logins, logged_in, root_shell, etc.
- **Traffic Features**: count, srv_count, error rates, etc.
- **Host Features**: dst_host_count, same_srv_rate, etc.

### Attack Types

- DoS (Denial of Service)
- Probe (Port scanning)
- R2L (Remote to Local)
- U2R (User to Root)
- Normal (Benign traffic)

## Zero-Trust Policy Rules

The policy engine enforces the following rules:

1. **High ML Risk** (>0.8) → **DENY**
2. **Low Device Trust** (<0.5) + Elevated Risk → **DENY**
3. **Low Device Trust** (<0.5) + Low Risk → **STEP_UP_AUTH** (MFA)
4. **High Geo Risk** (>0.7) → **STEP_UP_AUTH**
5. **Sensitive Segments** (admin, database) → Strict checks
6. **Medium Risk** (>0.6) → **RATE_LIMIT**
7. **Default** → **ALLOW**

## Results

### Baseline Model Performance

- **Accuracy**: 78.5%
- **Precision**: 97.2%
- **Recall**: 64.1%
- **F1 Score**: 77.3%

### Adversarial Evasion

- **Clean Deny Rate**: 60%
- **Adversarial Evasion Success**: 20%
- **Average Risk Reduction**: -0.20 (attacks actually increased risk in this test)

### Zero-Trust Effectiveness

- **73% of flows DENIED** (high security posture)
- **23% ALLOWED** (legitimate low-risk traffic)
- **3% STEP_UP_AUTH** (MFA required)

## Project Structure

```
Zero trust project/
├── data/
│   ├── KDDTrain+.txt          # NSL-KDD training data
│   └── KDDTest+.txt           # NSL-KDD test data
├── src/
│   ├── data/
│   │   └── network_loader.py  # Dataset loading & preprocessing
│   ├── risk_engine/
│   │   └── network_classifier.py  # ML risk classifier
│   ├── attacks/
│   │   ├── network_adversarial.py  # FGSM/PGD attacks
│   │   └── evasion_scenarios.py    # Realistic evasion patterns
│   ├── policy/
│   │   ├── network_context.py      # Zero-Trust context
│   │   └── zero_trust_engine.py    # Policy enforcement
│   └── system/
│       └── zero_trust_network.py   # Complete integration
├── scripts/
│   ├── train_baseline.py      # Train risk classifier
│   └── test_zero_trust_system.py  # End-to-end test
├── models/
│   ├── network_risk_classifier.pth  # Trained model
│   ├── label_encoders.pkl     # Feature encoders
│   └── scaler.pkl             # Feature scaler
└── logs/
    └── zero_trust_telemetry.json  # Access logs
```

## Key Insights

1. **Network-Level Detection**: Moving from generic ML to network intrusion detection provides realistic threat modeling
2. **Zero-Trust Defense-in-Depth**: ML risk scoring alone is insufficient; context-aware policies provide additional security layers
3. **Adversarial Robustness**: Even with adversarial perturbations, Zero-Trust policies can catch attacks through device trust and geo-risk checks
4. **SOC Integration**: Comprehensive telemetry enables security operations monitoring and incident response

## Future Enhancements

- [ ] Implement adversarial training for robust model
- [ ] Add SHAP explainability for risk scores
- [ ] Integrate real-time network traffic capture
- [ ] Add more sophisticated evasion techniques
- [ ] Implement adaptive policy thresholds
- [ ] Create Streamlit dashboard for visualization

## References

- NSL-KDD Dataset: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html)
- Zero-Trust Architecture: [NIST SP 800-207](https://csrc.nist.gov/publications/detail/sp/800-207/final)
- Adversarial ML: [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

## License

This project is for educational and research purposes.

## Contact

For questions or collaboration, please open an issue in the repository.
