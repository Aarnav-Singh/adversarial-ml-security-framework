# Quick Start Guide

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- 2GB free disk space

### Step 1: Clone or Download Project

```bash
cd "c:\Zero trust project"
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn streamlit plotly
```

Or use requirements.txt if available:

```bash
pip install -r requirements.txt
```

### Step 3: Verify Dataset

Ensure NSL-KDD dataset files exist:

- `data/KDDTrain+.txt` (19MB)
- `data/KDDTest+.txt` (3.4MB)

If missing, download from [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)

---

## Training the Model

### Train Network Risk Classifier

```bash
python scripts\train_baseline.py
```

**Expected Output:**

```
Epoch 1/20: Loss=0.234, Acc=0.756
Epoch 2/20: Loss=0.198, Acc=0.772
...
Epoch 20/20: Loss=0.145, Acc=0.785

Best Test Accuracy: 0.7852
Model saved to models/network_risk_classifier.pth
```

**Training Time:** ~2-3 minutes on CPU

---

## Running Tests

### End-to-End System Test

```bash
python scripts\test_zero_trust_system.py
```

**What it tests:**

1. Clean malicious traffic detection
2. Adversarial evasion attacks
3. Zero-Trust policy decisions
4. Telemetry logging

**Expected Results:**

- 60% clean deny rate
- 20% adversarial evasion success
- 73% overall deny rate
- Telemetry exported to `logs/zero_trust_telemetry.json`

---

## Launching the Dashboard

### Start Streamlit Dashboard

```bash
streamlit run src\dashboard\app.py
```

**Dashboard URL:** <http://localhost:8501>

### Navigate to Zero-Trust Tab

1. Open browser to <http://localhost:8501>
2. Click **🔵 Zero-Trust Network** tab (4th tab)
3. You're ready to go!

---

## First Experiments

### Experiment 1: Process Network Flows

**Left Panel:**

1. Select "Malicious Flows"
2. Set number to 10
3. Click **🚀 Process Network Flows**

**Right Panel:**

- See live processing
- View access decisions (ALLOW/DENY/MFA)
- Check decision breakdown

### Experiment 2: Test Adversarial Attack

**Left Panel:**

1. Set epsilon to 0.05
2. Click **🎯 Test Adversarial Evasion**

**Right Panel:**

- View evasion success rate
- Compare clean vs adversarial risk scores
- See box plot visualization

### Experiment 3: Adjust Policies

**Left Panel:**

1. Change "ML Risk Deny Threshold" to 0.7
2. Change "Min Device Trust" to 0.6
3. Process flows again

**Observe:**

- More flows allowed (lower threshold)
- Different decision distribution

---

## Programmatic Usage

### Example: Process Single Flow

```python
from src.system.zero_trust_network import ZeroTrustNetworkSystem
from src.data.network_loader import NetworkDataLoader

# Load data
loader = NetworkDataLoader()
loader.load_preprocessors('models')
X_test, y_test, _ = loader.load_and_preprocess('data/KDDTest+.txt', is_train=False)

# Initialize system
system = ZeroTrustNetworkSystem(model_path='models/network_risk_classifier.pth')

# Process flow
result = system.process_network_request(X_test[0], flow_index=0)

print(f"Decision: {result['decision'].value}")
print(f"ML Risk: {result['ml_risk_score']:.3f}")
print(f"Reason: {result['reason']}")
```

### Example: Generate Adversarial Attack

```python
from src.attacks.network_adversarial import NetworkAdversarialAttacker

# Get malicious flows
X_malicious = X_test[y_test == 1][:10]

# Create attacker
bounds = loader.get_feature_bounds(X_test)
attacker = NetworkAdversarialAttacker(system.risk_model, bounds)

# Generate attack
X_adv = attacker.constrained_fgsm(X_malicious[0], epsilon=0.05)

# Evaluate
results = attacker.evaluate_attack(X_malicious, threshold=0.5)
print(f"Attack Success Rate: {results['attack_success_rate']:.2%}")
```

---

## Troubleshooting

### Model Not Found

**Error:** `FileNotFoundError: network_risk_classifier.pth`

**Solution:**

```bash
python scripts\train_baseline.py
```

### Dataset Not Found

**Error:** `FileNotFoundError: KDDTest+.txt`

**Solution:** Download NSL-KDD dataset and place in `data/` directory

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**

```bash
pip install torch numpy pandas scikit-learn
```

### Dashboard Won't Start

**Error:** `streamlit: command not found`

**Solution:**

```bash
pip install streamlit plotly
```

---

## Next Steps

1. **Explore Dashboard** - Try different traffic types and attack strengths
2. **Read User Guide** - Learn advanced features ([USER_GUIDE.md](USER_GUIDE.md))
3. **Review Architecture** - Understand system design ([ARCHITECTURE.md](ARCHITECTURE.md))
4. **Check API Docs** - Learn to extend the system ([API_REFERENCE.md](API_REFERENCE.md))

---

## Quick Reference

### Key Commands

```bash
# Train model
python scripts\train_baseline.py

# Run tests
python scripts\test_zero_trust_system.py

# Start dashboard
streamlit run src\dashboard\app.py
```

### Key Files

- `models/network_risk_classifier.pth` - Trained model
- `data/KDDTest+.txt` - Test dataset
- `logs/zero_trust_telemetry.json` - Access logs

### Key URLs

- Dashboard: <http://localhost:8501>
- Documentation: `docs/README.md`

---

*For detailed instructions, see [User Guide](USER_GUIDE.md)*
