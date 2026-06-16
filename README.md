# Zero-Trust Adversarial Intrusion Detection System

> **Research Project** · Department of Computer Science & Engineering · SRM Institute of Science and Technology
> 
> 📄 **Paper under review** — *Journal of Information Security and Applications*, Elsevier · Scopus Q1 · IF 3.7

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-0066CC)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
[![Dataset](https://img.shields.io/badge/Dataset-CICIDS--2017-0066CC)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Research Question

> *Can Zero-Trust context-aware policies mitigate adversarial evasion attacks against ML-based network intrusion detection systems — without requiring adversarial retraining of the underlying model?*

**Answer: Yes.** While FGSM/PGD adversarial attacks achieve a **65% bypass rate** against the ML classifier alone, the Zero-Trust contextual policy layer reduces the effective system bypass rate to **2.2%** — a 96.6% reduction — without any model retraining.

---

## Key Results

| Metric | UNSW-NB15 | CICIDS-2017 |
|---|---|---|
| ML Classifier Accuracy | **90.7%** | **99.9%** |
| Seeds (statistical validity) | 5 | 5 |

| Condition | Attack Bypass Rate |
|---|---|
| ML classifier alone (adversarial) | **65%** |
| ML classifier + Zero-Trust context layer | **2.2%** |
| **Reduction** | **96.6%** |

All accuracy metrics are averaged across 5 independent random seeds for statistical robustness.

---

## What This Project Does

ML-based intrusion detection systems are vulnerable to **adversarial evasion attacks** — carefully crafted perturbations to network traffic features that cause the model to misclassify malicious flows as benign. This project investigates whether the **contextual policy layer** of a Zero-Trust Network Architecture (ZTNA) can compensate when the ML component is actively fooled, without needing to retrain the model against adversarial examples.

The key insight is **dimensional orthogonality**: gradient-based adversarial attacks operate in the network feature space of the ML model, while Zero-Trust contextual signals (device trust, geographic risk, identity tier, time-of-day) exist in a completely separate information space sourced from external systems that cannot be manipulated by crafting network packet features. An attacker who suppresses their ML risk score through adversarial perturbation still cannot escape a DENY decision if their device enrollment status or geographic IP reputation flags them as suspicious.

---

## System Architecture

```
Network Traffic (UNSW-NB15 / CICIDS-2017 features)
          │
          ▼
┌─────────────────────────┐
│   ML Risk Classifier    │  ←  Neural Net: Input → 128 → 64 → 32 → 1 (PyTorch)
│   Risk Score: 0.0 – 1.0 │
└──────────┬──────────────┘
           │          ▲
           │          │  FGSM / PGD adversarial perturbation
           │          │  (domain-constrained, ε-bounded)
           ▼
┌─────────────────────────┐
│   Context Enrichment    │  ←  Device Trust · Geo-Risk · Identity Tier · Time-of-Day
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│   Zero-Trust Policy     │  ←  Priority-ordered rule engine
│   Engine                │
└──────────┬──────────────┘
           ▼
   ALLOW / DENY / STEP_UP_AUTH / RATE_LIMIT / ISOLATE
           │
           ▼
┌─────────────────────────┐
│   SOC Telemetry Log     │  ←  Structured JSON audit trail
│   Streamlit Dashboard   │
└─────────────────────────┘
```

---

## Datasets

| Dataset | Description | Features |
|---|---|---|
| **UNSW-NB15** | Modern network intrusion dataset from the Australian Centre for Cyber Security | 49 features, 9 attack categories |
| **CICIDS-2017** | Canadian Institute for Cybersecurity network traffic dataset | Benign + 14 attack types |

Both datasets were preprocessed through the same pipeline: feature selection, normalisation, and encoding of categorical fields before training.

---

## Adversarial Attack Implementation

**FGSM (Fast Gradient Sign Method)** — single-step attack, perturbs input features in the direction that maximally increases model loss:
```
x_adv = x + ε · sign(∇_x J(θ, x, y))
```

**PGD (Projected Gradient Descent)** — iterative, stronger attack; takes multiple small steps and projects back into the valid feature range after each step:
```
x_t+1 = Π_{x+S}(x_t + α · sign(∇_x J(θ, x_t, y)))
```

Both attacks are **domain-constrained**: perturbations are bounded to realistic network traffic feature ranges. Features with physical constraints (e.g., non-negative packet sizes, valid port ranges) are clamped after each perturbation step to ensure the adversarial examples remain plausible network traffic.

---

## Zero-Trust Context Signals

| Signal | Low Risk | Medium Risk | High Risk |
|---|---|---|---|
| **Device Trust** | Enrolled, compliant | Unknown device | Untrusted / blacklisted |
| **Geographic Risk** | Home region | Unusual region | High-risk region |
| **Identity Tier** | Privileged user | Standard user | Guest / service account |
| **Time-of-Day** | Business hours | Off-hours | Unusual hours |

The policy engine combines the ML risk score with these four context signals through a priority-ordered rule set. An adversarial packet that suppresses the ML risk score still fails the policy check if *any* context signal exceeds its threshold for the requested resource.

---

## Ablation Study

The ablation study isolates the contribution of each context signal to the overall adversarial robustness:

| Configuration | Adversarial Bypass Rate |
|---|---|
| ML Classifier Only | 65.0% |
| ML + Device Trust Context | ~38% |
| ML + Geographic Risk Context | ~45% |
| **Full System (All Context Signals)** | **2.2%** |

The full context combination is what drives the result — no single signal alone achieves comparable robustness.

---

## Quick Start

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/Aarnav-Singh/adversarial-ml-security-framework.git
cd adversarial-ml-security-framework

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets
# UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
# CICIDS-2017: https://www.unb.ca/cic/datasets/ids-2017.html
# Place files in the data/ folder

# 4. Train the ML classifier
python scripts/train_baseline.py

# 5. Launch the interactive SOC dashboard
streamlit run src/dashboard/app.py
# Open http://localhost:8501
```

The dashboard has three tabs:
- **SOC Console** — live traffic simulation and real-time decision log
- **Red Team** — adversarial attack testing with configurable ε
- **Blue Team** — defense analytics and context signal breakdown

---

## Reproducing the Paper Results

All experiments use fixed random seeds and are fully reproducible. Run in this order:

```bash
python scripts/train_multiseed.py      # Train across 5 seeds → CI metrics
python scripts/run_ablation.py         # 4-configuration ablation study
python scripts/run_epsilon_sweep.py    # FGSM + PGD across ε = 0.01 to 0.20
```

Results are saved to `results/` as JSON. The key output is `results/ablation_results.json`, which directly corresponds to the ablation table in the paper.

---

## Repository Structure

```
adversarial-ml-security-framework/
├── src/
│   ├── config.py                    # Central configuration & hyperparameters
│   ├── data/network_loader.py       # UNSW-NB15 / CICIDS-2017 loading & preprocessing
│   ├── risk_engine/                 # Neural network model definition & inference
│   ├── attacks/                     # FGSM, PGD, epsilon sweep, constraint validator
│   ├── policy/                      # Context enrichment & Zero-Trust rule engine
│   ├── system/                      # Full pipeline integration layer
│   ├── evaluation/                  # Multi-seed runner, statistics, reporting
│   ├── training/                    # Model training & adversarial retraining
│   ├── logging/                     # SOC telemetry & blue team analytics
│   └── dashboard/app.py             # Streamlit interactive dashboard
├── scripts/                         # Runnable experiment scripts
│   ├── train_baseline.py            # ← Start here
│   ├── train_multiseed.py
│   ├── run_ablation.py
│   ├── run_epsilon_sweep.py
│   └── test_zero_trust_system.py
├── docs/                            # Full documentation
├── data/                            # Place dataset files here (not committed)
├── models/                          # Trained model checkpoints (generated locally)
├── results/                         # Experiment outputs (generated by scripts)
├── figures/                         # Architecture diagrams & paper figures
├── requirements.txt
└── LICENSE
```

---

## Documentation

**For researchers:**
- [Research Methodology](docs/RESEARCH_METHODOLOGY.md) — Dataset preprocessing, model architecture, attack generation, evaluation metrics
- [Architecture](docs/ARCHITECTURE.md) — Component design, data flow, policy rule table
- [Threat Model](docs/THREAT_MODEL.md) — Attacker capabilities, domain constraints, security assumptions
- [Adversarial Attacks](docs/ADVERSARIAL_ATTACKS.md) — FGSM and PGD implementation details

**For developers:**
- [Usage Guide](docs/USAGE.md) — Dashboard walkthrough
- [API Reference](docs/API_REFERENCE.md) — Module and function documentation
- [Deployment Guide](docs/DEPLOYMENT.md) — Production deployment considerations
- [Troubleshooting](docs/TROUBLESHOOTING.md)

---

## Requirements

```
torch==2.0.1
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
streamlit==1.28.0
plotly==5.17.0
joblib==1.3.2
```

---

## Academic Context

This is the research paper basis:

**"Zero-Trust Context-Aware Defense Against Adversarial Evasion Attacks on ML-Based Network Intrusion Detection Systems"**
*Currently under review — Journal of Information Security and Applications, Elsevier, Scopus Q1, Impact Factor 3.7*

Available on request.

---

## License

Released under the [MIT License](LICENSE) for educational and research purposes. Adversarial attack implementations are included solely to evaluate and demonstrate defensive mechanisms — not for offensive use.

---

## Author

**Aarnav Singh** · [Portfolio](https://aarnav-singh.github.io) · [LinkedIn](https://www.linkedin.com/in/aarnav-singh-bb6076251/) · aarnavujji@gmail.com
