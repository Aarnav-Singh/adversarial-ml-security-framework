# Adversarial ML Security Framework (v1.0)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)

**ZT-Shield (Original Version)** is a professional-grade research framework designed to evaluate and enhance the adversarial robustness of Machine Learning models within **Zero-Trust (ZT) Network Architectures**.

The framework provides an end-to-end pipeline: from high-fidelity network traffic simulation and baseline detection to rigorous black-box attack evaluation and adversarial fortification.

---

## 🔬 Core Research Capabilities

- **Hybrid Detection Engine**: Combines **Isolation Forest** (for unsupervised anomaly gatekeeping) and **Random Forest** (for high-precision classification).
- **Advanced Attack Suite**: Implements **HopSkipJump (HSJ)** decision-boundary attacks and **Fast Gradient Method (FGM)** transfers.
- **Resilience Engineering**: Automated **Adversarial Fortification** (Retraining) to bridge the security gap between Baseline and Robust states.
- **Side-by-Side Analytics**: Visual verification of **Drift Resilience**, **Attack Resistance**, and **State Evolution** graphs.

---

## 📂 Project Structure

```text
├── src/
│   ├── simulation/     # Traffic & Attack generators
│   ├── training/       # Model dev & fortification logic
│   ├── attacks/        # HSJ & FGM implementation
│   ├── evaluation/     # Metric suites & reporting
│   └── dashboard/      # Research-Elite Streamlit Console
├── docs/               # Technical Deep-Dives
│   ├── WALKTHROUGH.md
│   ├── IMPLEMENTATION_DESIGN.md
│   └── GITHUB_GUIDE.md
├── tests/              # Automated unit tests
├── models/             # Pre-trained research assets
└── threat_model.md     # Formal Attacker Matrix
```

---

## 🛠 Setup & Execution

### 1. Requirements

Install the research-grade dependency suite:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap streamlit plotly adversarial-robustness-toolbox joblib
```

### 2. Launch the Research Console

The system is controlled via a centralized SOC-style dashboard:

```bash
streamlit run src/dashboard/app.py
```

---

## 📄 Documentation Index

- **[Formal Threat Model](file:///c:/Adversarial%20Model%20training/threat_model.md)**: Understand the attacker capabilities and research constraints.
- **[Verification Walkthrough](file:///c:/Adversarial%20Model%20training/docs/WALKTHROUGH.md)**: Follow the step-by-step guide to replicate all research results.
- **[Implementation Design](file:///c:/Adversarial%20Model%20training/docs/IMPLEMENTATION_DESIGN.md)**: Detailed look at the modular architecture.
- **[GitHub Upload Guide](file:///c:/Adversarial%20Model%20training/docs/GITHUB_GUIDE.md)**: Instructions for deploying this project to a public repository.

---

## 🛡️ Research Ethics & Usage

This framework is intended for **Academic Research** and **Defensive Security** purposes only. It demonstrates how to fortify security infrastructure against advanced ML evasion techniques.

---
*Developed for the ZT-Shield Open Source Project (2026).*
