# Adversarial ML Security Framework 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)

**Adversarial ML Security Framework** is a modular, learning-focused project exploring how machine learning systems behave under adversarial conditions within a Zero-Trust inspired architecture.

The project provides an experimental pipeline: from generating network traffic and training baseline detectors to evaluating robustness against black-box attacks and exploring model fortification techniques.

---

## Core Exploration Areas

- **Layered Detection**: Exploring the combination of **Isolation Forest** (for anomaly gatekeeping) and **Random Forest** (for classification).
- **Adversarial Evaluation**: Implementing **HopSkipJump (HSJ)** decision-boundary attacks and **Fast Gradient Method (FGM)** transfers.
- **Model Fortification**: Testing **Adversarial Retraining** to observe the security gap between Baseline and Robust states.
- **Comparative Analytics**: Visual analysis of **Drift Resilience**, **Attack Resistance**, and **State Evolution** graphs.

---

## Project Structure

```text
├── src/
│   ├── simulation/     # Traffic & Attack generators
│   ├── training/       # Model dev & fortification logic
│   ├── attacks/        # HSJ & FGM implementation
│   ├── evaluation/     # Metric suites & reporting
│   ├── logging/        # Structured logging & analysis
│   └── dashboard/      # Streamlit Analysis Console
├── docs/               # Technical Deep-Dives
├── logs/               # Consolidated Logs (See logs/sessions)
│   ├── sessions/       # Attack + Defense history (JSON/CSV)
│   ├── attacks/        # (Provisioned Placeholder)
│   └── analytics/      # Hardening reports
├── tests/              # Automated unit tests
├── models/             # Local model artifacts (not committed)
└── data/               # Benign & Combined traffic datasets
```

---

## Setup and Execution

### 1. Requirements

Install required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Initialization

Since datasets and models are generated locally, you must initialize the framework once before launching:

```bash
# Generate baseline traffic
python src/simulation/traffic_generator.py

# Simulate initial attacks
python src/simulation/attack_generator.py

# Train security models
python src/training/core_model.py
```

### 3. Launch the Analysis Console

The system is controlled via a centralized dashboard:

```bash
streamlit run src/dashboard/app.py
```

---

## Documentation Index

- **[Master Research Report](FINAL_PROJECT_REPORT.md)**: Executive summary of research findings and defense gains.
- **[Architecture Overview](ARCHITECTURE_OVERVIEW.md)**: System design, component interaction, and data flow.
- **[Project Chronology](PROJECT_CHRONOLOGY.md)**: Timeline of development phases, critical fixes, and features.
- **[Structured Logging Guide](LOGGING_SYSTEM.md)**: Detailed dive into the auditing architecture.
- **[Logging Usage Guide](LOGGING_USAGE.md)**: Developer documentation for logging integration.
- **[Experimental Walkthrough](WALKTHROUGH.md)**: Follow the step-by-step guide to replicate the observations.
- **[Implementation Design](IMPLEMENTATION_DESIGN.md)**: Detailed look at the modular ML architecture.
- **[Formal Threat Model](threat_model.md)**: Attacker capabilities and research constraints.
- **[GitHub Deployment Guide](GITHUB_GUIDE.md)**: Instructions for project version control.

---

## Research Ethics and Usage

This framework is intended for **Academic Research** and **Defensive Security** purposes only. It demonstrates how to fortify security infrastructure against advanced ML evasion techniques.

---
*Developed as part of an adversarial ML security study (2026).*
