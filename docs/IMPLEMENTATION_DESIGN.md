# System Architecture & Implementation Design

This document details the technical design of the **ZT-Shield** framework, providing a roadmap of the modular evolution from a baseline system to a research-grade defense implementation.

## 1. Core Architecture

The system follows a **Separation of Concerns (SoC)** model, decoupling data generation, model training, and attack evaluation.

```mermaid
graph TD
    A[Traffic Simulator] -->|Raw Data| B[Training Pipeline]
    B -->|Serialized Models| C[Detection Engine]
    C -->|API Hooks| D[Attack Laboratory]
    D -->|Evaluation Metrics| E[Research Dashboard]
    E -->|Feedback Loop| B
```

## 2. Component Breakdown

### A. Traffic Simulation (`src/simulation/`)

- **Traffic Generator**: Produces synthetic network logs (Packet size, Flow duration, Request frequency, etc.) with a configurable "Benign" profile.
- **Attack Generator**: Implements the logic for malicious injection, ensuring perturbations remain within the protocol bounds defined in the `threat_model.md`.

### B. Detection Engine (`src/core/`, `src/training/`)

- **Isolation Forest**: Handles unsupervised anomaly detection, acting as the "Zero-Trust" gatekeeper.
- **Random Forest**: The primary classifier trained to distinguish between Benign and Malicious patterns.
- **Defense Module**: Implements **Adversarial Fortification** via FGM-augmented training sets.

### C. Attack Laboratory (`src/attacks/`)

- **Black-Box HSJ**: A decision-boundary attack that uses query-feedback to find minimum perturbations.
- **Epsilon Sweeps**: Automated testing of model resilience across varying levels of noise (0.0 to 1.0).

### D. Research Dashboard (`src/dashboard/`)

- **Real-Time Feed**: Simulation of live Zero-Trust traffic.
- **Resilience Comparisons**: Side-by-side bar charts and linear stress tests comparing Baseline vs. Fortified models.
- **Explainability (SHAP)**: Global feature importance reporting to identify which network markers act as primary defense drivers.

## 3. Engineering Rigor

- **Deterministic Seeding**: Global seeds are managed in `src/core/utils.py` to ensure 100% reproducibility across different research environments.
- **Metric Decoupling**: Statistics (T-Tests, Confidence Intervals) are calculated in dedicated modules (`src/evaluation/statistics.py`) rather than being hardcoded in UI layers.

---
*Technical Documentation for ZT-Shield v1.0.*
