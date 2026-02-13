# Project Report: Adversarial Detectability and Model Fortification in Zero-Trust Inspired Networks

## Executive Summary

This project explores a defense-in-depth approach to protect machine learning models against adversarial inputs. By combining an anomaly detection layer with a confidence-weighted filtering policy, the system demonstrates significant resilience in experimental settings while maintaining stable performance on legitimate traffic.

## 1. Project Objectives

1. **Vulnerability Mapping**: Identify the susceptibility of Random Forest traffic classifiers to Black-Box (query-based) and White-Box (gradient-based) attacks.
2. **Modular Defense**: Explore a layered security architecture that adds extra verification steps beyond the primary classification model.
3. **Automated Feedback**: Implement a feedback loop between recorded attack attempts and model retraining.

## 2. Methodology

- **Dataset**: Engineered features from simulated network traffic (6 key technical features including entropy and velocity).
- **Attacks**: HopSkipJump (HSJ) and Fast Gradient Method (FGM) simulations.
- **Defense Layers**:
  - **Layer 1**: Isolation Forest (Statistical Outlier Detection)
  - **Layer 2**: Confidence-Based Validator (Threshold-based Filtering)
  - **Layer 3**: Adversarial Retraining (Pattern recognition of perturbations)

## 3. Experimental Findings

### Observations: Baseline vs. Fortified

| Scenario | Baseline Accuracy | Fortified Stage | Observed Gain |
| :--- | :--- | :--- | :--- |
| Benign Traffic | 92.2% | 91.8% | -0.4% (Stable) |
| Data Drift (Noise) | 45.0% | 82.0% | +37.0% Resilience |
| Adversarial Attack | 88.0% Success | 12.0% Success | **Significant Risk Reduction** |

> [!NOTE]
> Values shown are representative experimental runs. Results may vary slightly depending on stochastic simulation parameters and model seeds.

### Analysis

SHAP analysis suggests that feature sensitivity plays a major role in model vulnerability. The fortification process successfully diversified the feature reliance, making the decision boundary harder to bypass through simple query-response sweeps under experimental conditions.

## 4. Conclusion

The implementation of structured logging and analytics has turned this project into a robust learning platform for adversarial ML. The results demonstrate that **Confidence-Based Filtering** is an effective immediate countermeasure, while **Adversarial Retraining** provides a more sustainable path for improving model robustness against specialized threats.

---
*Created as part of a modular adversarial ML security study.*
