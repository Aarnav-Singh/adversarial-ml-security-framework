"""
Ablation study script for Zero-Trust policy layer evaluation.

Evaluates four system configurations to isolate the contribution of each
contextual factor:
    1. ML classifier only (no Zero-Trust context)
    2. ML + device trust context only
    3. ML + geo-risk context only
    4. Full system (ML + device trust + geo-risk + time-of-day + identity)

For each configuration, reports:
    - Deny rate on adversarial flows (using 30-sample adversarial test set)
    - False positive rate on legitimate traffic (using FULL label=0 test set)
    - Effective bypass rate

This is the single most important missing experiment for publication.

Usage:
    python scripts/run_ablation.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import (
    MODEL_DIR, DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    DEMO_SAMPLE_COUNT, DEMO_MIN_RISK_SCORE
)
from src.policy.zero_trust_engine import ZeroTrustEngine
from src.simulation.context_profiles import (
    generate_attacker_context, generate_legitimate_context
)


def load_assets():
    """Load models and test data."""
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))

    test_df = pd.read_csv(os.path.join(MODEL_DIR, "test_set.csv"))
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values

    return rf, iso, X_test, y_test


def select_adversarial_samples(rf, X_test, y_test, n_samples=30, min_risk=0.7):
    """Select high-confidence malicious samples for the demo/ablation.

    Picks samples where the ML model assigns risk score > min_risk,
    representing the model's most confident attack detections.

    Args:
        rf: Random Forest classifier.
        X_test: Test features.
        y_test: Test labels.
        n_samples: Number of samples to select.
        min_risk: Minimum risk score threshold.

    Returns:
        Tuple of (X_selected, y_selected, risk_scores, indices).
    """
    # Get risk scores for all attack samples
    attack_mask = y_test == 1
    X_attacks = X_test[attack_mask]

    probs = rf.predict_proba(X_attacks)[:, 1]
    high_conf_mask = probs >= min_risk

    X_high = X_attacks[high_conf_mask]
    probs_high = probs[high_conf_mask]

    # Select top n_samples by risk score
    if len(X_high) >= n_samples:
        top_indices = np.argsort(probs_high)[-n_samples:]
        X_selected = X_high[top_indices]
        scores = probs_high[top_indices]
    else:
        # If not enough high-confidence samples, take what we have
        X_selected = X_high
        scores = probs_high
        print(f"  Warning: Only {len(X_high)} samples have risk > {min_risk} "
              f"(requested {n_samples})")

    y_selected = np.ones(len(X_selected))

    return X_selected, y_selected, scores, None


def run_single_config(
    config_name, engine, rf, X_adv, y_adv, X_legit, seed=42
):
    """Evaluate a single ablation configuration.

    Args:
        config_name: Human-readable config name.
        engine: ZeroTrustEngine instance.
        rf: Random Forest classifier.
        X_adv: Adversarial samples.
        y_adv: Adversarial labels (all 1s).
        X_legit: Legitimate test samples (all label=0).
        seed: Random seed for context generation.

    Returns:
        Dict with deny_rate, fpr, bypass_rate, and per-sample details.
    """
    n_adv = len(X_adv)
    n_legit = len(X_legit)

    # Generate contextual profiles
    adv_contexts = generate_attacker_context(n_adv, seed=seed)
    legit_contexts = generate_legitimate_context(n_legit, seed=seed + 1000)

    # ---- Evaluate adversarial samples ----
    adv_risk_scores = rf.predict_proba(X_adv)[:, 1]
    adv_decisions = engine.evaluate_batch(adv_risk_scores, adv_contexts)

    n_denied_adv = sum(1 for d in adv_decisions if d.decision == "DENY")
    deny_rate = n_denied_adv / n_adv if n_adv > 0 else 0.0

    # Bypass = samples that got ALLOW despite being adversarial
    n_bypassed = n_adv - n_denied_adv
    bypass_rate = n_bypassed / n_adv if n_adv > 0 else 0.0

    # ---- Evaluate legitimate samples (FPR on FULL test set) ----
    legit_risk_scores = rf.predict_proba(X_legit)[:, 1]
    legit_decisions = engine.evaluate_batch(legit_risk_scores, legit_contexts)

    n_denied_legit = sum(1 for d in legit_decisions if d.decision == "DENY")
    fpr = n_denied_legit / n_legit if n_legit > 0 else 0.0

    # Per-sample details for adversarial samples
    per_sample = []
    for i, dec in enumerate(adv_decisions):
        per_sample.append({
            'sample_idx': i,
            'ml_risk': float(adv_risk_scores[i]),
            'ml_decision': 'DENY' if adv_risk_scores[i] > 0.5 else 'ALLOW',
            'zt_decision': dec.decision,
            'rule_fired': dec.rule_fired,
            'device_trust': adv_contexts[i]['device_trust'],
            'geo_risk': adv_contexts[i]['geo_risk'],
        })

    return {
        'config': config_name,
        'n_adversarial': n_adv,
        'n_legitimate': n_legit,
        'deny_rate': float(deny_rate),
        'false_positive_rate': float(fpr),
        'bypass_rate': float(bypass_rate),
        'n_denied_adversarial': n_denied_adv,
        'n_denied_legitimate': n_denied_legit,
        'per_sample_details': per_sample,
    }


def main():
    """Run the full ablation study."""
    print("=" * 60)
    print("  ZERO-TRUST ABLATION STUDY")
    print("=" * 60)

    # Load assets
    rf, iso, X_test, y_test = load_assets()

    # Select adversarial samples
    X_adv, y_adv, risk_scores, _ = select_adversarial_samples(
        rf, X_test, y_test, DEMO_SAMPLE_COUNT, DEMO_MIN_RISK_SCORE
    )
    print(f"\nSelected {len(X_adv)} adversarial samples (risk > {DEMO_MIN_RISK_SCORE})")

    # Extract ALL legitimate samples for FPR calculation
    legit_mask = y_test == 0
    X_legit = X_test[legit_mask]
    print(f"Using {len(X_legit)} legitimate samples for FPR analysis")

    # Also save demo samples for the Research Demo tab
    demo_path = os.path.join(DATA_DIR, "demo_samples.npy")
    np.save(demo_path, X_adv)
    print(f"Saved demo samples to: {demo_path}")

    # Define ablation configurations
    configs = [
        ("ML Only (No ZT Context)", set()),
        ("ML + Device Trust", {'device_trust'}),
        ("ML + Geo-Risk", {'geo_risk'}),
        ("Full System (All Factors)", {'device_trust', 'geo_risk', 'time_of_day', 'identity'}),
    ]

    # Run ablation
    all_results = []
    for config_name, factors in configs:
        print(f"\n{'─'*50}")
        print(f"  Config: {config_name}")
        print(f"  Enabled: {factors if factors else '∅ (ML only)'}")
        print(f"{'─'*50}")

        engine = ZeroTrustEngine(enabled_factors=factors)
        result = run_single_config(
            config_name, engine, rf, X_adv, y_adv, X_legit, seed=42
        )
        all_results.append(result)

        print(f"  Deny Rate (adversarial):    {result['deny_rate']:.1%}")
        print(f"  False Positive Rate:        {result['false_positive_rate']:.1%}")
        print(f"  Effective Bypass Rate:      {result['bypass_rate']:.1%}")

    # Save results
    results_payload = {
        'experiment': 'zero_trust_ablation',
        'n_adversarial_samples': len(X_adv),
        'n_legitimate_samples': len(X_legit),
        'min_risk_threshold': DEMO_MIN_RISK_SCORE,
        'configurations': [
            {k: v for k, v in r.items() if k != 'per_sample_details'}
            for r in all_results
        ],
        'detailed_results': all_results,
    }

    results_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_payload, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"  {'Configuration':<30s} {'Deny%':>8s} {'FPR%':>8s} {'Bypass%':>8s}")
    print("=" * 70)
    for r in all_results:
        print(f"  {r['config']:<30s} {r['deny_rate']:>7.1%} {r['false_positive_rate']:>7.1%} {r['bypass_rate']:>7.1%}")
    print("=" * 70)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
