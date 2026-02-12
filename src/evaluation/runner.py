import os
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import accuracy_score

from src.config import MODEL_DIR
from src.core.defense import ensemble_defense_predict
from src.core.metrics import calculate_evasion_rate, get_perturbation_norms
from src.core.utils import set_seed, benchmark_latency
from src.evaluation.statistics import calculate_statistical_significance, calculate_confidence_interval

logger = logging.getLogger(__name__)

def load_system_assets(model_name="random_forest.pkl"):
    """Loads models and data from the centralized model directory."""
    rf = joblib.load(os.path.join(MODEL_DIR, model_name))
    iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    
    # Load Test Set
    test_df = pd.read_csv(os.path.join(MODEL_DIR, "test_set.csv"))
    X_test = test_df.drop(columns=['label']).values.astype(np.float32)
    y_test = test_df['label'].values
    
    # Load Training Set (for surrogate)
    train_df = pd.read_csv(os.path.join(MODEL_DIR, "train_set.csv"))
    X_train = train_df.drop(columns=['label']).values.astype(np.float32)
    y_train = train_df['label'].values
    
    clip_values = joblib.load(os.path.join(MODEL_DIR, "feature_bounds.pkl"))
    
    return rf, iso, X_test, y_test, X_train, y_train, clip_values

def evaluate_attack_vector(rf, iso, X_orig, X_adv, y_sample):
    """
    Computes professional research results for a single attack run.
    """
    # Baseline Results
    preds_base = rf.predict(X_adv)
    acc_base = accuracy_score(y_sample, preds_base)
    evasion_base = calculate_evasion_rate(y_sample, preds_base)
    
    # Defended Results
    # We use benchmark_latency wrapper from defense.py
    preds_def, latency_ms = ensemble_defense_predict(rf, iso, X_adv)
    acc_def = accuracy_score(y_sample, preds_def)
    evasion_def = calculate_evasion_rate(y_sample, preds_def)
    
    # Perturbation
    l2, linf = get_perturbation_norms(X_orig, X_adv)
    
    return {
        "acc_base": acc_base,
        "acc_def": acc_def,
        "evasion_base": evasion_base,
        "evasion_def": evasion_def,
        "l2": l2,
        "linf": linf,
        "latency_ms": latency_ms
    }

def run_research_suite(attack_func, rf, iso, X_test, y_test, clip_values, multi_seed=False, seeds=[42, 43, 44], **kwargs):
    """
    Orchestrates a robust multi-seed evaluation suite.
    """
    results_log = []
    
    active_seeds = seeds if multi_seed else [42]
    
    for seed in active_seeds:
        set_seed(seed)
        # Run the provided attack function
        X_adv, X_sample, y_sample, *extra = attack_func(rf, X_test, y_test, clip_values, **kwargs)
        
        res = evaluate_attack_vector(rf, iso, X_sample, X_adv, y_sample)
        # Add extra info like query counts if present
        if extra:
            res["avg_queries"] = extra[0]
            
        results_log.append(res)
        
    # Aggregate Statistics
    summary = {
        "mean_evasion_base": np.mean([r['evasion_base'] for r in results_log]),
        "mean_evasion_def": np.mean([r['evasion_def'] for r in results_log]),
        "std_evasion_def": np.std([r['evasion_def'] for r in results_log]),
        "mean_robust_acc_def": np.mean([r['acc_def'] for r in results_log]),
        "mean_latency_ms": np.mean([r['latency_ms'] for r in results_log]),
        "mean_l2": np.mean([r['l2'] for r in results_log])
    }
    
    # Add Stat Rigor (T-Test) if multi-seed
    if multi_seed:
        base_evasions = [r['evasion_base'] for r in results_log]
        def_evasions = [r['evasion_def'] for r in results_log]
        stats = calculate_statistical_significance(base_evasions, def_evasions)
        summary.update(stats)
        summary["ci_95"] = calculate_confidence_interval(def_evasions)
        
    return summary, results_log
