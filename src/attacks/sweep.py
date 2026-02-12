from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import logging

from src.core.defense import ensemble_defense_predict
from src.config import EPS_VALUES

logger = logging.getLogger(__name__)

def run_epsilon_sweep(rf, iso_forest, surrogate_model, X_test, y_test, clip_values, sample_size=50):
    """
    Generates robustness curves for both Baseline and Defended models.
    Sweeps across epsilon values to measure accuracy degradation.
    """
    logger.info(f"Initiating Epsilon Sweep across {len(EPS_VALUES)} values...")
    
    classifier = PyTorchClassifier(
        model=surrogate_model, 
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(surrogate_model.parameters(), lr=0.01),
        input_shape=(X_test.shape[1],), 
        nb_classes=2,
        clip_values=clip_values
    )
    
    # Use a fixed sample for the sweep to ensure curve consistency
    indices = np.random.RandomState(42).permutation(len(X_test))[:sample_size]
    X_sample, y_sample = X_test[indices], y_test[indices]
    
    sweep_results = []
    
    for eps in EPS_VALUES:
        attack = FastGradientMethod(estimator=classifier, eps=eps)
        X_adv = attack.generate(x=X_sample)
        
        # Baseline Accuracy
        preds_base = rf.predict(X_adv)
        acc_base = accuracy_score(y_sample, preds_base)
        
        # Defended Accuracy (Ensemble)
        preds_def = ensemble_defense_predict(rf, iso_forest, X_adv)
        acc_def = accuracy_score(y_sample, preds_def)
        
        sweep_results.append({
            "epsilon": eps,
            "Baseline": acc_base,
            "ZT-Shield (Defended)": acc_def
        })
        logger.debug(f"Eps {eps} | Base: {acc_base*100:.1f}% | Def: {acc_def*100:.1f}%")
        
    return pd.DataFrame(sweep_results)
