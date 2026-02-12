import numpy as np
from src.core.utils import benchmark_latency
from src.config import CONFIDENCE_THRESHOLD

@benchmark_latency
def ensemble_defense_predict(rf, iso_forest, X, conf_threshold=CONFIDENCE_THRESHOLD):
    """
    Hybrid Defense (Zero-Trust Logic).
    Combines Random Forest, Isolation Forest (Anomalies), and Prediction Confidence.
    """
    rf_probs = rf.predict_proba(X)
    rf_preds = rf.predict(X)
    if_preds = iso_forest.predict(X) 
    
    defended_preds = []
    for i in range(len(X)):
        prob_attack = rf_probs[i][1]
        is_anomaly = (if_preds[i] == -1)
        # Check if model is uncertain (near 0.5 boundary)
        is_uncertain = (0.5 - conf_threshold < prob_attack < 0.5 + conf_threshold)
        
        # Deny if RF says attack OR IF says anomaly OR RF is uncertain
        if rf_preds[i] == 1 or is_anomaly or is_uncertain:
            decision = 1 
        else:
            decision = 0 
        defended_preds.append(decision)
        
    return np.array(defended_preds)
