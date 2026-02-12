import numpy as np
from sklearn.metrics import accuracy_score

def calculate_evasion_rate(y_true, y_pred):
    """Evasion Rate = % of Attacks (1) misclassified as Benign (0)."""
    attack_indices = np.where(y_true == 1)[0]
    if len(attack_indices) == 0: return 0.0
    evasions = np.sum(y_pred[attack_indices] == 0)
    return evasions / len(attack_indices)

def get_perturbation_norms(X_orig, X_adv):
    """Calculates L2 and Linf norms of perturbations."""
    perturbations = X_adv - X_orig
    l2_norms = np.linalg.norm(perturbations, axis=1)
    linf_norms = np.linalg.norm(perturbations, ord=np.inf, axis=1)
    return np.mean(l2_norms), np.mean(linf_norms)
