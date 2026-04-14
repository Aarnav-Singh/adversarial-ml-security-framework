import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def calculate_macro_f1(y_true, y_pred):
    """Calculate Macro-averaged F1 score"""
    return f1_score(y_true, y_pred, average='macro')

def calculate_weighted_f1(y_true, y_pred):
    """Calculate Weighted-averaged F1 score"""
    return f1_score(y_true, y_pred, average='weighted')

def calculate_pr_auc(y_true, y_scores):
    """Calculate Precision-Recall Area Under Curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

def calculate_far(y_true, y_pred):
    """Calculate False Alarm Rate (FPR)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0

def calculate_constraint_validity(x_adv, constraint_fn):
    """
    Calculate the percentage of adversarial samples that satisfy domain constraints.
    This typically checks if project(x_adv) == x_adv (or within a tolerance).
    """
    # This is a conceptual implementation
    # A sample is valid if it doesn't change after being projected back to the valid space
    import torch
    if isinstance(x_adv, np.ndarray):
        x_adv_tensor = torch.FloatTensor(x_adv)
    else:
        x_adv_tensor = x_adv
        
    x_projected = constraint_fn.project(x_adv_tensor, x_adv_tensor) # x_orig == x_adv for self-check
    
    # Check for equality within floating point tolerance
    diff = torch.abs(x_adv_tensor - x_projected)
    valid_mask = torch.all(diff < 1e-5, dim=1)
    
    return valid_mask.float().mean().item()

def get_full_eval_report(y_true, y_pred, y_scores=None):
    """Generate a comprehensive evaluation report"""
    metrics = {
        'macro_f1': calculate_macro_f1(y_true, y_pred),
        'weighted_f1': calculate_weighted_f1(y_true, y_pred),
        'far': calculate_far(y_true, y_pred)
    }
    if y_scores is not None:
        metrics['pr_auc'] = calculate_pr_auc(y_true, y_scores)
        
    return metrics
