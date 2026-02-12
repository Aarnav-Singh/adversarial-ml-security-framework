import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

def calculate_confidence_interval(data, confidence=0.95):
    """Computes the 95% Confidence Interval for a metric."""
    n = len(data)
    if n < 2: return 0.0
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    # Z-score for 95% is approx 1.96
    margin = 1.96 * std_err
    return margin

def calculate_statistical_significance(baseline_scores, defended_scores):
    """Performs a Paired T-Test and computes Effect Size (Cohen's d)."""
    t_stat, p_value = ttest_rel(baseline_scores, defended_scores)
    
    # Cohen's d: (mean_diff) / std_diff
    diff = np.array(defended_scores) - np.array(baseline_scores)
    if np.std(diff) == 0:
        cohens_d = 0.0
    else:
        cohens_d = np.mean(diff) / np.std(diff)
        
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "is_significant": p_value < 0.01
    }
