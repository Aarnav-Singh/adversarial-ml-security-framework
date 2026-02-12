import numpy as np
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import SklearnClassifier
import logging

logger = logging.getLogger(__name__)

class QueryCountingWrapper:
    """Wraps a model to count total queries during an attack."""
    def __init__(self, model):
        self.model = model
        self.query_count = 0

    def predict(self, X):
        self.query_count += len(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        self.query_count += len(X)
        return self.model.predict_proba(X)

def run_blackbox_attack(model, X_test, y_test, clip_values, sample_size=100, max_iter=50):
    """
    Runs HopSkipJump attack with Query Complexity tracking.
    """
    logger.info(f"Initiating Black-Box HSJ Attack (Sample Size: {sample_size}, Max Iter: {max_iter})")
    
    # Wrap model for query counting
    policed_model = QueryCountingWrapper(model)
    classifier = SklearnClassifier(model=policed_model, clip_values=clip_values)
    
    indices = np.random.permutation(len(X_test))[:sample_size]
    X_sample, y_sample = X_test[indices], y_test[indices]
    
    attack = HopSkipJump(
        classifier=classifier, 
        max_iter=max_iter, 
        max_eval=100, 
        init_eval=10,
        verbose=False
    )
    
    X_adv = attack.generate(x=X_sample)
    
    total_queries = policed_model.query_count
    avg_queries = total_queries / sample_size
    
    logger.info(f"Attack Complete. Total Queries: {total_queries} (Avg: {avg_queries:.1f} per sample)")
    
    return X_adv, X_sample, y_sample, avg_queries
