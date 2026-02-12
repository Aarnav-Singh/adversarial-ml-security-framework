from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from src.config import FGM_EPS

logger = logging.getLogger(__name__)

def run_whitebox_attack(surrogate_model, X_test, y_test, clip_values, sample_size=100, eps=FGM_EPS):
    """
    Runs White-Box Transfer Attack (FGM) via a Surrogate model.
    """
    logger.info(f"Initiating White-Box FGM Attack (Transfer) [Eps: {eps}, Sample: {sample_size}]")
    
    classifier = PyTorchClassifier(
        model=surrogate_model, 
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(surrogate_model.parameters(), lr=0.01),
        input_shape=(X_test.shape[1],), 
        nb_classes=2,
        clip_values=clip_values
    )
    
    indices = np.random.permutation(len(X_test))[:sample_size]
    X_sample, y_sample = X_test[indices], y_test[indices]
    
    attack = FastGradientMethod(estimator=classifier, eps=eps)
    X_adv = attack.generate(x=X_sample)
    
    return X_adv, X_sample, y_sample
