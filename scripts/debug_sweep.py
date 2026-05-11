import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dashboard.app import get_eval_data
from src.training.surrogate import train_surrogate
from src.attacks.sweep import run_epsilon_sweep
from src.core.defense import ensemble_defense_predict
import src.config as config
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import torch.optim as optim

def test_sweep():
    print("Loading data...")
    rf_model, iso_model, X_test, y_test, X_train, y_train, clip_values = get_eval_data("random_forest.pkl")
    
    print(f"Training surrogate on {X_train.shape}...")
    surr, val_acc = train_surrogate(X_train, y_train)
    print(f"Surrogate val acc: {val_acc}")
    
    print("Running sweep...")
    eps_values = [0.01, 0.05, 0.1, 0.2]
    df = run_epsilon_sweep(
        rf_model, iso_model, surr, X_test, y_test, clip_values,
        eps_values=eps_values,
        ensemble_defense_predict_func=ensemble_defense_predict,
        sample_size=10,
        enable_debug_logging=True
    )
    print(df)
    
    # Manual check of perturbation size
    print("Manual FGM check:")
    classifier = PyTorchClassifier(
        model=surr,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(surr.parameters(), lr=0.001),
        input_shape=(X_test.shape[1],),
        nb_classes=2,
        clip_values=clip_values
    )
    attack = FastGradientMethod(estimator=classifier, eps=0.2)
    X_sample = X_test[:10]
    X_adv = attack.generate(x=X_sample)
    
    diff = np.abs(X_adv - X_sample)
    print("Max diff per feature:", diff.max(axis=0))
    print("Mean diff per feature:", diff.mean(axis=0))
    
    print("Original RF preds:", rf_model.predict(X_sample))
    print("Adv RF preds:", rf_model.predict(X_adv))
    print("True labels:", y_test[:10])

if __name__ == "__main__":
    test_sweep()
