import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.preprocessing.base import fix_seeds
from src.preprocessing.unsw_nb15 import UNSWNB15Loader
from src.preprocessing.cicids_2017 import CICIDS2017Loader
from src.risk_engine.network_classifier import NetworkRiskClassifier
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.attacks.constraints.unsw_nb15_constraints import UNSWNB15Constraints
from src.attacks.constraints.cicids_constraints import CICIDS2017Constraints
from src.evaluation.metrics import get_full_eval_report, calculate_constraint_validity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEEDS = [0, 42, 123, 456, 789]
EPSILONS = [0.01, 0.02, 0.05, 0.10, 0.20]

def train_model(model, X_train, y_train, epochs=10, batch_size=256, device='cpu'):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.view_as(outputs))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    return model

def run_dataset_experiment(dataset_name, loader_class, constraint_class):
    logger.info(f"\n{'='*20} RUNNING EXPERIMENT: {dataset_name} {'='*20}")
    
    loader = loader_class()
    X_train, X_test, y_train, y_test, input_dim = loader.load_and_preprocess()
    
    # Feature names for constraints (assuming they exist in loader or derived)
    # For now we use the feature names from the loader if available
    feature_names = getattr(loader, 'feature_names', [f'feat_{i}' for i in range(input_dim)])
    constraints = constraint_class(feature_names)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_results = []
    
    for seed in SEEDS:
        logger.info(f"--- RUNNING SEED: {seed} ---")
        fix_seeds(seed)
        
        model = NetworkRiskClassifier(input_dim=input_dim)
        model = train_model(model, X_train, y_train, device=device)
        model.eval()
        
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        # 1. Baseline Evaluation
        with torch.no_grad():
            clean_scores = model(X_test_tensor).cpu().numpy()
            clean_preds = (clean_scores > 0.5).astype(int)
        
        seed_results = {
            'seed': seed,
            'baseline': get_full_eval_report(y_test, clean_preds, clean_scores)
        }
        
        # 2. Adversarial Epsilon Sweep
        seed_results['adversarial'] = {}
        for eps in EPSILONS:
            logger.info(f"  Evaluating Epsilon: {eps}")
            # FGSM
            X_adv_fgsm = fgsm_attack(model, device, X_test_tensor, y_test_tensor, eps, constraints)
            with torch.no_grad():
                fgsm_scores = model(X_adv_fgsm).cpu().numpy()
                fgsm_preds = (fgsm_scores > 0.5).astype(int)
            
            # PGD
            X_adv_pgd = pgd_attack(model, device, X_test_tensor, y_test_tensor, eps, alpha=eps/4, num_iter=10, constraint_fn=constraints)
            with torch.no_grad():
                pgd_scores = model(X_adv_pgd).cpu().numpy()
                pgd_preds = (pgd_scores > 0.5).astype(int)
                
            seed_results['adversarial'][eps] = {
                'fgsm': get_full_eval_report(y_test, fgsm_preds, fgsm_scores),
                'pgd': get_full_eval_report(y_test, pgd_preds, pgd_scores),
                'validity': {
                    'fgsm': calculate_constraint_validity(X_adv_fgsm, constraints),
                    'pgd': calculate_constraint_validity(X_adv_pgd, constraints)
                }
            }
        
        all_results.append(seed_results)
    
    # Calculate and report mean/std
    # (Implementation of summary logic here)
    logger.info(f"Finished {dataset_name} across {len(SEEDS)} seeds.")
    return all_results

if __name__ == "__main__":
    # Note: These will only run if datasets are present
    try:
        unsw_results = run_dataset_experiment('UNSW-NB15', UNSWNB15Loader, UNSWNB15Constraints)
        import json
        os.makedirs('results', exist_ok=True)
        with open('results/unsw_nb15_results.json', 'w') as f:
            json.dump(unsw_results, f, indent=4)
        logger.info("Saved UNSW-NB15 results to results/unsw_nb15_results.json")
    except Exception as e:
        logger.error(f"Failed UNSW-NB15: {e}")
        
    try:
        cicids_results = run_dataset_experiment('CICIDS-2017', CICIDS2017Loader, CICIDS2017Constraints)
        import json
        os.makedirs('results', exist_ok=True)
        with open('results/cicids_2017_results.json', 'w') as f:
            json.dump(cicids_results, f, indent=4)
        logger.info("Saved CICIDS-2017 results to results/cicids_2017_results.json")
    except Exception as e:
        logger.error(f"Failed CICIDS-2017: {e}")
