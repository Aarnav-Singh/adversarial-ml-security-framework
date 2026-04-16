"""
Train Network Risk Classifier on UNSW-NB15 Dataset
Updated baseline training for the current dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.unsw_nb15 import UNSWNB15Loader
from src.risk_engine.network_classifier import NetworkRiskClassifier


def train_network_classifier():
    """Train network intrusion detection classifier on UNSW-NB15"""
    
    print("="*60)
    print("Training Network Risk Classifier on UNSW-NB15 Dataset")
    print("="*60)
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'unsw-nb15')
    loader = UNSWNB15Loader(data_dir=data_dir)
    
    print("\n[1/4] Loading and preprocessing UNSW-NB15 data...")
    X_train, X_test, y_train, y_test, input_dim = loader.load_and_preprocess()
    
    print(f"  Input features: {input_dim}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Train class dist: Normal={sum(y_train==0)}, Attack={sum(y_train==1)}")
    print(f"  Test class dist: Normal={sum(y_test==0)}, Attack={sum(y_test==1)}")
    
    # Save the scaler
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    loader.save_scaler(os.path.join(models_dir, 'unsw_scaler.pkl'))
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    print(f"\n[2/4] Initializing model (input_dim={input_dim})...")
    
    # Initialize model
    model = NetworkRiskClassifier(input_dim=input_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 25
    print(f"\n[3/4] Training for {epochs} epochs...")
    print("-"*60)
    
    best_accuracy = 0.0
    model_path = os.path.join(models_dir, 'network_risk_classifier.pth')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_preds = (test_outputs > 0.5).float()
            accuracy = (test_preds == y_test_t).float().mean()
            
            # Precision, recall, F1 for attack class
            true_positives = ((test_preds == 1) & (y_test_t == 1)).sum().float()
            false_positives = ((test_preds == 1) & (y_test_t == 0)).sum().float()
            false_negatives = ((test_preds == 0) & (y_test_t == 1)).sum().float()
            
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Acc: {accuracy:.4f} | Prec: {precision:.4f} | "
              f"Rec: {recall:.4f} | F1: {f1:.4f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_path)
    
    # Also save demo samples from UNSW-NB15 test data
    print("\n[4/4] Generating demo samples...")
    attack_mask = y_test == 1
    rng = np.random.default_rng(42)
    n_demo = min(50, int(attack_mask.sum()))
    indices = rng.choice(int(attack_mask.sum()), n_demo, replace=False)
    X_demo = X_test[attack_mask][indices]
    
    demo_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'demo_samples.npy')
    np.save(demo_path, X_demo)
    print(f"  Saved {n_demo} demo samples to {demo_path}")
    
    print("-"*60)
    print(f"\nTraining complete!")
    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print("="*60)
    
    return model


if __name__ == "__main__":
    train_network_classifier()
