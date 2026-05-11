"""
Train Baseline Network Risk Classifier
Trains intrusion detection model on CICIDS-2017 dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.cicids_loader import load_cicids2017
from src.risk_engine.network_classifier import NetworkRiskClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cicids-2017')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def train_network_classifier():
    """Train network intrusion detection classifier on CICIDS-2017."""

    print("=" * 60)
    print("Training Network Risk Classifier on CICIDS-2017 Dataset")
    print("=" * 60)

    print("\n[1/5] Loading CICIDS-2017 data...")
    X_train, X_test, y_train, y_test = load_cicids2017(
        data_dir=DATA_DIR,
        max_samples=200_000,  # cap for speed; remove to use full dataset
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"  Attack ratio — train: {y_train.mean():.2%}  |  test: {y_test.mean():.2%}")

    print("\n[2/5] Converting to tensors...")
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t  = torch.FloatTensor(X_test)
    y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)

    print(f"\n[3/5] Initializing model (input_dim={X_train.shape[1]})...")
    model     = NetworkRiskClassifier(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n[4/5] Training for 20 epochs...")
    print("-" * 60)

    epochs       = 20
    best_accuracy = 0.0
    model_path   = os.path.join(MODELS_DIR, 'network_risk_classifier.pth')
    os.makedirs(MODELS_DIR, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss    = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_preds   = (test_outputs > 0.5).float()
            accuracy     = (test_preds == y_test_t).float().mean()

            tp = ((test_preds == 1) & (y_test_t == 1)).sum().float()
            fp = ((test_preds == 1) & (y_test_t == 0)).sum().float()
            fn = ((test_preds == 0) & (y_test_t == 1)).sum().float()

            precision = tp / (tp + fp + 1e-10)
            recall    = tp / (tp + fn + 1e-10)
            f1        = 2 * (precision * recall) / (precision + recall + 1e-10)

        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Acc: {accuracy:.4f} | Prec: {precision:.4f} | "
            f"Rec: {recall:.4f} | F1: {f1:.4f}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_path)

    print("-" * 60)
    print(f"\n[5/5] Training complete!")
    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Model saved to:     {model_path}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    train_network_classifier()
