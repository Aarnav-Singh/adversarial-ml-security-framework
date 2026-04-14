import torch
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.preprocessing.unsw_nb15 import UNSWNB15Loader
from src.preprocessing.cicids_2017 import CICIDS2017Loader
from src.models.classifier import build_model
from src.attacks.constraints.unsw_nb15_constraints import UNSWNB15Constraints
from src.attacks.constraints.cicids_constraints import CICIDS2017Constraints

def verify_unsw():
    print("=== UNSW-NB15 SHAPE CHECK ===")
    try:
        loader = UNSWNB15Loader()
        X_train, X_test, y_train, y_test, input_dim = loader.load_and_preprocess()
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape:  {X_test.shape}")
        print(f"y_train dist:  {np.mean(y_train):.3f} attack ratio")
        print(f"input_dim:     {input_dim}")
        
        assert input_dim == 42, f"Expected 42, got {input_dim}"
        assert X_train.shape[1] == 42, f"Expected 42 features, got {X_train.shape[1]}"
        assert X_train.max() <= 1.0001 and X_train.min() >= -0.0001, "Scaling failed"
        print("[PASSED] UNSW-NB15 Preprocessing")
        return True
    except Exception as e:
        print(f"[FAILED] UNSW-NB15 Preprocessing: {e}")
        return False

def verify_cicids():
    print("\n=== CICIDS-2017 SHAPE CHECK ===")
    try:
        loader = CICIDS2017Loader()
        X_train, X_test, y_train, y_test, input_dim = loader.load_and_preprocess()
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape:  {X_test.shape}")
        print(f"y_train dist:  {np.mean(y_train):.3f} attack ratio")
        print(f"input_dim:     {input_dim}")
        
        assert X_train.shape[1] == input_dim, "Dim mismatch"
        assert X_train.max() <= 1.0001 and X_train.min() >= -0.0001, "Scaling failed"
        assert not np.isinf(X_train).any(), "Inf present"
        assert not np.isnan(X_train).any(), "NaN present"
        print("[PASSED] CICIDS-2017 Preprocessing")
        return True
    except Exception as e:
        print(f"[FAILED] CICIDS-2017 Preprocessing: {e}")
        return False

def verify_models():
    print("\n=== MODEL BUILD CHECK ===")
    try:
        model_unsw = build_model(input_dim=42)
        model_cicids = build_model(input_dim=76)
        
        unsw_params = sum(p.numel() for p in model_unsw.parameters())
        cicids_params = sum(p.numel() for p in model_cicids.parameters())
        print(f"UNSW params:  {unsw_params:,}")
        print(f"CICIDS params: {cicids_params:,}")
        
        x_unsw = torch.randn(32, 42)
        out_unsw = model_unsw(x_unsw)
        assert out_unsw.shape == (32, 1)
        
        x_cicids = torch.randn(32, 76)
        out_cicids = model_cicids(x_cicids)
        assert out_cicids.shape == (32, 1)
        
        print("[PASSED] Model Builds")
        return True
    except Exception as e:
        print(f"[FAILED] Model Builds: {e}")
        return False

def verify_constraints():
    print("\n=== CONSTRAINT TEST ===")
    try:
        # UNSW
        c_unsw = UNSWNB15Constraints(feature_names=[f'f{i}' for i in range(42)])
        x = torch.rand(10, 42)
        x_p = x + 0.3
        x_proj = c_unsw.project(x_p)
        assert x_proj.max() <= 1.0001 and x_proj.min() >= -0.0001
        
        # CICIDS
        c_cicids = CICIDS2017Constraints(feature_names=[f'f{i}' for i in range(76)])
        grad = torch.ones(10, 76)
        m_grad = c_cicids.apply_gradient_mask(grad)
        
        print("[PASSED] Constraints Logic (Structure)")
        return True
    except Exception as e:
        print(f"[FAILED] Constraints Logic: {e}")
        return False

if __name__ == "__main__":
    m_ok = verify_models()
    c_ok = verify_constraints()
    
    # Preprocessing checks will fail if data is missing
    u_ok = verify_unsw()
    ci_ok = verify_cicids()
    
    if m_ok and c_ok and u_ok and ci_ok:
        print("\n[COMPLETE] ALL CHECKS PASSED. Ready for experiment loop.")
    else:
        print("\n[WARNING] SOME CHECKS FAILED. See log above.")
