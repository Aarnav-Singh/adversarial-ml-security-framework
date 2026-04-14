import numpy as np
import torch
import random
import os
import joblib
import logging

logger = logging.getLogger(__name__)

def fix_seeds(seed=42):
    """Fix all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed fixed to {seed}")

class NetworkBaseLoader:
    """Base class for shared preprocessing logic across datasets"""
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.scaler = None
        
    def save_scaler(self, path):
        """Persist the fitted scaler"""
        if self.scaler:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.scaler, path)
            logger.info(f"Saved scaler for {self.dataset_name} to {path}")
            
    def load_scaler(self, path):
        """Load a persisted scaler"""
        if os.path.exists(path):
            self.scaler = joblib.load(path)
            logger.info(f"Loaded scaler for {self.dataset_name} from {path}")
            return self.scaler
        return None
