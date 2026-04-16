import os
import sys
import logging
import json
import numpy as np

sys.path.append(os.getcwd())
from src.preprocessing.cicids_2017 import CICIDS2017Loader
from src.attacks.constraints.cicids_constraints import CICIDS2017Constraints
import src.evaluation.run_experiment
from src.evaluation.run_experiment import run_dataset_experiment

# SIGNIFICANTLY shrink the experiment computational footprint:
src.evaluation.run_experiment.SEEDS = [42]           # Run 1 deterministic seed instead of 5
src.evaluation.run_experiment.EPSILONS = [0.05, 0.10] # Reduced adversarial intensity sweep points

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We need to monkey-patch the SMOTE and load behavior temporarily because 
# SMOTE-ing 1.3 million samples takes hours on a standard CPU.
original_load = CICIDS2017Loader.load_and_preprocess

def optimized_load_and_preprocess(self, test_size=0.3):
    logger.info("Using OPTIMIZED load routine - sampling data to prevent memory overflow...")
    
    import glob
    import pandas as pd
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    
    csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
    frames = []
    
    # Only load a subset of rows from each file to drastically cut sizes early
    for f in sorted(csv_files):
        df_temp = pd.read_csv(f, low_memory=False)
        # Randomly sample 5% of each file to retain attack distribution but drop volume
        if len(df_temp) > 10000:
            df_temp = df_temp.sample(frac=0.05, random_state=42)
            
        df_temp.columns = df_temp.columns.str.strip()
        frames.append(df_temp)
        logger.info(f"  Downsampled {os.path.basename(f)}: {len(df_temp)}")
        
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Optimized concatenated rows: {len(df)}")
    
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()
        
    df['label_binary'] = (df['Label'] != 'BENIGN').astype(int)
    current_drop = [c for c in self.drop_cols if c in df.columns] + ['Label']
    df = df.drop(columns=current_drop)
    df = df.drop_duplicates()
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.columns:
        if df[col].dtype != 'object' and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    zero_var_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=zero_var_cols)
    
    X = df.drop(columns=['label_binary']).values.astype(np.float32)
    y = df['label_binary'].values.astype(np.int32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    X_train = self.scaler.fit_transform(X_train)
    X_test = self.scaler.transform(X_test)
    
    logger.info(f"Class distribution before SMOTE: {np.bincount(y_train)}")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    logger.info(f"Class distribution after SMOTE: {np.bincount(y_train_res)}")
    
    # Cap the final test set to 5000 max for speeding up adversarial generation (PGD/FGSM are slow)
    if len(X_test) > 5000:
        idx = np.random.choice(len(X_test), 5000, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]
        
    return X_train_res, X_test, y_train_res, y_test, X_train.shape[1]

# Apply the patch
CICIDS2017Loader.load_and_preprocess = optimized_load_and_preprocess

# Patch epochs to 5 for speed
import src.evaluation.run_experiment
original_train = src.evaluation.run_experiment.train_model
def fast_train(model, X_train, y_train, epochs=3, batch_size=256, device='cpu'):
    logger.info("Training with reduced epochs (3) for speed...")
    return original_train(model, X_train, y_train, epochs, batch_size, device)

src.evaluation.run_experiment.train_model = fast_train


if __name__ == "__main__":
    try:
        logger.info("Starting (Optimized) CICIDS-2017 experiment generation...")
        cicids_results = run_dataset_experiment('CICIDS-2017', CICIDS2017Loader, CICIDS2017Constraints)
        os.makedirs('results', exist_ok=True)
        with open('results/cicids_2017_results.json', 'w') as f:
            json.dump(cicids_results, f, indent=4)
        logger.info("Saved CICIDS-2017 results to results/cicids_2017_results.json successfully!")
    except Exception as e:
        logger.error(f"Failed CICIDS-2017: {e}")
        import traceback
        traceback.print_exc()
