import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
from .base import NetworkBaseLoader, fix_seeds

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False
    logging.getLogger(__name__).warning("imbalanced-learn not installed — SMOTE disabled")

logger = logging.getLogger(__name__)

class CICIDS2017Loader(NetworkBaseLoader):
    def __init__(self, data_dir='data/cicids-2017'):
        super().__init__('CICIDS-2017')
        self.data_dir = data_dir
        self.scaler = MinMaxScaler()
        # Columns to drop based on instructions (identifiers and duplicates)
        self.drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Source Port', 
                          'Destination Port', 'Timestamp', 'Fwd Header Length.1']
        
    def load_and_preprocess(self, test_size=0.3):
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}. Run scripts/download_datasets.py")
            
        logger.info(f"Loading {len(csv_files)} files for CICIDS-2017...")
        frames = []
        for f in sorted(csv_files):
            df_temp = pd.read_csv(f, low_memory=False)
            # Strip whitespace from column names immediately
            df_temp.columns = df_temp.columns.str.strip()
            frames.append(df_temp)
            logger.info(f"  Loaded {os.path.basename(f)}: {len(df_temp)} rows")
            
        df = pd.concat(frames, ignore_index=True)
        logger.info(f"Total concatenated rows: {len(df)}")
        
        # 1. Strip whitespace from all string columns (values)
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip()
            
        # 2. Binary Labeling
        df['label_binary'] = (df['Label'] != 'BENIGN').astype(int)
        
        # 3. Drop Identifiers and original Label
        current_drop = [c for c in self.drop_cols if c in df.columns] + ['Label']
        df = df.drop(columns=current_drop)
        
        # 4. Remove Duplicates (~308,381 expected)
        count_before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {count_before - len(df)} duplicate rows")
        
        # 5. Handle infinite and NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 6. Vectorized median imputation (single pass, much faster than per-col loop)
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # 7. Drop zero-variance columns (vectorized nunique)
        zero_var_cols = df.columns[df.nunique() <= 1].tolist()
        logger.info(f"Dropping {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")
        df = df.drop(columns=zero_var_cols)
        
        # Split features and label
        X = df.drop(columns=['label_binary']).values.astype(np.float32)
        y = df['label_binary'].values.astype(np.int32)
        
        # Split before scaling as per best practice
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 8. Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # 9. SMOTE on training data only (if available)
        logger.info(f"Class distribution before SMOTE: {np.bincount(y_train)}")
        if _SMOTE_AVAILABLE:
            sm = SMOTE(random_state=42)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            logger.info(f"Class distribution after SMOTE: {np.bincount(y_train_res)}")
        else:
            X_train_res, y_train_res = X_train, y_train
            logger.warning("SMOTE skipped — install imbalanced-learn for oversampling")

        return X_train_res, X_test, y_train_res, y_test, X_train.shape[1]
