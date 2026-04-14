import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import logging
from .base import NetworkBaseLoader, fix_seeds

logger = logging.getLogger(__name__)

class UNSWNB15Loader(NetworkBaseLoader):
    def __init__(self, data_dir='data/unsw-nb15'):
        super().__init__('UNSW-NB15')
        self.data_dir = data_dir
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.categorical_cols = ['proto', 'service', 'state']
        self.drop_cols = ['id', 'srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime', 'attack_cat']
        
    def load_and_preprocess(self, train_file='UNSW_NB15_training-set.csv', test_file='UNSW_NB15_testing-set.csv'):
        # Construct paths
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"UNSW-NB15 files not found in {self.data_dir}. Run scripts/download_datasets.py")
            
        logger.info(f"Loading UNSW-NB15 training data from {train_path}...")
        df_train = pd.read_csv(train_path)
        logger.info(f"Loading UNSW-NB15 testing data from {test_path}...")
        df_test = pd.read_csv(test_path)
        
        # Merge for unified encoding if needed, or follow train-only fit logic
        # Instructions say: "Label encoding... not OHE, keeping input dim at 42"
        # Binary label: label column.
        
        def process_df(df, is_train=True):
            # Drop identifiers
            current_drop = [c for c in self.drop_cols if c in df.columns]
            df = df.drop(columns=current_drop)
            
            # Handle NaN
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
                else:
                    df[col] = df[col].fillna(df[col].mean())
            
            # Label Encoding
            for col in self.categorical_cols:
                if col in df.columns:
                    if is_train:
                        self.label_encoders[col] = LabelEncoder()
                        df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    else:
                        # Handle unseen labels by mapping them to closest known or mode
                        known_classes = set(self.label_encoders[col].classes_)
                        df[col] = df[col].apply(lambda x: x if x in known_classes else 'unknown')
                        if 'unknown' not in known_classes:
                            # If 'unknown' wasn't in train, map to first class
                            df[col] = df[col].apply(lambda x: x if x in known_classes else list(known_classes)[0])
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
            return df

        df_train = process_df(df_train, is_train=True)
        df_test = process_df(df_test, is_train=False)
        
        X_train = df_train.drop(columns=['label']).values.astype(np.float32)
        y_train = df_train['label'].values.astype(np.int32)
        X_test = df_test.drop(columns=['label']).values.astype(np.float32)
        y_test = df_test['label'].values.astype(np.int32)
        
        # Scaling - Fit on train only
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # SMOTE on training data only
        logger.info(f"Class distribution before SMOTE: {np.bincount(y_train)}")
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        logger.info(f"Class distribution after SMOTE: {np.bincount(y_train_res)}")
        
        return X_train_res, X_test, y_train_res, y_test, X_train.shape[1]
