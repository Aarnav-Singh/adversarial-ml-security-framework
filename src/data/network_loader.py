"""
Network Data Loader
Generic loader interface for network intrusion detection datasets.
Provides a unified API for loading NSL-KDD, UNSW-NB15, and CICIDS-2017 data.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class NetworkDataLoader:
    """
    Generic network intrusion detection dataset loader.

    Supports NSL-KDD (text format) and generic CSV datasets.
    Provides a unified preprocessing pipeline with save/load for reproducibility.
    """

    # NSL-KDD column names (41 features + label + difficulty)
    NSL_KDD_COLUMNS = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    # Categorical features for encoding
    CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._is_fitted = False

    def load_and_preprocess(
        self,
        filepath: str,
        is_train: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load and preprocess a network dataset file.

        Supports NSL-KDD text format (.txt) and generic CSV (.csv).

        Args:
            filepath: Path to the dataset file.
            is_train: If True, fit the scaler/encoders on this data.
            test_size: Fraction for test split (only used when is_train=True).
            random_state: Random seed.

        Returns:
            Tuple of (X, y, feature_names):
                X: Scaled feature matrix (n_samples, n_features).
                y: Binary labels (0=benign, 1=attack).
                feature_names: List of feature names or None.
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext in ('.txt', '.csv'):
            df = self._load_nsl_kdd(filepath) if ext == '.txt' else pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        X, y, feature_names = self._preprocess(df, is_train=is_train)
        return X, y, feature_names

    def _load_nsl_kdd(self, filepath: str) -> pd.DataFrame:
        """Load NSL-KDD dataset from text format."""
        df = pd.read_csv(filepath, names=self.NSL_KDD_COLUMNS, header=None)
        # Drop difficulty column
        if 'difficulty' in df.columns:
            df = df.drop('difficulty', axis=1)
        return df

    def _preprocess(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Encode categoricals, scale, and binarize labels."""
        df = df.copy()

        # Binary label: 'normal' -> 0, everything else -> 1
        if 'label' in df.columns:
            y = (df['label'].str.strip().str.lower() != 'normal').astype(int).values
            df = df.drop('label', axis=1)
        else:
            y = np.zeros(len(df), dtype=int)

        # Encode categorical features
        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                if is_train:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le is not None:
                        # Handle unseen labels gracefully
                        df[col] = df[col].astype(str).apply(
                            lambda v: le.transform([v])[0]
                            if v in le.classes_ else 0
                        )
                    else:
                        df[col] = 0

        feature_names = list(df.columns)

        # Convert to float
        X = df.values.astype(np.float32)

        # Scale
        if is_train:
            X = self.scaler.fit_transform(X)
            self._is_fitted = True
        else:
            X = self.scaler.transform(X) if self._is_fitted else X

        return X, y, feature_names

    def save_preprocessors(self, save_dir: str) -> None:
        """Save scaler and encoders for later use."""
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(save_dir, 'network_scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(save_dir, 'network_encoders.pkl'))
        logger.info(f"Saved preprocessors to {save_dir}")

    def load_preprocessors(self, save_dir: str) -> None:
        """Load previously saved preprocessors."""
        scaler_path = os.path.join(save_dir, 'network_scaler.pkl')
        enc_path = os.path.join(save_dir, 'network_encoders.pkl')

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")

        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(enc_path) if os.path.exists(enc_path) else {}
        self._is_fitted = True
        logger.info(f"Loaded preprocessors from {save_dir}")

    def get_feature_bounds(self, X: np.ndarray) -> dict:
        """
        Get per-feature min/max bounds from a data array.

        Returns a dict with 'min' and 'max' arrays suitable for
        NetworkAdversarialAttacker.
        """
        return {
            'min': X.min(axis=0),
            'max': X.max(axis=0),
        }
