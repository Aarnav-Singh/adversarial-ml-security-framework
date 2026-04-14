"""
Dataset Download Helper Script
This script provides information and automated download links for the required datasets.
Note: CICIDS-2017 often requires manual registration and download from the official site.
"""

import os
import requests
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASETS = {
    'unsw-nb15': {
        'url': 'https://cloudstor.aarnet.edu.au/plus/s/2Eueiaobv9uEAtR/download', # Direct download from UNSW Canberra (example)
        'target_dir': 'data/unsw-nb15',
        'files': ['UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv', 'UNSW-NB15_features.csv']
    },
    'cicids-2017': {
        'url': 'https://www.unb.ca/cic/datasets/ids-2017.html', # Reference URL
        'target_dir': 'data/cicids-2017',
        'requires_manual': True
    }
}

def setup_directories():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for ds_name, info in DATASETS.items():
        path = os.path.join(base_dir, info['target_dir'])
        os.makedirs(path, exist_ok=True)
        logger.info(f"Directory ready: {path}")

def download_unsw():
    logger.info("Attempting to download UNSW-NB15 CSVs...")
    # NOTE: In a real scenario, this would download the zip and extract
    logger.info("Please visit https://research.unsw.edu.au/projects/unsw-nb15-dataset to download:")
    logger.info("1. UNSW_NB15_training-set.csv")
    logger.info("2. UNSW_NB15_testing-set.csv")
    logger.info("3. UNSW-NB15_features.csv")
    logger.info("Place them in data/unsw-nb15/")

def info_cicids():
    logger.info("CICIDS-2017 DOWNLOAD INSTRUCTIONS:")
    logger.info("1. Visit https://www.unb.ca/cic/datasets/ids-2017.html")
    logger.info("2. Register and download the 'Generated Labelled Network Traffic' (CSV files)")
    logger.info("3. Extract all daily CSV files (Monday through Friday) into data/cicids-2017/")

if __name__ == "__main__":
    setup_directories()
    download_unsw()
    info_cicids()
