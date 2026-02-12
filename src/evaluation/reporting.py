import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def export_results_to_json(summary, results_log, output_dir="results"):
    """
    Saves the research findings to a structured JSON file.
    Crucial for reproducibility and large-scale experimentation.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prune results_log of numpy arrays for JSON serialization
    serializable_log = []
    for entry in results_log:
        pruned = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in entry.items()}
        serializable_log.append(pruned)
        
    report = {
        "metadata": {
            "timestamp": timestamp,
            "version": "Research-Elite-v2.0"
        },
        "summary": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in summary.items()},
        "per_run_data": serializable_log
    }
    
    with open(filepath, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Research report exported to {filepath}")
    return filepath
