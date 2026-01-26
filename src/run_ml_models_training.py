# src/run_ml_models_training.py
"""
Before running this script, run src/data_preprocessing.py.
To run the ML models training pipeline:
1. Models Training (XGBoost and Decision Tree)
2. Models Evaluation
Returns:
- Trained models saved to path specified in runs_config.yaml
- Evaluation metrics and confusion matrices saved to path specified in runs_config.yaml
"""

import subprocess
import sys


def run_script(script_path):
    """Run a Python script located at script_path."""
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error running {script_path}.")
    else:
        print(f"Successfully ran {script_path}.")


if __name__ == "__main__":
    run_script("src/ml/dt_tfidf.py")
    run_script("src/ml/xgb_tfidf.py")
    run_script("src/ml/dt_embeddings.py")
    run_script("src/ml/xgb_embeddings.py")

    print("🎉 Models training pipeline completed.")
