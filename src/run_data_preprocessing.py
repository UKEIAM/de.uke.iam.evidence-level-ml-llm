# src/run_data_preprocessing.py
"""
Script to run the data preprocessing pipeline:
1. Data Cleaning
2. Data Splitting
3. TF-IDF Vectorization
Returns:
- Cleaned data saved to path specified in data_config.yaml
- Train and test splits saved to path specified in splits_config.yaml
- TF-IDF features saved to path specified in splits_config.yaml
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
    # Step 1: Fetch CIViC Data
    print("🚀 Starting data preprocessing pipeline...")
    run_script("src/data/fetch_civic_data.py")

    # Step 1: Data Cleaning
    run_script("src/data/data_cleaning.py")

    # Step 2: Data Splitting
    run_script("src/data/splits.py")

    # Step 3: TF-IDF Vectorization
    run_script("src/features/tfidf_vectorization.py")
    print("📚 Data preprocessing pipeline completed.")
