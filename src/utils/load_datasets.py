# src/utils/load_datasets.py
"""Utility functions to load datasets for training and evaluation."""

import yaml
import os
import pandas as pd

with open("config/splits_config.yaml", "r") as f:
    splits_config = yaml.safe_load(f)


def load_tfidf_features():
    """Loads the TF-IDF vectorized datasets for training and testing."""
    # Load TF-IDF features and labels
    train_tfidf_path = os.path.join(
        splits_config["splits_saving_path"], "train_tfidf_features.csv"
    )
    test_tfidf_path = os.path.join(
        splits_config["splits_saving_path"], "test_tfidf_features.csv"
    )
    train_tfidf_df = pd.read_csv(train_tfidf_path, sep="\t")
    test_tfidf_df = pd.read_csv(test_tfidf_path, sep="\t")
    print("TF-IDF datasets loaded successfully.")

    return train_tfidf_df, test_tfidf_df


def load_splits():
    """Loads the train and test splits datasets."""
    train_split_path = os.path.join(
        splits_config["splits_saving_path"], "train_dataset.csv"
    )
    test_split_path = os.path.join(
        splits_config["splits_saving_path"], "test_dataset.csv"
    )
    train_split_df = pd.read_csv(train_split_path, sep="\t")
    test_split_df = pd.read_csv(test_split_path, sep="\t")
    print("Splits datasets loaded successfully.")

    return train_split_df, test_split_df
