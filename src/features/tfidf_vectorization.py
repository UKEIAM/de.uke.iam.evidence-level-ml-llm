"""
TF-IDF Vectorization of text data from train and test splits.
Saves the TF-IDF features as CSV files on path specified in the splits_config file.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import yaml

with open("config/splits_config.yaml", "r") as f:
    splits_config = yaml.safe_load(f)
splits_path = splits_config["splits_saving_path"]

with open("config/runs_config.yaml", "r") as f:
    runs_config = yaml.safe_load(f)

# Load split data
train_df = pd.read_csv(f"{splits_path}/train_dataset.csv", sep="\t")
test_df = pd.read_csv(f"{splits_path}/test_dataset.csv", sep="\t")

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=runs_config["tfidf"]["max_features"],
    ngram_range=tuple(runs_config["tfidf"]["ngram_range"]),
    stop_words="english",
)

# Fit and transform the training data, transform the test data
X_train_tfidf = tfidf.fit_transform(train_df[runs_config["data"]["text_column"]])
X_test_tfidf = tfidf.transform(test_df[runs_config["data"]["text_column"]])
print("TF-IDF vectorization completed.")

# Convert to DataFrame for easier handling
train_tfidf_df = pd.DataFrame(
    X_train_tfidf.toarray(), columns=tfidf.get_feature_names_out()
)
test_tfidf_df = pd.DataFrame(
    X_test_tfidf.toarray(), columns=tfidf.get_feature_names_out()
)

# Save the TF-IDF features as DataFrames
train_tfidf_df.to_csv(f"{splits_path}/train_tfidf_features.csv", sep="\t", index=False)
test_tfidf_df.to_csv(f"{splits_path}/test_tfidf_features.csv", sep="\t", index=False)
print(
    f"TF-IDF features saved to {splits_path}/train_tfidf_features.csv and {splits_path}/test_tfidf_features.csv"
)
