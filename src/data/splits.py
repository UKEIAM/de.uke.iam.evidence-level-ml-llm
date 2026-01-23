"""
Script to split the preprocessed dataset into train, test and validation sets
with stratification based on the 'evidence_level' label.
Saves the splits as CSV files in the path specified in the splits_config file.
"""

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import os
import yaml

with open("config/splits_config.yaml", "r") as f:
    splits_config = yaml.safe_load(f)

OUTPUT_DIR = splits_config["splits_saving_path"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load preprocessed data
with open("config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

df_clean = pd.read_csv(
    data_config["preprocessed_csv_path"],
    sep="\t",
)

# Load label encoder
label_encoder = LabelEncoder()

# Split sizes
train_size = splits_config["train_size"]
test_size = splits_config["test_size"]


def weighted_train_test_val_split(df):
    """Retunrs train, test and validation sets as dataframes. Outputs visualization with disribution of labels per set."""
    X = df["text"]
    y = df["evidence_level"]

    # Encoding labels
    y_encoded = label_encoder.fit_transform(y)

    # weigths
    weights = compute_sample_weight(class_weight="balanced", y=y_encoded)

    # Split to train and test; then split train again into validation and train sets
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y_encoded,
        weights,
        test_size=splits_config["test_size"],
        random_state=splits_config.get("random_state", 42),
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def plot_splits_distribution(y_train, y_test):
    # Checking distribution of labels per set
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    distributions = []
    axs[0].set_title("Train")
    axs[1].set_title("Test")
    train_distributions = axs[0].hist(y_train, bins=5)
    test_distributions = axs[1].hist(y_test, bins=5)

    for distributions, ax in zip([train_distributions, test_distributions], axs):
        for j in range(5):
            # Display the counts on each column of the histograms
            ax.text(
                distributions[1][j],
                distributions[0][j],
                str(int(distributions[0][j])),
                weight="bold",
            )
    plt.savefig(OUTPUT_DIR + "/splits_distributions.png")
    plt.close()
    print(f"Saved splits distribution plot to {OUTPUT_DIR}/splits_distributions.png")


def save_datasets(X_train, y_train, X_test, y_test):
    # Converting sets to dataframes
    train = {"text": X_train, "label": y_train}
    train_df = pd.DataFrame(train)

    test = {"text": X_test, "label": y_test}
    test_df = pd.DataFrame(test)

    # Saving datasets
    train_df.to_csv(OUTPUT_DIR + "/train_dataset.csv", sep="\t", index=False)
    test_df.to_csv(OUTPUT_DIR + "/test_dataset.csv", sep="\t", index=False)
    print(f"Saved train and test sets to {OUTPUT_DIR}")


# Execute splitting, plotting and saving
X_train, X_test, y_train, y_test = weighted_train_test_val_split(df_clean)

plot_splits_distribution(y_train, y_test)

save_datasets(
    X_train,  # train texts
    y_train,  # train labels
    X_test,  # test texts
    y_test,  # test labels
)
