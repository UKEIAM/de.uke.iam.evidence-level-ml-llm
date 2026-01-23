"""
This script cleans the raw data by removing entries with missing abstracts,
dropping duplicates, and combining titles with abstracts into a single text field.
The cleaned data is then saved to a new CSV file (specified in the data_config).
"""

import pandas as pd
import yaml
import os

with open("config/data_config.yaml", "r") as f:
    data_config = yaml.safe_load(f)

df = pd.read_csv(
    data_config["raw_csv_path"],
    sep="\t",
)

df = df[df["abstract_fetched"].notna()]

# Drop duplicates based on 'reference_id'
df["text"] = df["title_fetched"] + " " + df["abstract_fetched"].str.strip()

# Keep only relevant columns
df = df[["reference_id", "text", "evidence_level"]]
print(
    f"Cleaned data has {len(df)} entries after removing missing abstracts and duplicates."
)

# Save cleaned data
OUTPUT_DIR = data_config["preprocessed_path"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

df.to_csv(
    os.path.join(OUTPUT_DIR, data_config["preprocessed_filename"]),
    sep="\t",
    index=False,
)
print(
    f"Cleaned data saved to {os.path.join(OUTPUT_DIR, data_config['preprocessed_filename'])}"
)
