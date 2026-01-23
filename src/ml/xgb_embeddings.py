import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.load_datasets import load_splits
from sklearn.utils.class_weight import compute_sample_weight
from ml.base_models import XGBTrainer
from features.embeddings import prepare_embeddings
import yaml
import os
from datetime import datetime

with open("config/runs_config.yaml", "r") as f:
    runs_config = yaml.safe_load(f)

# Load splits datasets
train_split_df, test_split_df = load_splits()

# Prepare embeddings for training and testing data
X_train, X_test = prepare_embeddings(train_split_df, test_split_df, runs_config)

# Prepare labels
y_train = train_split_df["label"].values
y_test = test_split_df["label"].values

# Compute number of classes and sample weights
num_classes = len(train_split_df["label"].unique())
weights = compute_sample_weight(class_weight="balanced", y=train_split_df["label"])

# Load trainer
trainer = XGBTrainer()

# Train the model
model = trainer.train(X_train, y_train, num_classes, weights)

# Evaluate the model (print classification report)
trainer.evaluate(model, X_test, y_test)

# Save the model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"xgb_model_{timestamp}.json"
run_name = runs_config.get("embedding", {}).get("run_name", "")
file_name = f"xgb_{run_name}_model_{timestamp}.json" if run_name else model_filename

# Ensure output directory exists
output_dir = runs_config["xgb"]["output_dir"]
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(runs_config["xgb"]["output_dir"], file_name)
trainer.save_model(model, model_path)
