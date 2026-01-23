import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import yaml
from utils.load_datasets import load_tfidf_features, load_splits
from sklearn.utils.class_weight import compute_sample_weight
from ml.base_models import DecisionTreeTrainer
import os
from datetime import datetime

with open("config/runs_config.yaml", "r") as f:
    runs_config = yaml.safe_load(f)

# Load TF-IDF datasets and labels
train_tfidf_df, test_tfidf_df = load_tfidf_features()
train_labels_df, test_labels_df = load_splits()

# Prepare data for Decision Tree
X_train = train_tfidf_df.values
y_train = train_labels_df["label"].values
X_test = test_tfidf_df.values
y_test = test_labels_df["label"].values

# Compute number of classes and sample weights
num_classes = len(train_labels_df["label"].unique())
weights = compute_sample_weight(class_weight="balanced", y=train_labels_df["label"])

# Load trainer
trainer = DecisionTreeTrainer()
# Train the model
model = trainer.train(X_train, y_train)
# Evaluate the model (print classification report)
trainer.evaluate(model, X_test, y_test)
# Save the model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"dt_model_{timestamp}.json"

# Ensure output directory exists
output_dir = runs_config["dt"]["output_dir"]
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(runs_config["dt"]["output_dir"], model_filename)
trainer.save_model(model, model_path)
