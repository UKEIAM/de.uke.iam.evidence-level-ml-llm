import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.metrics import compute_performance_metrics, save_performance_metrics
from evaluation.save_cm import save_confusion_matrix
from evaluation.save_test_preds import save_test_predictions
from utils.load_datasets import load_splits
from sklearn.utils.class_weight import compute_sample_weight
from ml.base_models import DecisionTreeTrainer
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
trainer = DecisionTreeTrainer()

# Train the model
model = trainer.train(X_train, y_train)

# --- Evaluate model on test data ---
y_pred = model.predict(X_test)

# Directories for saving outputs
output_dir = runs_config["dt"]["output_dir"]
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = runs_config.get("embeddings", {}).get("run_name", "")

# Save model
model_filename = f"dt_{run_name}_model_{timestamp}.json"
model_path = os.path.join(runs_config["dt"]["output_dir"], model_filename)
trainer.save_model(model, model_path)

# Save predictions
output_preds_path = os.path.join(
    runs_config["dt"]["output_dir"], f"dt_{run_name}_predictions_{timestamp}.csv"
)
save_test_predictions(y_pred=y_pred, y_test=y_test, output_path=output_preds_path)

# Save confusion matrix
cm_path = os.path.join(
    runs_config["dt"]["output_dir"],
    f"dt_{run_name}_confusion_matrix_{timestamp}.png",
)
save_confusion_matrix(
    y_test=y_test,
    y_pred=y_pred,
    output_path=cm_path,
)

# Save performance metrics
metrics = compute_performance_metrics(y_true=y_test, y_pred=y_pred)
metrics_path = os.path.join(
    runs_config["dt"]["output_dir"],
    f"dt_{run_name}_performance_metrics_{timestamp}.txt",
)
save_performance_metrics(metrics=metrics, output_path=metrics_path)

# Print classification report
trainer.evaluate(y_pred=y_pred, y_test=y_test)
