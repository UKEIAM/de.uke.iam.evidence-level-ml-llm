import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evaluation.save_cm import save_confusion_matrix
from evaluation.metrics import compute_performance_metrics, save_performance_metrics
from evaluation.save_test_preds import save_test_predictions
import yaml
from utils.load_datasets import load_tfidf_features, load_splits
from sklearn.utils.class_weight import compute_sample_weight
from ml.base_models import XGBTrainer
import os
from datetime import datetime

print("📚 Starting XGBoost with TF-IDF features training and evaluation...")
with open("config/runs_config.yaml", "r") as f:
    runs_config = yaml.safe_load(f)

# Load TF-IDF datasets and labels
train_tfidf_df, test_tfidf_df = load_tfidf_features()
train_labels_df, test_labels_df = load_splits()

# Prepare data for XGBoost
X_train = train_tfidf_df.values
y_train = train_labels_df["label"].values
X_test = test_tfidf_df.values
y_test = test_labels_df["label"].values

# Compute number of classes and sample weights
num_classes = len(train_labels_df["label"].unique())
weights = compute_sample_weight(class_weight="balanced", y=train_labels_df["label"])

# Load trainer
trainer = XGBTrainer()

# Train the model
model = trainer.train(X_train, y_train, num_classes, weights)

# --- Evaluate model on test data ---
y_pred = model.predict(X_test)

# Directories for saving outputs
output_dir = runs_config["xgb"]["output_dir"]
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = runs_config.get("tfidf", {}).get("run_name", "")

# Save model
model_filename = f"xgb_{run_name}_model_{timestamp}.json"
model_path = os.path.join(runs_config["xgb"]["output_dir"], model_filename)
trainer.save_model(model, model_path)

# Save predictions
output_preds_path = os.path.join(
    runs_config["xgb"]["output_dir"], f"xgb_{run_name}_predictions_{timestamp}.csv"
)
save_test_predictions(y_pred=y_pred, y_test=y_test, output_path=output_preds_path)

# Save confusion matrix
cm_path = os.path.join(
    runs_config["xgb"]["output_dir"],
    f"xgb_{run_name}_confusion_matrix_{timestamp}.png",
)
save_confusion_matrix(
    y_test=y_test,
    y_pred=y_pred,
    output_path=cm_path,
)

# Save performance metrics
metrics = compute_performance_metrics(y_true=y_test, y_pred=y_pred)
metrics_path = os.path.join(
    runs_config["xgb"]["output_dir"],
    f"xgb_{run_name}_performance_metrics_{timestamp}.txt",
)
save_performance_metrics(metrics=metrics, output_path=metrics_path)

# Print classification report
trainer.evaluate(y_pred=y_pred, y_test=y_test)
print("===== 📚 XGBoost with TF-IDF features training and evaluation completed.=====")
