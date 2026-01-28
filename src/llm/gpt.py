import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import yaml
from utils.gpt_schema import gpt_schema
from utils.prompts import gpt_prompt
from llm.inference import LLMClassifier
from evaluation.metrics import compute_performance_metrics, save_performance_metrics
from evaluation.save_cm import save_confusion_matrix
from sklearn.preprocessing import LabelEncoder
import datetime

# Load config
with open("config/runs_config.yaml", "r") as f:
    runs_config = yaml.safe_load(f)

# Setup
load_dotenv(dotenv_path=".env")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
model = runs_config["llm"]["gpt"]["model"]
schema = gpt_schema()
max_retries = runs_config["llm"].get("max_retries", 3)

# Load test data
test_data_path = runs_config["data"]["test_path"]
test_df = pd.read_csv(test_data_path, sep="\t")

# Initialize classifier
classifier = LLMClassifier(
    model_name=model,
    client=client,
    schema=schema,
    prompt_fn=gpt_prompt,
    runs_config=runs_config,
    max_retries=max_retries,
)

# Run zero-shot inference
print("Running zero-shot inference...")
zero_shot_results = classifier.classify_dataset(test_df, prompt_type="zero_shot")
classifier.save_results(
    zero_shot_results, runs_config["llm"]["gpt"]["output_dir"], "zero_shot"
)

# Run few-shot inference
print("\nRunning few-shot inference...")
few_shot_results = classifier.classify_dataset(test_df, prompt_type="few_shot")
classifier.save_results(
    few_shot_results, runs_config["llm"]["gpt"]["output_dir"], "few_shot"
)

print("GPT inference completed.")

# --- Evaluate performance on test dataset ---
# Directories for saving outputs
output_dir = runs_config["llm"]["gpt"]["output_dir"]
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Prepare labels
le = LabelEncoder()
y_pred_zero_shot = le.fit_transform(zero_shot_results["llm_pred"])
y_pred_few_shot = le.transform(few_shot_results["llm_pred"])
y_test = le.transform(test_df["evidence_level"])

# Save confusion matrices
run_names = ["zero_shot", "few_shot"]

for run_name, y_pred in zip(run_names, [y_pred_zero_shot, y_pred_few_shot]):
    # Save confusion matrix
    cm_path = os.path.join(
        runs_config["llm"]["gpt"]["output_dir"],
        f"gpt_{run_name}_confusion_matrix_{timestamp}.png",
    )

    save_confusion_matrix(
        y_test=y_test,
        y_pred=y_pred,
        output_path=cm_path,
    )

    # Save performance metrics
    metrics_path = os.path.join(
        runs_config["llm"]["gpt"]["output_dir"],
        f"gpt_{run_name}_performance_metrics_{timestamp}.txt",
    )
    metrics = compute_performance_metrics(y_true=y_test, y_pred=y_pred)
    save_performance_metrics(metrics=metrics, output_path=metrics_path)

print("Evaluation of GPT inference completed.")
