import json
from openai import max_retries
import pandas as pd
from datetime import datetime, time
import os
from google.api_core.exceptions import ServiceUnavailable


class LLMClassifier:
    """Generic LLM classifier for evidence level prediction."""

    def __init__(
        self, model_name, client, schema, prompt_fn, runs_config, max_retries=3
    ):
        self.model_name = model_name
        self.client = client
        self.schema = schema
        self.prompt_fn = prompt_fn
        self.runs_config = runs_config
        self.max_retries = max_retries

    def classify_dataset(self, df, prompt_type="zero_shot"):
        """
        Classify all rows in a DataFrame.

        Args:
            df: DataFrame with 'title_fetched' and 'abstract_fetched' columns
            prompt_type: 'zero_shot' or 'few_shot'

        Returns:
            DataFrame with added columns: llm_pred, llm_explanation, llm_confidence
        """
        # Initialize result columns
        results = []

        for index, row in df.iterrows():
            title = row["title_fetched"]
            abstract = row["abstract_fetched"]

            retries = 0
            while retries < self.max_retries:
                try:
                    print(f"Processing publication {index + 1}/{len(df)}")

                    # Call LLM
                    llm_output_json = self.prompt_fn(
                        title,
                        abstract,
                        self.client,
                        self.model_name,
                        prompt_type,
                        self.schema,
                        self.runs_config["llm"]["temperature"],
                    )
                    results.append(json.loads(llm_output_json))
                    break  # Exit retry loop on success
                except ServiceUnavailable:
                    print(
                        f"Service unavailable for index {index}. Retrying in 3 seconds (Attempt {retries + 1}/{max_retries})."
                    )
                    time.sleep(3)
                    retries += 1
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                    results.append(
                        {
                            "reference_id": row.get("reference_id", index),
                            "llm_pred": None,
                            "llm_explanation": f"Error: {str(e)}",
                            "llm_confidence": None,
                            "true_label": row.get("evidence_level", None),
                        }
                    )
                    break

        return pd.DataFrame(results)

    def save_results(self, results_df, output_dir, prompt_type):
        """Save results with timestamp."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name.replace('/', '_')}_{prompt_type}_predictions_{timestamp}.csv"
        output_path = os.path.join(output_dir, filename)
        results_df.to_csv(output_path, sep="\t", index=False)
        print(f"Results saved to {output_path}")
        return output_path
