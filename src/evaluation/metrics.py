from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    cohen_kappa_score,
)


def compute_performance_metrics(y_true, y_pred):
    """Computes common classification metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: Dictionary containing accuracy, F1-score, and Cohen's kappa.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    kappa = cohen_kappa_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "cohen_kappa": kappa,
        "mean_absolute_error": mae,
    }


def save_performance_metrics(metrics: dict, output_path: str):
    """Saves performance metrics to a text file.

    Args:
        metrics (dict): Dictionary containing performance metrics.
        output_path (str): Path to save the metrics file.
    """
    with open(output_path, "w") as f:
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.2f}\n")
    print(f"Performance metrics saved to {output_path}")
