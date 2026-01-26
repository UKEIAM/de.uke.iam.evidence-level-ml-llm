import pandas as pd


def save_test_predictions(y_pred, y_test, output_path: str):
    """Saves the test predictions to a CSV file.

    Args:
        y_pred (array-like): Predicted labels.
        y_test (array-like): True labels.
        output_path (str): Path to save the CSV file.
    """
    test_preds = pd.DataFrame({"true_label": y_test, "predicted_label": y_pred})

    test_preds.to_csv(output_path, index=False)
    print(f"Test predictions saved to {output_path}")
