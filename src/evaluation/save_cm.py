import seaborn as sns
import matplotlib.pyplot as plt


def save_confusion_matrix(y_pred, y_test, output_path: str):
    """Saves the confusion matrix as a heatmap image.

    Args:
        y_pred (array-like): Predicted labels.
        y_test (array-like): True labels.
        output_path (str): Path to save the heatmap image.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")
