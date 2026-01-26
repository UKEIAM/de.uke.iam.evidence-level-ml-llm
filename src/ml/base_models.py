# src/ml/base_models.py
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import yaml
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import os
import pickle


with open("config/runs_config.yaml", "r") as f:
    runs_config = yaml.safe_load(f)


class XGBTrainer:
    """XGBoost Trainer with shared training, evaluation, and saving logic."""

    def __init__(self, params=None):
        self.params = params

    def train(self, X_train, y_train, num_classes, weights):
        """Shared xgb training logic"""
        print("Starting XGBoost model training...")
        model = xgb.XGBClassifier(
            objective=runs_config["xgb"]["objective"],
            num_class=num_classes,
            learning_rate=runs_config["xgb"]["learning_rate"],
            max_depth=runs_config["xgb"]["max_depth"],
            n_estimators=runs_config["xgb"]["n_estimators"],
            subsample=runs_config["xgb"]["subsample"],
            colsample_bytree=runs_config["xgb"]["colsample_bytree"],
            random_state=runs_config.get("xgb", {}).get("random_state", 42),
        )
        model.fit(X_train, y_train, sample_weight=weights)
        print("XGBoost model training completed.")
        return model

    def evaluate(self, y_test, y_pred):
        """Shared xgb evaluation logic"""
        le = LabelEncoder()
        le.fit(y_test)

        print(
            classification_report(
                le.inverse_transform(y_test), le.inverse_transform(y_pred)
            )
        )
        print("Evaluation completed.")

    def save_model(self, model, path):
        """Shared save logic"""
        model.save_model(path)
        print(f"Model saved to {path}")


class DecisionTreeTrainer:
    def __init__(self, params=None):
        self.params = params

    def train(self, X_train, y_train):
        """Shared training logic"""
        print("Starting Decision Tree model training...")
        model = DecisionTreeClassifier(
            criterion=runs_config["dt"]["criterion"],
            max_depth=runs_config["dt"]["max_depth"],
            min_samples_split=runs_config["dt"]["min_samples_split"],
            class_weight="balanced",  # account for class imbalance
            random_state=runs_config.get("dt", {}).get("random_state", 42),
        )
        model.fit(X_train, y_train)
        return model

    def evaluate(self, y_test, y_pred):
        """Shared evaluation logic"""
        print(classification_report(y_test, y_pred))
        print("Evaluation completed.")

    def save_model(self, model, path):
        """Shared save logic for sklearn models"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Change extension to .pkl for pickle files
        if path.endswith(".json"):
            path = path.replace(".json", ".pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {path}")
