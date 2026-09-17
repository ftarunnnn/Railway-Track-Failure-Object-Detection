import os
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils.logger import get_logger

logger = get_logger("MLFailurePredictor")

class MLFailurePredictor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.features_path = self.config["paths"]["features_data"]
        self.models_dir = self.config["paths"]["models_dir"]
        self.model_save_path = os.path.join(self.models_dir, "ml_failure_model.pkl")
        
        os.makedirs(self.models_dir, exist_ok=True)
        self.model = None

    def train_and_evaluate(self):
        """
        Executes Phase 5 ML Training:
        1. Load engineered features
        2. Split Train / Test
        3. Train Random Forest / HistGradientBoosting Classifier
        4. Predict binary failure risk & output probability %
        5. Evaluate accuracy, precision, recall, f1, roc_auc
        6. Save trained model
        """
        logger.info(f"Loading feature dataset from {self.features_path}...")
        df = pd.read_csv(self.features_path)
        
        X = df.drop(columns=["failure_status"])
        y = df["failure_status"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config["ml_model"]["test_size"],
            random_state=self.config["ml_model"]["random_state"],
            stratify=y
        )
        
        logger.info(f"Training ML Failure Classifier on {len(X_train)} samples...")
        self.model = HistGradientBoostingClassifier(
            max_iter=self.config["ml_model"]["n_estimators"],
            max_depth=self.config["ml_model"]["max_depth"],
            random_state=self.config["ml_model"]["random_state"]
        )
        self.model.fit(X_train, y_train)
        
        # Predictions & Probabilities
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        logger.info("=== Phase 5 ML Model Performance ===")
        for k, v in metrics.items():
            logger.info(f"  {k.upper()}: {v:.4f}")
            
        # Save trained artifact
        with open(self.model_save_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": list(X.columns),
                "metrics": metrics
            }, f)
            
        logger.info(f"Successfully saved trained ML model to {self.model_save_path}")
        return metrics

    def predict_single(self, sensor_features: dict) -> dict:
        """
        Inference helper predicting failure risk & probability for a single telemetry record.
        """
        if self.model is None:
            if os.path.exists(self.model_save_path):
                with open(self.model_save_path, "rb") as f:
                    saved = pickle.load(f)
                    self.model = saved["model"]
                    self.feature_names = saved["feature_names"]
            else:
                raise FileNotFoundError("Model file not found. Please train model first.")
                
        df_single = pd.DataFrame([sensor_features])
        # Reorder features to match training
        for col in self.feature_names:
            if col not in df_single.columns:
                df_single[col] = 0.0
        df_single = df_single[self.feature_names]
        
        prob = self.model.predict_proba(df_single)[0, 1]
        pred_class = int(prob >= 0.5)
        
        risk_level = "CRITICAL" if prob > 0.75 else ("WARNING" if prob > 0.40 else "NORMAL")
        
        return {
            "failure_status": pred_class,
            "failure_probability": float(prob),
            "risk_level": risk_level
        }

if __name__ == "__main__":
    predictor = MLFailurePredictor()
    predictor.train_and_evaluate()
