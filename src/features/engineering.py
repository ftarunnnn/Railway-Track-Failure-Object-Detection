import os
import yaml
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("FeatureEngineering")

class FeatureEngineer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.input_path = self.config["paths"]["processed_sensor_data"]
        self.output_path = self.config["paths"]["features_data"]
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def extract_features(self) -> pd.DataFrame:
        """
        Executes Phase 4 Feature Engineering:
        1. Loads cleaned sensor dataset
        2. Rolling window statistical features for vibration signal (mean, std, min, max, peak-to-peak)
        3. Temperature anomaly & rate of change features
        4. Maintenance frequency ratio
        5. Stress-to-vibration interaction index
        6. Saves engineered feature dataset
        """
        logger.info(f"Extracting features from {self.input_path}...")
        df = pd.read_csv(self.input_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        
        # 1. Rolling Window Vibration Statistics (Window = 5 samples ~ 1.25 hours)
        df["vib_rolling_mean"] = df["vibration_g"].rolling(window=5, min_periods=1).mean()
        df["vib_rolling_std"] = df["vibration_g"].rolling(window=5, min_periods=1).std().fillna(0)
        df["vib_peak_to_peak"] = df["vibration_g"].rolling(window=5, min_periods=1).max() - df["vibration_g"].rolling(window=5, min_periods=1).min()
        
        # 2. Temperature Anomaly & Stress Interactions
        temp_mean = df["temperature_c"].mean()
        df["temp_anomaly"] = np.abs(df["temperature_c"] - temp_mean)
        df["stress_vibration_ratio"] = df["track_stress_index"] / (df["vibration_g"] + 1e-5)
        
        # 3. Maintenance Frequency Ratio (Higher value = higher risk due to aging service)
        df["maint_freq_ratio"] = df["maintenance_days"] / 365.0
        
        # 4. Feature Selection / Dropping redundant non-numeric columns
        drop_cols = ["timestamp"]
        feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        feature_df.to_csv(self.output_path, index=False)
        logger.info(f"Saved engineered feature dataset ({feature_df.shape[1]} columns, {len(feature_df)} rows) to {self.output_path}")
        return feature_df

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.extract_features()
