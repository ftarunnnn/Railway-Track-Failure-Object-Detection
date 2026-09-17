import os
import yaml
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("DataPreprocessing")

class DataPreprocessor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.raw_sensor_path = self.config["paths"]["raw_sensor_data"]
        self.processed_sensor_path = self.config["paths"]["processed_sensor_data"]
        os.makedirs(os.path.dirname(self.processed_sensor_path), exist_ok=True)

    def preprocess_sensor_data(self) -> pd.DataFrame:
        """
        Executes Phase 2 Data Preprocessing:
        1. Load raw dataset
        2. Remove duplicates
        3. Handle missing values (median imputation)
        4. Detect & handle extreme outliers via IQR filtering
        5. Encode derived categorical indicators (e.g., risk level category)
        6. Save cleaned dataset
        """
        logger.info(f"Loading raw sensor data from {self.raw_sensor_path}...")
        df = pd.read_csv(self.raw_sensor_path)
        initial_len = len(df)
        
        # 1. Deduplication
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        logger.info(f"Deduplicated dataset: {initial_len} -> {len(df)} rows")
        
        # 2. Handle missing values
        num_cols = ["vibration_g", "temperature_c", "pressure_psi", "maintenance_days"]
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputed {col} missing values with median ({median_val:.2f})")
                
        # 3. Outlier Clipping / Filtering (IQR method)
        for col in ["vibration_g", "temperature_c", "pressure_psi"]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 2.5 * iqr
            upper_bound = q3 + 2.5 * iqr
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        # 4. Categorical Encoding (Add Maintenance Category)
        # e.g., 'recent' (<= 60 days), 'moderate' (61 - 180 days), 'overdue' (> 180 days)
        df["maintenance_cat"] = pd.cut(
            df["maintenance_days"],
            bins=[-1, 60, 180, 1000],
            labels=["recent", "moderate", "overdue"]
        )
        
        # One-hot encode maintenance category
        df = pd.get_dummies(df, columns=["maintenance_cat"], prefix="maint", dtype=int)
        
        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Save processed dataset
        df.to_csv(self.processed_sensor_path, index=False)
        logger.info(f"Successfully saved cleaned sensor dataset ({len(df)} rows) to {self.processed_sensor_path}")
        return df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_sensor_data()
