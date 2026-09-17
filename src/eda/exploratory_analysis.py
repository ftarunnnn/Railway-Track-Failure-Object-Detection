import os
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import get_logger

logger = get_logger("EDAAnalysis")

class EDAAnalyzer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.data_path = self.config["paths"]["processed_sensor_data"]
        self.figures_dir = self.config["paths"]["figures_dir"]
        self.reports_dir = self.config["paths"]["reports_dir"]
        
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def run_eda(self):
        """
        Executes Phase 3 EDA:
        1. Loads processed sensor dataset
        2. Computes summary statistics
        3. Analyzes class balance (Normal vs Failure)
        4. Generates visual distribution & correlation heatmaps
        5. Saves analysis metrics JSON & figure plots
        """
        logger.info(f"Performing Exploratory Data Analysis on {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # 1. Target Class Imbalance Analysis
        class_counts = df["failure_status"].value_counts().to_dict()
        class_percentages = df["failure_status"].value_counts(normalize=True).to_dict()
        
        logger.info(f"Target Distribution: {class_counts} ({class_percentages})")
        
        # Plot 1: Target Class Imbalance
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x="failure_status", palette=["#2ecc71", "#e74c3c"])
        plt.title("Track Failure Status Distribution (0: Normal, 1: Failure Risk)")
        plt.xlabel("Failure Status")
        plt.ylabel("Sample Count")
        plt.tight_layout()
        imbalance_plot_path = os.path.join(self.figures_dir, "failure_status_imbalance.png")
        plt.savefig(imbalance_plot_path, dpi=200)
        plt.close()
        
        # Plot 2: Vibration & Temperature Distributions by Failure Status
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(data=df, x="failure_status", y="vibration_g", ax=axes[0], palette=["#2ecc71", "#e74c3c"])
        axes[0].set_title("Vibration (g) vs Failure Status")
        
        sns.boxplot(data=df, x="failure_status", y="temperature_c", ax=axes[1], palette=["#2ecc71", "#e74c3c"])
        axes[1].set_title("Temperature (°C) vs Failure Status")
        plt.tight_layout()
        dist_plot_path = os.path.join(self.figures_dir, "vibration_temperature_distribution.png")
        plt.savefig(dist_plot_path, dpi=200)
        plt.close()
        
        # Plot 3: Feature Correlation Heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(8, 6))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Telemetry Feature Correlation Heatmap")
        plt.tight_layout()
        corr_plot_path = os.path.join(self.figures_dir, "feature_correlation_heatmap.png")
        plt.savefig(corr_plot_path, dpi=200)
        plt.close()
        
        # Save EDA Summary JSON
        summary = {
            "total_records": len(df),
            "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
            "class_percentages": {str(k): float(v) for k, v in class_percentages.items()},
            "feature_means": numeric_df.mean().to_dict(),
            "feature_stds": numeric_df.std().to_dict(),
            "figures_generated": [
                "failure_status_imbalance.png",
                "vibration_temperature_distribution.png",
                "feature_correlation_heatmap.png"
            ]
        }
        
        summary_json_path = os.path.join(self.reports_dir, "eda_summary.json")
        with open(summary_json_path, "w") as f:
            json.dump(summary, f, indent=4)
            
        logger.info(f"Saved EDA figures to {self.figures_dir} and summary to {summary_json_path}")

if __name__ == "__main__":
    analyzer = EDAAnalyzer()
    analyzer.run_eda()
