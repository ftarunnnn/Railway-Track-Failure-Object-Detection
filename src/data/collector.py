import os
import yaml
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from src.utils.logger import get_logger

logger = get_logger("DataCollector")

class DataCollector:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.raw_sensor_path = self.config["paths"]["raw_sensor_data"]
        self.raw_images_dir = self.config["paths"]["raw_images_dir"]
        self.classes = self.config["image_data"]["classes"]
        
        os.makedirs(os.path.dirname(self.raw_sensor_path), exist_ok=True)
        os.makedirs(self.raw_images_dir, exist_ok=True)

    def generate_sensor_data(self, num_samples: int = 2000) -> pd.DataFrame:
        """
        Generates realistic track telemetry data:
        - vibration_g: normal (0.5 - 2.5 g), failure (> 3.5 g)
        - temperature_c: rail temp (-10 to 60°C)
        - pressure_psi: joint pressure (3000 - 5500 PSI)
        - maintenance_days: days since service (1 - 365)
        - track_stress_index: calculated load stress
        - failure_status: 0 (Normal), 1 (Failure Risk)
        """
        np.random.seed(self.config["sensor_data"]["random_seed"])
        logger.info(f"Generating {num_samples} telemetry sensor records...")
        
        timestamps = pd.date_range(start="2026-01-01", periods=num_samples, freq="15min")
        
        # Base distributions
        vibration = np.random.normal(loc=1.5, scale=0.5, size=num_samples)
        temperature = np.random.normal(loc=28.0, scale=8.0, size=num_samples)
        pressure = np.random.normal(loc=4500, scale=300, size=num_samples)
        maintenance = np.random.randint(1, 365, size=num_samples)
        
        # Compute track stress index
        stress_index = (vibration * 0.4) + (temperature * 0.05) + (maintenance * 0.02) + np.random.normal(0, 0.1, num_samples)
        
        # Determine failure risk based on physics rules
        failure_prob = 1 / (1 + np.exp(-(stress_index - 3.2)))
        failure_status = (np.random.rand(num_samples) < failure_prob).astype(int)
        
        # Inject realistic noise & missing values (5%)
        vib_with_nan = vibration.copy()
        temp_with_nan = temperature.copy()
        mask_nan_vib = np.random.rand(num_samples) < 0.03
        mask_nan_temp = np.random.rand(num_samples) < 0.02
        vib_with_nan[mask_nan_vib] = np.nan
        temp_with_nan[mask_nan_temp] = np.nan
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "vibration_g": vib_with_nan,
            "temperature_c": temp_with_nan,
            "pressure_psi": pressure,
            "maintenance_days": maintenance,
            "track_stress_index": stress_index,
            "failure_status": failure_status
        })
        
        df.to_csv(self.raw_sensor_path, index=False)
        logger.info(f"Saved raw sensor telemetry dataset to {self.raw_sensor_path}")
        return df

    def generate_synthetic_images(self, num_images: int = 50):
        """
        Generates synthetic railway track images with background textures and defect bounding boxes.
        Saves images (.jpg) and YOLO annotations (.txt).
        """
        logger.info(f"Generating {num_images} synthetic track images with annotations...")
        img_size = tuple(self.config["image_data"]["image_size"])
        
        for i in range(num_images):
            # Create track background (ballast gravel & parallel rails)
            bg_color = (np.random.randint(70, 110), np.random.randint(70, 100), np.random.randint(60, 90))
            img = Image.new("RGB", img_size, color=bg_color)
            draw = ImageDraw.Draw(img)
            
            # Draw railway steel tracks (two vertical metallic gray lines)
            rail_width = 35
            left_rail_x = 180
            right_rail_x = 425
            
            # Left rail
            draw.rectangle([left_rail_x, 0, left_rail_x + rail_width, img_size[1]], fill=(170, 180, 190))
            # Right rail
            draw.rectangle([right_rail_x, 0, right_rail_x + rail_width, img_size[1]], fill=(170, 180, 190))
            
            # Draw sleepers (horizontal wooden ties)
            for y in range(50, img_size[1], 100):
                draw.rectangle([80, y, 560, y + 25], fill=(90, 55, 35))
            
            # Determine defect class (0: Normal, 1: Broken Rail, 2: Crack, 3: Missing Fishplate, 4: Obstacle)
            class_id = np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.2, 0.2, 0.15, 0.15])
            
            annotations = []
            if class_id != 0:
                # Place defect on one of the rails or on track bed
                if class_id in [1, 2, 3]:
                    # Defect on rail
                    rail_x = left_rail_x if np.random.rand() > 0.5 else right_rail_x
                    box_w = np.random.randint(40, 70)
                    box_h = np.random.randint(40, 80)
                    box_x = rail_x - 10
                    box_y = np.random.randint(100, 500)
                    
                    # Draw defect visual feature
                    defect_color = (200, 40, 40) if class_id == 1 else (40, 40, 40)
                    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=defect_color)
                else:
                    # Obstacle between rails
                    box_w = np.random.randint(60, 120)
                    box_h = np.random.randint(60, 120)
                    box_x = np.random.randint(left_rail_x + 40, right_rail_x - 100)
                    box_y = np.random.randint(100, 500)
                    draw.ellipse([box_x, box_y, box_x + box_w, box_y + box_h], fill=(180, 140, 30))
                
                # Convert bbox to YOLO normalized format: [class_id, x_center, y_center, width, height]
                x_center = (box_x + box_w / 2.0) / img_size[0]
                y_center = (box_y + box_h / 2.0) / img_size[1]
                w_norm = box_w / img_size[0]
                h_norm = box_h / img_size[1]
                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            # Save image and label
            img_filename = f"track_{i:04d}.jpg"
            lbl_filename = f"track_{i:04d}.txt"
            
            img.save(os.path.join(self.raw_images_dir, img_filename))
            with open(os.path.join(self.raw_images_dir, lbl_filename), "w") as lf:
                lf.write("\n".join(annotations))
                
        logger.info(f"Successfully generated images and labels in {self.raw_images_dir}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.generate_sensor_data(2000)
    collector.generate_synthetic_images(60)
