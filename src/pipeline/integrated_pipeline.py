import os
import yaml
from PIL import Image
from src.models.ml_failure_predictor import MLFailurePredictor
from src.vision.cnn_classifier import CNNClassifierEngine
from src.vision.yolo_detector import YOLODetectorEngine
from src.utils.logger import get_logger

logger = get_logger("IntegratedPipeline")

class IntegratedRailwayInspectionPipeline:
    """
    Phase 9 Unified Multi-Modal Pipeline:
    Integrates Telemetry Sensor ML Failure Prediction with Camera Vision CNN & YOLO Defect/Obstacle Detection.
    """
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.ml_predictor = MLFailurePredictor(config_path)
        self.cnn_classifier = CNNClassifierEngine(config_path)
        self.yolo_detector = YOLODetectorEngine(config_path)

    def inspect(self, sensor_features: dict, image: Image.Image) -> dict:
        """
        Executes unified multi-modal inspection:
        1. Predicts sensor failure risk % using ML model
        2. Classifies image defect type using CNN classifier
        3. Detects localized bounding box objects using YOLO detector
        4. Computes integrated Railway Track Health Index (0-100) & Combined Threat Level
        """
        logger.info("Running Phase 9 Integrated ML + DL Multi-Modal Inspection...")
        
        # 1. ML Sensor Telemetry Risk Prediction
        sensor_res = self.ml_predictor.predict_single(sensor_features)
        ml_prob = sensor_res["failure_probability"]
        ml_risk_level = sensor_res["risk_level"]
        
        # 2. CNN Image Defect Classification
        cnn_res = self.cnn_classifier.predict(image)
        defect_class = cnn_res["class_name"]
        defect_conf = cnn_res["confidence"]
        
        # 3. YOLO Object & Obstacle Bounding Box Detection
        detections, annotated_image = self.yolo_detector.detect(image)
        
        # 4. Integrated Health Index Calculation (100 = Perfect Health, 0 = Critical Hazard)
        # Base penalty from sensor risk
        sensor_penalty = ml_prob * 50.0
        
        # Penalty from visual defects & obstacles
        visual_penalty = 0.0
        if defect_class != "normal":
            visual_penalty += 30.0 * defect_conf
            
        num_obstacles = sum(1 for d in detections if d["class_name"] == "obstacle")
        num_cracks_breaks = sum(1 for d in detections if d["class_name"] in ["broken_rail", "crack", "missing_fishplate"])
        
        visual_penalty += (num_obstacles * 25.0) + (num_cracks_breaks * 20.0)
        
        track_health_index = float(max(0.0, min(100.0, 100.0 - (sensor_penalty + visual_penalty))))
        
        if track_health_index < 40.0:
            overall_status = "CRITICAL_HAZARD"
            status_color = "red"
        elif track_health_index < 70.0:
            overall_status = "WARNING_MAINTENANCE_REQUIRED"
            status_color = "orange"
        else:
            overall_status = "SAFE_NORMAL_OPERATION"
            status_color = "green"
            
        result = {
            "track_health_index": round(track_health_index, 1),
            "overall_status": overall_status,
            "status_color": status_color,
            "sensor_telemetry_analysis": {
                "failure_probability_pct": round(ml_prob * 100, 1),
                "sensor_risk_level": ml_risk_level
            },
            "visual_inspection_analysis": {
                "defect_type": defect_class,
                "defect_confidence": round(defect_conf, 3),
                "detected_objects": detections,
                "total_objects_found": len(detections)
            },
            "annotated_image": annotated_image
        }
        
        logger.info(f"Integrated Inspection Result: Health Index {track_health_index:.1f}/100 -> Status: {overall_status}")
        return result

if __name__ == "__main__":
    pipeline = IntegratedRailwayInspectionPipeline()
    sample_sensor = {
        "vibration_g": 3.2,
        "temperature_c": 45.0,
        "pressure_psi": 3800,
        "maintenance_days": 120,
        "track_stress_index": 3.8
    }
    sample_img = Image.new("RGB", (640, 640), color=(100, 100, 100))
    res = pipeline.inspect(sample_sensor, sample_img)
