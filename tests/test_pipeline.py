import os
import pytest
import numpy as np
import pandas as pd
from PIL import Image
from fastapi.testclient import TestClient

from src.data.preprocessing import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.ml_failure_predictor import MLFailurePredictor
from src.vision.image_preprocessing import ImagePreprocessor
from src.vision.cnn_classifier import CNNClassifierEngine
from src.vision.yolo_detector import YOLODetectorEngine
from src.pipeline.integrated_pipeline import IntegratedRailwayInspectionPipeline
from app.api import app

client = TestClient(app)

def test_data_preprocessing():
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess_sensor_data()
    assert len(df) > 0
    assert df["vibration_g"].isnull().sum() == 0
    assert df["temperature_c"].isnull().sum() == 0

def test_feature_engineering():
    engineer = FeatureEngineer()
    df = engineer.extract_features()
    assert "vib_rolling_mean" in df.columns
    assert "temp_anomaly" in df.columns
    assert "maint_freq_ratio" in df.columns

def test_ml_failure_predictor():
    predictor = MLFailurePredictor()
    metrics = predictor.train_and_evaluate()
    assert metrics["accuracy"] > 0.70
    assert metrics["f1"] > 0.70

def test_vision_pipeline():
    img_pre = ImagePreprocessor()
    test_img = Image.new("RGB", (640, 640), color=(100, 100, 100))
    tensor_img = img_pre.preprocess_pil(test_img)
    assert tensor_img.shape == (3, 640, 640)

def test_integrated_pipeline():
    pipeline = IntegratedRailwayInspectionPipeline()
    sample_sensor = {
        "vibration_g": 2.1,
        "temperature_c": 35.0,
        "pressure_psi": 4200,
        "maintenance_days": 90,
        "track_stress_index": 2.8
    }
    sample_img = Image.new("RGB", (640, 640), color=(100, 100, 100))
    res = pipeline.inspect(sample_sensor, sample_img)
    assert "track_health_index" in res
    assert 0.0 <= res["track_health_index"] <= 100.0

def test_api_endpoints():
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"
    
    res_health = client.get("/health")
    assert res_health.status_code == 200
    
    sensor_payload = {
        "vibration_g": 2.1,
        "temperature_c": 35.0,
        "pressure_psi": 4200.0,
        "maintenance_days": 90,
        "track_stress_index": 2.8
    }
    res_sensor = client.post("/predict_sensor", json=sensor_payload)
    assert res_sensor.status_code == 200
    assert "failure_probability" in res_sensor.json()["result"]
