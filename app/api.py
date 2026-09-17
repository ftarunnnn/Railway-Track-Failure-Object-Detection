import io
import base64
import yaml
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.pipeline.integrated_pipeline import IntegratedRailwayInspectionPipeline
from src.utils.logger import get_logger

logger = get_logger("FastAPIBackend")

app = FastAPI(
    title="Railway Track Failure & Object Detection API",
    description="Multi-modal API providing ML Sensor Failure Risk & DL YOLO/CNN Object Detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = IntegratedRailwayInspectionPipeline()

class SensorInput(BaseModel):
    vibration_g: float = 2.1
    temperature_c: float = 35.0
    pressure_psi: float = 4200.0
    maintenance_days: int = 90
    track_stress_index: float = 2.8

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Railway Track Failure & Object Detection Engine",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict_sensor", "/predict_image", "/predict_integrated"]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "pipeline": "ready"}

@app.post("/predict_sensor")
def predict_sensor_telemetry(data: SensorInput):
    try:
        features = data.model_dump()
        res = pipeline.ml_predictor.predict_single(features)
        return {"status": "success", "result": res}
    except Exception as e:
        logger.error(f"Sensor prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_image")
async def predict_track_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        cnn_res = pipeline.cnn_classifier.predict(image)
        detections, annotated_img = pipeline.yolo_detector.detect(image)
        
        # Convert annotated PIL image to base64 string
        buffered = io.BytesIO()
        annotated_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "status": "success",
            "cnn_defect_classification": cnn_res,
            "yolo_object_detections": detections,
            "annotated_image_base64": f"data:image/jpeg;base64,{img_str}"
        }
    except Exception as e:
        logger.error(f"Image prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_integrated")
async def predict_integrated(
    vibration_g: float = Form(2.1),
    temperature_c: float = Form(35.0),
    pressure_psi: float = Form(4200.0),
    maintenance_days: int = Form(90),
    track_stress_index: float = Form(2.8),
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        sensor_data = {
            "vibration_g": vibration_g,
            "temperature_c": temperature_c,
            "pressure_psi": pressure_psi,
            "maintenance_days": maintenance_days,
            "track_stress_index": track_stress_index
        }
        
        result = pipeline.inspect(sensor_data, image)
        
        # Base64 encode annotated image for JSON response
        buffered = io.BytesIO()
        result["annotated_image"].save(buffered, format="JPEG")
        result["annotated_image_base64"] = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
        del result["annotated_image"]
        
        return {"status": "success", "inspection": result}
    except Exception as e:
        logger.error(f"Integrated prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
