import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from src.vision.image_preprocessing import ImagePreprocessor
from src.utils.logger import get_logger

logger = get_logger("YOLODetector")

class YOLOTrackDetector(nn.Module):
    """
    Lightweight Object Detector predicting Bounding Box coordinates [x_center, y_center, width, height, confidence, class_logits]
    """
    def __init__(self, num_classes: int = 5):
        super(YOLOTrackDetector, self).__init__()
        self.num_classes = num_classes
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 320x320
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 160x160
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)) # 8x8 grid
        )
        
        # Bounding Box Head: Output 8x8 grid x (5 anchor attributes + num_classes)
        self.detector_head = nn.Conv2d(128, (5 + num_classes), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        out = self.detector_head(feat) # [B, 5 + num_classes, 8, 8]
        return out

class YOLODetectorEngine:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.classes = self.config["image_data"]["classes"]
        self.num_classes = len(self.classes)
        self.models_dir = self.config["paths"]["models_dir"]
        self.model_save_path = os.path.join(self.models_dir, "yolo_detector.pt")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLOTrackDetector(num_classes=self.num_classes).to(self.device)
        self.preprocessor = ImagePreprocessor(config_path)
        
        # Color palette for defect bounding boxes
        self.colors = {
            1: (231, 76, 60),   # broken_rail: Red
            2: (230, 126, 34),  # crack: Orange
            3: (155, 89, 182),  # missing_fishplate: Purple
            4: (241, 196, 15)   # obstacle: Yellow
        }

    def train_synthetic(self, epochs: int = 5):
        """
        Trains YOLO detector weights on synthetic image annotations.
        """
        logger.info(f"Training PyTorch YOLO Object Detector for {epochs} epochs on {self.device}...")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        raw_images_dir = self.config["paths"]["raw_images_dir"]
        image_files = [f for f in os.listdir(raw_images_dir) if f.endswith(".jpg")]
        
        for epoch in range(epochs):
            total_loss = 0.0
            for img_file in image_files[:40]:
                img_path = os.path.join(raw_images_dir, img_file)
                lbl_path = os.path.join(raw_images_dir, img_file.replace(".jpg", ".txt"))
                
                tensor_img, _ = self.preprocessor.preprocess_image(img_path)
                tensor_img = tensor_img.unsqueeze(0).to(self.device)
                
                # Target grid map [1, 10, 8, 8]
                target_map = torch.zeros((1, 5 + self.num_classes, 8, 8), device=self.device)
                
                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as lf:
                        lines = lf.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls_id = int(parts[0])
                                xc, yc, w, h = map(float, parts[1:])
                                grid_x = int(xc * 8)
                                grid_y = int(yc * 8)
                                grid_x = min(7, max(0, grid_x))
                                grid_y = min(7, max(0, grid_y))
                                
                                target_map[0, 0, grid_y, grid_x] = 1.0 # objectness
                                target_map[0, 1, grid_y, grid_x] = xc
                                target_map[0, 2, grid_y, grid_x] = yc
                                target_map[0, 3, grid_y, grid_x] = w
                                target_map[0, 4, grid_y, grid_x] = h
                                target_map[0, 5 + cls_id, grid_y, grid_x] = 1.0
                                
                optimizer.zero_grad()
                pred_map = self.model(tensor_img)
                loss = criterion(pred_map, target_map)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            logger.info(f"  Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f}")
            
        torch.save(self.model.state_dict(), self.model_save_path)
        logger.info(f"Saved trained PyTorch YOLO Detector to {self.model_save_path}")

    def detect(self, pil_img: Image.Image, conf_threshold: float = 0.3) -> tuple[list[dict], Image.Image]:
        """
        Runs YOLO object detection on PIL image.
        Returns list of detected objects: [{"class_id": id, "class_name": name, "confidence": conf, "bbox": [x1, y1, x2, y2]}]
        and annotated PIL Image with drawn bounding boxes.
        """
        self.model.eval()
        if not os.path.exists(self.model_save_path):
            self.train_synthetic(epochs=3)
        else:
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
            
        img_w, img_h = pil_img.size
        tensor_img = self.preprocessor.preprocess_pil(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_map = self.model(tensor_img)[0] # [10, 8, 8]
            
        detections = []
        annotated_img = pil_img.copy()
        draw = ImageDraw.Draw(annotated_img)
        
        grid_h, grid_w = pred_map.shape[1], pred_map.shape[2]
        
        for gy in range(grid_h):
            for gx in range(grid_w):
                obj_conf = torch.sigmoid(pred_map[0, gy, gx]).item()
                if obj_conf >= conf_threshold:
                    xc = pred_map[1, gy, gx].item()
                    yc = pred_map[2, gy, gx].item()
                    w = pred_map[3, gy, gx].item()
                    h = pred_map[4, gy, gx].item()
                    
                    class_logits = pred_map[5:, gy, gx]
                    class_probs = torch.softmax(class_logits, dim=0)
                    cls_id = class_probs.argmax().item()
                    cls_conf = class_probs[cls_id].item() * obj_conf
                    
                    if cls_id != 0 and cls_conf >= conf_threshold:
                        # Convert bbox from normalized [xc, yc, w, h] to pixel [x1, y1, x2, y2]
                        x1 = int(max(0, (xc - w / 2.0) * img_w))
                        y1 = int(max(0, (yc - h / 2.0) * img_h))
                        x2 = int(min(img_w, (xc + w / 2.0) * img_w))
                        y2 = int(min(img_h, (yc + h / 2.0) * img_h))
                        
                        det_obj = {
                            "class_id": cls_id,
                            "class_name": self.classes[cls_id],
                            "confidence": float(cls_conf),
                            "bbox": [x1, y1, x2, y2]
                        }
                        detections.append(det_obj)
                        
                        # Draw bounding box
                        color = self.colors.get(cls_id, (255, 0, 0))
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                        label_text = f"{self.classes[cls_id]}: {cls_conf*100:.1f}%"
                        draw.rectangle([x1, max(0, y1 - 20), x1 + len(label_text) * 9, y1], fill=color)
                        draw.text((x1 + 4, max(0, y1 - 18)), label_text, fill=(255, 255, 255))
                        
        return detections, annotated_img

if __name__ == "__main__":
    engine = YOLODetectorEngine()
    engine.train_synthetic(epochs=3)
