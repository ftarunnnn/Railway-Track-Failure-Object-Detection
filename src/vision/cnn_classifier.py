import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from src.vision.image_preprocessing import ImagePreprocessor
from src.utils.logger import get_logger

logger = get_logger("CNNClassifier")

class TrackDefectCNN(nn.Module):
    """
    Lightweight Deep CNN Classifier for Railway Defect Recognition.
    """
    def __init__(self, num_classes: int = 5):
        super(TrackDefectCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 320x320
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 160x160
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)) # 4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits

class CNNClassifierEngine:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.classes = self.config["image_data"]["classes"]
        self.num_classes = len(self.classes)
        self.models_dir = self.config["paths"]["models_dir"]
        self.model_save_path = os.path.join(self.models_dir, "cnn_classifier.pt")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrackDefectCNN(num_classes=self.num_classes).to(self.device)
        self.preprocessor = ImagePreprocessor(config_path)

    def train_synthetic(self, epochs: int = 5):
        """
        Trains CNN model on synthetic images dataset.
        """
        logger.info(f"Training PyTorch CNN Track Classifier for {epochs} epochs on {self.device}...")
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Load sample raw images
        raw_images_dir = self.config["paths"]["raw_images_dir"]
        image_files = [f for f in os.listdir(raw_images_dir) if f.endswith(".jpg")]
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for img_file in image_files[:40]:
                img_path = os.path.join(raw_images_dir, img_file)
                lbl_path = os.path.join(raw_images_dir, img_file.replace(".jpg", ".txt"))
                
                # Determine label from label file or default to 0
                label = 0
                if os.path.exists(lbl_path):
                    with open(lbl_path, "r") as lf:
                        lines = lf.readlines()
                        if lines:
                            label = int(lines[0].split()[0])
                            
                tensor_img, _ = self.preprocessor.preprocess_image(img_path)
                tensor_img = tensor_img.unsqueeze(0).to(self.device)
                target = torch.tensor([label], device=self.device)
                
                optimizer.zero_grad()
                output = self.model(tensor_img)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1).item()
                if pred == label:
                    correct += 1
                total += 1
                
            acc = correct / max(1, total)
            logger.info(f"  Epoch [{epoch+1}/{epochs}] Loss: {total_loss/max(1, total):.4f}, Accuracy: {acc*100:.2f}%")
            
        torch.save(self.model.state_dict(), self.model_save_path)
        logger.info(f"Saved trained PyTorch CNN Classifier to {self.model_save_path}")

    def predict(self, pil_img: Image.Image) -> dict:
        self.model.eval()
        if not os.path.exists(self.model_save_path):
            self.train_synthetic(epochs=3)
        else:
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
            
        tensor_img = self.preprocessor.preprocess_pil(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            conf = probs[pred_idx].item()
            
        return {
            "class_id": pred_idx,
            "class_name": self.classes[pred_idx],
            "confidence": float(conf)
        }

if __name__ == "__main__":
    engine = CNNClassifierEngine()
    engine.train_synthetic(epochs=3)
