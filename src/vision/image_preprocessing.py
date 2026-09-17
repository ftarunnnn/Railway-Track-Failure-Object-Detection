import os
import yaml
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from src.utils.logger import get_logger

logger = get_logger("ImagePreprocessing")

class ImagePreprocessor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.target_size = tuple(self.config["image_data"]["image_size"]) # (640, 640)
        
        self.transform_tensor = T.Compose([
            T.Resize(self.target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path: str) -> tuple[torch.Tensor, Image.Image]:
        """
        Loads an image, resizes to target_size (640x640), normalizes RGB channels, returns PyTorch Tensor and original PIL Image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        pil_img = Image.open(image_path).convert("RGB")
        tensor_img = self.transform_tensor(pil_img)
        return tensor_img, pil_img

    def preprocess_pil(self, pil_img: Image.Image) -> torch.Tensor:
        return self.transform_tensor(pil_img.convert("RGB"))

if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    test_img_path = "data/raw/images/track_0000.jpg"
    if os.path.exists(test_img_path):
        tensor_img, orig = preprocessor.preprocess_image(test_img_path)
        logger.info(f"Successfully preprocessed image: shape {tensor_img.shape}, min {tensor_img.min():.2f}, max {tensor_img.max():.2f}")
