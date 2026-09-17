import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from src.utils.logger import get_logger

logger = get_logger("ImageAugmentation")

class TrackAugmenter:
    def __init__(self):
        pass

    def augment(self, pil_img: Image.Image) -> Image.Image:
        """
        Applies computer vision data augmentation for railway inspection:
        - Random Horizontal Flip
        - Random Rotation (-15 to +15 deg)
        - Brightness / Contrast jitter
        - Weather simulation (soft blur for fog/rain)
        """
        augmented = pil_img.copy()
        
        # 1. Random Horizontal Flip
        if random.random() > 0.5:
            augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)
            
        # 2. Random Rotation (-15 to 15 deg)
        angle = random.uniform(-15, 15)
        augmented = augmented.rotate(angle, resample=Image.BILINEAR)
        
        # 3. Brightness / Contrast Jitter
        brightness_factor = random.uniform(0.7, 1.3)
        contrast_factor = random.uniform(0.8, 1.2)
        
        enhancer = ImageEnhance.Brightness(augmented)
        augmented = enhancer.enhance(brightness_factor)
        
        enhancer = ImageEnhance.Contrast(augmented)
        augmented = enhancer.enhance(contrast_factor)
        
        # 4. Soft blur weather simulation (20% chance)
        if random.random() < 0.2:
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            
        return augmented

if __name__ == "__main__":
    augmenter = TrackAugmenter()
    test_img = Image.new("RGB", (640, 640), color=(100, 100, 100))
    aug_img = augmenter.augment(test_img)
    logger.info(f"Augmentation test passed: output image size {aug_img.size}")
