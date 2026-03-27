import cv2
import numpy as np
from typing import Tuple, Optional
from .corrections import ImageCorrector

class IDDocumentEnhancer:
    
    def __init__(self):
        self.corrector = ImageCorrector({
            'enable_skew_correction': True,
            'max_skew_angle': 45
        })
    
    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        image = self._resize_if_needed(image, min_width=1600)
        
        image = self.deskew_document(image)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        sharpened = self._sharpen_image(enhanced)
        
        binary = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            10
        )
        
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def deskew_document(self, image: np.ndarray) -> np.ndarray:
        return self.corrector.correct_skew(image)
    
    def _resize_if_needed(self, image: np.ndarray, min_width: int = 1600) -> np.ndarray:
        height, width = image.shape[:2]
        
        if width < min_width:
            scale = min_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            return resized
        
        return image
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
