import cv2
import numpy as np
from typing import Dict, List, Optional
from .corrections import ImageCorrector

class PreprocessingPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.corrector = ImageCorrector(config)
        self.steps_applied = []
    
    def process(self, image: np.ndarray, save_intermediates: bool = False) -> Dict:
        self.steps_applied = []
        intermediates = {}
        
        current_image = image.copy()
        
        if save_intermediates:
            intermediates['original'] = image.copy()
        
        if self.config.get('enable_noise_removal', True):
            current_image = self.corrector.remove_noise(current_image)
            self.steps_applied.append('noise_removal')
            if save_intermediates:
                intermediates['denoised'] = current_image.copy()
        
        if self.config.get('enable_skew_correction', True):
            current_image = self.corrector.correct_skew(current_image)
            self.steps_applied.append('skew_correction')
            if save_intermediates:
                intermediates['deskewed'] = current_image.copy()
        
        if self.config.get('enable_perspective_correction', True):
            current_image = self.corrector.correct_perspective(current_image)
            self.steps_applied.append('perspective_correction')
            if save_intermediates:
                intermediates['perspective_corrected'] = current_image.copy()
        
        if self.config.get('enable_illumination_normalization', True):
            current_image = self.corrector.normalize_illumination(current_image)
            self.steps_applied.append('illumination_normalization')
            if save_intermediates:
                intermediates['illumination_normalized'] = current_image.copy()
        
        return {
            'processed_image': current_image,
            'steps_applied': self.steps_applied,
            'intermediates': intermediates
        }
    
    def process_for_ocr(self, image: np.ndarray) -> np.ndarray:
        result = self.process(image)
        processed = result['processed_image']
        
        binary = self.corrector.apply_adaptive_threshold(processed)
        
        return binary
    
    def get_steps_applied(self) -> List[str]:
        return self.steps_applied.copy()
