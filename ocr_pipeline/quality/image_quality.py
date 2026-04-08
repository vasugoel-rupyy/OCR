"""Image quality assessment module (Approach 1).

Implements blur detection, brightness scoring, resolution checking,
contrast ratio calculation, and edge density metrics.
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Container for image quality metrics."""
    blur_score: float
    brightness_score: float
    resolution_score: float
    contrast_score: float
    edge_density: float
    glare_ratio: float
    composite_score: float
    passed: bool
    failure_reasons: list
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        from dataclasses import asdict
        return asdict(self)


class ImageQualityAssessor:
    """Assesses image quality using multiple metrics."""
    
    def __init__(self, config: Dict):
        """Initialize quality assessor with configuration.
        
        Args:
            config: Quality configuration dictionary
        """
        self.config = config
        self.min_blur_score = config.get('min_blur_score', 100.0)
        self.min_brightness = config.get('min_brightness', 30)
        self.max_brightness = config.get('max_brightness', 225)
        self.min_contrast_ratio = config.get('min_contrast_ratio', 1.5)
        self.min_edge_density = config.get('min_edge_density', 0.01)
        self.min_resolution = config.get('min_resolution', 300)
        self.max_glare_ratio = config.get('max_glare_ratio', 0.05)
        
        weights = config.get('weights', {})
        self.weight_blur = weights.get('blur', 0.30)  
        self.weight_brightness = weights.get('brightness', 0.20)
        self.weight_resolution = weights.get('resolution', 0.20)  
        self.weight_contrast = weights.get('contrast', 0.20)
        self.weight_glare = weights.get('glare', 0.10)  
    
    def assess(self, image: np.ndarray) -> QualityMetrics:
        """Assess overall image quality.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            QualityMetrics object with all scores
        """
        blur_score = self.calculate_blur_score(image)
        brightness_score = self.calculate_brightness_score(image)
        resolution_score = self.calculate_resolution_score(image)
        contrast_score = self.calculate_contrast_score(image)
        edge_density = self.calculate_edge_density(image)
        glare_ratio = self.calculate_glare_ratio(image)
        
        normalized_blur = self._normalize_blur(blur_score)
        normalized_brightness = self._normalize_brightness(brightness_score)
        normalized_resolution = self._normalize_resolution(resolution_score)
        normalized_contrast = self._normalize_contrast(contrast_score)
        normalized_glare = self._normalize_glare(glare_ratio)
        
        composite_score = (
            self.weight_blur * normalized_blur +
            self.weight_brightness * normalized_brightness +
            self.weight_resolution * normalized_resolution +
            self.weight_contrast * normalized_contrast +
            self.weight_glare * normalized_glare
        )
        
        failure_reasons = []
        
        if blur_score < self.min_blur_score:
            failure_reasons.append(f"Blur score {blur_score:.2f} below threshold {self.min_blur_score}")
        
        if brightness_score < self.min_brightness or brightness_score > self.max_brightness:
            failure_reasons.append(f"Brightness {brightness_score:.2f} outside range [{self.min_brightness}, {self.max_brightness}]")
        
        if contrast_score < self.min_contrast_ratio:
            failure_reasons.append(f"Contrast ratio {contrast_score:.2f} below threshold {self.min_contrast_ratio}")
        
        if edge_density < self.min_edge_density:
            failure_reasons.append(f"Edge density {edge_density:.4f} below threshold {self.min_edge_density}")
            
        if glare_ratio > self.max_glare_ratio:
            failure_reasons.append(f"Glare ratio {glare_ratio:.1%} exceeds threshold {self.max_glare_ratio:.1%}")
        
        passed = len(failure_reasons) == 0
        
        return QualityMetrics(
            blur_score=blur_score,
            brightness_score=brightness_score,
            resolution_score=resolution_score,
            contrast_score=contrast_score,
            edge_density=edge_density,
            glare_ratio=glare_ratio,
            composite_score=composite_score,
            passed=passed,
            failure_reasons=failure_reasons
        )
    
    def calculate_blur_score(self, image: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance.
        
        Higher values indicate sharper images.
        
        Args:
            image: Input image
            
        Returns:
            Blur score (Laplacian variance)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return float(variance)
    
    def calculate_brightness_score(self, image: np.ndarray) -> float:
        """Calculate mean brightness of image.
        
        Args:
            image: Input image
            
        Returns:
            Mean pixel intensity [0, 255]
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(np.mean(gray))
    
    def calculate_resolution_score(self, image: np.ndarray) -> float:
        """Calculate effective resolution score.
        
        Args:
            image: Input image
            
        Returns:
            Resolution score (total pixels)
        """
        height, width = image.shape[:2]
        return float(height * width)
    
    def calculate_contrast_score(self, image: np.ndarray) -> float:
        """Calculate contrast ratio.
        
        Args:
            image: Input image
            
        Returns:
            Contrast ratio (std / mean)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        mean = np.mean(gray)
        std = np.std(gray)
        
        if mean == 0:
            return 0.0
        
        return float(std / mean)
    
    def calculate_edge_density(self, image: np.ndarray) -> float:
        """Calculate edge density using Canny edge detection.
        
        Args:
            image: Input image
            
        Returns:
            Edge density (ratio of edge pixels to total pixels)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150)
        
        total_pixels = edges.size
        edge_pixels = np.count_nonzero(edges)
        
        return float(edge_pixels / total_pixels)

    def calculate_glare_ratio(self, image: np.ndarray) -> float:
        """Calculate ratio of overexposed pixels indicating glare.
        
        Args:
            image: Input image
            
        Returns:
            Ratio of pixels exceeding glare threshold
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        threshold = self.config.get('glare_threshold', 253)
        overexposed = np.count_nonzero(gray >= threshold)
        return float(overexposed / gray.size)

    def _normalize_blur(self, blur_score: float) -> float:
        """Normalize blur score to [0, 1]."""
        return min(1.0, blur_score / 1000.0)
    
    def _normalize_brightness(self, brightness: float) -> float:
        """Normalize brightness to [0, 1]."""
        if brightness < self.min_brightness or brightness > self.max_brightness:
            return 0.0
        
        distance_from_optimal = abs(brightness - 127.5)
        return 1.0 - (distance_from_optimal / 127.5) * 0.5
        


    def _normalize_resolution(self, resolution: float) -> float:
        """Normalize resolution to [0, 1]."""
        min_pixels = 307200
        good_pixels = 2073600
        
        if resolution < min_pixels:
            return resolution / min_pixels
        elif resolution < good_pixels:
            return 0.5 + 0.5 * (resolution - min_pixels) / (good_pixels - min_pixels)
        else:
            return 1.0
        
    def _normalize_glare(self, glare_ratio: float) -> float:
        """Normalize glare ratio to [0, 1]. High glare -> low score."""
        max_acceptable = self.config.get('max_glare_ratio', 0.05)
        if glare_ratio > max_acceptable:
            return max(0.0, 1.0 - (glare_ratio / max_acceptable))
        return 1.0 - (glare_ratio / max_acceptable * 0.5)

    def _normalize_contrast(self, contrast: float) -> float:
        """Normalize contrast to [0, 1]."""
        if contrast < 0.1:
            return 0.0
        elif contrast < 0.5:
            return contrast / 0.5
        else:
            return 1.0


def assess_image_quality(image: np.ndarray, config: Dict) -> QualityMetrics:
    """Convenience function to assess image quality.
    
    Args:
        image: Input image
        config: Quality configuration
        
    Returns:
        QualityMetrics object
    """
    assessor = ImageQualityAssessor(config)
    return assessor.assess(image)
