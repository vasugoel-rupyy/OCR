import logging
from typing import List, Tuple
import cv2
import numpy as np

from .region import Region, BoundingBox

class DocumentDetector:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        contour_config = self.config.get('contour_detection', {})
        self.min_area_ratio = contour_config.get('min_area_ratio', 0.20)
        self.max_area_ratio = contour_config.get('max_area_ratio', 0.95)
        self.min_aspect_ratio = contour_config.get('min_aspect_ratio', 0.3)
        self.max_aspect_ratio = contour_config.get('max_aspect_ratio', 3.0)
        self.approx_epsilon = contour_config.get('approx_epsilon', 0.02)
    
    def detect_contours(self, image: np.ndarray) -> List[Region]:
        self.logger.debug("Starting contour-based document detection")
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.logger.debug(f"Found {len(contours)} contours")
        
        regions = []
        image_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            if self._is_valid_document_contour(contour, image.shape, image_area):
                region = self._extract_region_from_contour(image, contour, image_area)
                if region:
                    regions.append(region)
        
        self.logger.info(f"Detected {len(regions)} valid document regions using contours")
        return regions
    
    def _is_valid_document_contour(self, contour: np.ndarray, 
                                   image_shape: Tuple, 
                                   image_area: int) -> bool:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        
        if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
            return False
        
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, self.approx_epsilon * peri, True)
        
        if not (4 <= len(approx) <= 6):
            return False
        
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False
        
        margin = 5
        if x < margin or y < margin:
            return False
        if x + w > image_shape[1] - margin or y + h > image_shape[0] - margin:
            if area_ratio < 0.8:
                return False
        
        return True
    
    def _extract_region_from_contour(self, image: np.ndarray, 
                                     contour: np.ndarray,
                                     image_area: int) -> Region:
        try:
            x, y, w, h = cv2.boundingRect(contour)
            
            region_image = image[y:y+h, x:x+w].copy()
            
            area = cv2.contourArea(contour)
            area_ratio = area / image_area
            
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_epsilon * peri, True)
            rectangularity = len(approx) / 4.0
            
            confidence = min(1.0, area_ratio * 0.7 + (1.0 / rectangularity) * 0.3)
            
            return Region(
                bbox=BoundingBox(x, y, w, h),
                image=region_image,
                confidence=confidence,
                detection_method='contour',
                area_ratio=area_ratio,
                contour=contour
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to extract region from contour: {e}")
            return None
