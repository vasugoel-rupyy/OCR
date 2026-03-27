from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> tuple:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0
        return self.width / self.height
    
    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.width, self.height)
    
    def to_corners(self) -> tuple:
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def overlaps_with(self, other: 'BoundingBox', threshold: float = 0.5) -> bool:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou >= threshold


@dataclass
class Region:
    bbox: BoundingBox
    image: np.ndarray
    confidence: float
    detection_method: str
    area_ratio: float
    contour: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        if not 0.0 <= self.area_ratio <= 1.0:
            raise ValueError(f"Area ratio must be between 0 and 1, got {self.area_ratio}")
        
        if self.detection_method not in ['contour', 'text_cluster', 'merged', 'full_image']:
            raise ValueError(f"Invalid detection method: {self.detection_method}")
    
    def to_dict(self) -> dict:
        return {
            'bbox': self.bbox.to_tuple(),
            'confidence': self.confidence,
            'detection_method': self.detection_method,
            'area_ratio': self.area_ratio,
            'aspect_ratio': self.bbox.aspect_ratio
        }
