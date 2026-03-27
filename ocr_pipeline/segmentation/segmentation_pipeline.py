import logging
from typing import List, Optional
import numpy as np

from .region import Region, BoundingBox
from .document_detector import DocumentDetector
from .text_clustering import TextClusterer

class SegmentationPipeline:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        self.enabled = self.config.get('enabled', True)
        
        self.document_detector = DocumentDetector(self.config)
        self.text_clusterer = TextClusterer(self.config)
        
        self.max_regions = self.config.get('max_regions', 5)
        self.min_region_confidence = self.config.get('min_region_confidence', 0.3)
        
        contour_config = self.config.get('contour_detection', {})
        text_config = self.config.get('text_clustering', {})
        self.use_contour_detection = contour_config.get('enabled', True)
        self.use_text_clustering = text_config.get('enabled', True)
    
    def detect_regions(self, image: np.ndarray, 
                      ocr_boxes: Optional[List] = None) -> List[Region]:
        if not self.enabled:
            return self._create_full_image_region(image)
        
        self.logger.info("Starting document region detection")
        
        all_regions = []
        
        if self.use_contour_detection:
            try:
                contour_regions = self.document_detector.detect_contours(image)
                all_regions.extend(contour_regions)
            except Exception as e:
                self.logger.warning(f"Contour detection failed: {e}")
        
        if self.use_text_clustering and ocr_boxes:
            try:
                text_regions = self.text_clusterer.cluster_text_regions(
                    ocr_boxes, 
                    image.shape
                )
                for region in text_regions:
                    region.image = self._extract_region_image(image, region.bbox)
                all_regions.extend(text_regions)
            except Exception as e:
                self.logger.warning(f"Text clustering failed: {e}")
        
        if not all_regions:
            return self._create_full_image_region(image)
        
        regions = self._deduplicate_regions(all_regions)
        regions = self._filter_regions(regions)
        regions.sort(key=lambda r: r.confidence, reverse=True)
        
        if len(regions) > self.max_regions:
            regions = regions[:self.max_regions]
        
        return regions
    
    def _create_full_image_region(self, image: np.ndarray) -> List[Region]:
        h, w = image.shape[:2]
        return [Region(
            bbox=BoundingBox(0, 0, w, h),
            image=image.copy(),
            confidence=1.0,
            detection_method='full_image',
            area_ratio=1.0
        )]
    
    def _extract_region_image(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        return image[bbox.y:bbox.y+bbox.height, bbox.x:bbox.x+bbox.width].copy()
    
    def _deduplicate_regions(self, regions: List[Region]) -> List[Region]:
        if len(regions) <= 1:
            return regions
        
        sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        unique_regions = []
        
        for region in sorted_regions:
            is_duplicate = False
            for kept_region in unique_regions:
                if region.bbox.overlaps_with(kept_region.bbox, threshold=0.7):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_regions.append(region)
        
        return unique_regions
    
    def _filter_regions(self, regions: List[Region]) -> List[Region]:
        filtered = []
        
        for region in regions:
            if region.confidence < self.min_region_confidence:
                continue
            
            if region.area_ratio < 0.05:
                continue
            
            aspect_ratio = region.bbox.aspect_ratio
            if aspect_ratio < 0.1 or aspect_ratio > 10.0:
                continue
            
            filtered.append(region)
        
        return filtered
