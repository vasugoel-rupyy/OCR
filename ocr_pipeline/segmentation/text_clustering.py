import logging
from typing import List, Tuple
import numpy as np

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    DBSCAN = None
    HAS_SKLEARN = False

from .region import Region, BoundingBox

class TextClusterer:
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        if not HAS_SKLEARN:
            self.logger.warning("scikit-learn not found. Text clustering will be disabled.")
        
        text_config = self.config.get('text_clustering', {})
        self.min_cluster_size = text_config.get('min_cluster_size', 5)
        self.eps = text_config.get('eps', 50)
        self.min_samples = text_config.get('min_samples', 3)
    
    def cluster_text_regions(self, ocr_boxes: List[Tuple], 
                            image_shape: Tuple) -> List[Region]:
        if not HAS_SKLEARN:
            return []
            
        if not ocr_boxes or len(ocr_boxes) < self.min_cluster_size:
            return []
        
        centers = []
        for box in ocr_boxes:
            if len(box) == 4:
                x, y, w, h = box
                centers.append([x + w/2, y + h/2])
        
        if len(centers) < self.min_cluster_size:
            return []
        
        centers = np.array(centers)
        
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(centers)
        
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(ocr_boxes[idx])
        
        regions = []
        image_area = image_shape[0] * image_shape[1]
        
        for cluster_id, boxes in clusters.items():
            if len(boxes) >= self.min_cluster_size:
                region = self._create_region_from_cluster(boxes, image_shape, image_area)
                if region:
                    regions.append(region)
        
        regions = self._merge_overlapping_clusters(regions)
        
        return regions
    
    def _create_region_from_cluster(self, boxes: List[Tuple], 
                                    image_shape: Tuple,
                                    image_area: int) -> Region:
        try:
            min_x = min(box[0] for box in boxes)
            min_y = min(box[1] for box in boxes)
            max_x = max(box[0] + box[2] for box in boxes)
            max_y = max(box[1] + box[3] for box in boxes)
            
            padding = 20
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(image_shape[1], max_x + padding)
            max_y = min(image_shape[0], max_y + padding)
            
            width = max_x - min_x
            height = max_y - min_y
            
            if width <= 0 or height <= 0:
                return None
            
            area = width * height
            area_ratio = area / image_area
            
            text_density = len(boxes) / area if area > 0 else 0
            confidence = min(1.0, text_density * 1000)
            
            region_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            return Region(
                bbox=BoundingBox(int(min_x), int(min_y), int(width), int(height)),
                image=region_image,
                confidence=confidence,
                detection_method='text_cluster',
                area_ratio=area_ratio
            )
        
        except Exception as e:
            self.logger.warning(f"Failed to create region from cluster: {e}")
            return None
    
    def _merge_overlapping_clusters(self, regions: List[Region]) -> List[Region]:
        if len(regions) <= 1:
            return regions
        
        merged = []
        used = set()
        
        for i, region1 in enumerate(regions):
            if i in used:
                continue
            
            overlapping = [region1]
            for j, region2 in enumerate(regions[i+1:], start=i+1):
                if j in used:
                    continue
                
                if region1.bbox.overlaps_with(region2.bbox, threshold=0.3):
                    overlapping.append(region2)
                    used.add(j)
            
            if len(overlapping) > 1:
                merged_region = self._merge_regions(overlapping)
                if merged_region:
                    merged.append(merged_region)
            else:
                merged.append(region1)
        
        return merged
    
    def _merge_regions(self, regions: List[Region]) -> Region:
        min_x = min(r.bbox.x for r in regions)
        min_y = min(r.bbox.y for r in regions)
        max_x = max(r.bbox.x + r.bbox.width for r in regions)
        max_y = max(r.bbox.y + r.bbox.height for r in regions)
        
        width = max_x - min_x
        height = max_y - min_y
        
        avg_confidence = sum(r.confidence for r in regions) / len(regions)
        avg_area_ratio = sum(r.area_ratio for r in regions) / len(regions)
        
        region_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        return Region(
            bbox=BoundingBox(min_x, min_y, width, height),
            image=region_image,
            confidence=avg_confidence,
            detection_method='merged',
            area_ratio=avg_area_ratio
        )
