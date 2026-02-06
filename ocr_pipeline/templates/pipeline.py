import logging
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
import time

from .matcher import TemplateMatcher
from .extractor import TemplateExtractor
from .definitions import Template
from ..ocr.models import OCRResult

logger = logging.getLogger("ocr_pipeline.templates.pipeline")

class TemplatePipeline:
    """
    Orchestrates the template-based OCR process.
    Steps:
    1. Match image to template.
    2. If matched, extract regions concurrently.
    3. Return structured data.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.matcher = TemplateMatcher(config)
        self.extractor = TemplateExtractor(config)
        
    def process(self, image: np.ndarray, ocr_result: Optional[OCRResult] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]], float]:
        """
        Attempt to process the document using templates.
        
        Args:
            image: Input image (numpy array)
            ocr_result: Optional existing OCR result (to speed up matching)
            
        Returns:
            Tuple(Template_Name, Extracted_Data, Confidence_Score)
            If no match or failure, returns (None, None, 0.0)
        """
        start_time = time.time()
        
        # 1. Match Template
        template, match_score = self.matcher.match(image, ocr_result)
        
        if not template:
            logger.info("No template matched.")
            return None, None, 0.0
            
        logger.info(f"Proceeding with template '{template.name}' (Score: {match_score:.2f})")
        
        # 2. Extract Regions
        try:
            extracted_data = self.extractor.extract(image, template)
            
            # 3. Basic Validation / Scoring
            # Score = Average confidence of fields?
            # Or just use match_score?
            # Let's verify required fields
            missing_required = []
            for region in template.regions:
                if region.required and region.name not in extracted_data:
                    missing_required.append(region.name)
                elif region.required and not extracted_data[region.name]:
                    missing_required.append(region.name) # Empty value
            
            if missing_required:
                logger.warning(f"Template extraction missing required fields: {missing_required}")
                # Penalize score or reject?
                # For hybrid mode, we might want to return what we have but with low confidence
                # so the fallback can kick in.
                return template.name, extracted_data, 0.4 # Low confidence
                
            logger.info("Template extraction successful.")
            return template.name, extracted_data, 0.95 # High confidence
            
        except Exception as e:
            logger.error(f"Template processing failed: {e}", exc_info=True)
            return template.name, None, 0.0
