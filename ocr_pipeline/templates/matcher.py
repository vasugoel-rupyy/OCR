import cv2
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple
from .definitions import Template
from .library import TEMPLATE_LIBRARY
from ..ocr.engine import PaddleOCREngine
from ..ocr.models import OCRResult

logger = logging.getLogger("ocr_pipeline.templates.matcher")

class TemplateMatcher:
    """Matches an input document image to a known template."""
    
    def __init__(self, config: Dict = None):
        self.templates = TEMPLATE_LIBRARY
        self.config = config or {}
        # We might need an OCR engine for keyword matching if text isn't provided
        # We defer initialization to avoid heavy load if not needed
        self._ocr_engine = None

    @property
    def ocr_engine(self):
        if self._ocr_engine is None:
            # Initialize with default/light config
            from ..ocr.engine import PaddleOCREngine
            # Assuming default config is fine
            self._ocr_engine = PaddleOCREngine(self.config.get('ocr', {}))
        return self._ocr_engine

    def match(self, image: np.ndarray, ocr_result: Optional[OCRResult] = None) -> Tuple[Optional[Template], float]:
        """
        Identify the best matching template for the image.
        
        Args:
            image: Preprocessed image (expected to be somewhat deskewed/cropped to document boundary)
            ocr_result: Optional existing OCR result to speed up keyword matching
            
        Returns:
            (Best Template, Confidence Score) or (None, 0.0)
        """
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return None, 0.0
            
        aspect_ratio = w / h
        
        logger.info(f"Matching image with Aspect Ratio: {aspect_ratio:.2f}")
        
        # 1. Filter by Aspect Ratio
        # Allow some tolerance (e.g. +/- 10%)
        candidates = []
        for tmpl in self.templates:
            # Check AR
            ratio_diff = abs(tmpl.width_height_ratio - aspect_ratio)
            # Tolerance: 0.15 (15%)
            if ratio_diff < 0.2: 
                candidates.append(tmpl)
                
        if not candidates:
            logger.info("No templates match the aspect ratio.")
            return None, 0.0
            
        if len(candidates) == 1 and not candidates[0].anchor_keywords:
            # Exact AR match and no keywords needed? (Rare)
            return candidates[0], 0.9

        # 2. Check Anchor Keywords
        # We need text. If not provided, run OCR.
        if ocr_result is None:
            logger.info("Running coarse OCR for template matching...")
            # We could resize image to speed up? But might lose small text.
            # For now, run full OCR.
            ocr_result = self.ocr_engine.extract_text(image)
        
        full_text_lower = ocr_result.full_text.lower()
        logger.info(f"Template Matcher found text: {full_text_lower[:100]}...") # Log first 100 chars
        
        best_template = None
        best_score = 0.0
        
        for tmpl in candidates:
            # Score based on keyword presence
            if not tmpl.anchor_keywords:
                # If no keywords defined, treat AR match as moderate confidence
                score = 0.5
            else:
                matches = [kw for kw in tmpl.anchor_keywords if kw in full_text_lower]
                match_count = len(matches)
                # Score = ratio of found keywords
                score = match_count / len(tmpl.anchor_keywords)
                
                # Bonus for AR match tightness
                ar_bonus = 1.0 - (abs(tmpl.width_height_ratio - aspect_ratio) / 0.2)
                score = (score * 0.7) + (ar_bonus * 0.3)
                
                logger.info(f"Checking {tmpl.name}: Found {match_count}/{len(tmpl.anchor_keywords)} keywords {matches}. Raw Score: {score:.2f}")
            
            logger.debug(f"Template {tmpl.name} score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_template = tmpl
                
        if best_template and best_score >= best_template.threshold_score:
            logger.info(f"Matched template: {best_template.name} with score {best_score:.2f}")
            return best_template, best_score
            
        logger.info(f"No template met threshold (Max: {best_score:.2f})")
        return None, best_score
