import cv2
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List
import multiprocessing
import time

from .definitions import Template, Region
from ..ocr.engine import PaddleOCREngine
from ..ocr.models import OCRResult

logger = logging.getLogger("ocr_pipeline.templates.extractor")

def _worker_extract_region(image_crop: np.ndarray, region_name: str, config: Dict) -> Dict:
    """
    Worker function to run in a separate process.
    Instantiates its own OCR engine to avoid pickling issues/GIL.
    
    Args:
        image_crop: Cropped image numpy array
        region_name: Name of region (for logging/tagged result)
        config: OCR config dictionary
        
    Returns:
        Dict with 'name', 'text', 'confidence', etc.
    """
    try:
        # Instantiate engine
        # Note: This loads model into memory. 
        # For high throughput, we might want a persistent worker pool with initialized engines,
        # but ProcessPoolExecutor usually spawns fresh or reuses. 
        # PaddleOCR lazy loading might help.
        engine = PaddleOCREngine(config)
        result = engine.extract_text(image_crop)
        
        return {
            "region": region_name,
            "text": result.full_text,
            "confidence": result.mean_confidence, # 0-100 usually, engine returns... check engine.py
            # engine.py: confidence = float(confidence) * 100
            "words": [w.__dict__ for w in result.words] # Serializable
        }
    except Exception as e:
        # Retrieve traceback if needed
        return {
            "region": region_name,
            "error": str(e)
        }


class TemplateExtractor:
    """Extracts fields from a document based on a template using concurrent processing."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        # Max workers for region extraction
        self.max_workers = self.config.get('batch', {}).get('max_workers', 4)

    def extract(self, image: np.ndarray, template: Template) -> Dict[str, Any]:
        """
        Extract processed fields from the image using the template.
        
        Args:
            image: Full document image
            template: Matched template
            
        Returns:
            Dictionary of extracted fields {region_name: value}
        """
        h, w = image.shape[:2]
        tasks = []
        
        # Prepare crops
        crops = {}
        for region in template.regions:
            x1, y1, x2, y2 = region.to_absolute(w, h)
            
            # Ensure within bounds
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid crop for region {region.name}: {x1},{y1},{x2},{y2}")
                continue
                
            crop = image[y1:y2, x1:x2]
            crops[region.name] = crop

        # Execute concurrently
        extracted_data = {}
        regions_processed = 0
        
        # Use 'spawn' or 'fork' depending on OS? Linux 'fork' is default and fastest.
        # But PaddleOCR/Tensorflow/PyTorch + fork = deadlock sometimes.
        # Safest is 'spawn', but slower start.
        # Let's try default first. If it hangs, we switch to spawn.
        
        start_time = time.time()
        
        # We need to pass the OCR part of config
        ocr_config = self.config # Pass full config, worker extracts 'ocr' section
        
        with ProcessPoolExecutor(max_workers=min(len(crops), self.max_workers)) as executor:
            future_to_region = {
                executor.submit(_worker_extract_region, crop, name, ocr_config): name
                for name, crop in crops.items()
            }
            
            for future in as_completed(future_to_region):
                region_name = future_to_region[future]
                try:
                    res = future.result()
                    if 'error' in res:
                        logger.error(f"Error extracting region {region_name}: {res['error']}")
                    else:
                        # Normalize/Clean text based on region type
                        text = res['text'].strip()
                        extracted_data[region_name] = text
                        logger.debug(f"Region {region_name}: {text} (Conf: {res.get('confidence', 0):.1f})")
                except Exception as e:
                    logger.error(f"Exception in worker for {region_name}: {e}")
                    
        total_time = time.time() - start_time
        logger.info(f"Template extraction finished in {total_time:.2f}s")
        
        return extracted_data
