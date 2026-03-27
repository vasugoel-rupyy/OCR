from typing import Dict, List, Tuple
from ..ocr.models import OCRResult, WordData

class KeyValueExtractor:
    
    def __init__(self, config: Dict):
        self.config = config
        
    def validate_kv_pairs(self, ocr_result: OCRResult, document_type: str) -> float:
        text = ocr_result.full_text.lower()
        score = 0.5
        
        if document_type == 'invoice':
            if 'total' in text and any(c.isdigit() for c in text):
                score += 0.3
            if 'invoice' in text:
                score += 0.2
                
        elif document_type == 'id_document':
            if 'dob' in text or 'birth' in text:
                score += 0.25
            if any(k in text for k in ['id', 'no.', 'number']):
                score += 0.25
                
        return min(1.0, score)
