from typing import Dict, List, Tuple
from rapidfuzz import process, fuzz
from ..ocr.models import OCRResult

class AnchorValidator:
    
    def __init__(self, config: Dict):
        self.config = config.get('anchors', {})
        
    def validate_anchors(self, text: str, document_type: str) -> Tuple[float, Dict[str, float]]:
        if document_type not in self.config:
            return 0.0, {'error': f'No anchor config for {document_type}'}
            
        doc_config = self.config[document_type]
        required_anchors = doc_config.get('required', [])
        optional_anchors = doc_config.get('optional', [])
        threshold = doc_config.get('threshold', 80)
        
        text_lower = text.lower()
        
        found_required = 0
        found_optional = 0
        matches = {}
        
        for anchor in required_anchors:
            if anchor in text_lower:
                matches[anchor] = 100.0
                found_required += 1
                continue
                
            score = fuzz.partial_token_sort_ratio(anchor, text_lower)
            if score >= threshold:
                matches[anchor] = score
                found_required += 1
            else:
                matches[anchor] = score
        
        for anchor in optional_anchors:
            if anchor in text_lower:
                matches[anchor] = 100.0
                found_optional += 1
                continue
                
            score = fuzz.partial_token_sort_ratio(anchor, text_lower)
            if score >= threshold:
                matches[anchor] = score
                found_optional += 1
                
        total_required = len(required_anchors)
        
        if total_required > 0:
            required_ratio = found_required / total_required
        else:
            required_ratio = 1.0
            
        optional_bonus = min(0.2, found_optional * 0.05)
        
        final_score = min(1.0, required_ratio + optional_bonus)
        
        details = {
            'found_required': found_required,
            'total_required': total_required,
            'found_optional': found_optional,
            'matches': matches
        }
        
        return final_score, details
