from typing import Dict, Tuple
import re

class DistributionAnalyzer:
    
    def __init__(self, config: Dict):
        self.config = config.get('distribution', {})
        
    def analyze(self, text: str, document_type: str) -> Tuple[float, Dict]:
        if not text:
            return 0.0, {'error': 'No text'}
            
        if document_type not in self.config:
            return 1.0, {}
            
        profile = self.config[document_type]
        min_numeric = profile.get('min_numeric_ratio', 0.0)
        max_special = profile.get('max_special_char_ratio', 1.0)
        
        total_chars = len(text)
        if total_chars == 0:
            return 0.0, {}
            
        numeric_count = sum(c.isdigit() for c in text)
        alphanumeric_count = sum(c.isalnum() or c.isspace() for c in text)
        special_char_count = total_chars - alphanumeric_count
        
        numeric_ratio = numeric_count / total_chars
        special_char_ratio = special_char_count / total_chars
        
        score = 1.0
        
        if numeric_ratio < min_numeric:
            deviation = (min_numeric - numeric_ratio) / min_numeric
            score -= deviation * 0.5
            
        if special_char_ratio > max_special:
            deviation = (special_char_ratio - max_special) / (1.0 - max_special)
            score -= deviation * 0.8
            
        metrics = {
            'numeric_ratio': numeric_ratio,
            'special_char_ratio': special_char_ratio,
            'total_chars': total_chars
        }
        
        return max(0.0, score), metrics
