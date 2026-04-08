"""Document classification logic for Indian documents."""

from typing import Dict, List, Tuple
import re
import logging

logger = logging.getLogger('ocr_pipeline')

class DocumentClassifier:
    """Classifies Indian documents (Aadhaar, PAN, Vehicle RC) based on content."""
    
    def __init__(self):
        """Initialize classifier with keyword maps and patterns."""
        self.type_keywords = {
            'aadhaar': [
                'aadhaar', 'आधार', 'uidai', 'government of india',
                'भारत सरकार', 'unique identification', 'unique identification authority',
                'enrollment', 'resident', 'dob', 'date of birth', 'male', 'female',
                'gender', 'address', 'पता'
            ],
            'pan': [
                'income tax', 'permanent account number', 'pan',
                'income tax department', 'govt. of india', 'government of india',
                'आयकर विभाग', 'स्थायी खाता संख्या', 'father', 'signature',
                'fathers name', 'father\'s name'
            ],
            'vehicle_rc': [
                'registration certificate', 'vehicle', 'registration number',
                'engine no', 'chassis no', 'registering authority', 'owner',
                'रजिस्ट्रेशन', 'वाहन', 'इंजन', 'चेसिस', 'maker', 'model',
                'vehicle class', 'reg no', 'rc', 'rto'
            ],
            'disbursement_order': [
                'disbursement order', 'loan disbursement', 'sanction letter',
                'credit facility release', 'loan release advice', 'disbursement memo',
                'loan booking confirmation', 'loan account created', 'first disbursement',
                'amount credited to account', 'ऋण वितरण', 'वितरण आदेश', 'स्वीकृत ऋण',
                'ऋण जारी', 'ऋण खाता', 'बैंक द्वारा वितरित',
                'bank of baroda', 'baroda', 'bob', 'lead acknowledgement receipt'
            ]
        }
        
        self.type_patterns = {
            'aadhaar': [
                r'\b\d{4}\s+\d{4}\s+\d{4}\b',  
                r'\b\d{12}\b',  
                r'(?:aadhaar|आधार)',  
                r'UIDAI',  
            ],
            'pan': [
                r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',  
                r'[I1|]NCOME\s*TAX\s*DEP[A-Z]*',
                r'NCOME\s*T[A-X]+',  
                r'P[AE]RM[A-Z]*\s*ACC[A-Z]*\s*NUM[A-Z]*',
                r'(?:father\'?s?\s+name)',  
                r'GOVT\.?\s*O[Ff]\s*IND[A-Z]*', 
            ],
            'vehicle_rc': [
                r'\b[A-Z]{2}\s*[-]?\s*\d{2}\s*[-]?\s*[A-Z]{1,2}\s*[-]?\s*\d{4}\b',  
                r'(?:registration\s+certificate|vehicle\s+informa)',
                r'(?:chassis|engine\s+no)',
                r'(?:fuel|seating|unladen|wheel\s*base)',
                r'(?:mfg\s*date|form\s+23)',
                r'(?:model|maker|manufacturer)',
            ],
            'disbursement_order': [
                r'(?i)(?:disbursement\s+order|loan\s+disbursement|sanction\s+letter)',
                r'(?i)(?:credit\s+facility\s+release|loan\s+release\s+advice|disbursement\s+memo)',
                r'(?i)(?:loan\s+booking\s+confirmation|amount\s+credited\s+to\s+account)',
                r'(?i)(?:first\s+disbursement)',
            ],
        }
    
    def classify_with_scores(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Classify document and return scores.
        
        Args:
            text: OCR extracted text
            
        Returns:
            Tuple of (best_type, scores_dict)
        """
        text_lower = text.lower()
        scores = {dtype: 0 for dtype in self.type_keywords}
        
        logger.debug(f"Classifying text (len={len(text)}): {text[:100]}...")
        
        for dtype, keywords in self.type_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    weight = 2 if len(keyword.split()) > 1 else 1
                    scores[dtype] += weight
        
        for dtype, patterns in self.type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[dtype] += 5  
        
        logger.info(f"Classification scores: {scores}")
        
        max_score = max(scores.values())
        
        if max_score == 0:
            logger.warning("No classification signals found, defaulting to 'aadhaar'")
            return 'aadhaar', scores
        
        classified_type = max(scores, key=scores.get)
        
        if list(scores.values()).count(max_score) > 1:
            logger.info(f"Tie detected, using priority order")
            priority_order = ['disbursement_order', 'vehicle_rc', 'pan', 'aadhaar']
            for dtype in priority_order:
                if scores[dtype] == max_score:
                    classified_type = dtype
                    break
        
        if classified_type == 'disbursement_order':
            matched_indicators = 0
            for keyword in self.type_keywords['disbursement_order']:
                if keyword.lower() in text_lower:
                    matched_indicators += 1
            for p in self.type_patterns['disbursement_order']:
                if re.search(p, text, re.IGNORECASE):
                    matched_indicators += 1
                    
            do_confidence = min(1.0, matched_indicators / 3.0) 
            
            scores['disbursement_order_confidence'] = do_confidence

        logger.info(f"Classified as: {classified_type} (score: {scores[classified_type]})")
        return classified_type, scores

    def classify(self, text: str) -> str:
        """Classify document based on text content (Legacy wrapper)."""
        dtype, _ = self.classify_with_scores(text)
        return dtype

