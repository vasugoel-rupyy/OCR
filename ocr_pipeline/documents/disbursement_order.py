"""Disbursement Order (DO) document processor.

Operates as a Semantic Extraction Fallback Module (SEFM) using
raw OCR text without bounding boxes or template matching.
"""

from typing import Dict, Any, Optional, Tuple, List
import re
from difflib import SequenceMatcher
import logging

from ..ocr.models import OCRResult
from .base import BaseDocumentProcessor, Decision, BaseDocument, FieldValue

class DisbursementOrderDocument(BaseDocument):
    """Pydantic model for Disbursement Order."""
    loan_amount: Optional[FieldValue] = None
    disbursed_amount: Optional[FieldValue] = None
    rate_of_interest: Optional[FieldValue] = None
    tenure_months: Optional[FieldValue] = None
    customer_name: Optional[FieldValue] = None
    bank_name: Optional[FieldValue] = None
    ifsc: Optional[FieldValue] = None
    bank_branch_region: Optional[FieldValue] = None

logger = logging.getLogger('ocr_pipeline.documents.disbursement_order')

class DisbursementOrderProcessor(BaseDocumentProcessor):
    """Semantic Extractor for Disbursement Orders."""
    
    FallbackProcessorType = "SEMANTIC_TEXT_ONLY"

    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        
        # Vocabularies
        self.keywords = {
            'loan_amount': [
                'sanctioned amount', 'loan amount', 'approved amount',
                'facility amount', 'credit limit', 'loan value', 'principal amount',
                'ऋण राशि', 'स्वीकृत राशि', 'स्वीकृत ऋण', 'मूलधन',
                'credit facility', 'loan sanction', 'term loan amount',
                'amount loan', 'loan', 'amount'
            ],
            'disbursed_amount': [
                'disbursed amount', 'net disbursement', 'amount released',
                'amount credited', 'payout amount', 'released amount',
                'वितरित राशि', 'जारी राशि', 'खाते में जमा राशि', 'भुगतान राशि',
                'net proceeds', 'disbursement value', 'finance amount', 'total payable', 'amount payable',
                'loan disbursement', 'case disbursed', 'net disbursement amount',
                'disbursement amount loan', 'amount net disbursement', 'disbursement'
            ],
            'rate_of_interest': [
                'interest rate', 'roi', 'applicable interest', 'lending rate',
                'floating rate', 'fixed rate',
                'ब्याज दर', 'वार्षिक ब्याज', 'लागू ब्याज'
            ],
            'tenure': [
                'tenure', 'loan period', 'repayment period', 'term', 'duration',
                'ऋण अवधि', 'पुनर्भुगतान अवधि', 'अवधि',
                'tenure months', 'emi'
            ],
            'customer_name': [
                'borrower name', 'applicant name', 'customer name', 'account holder',
                'ग्राहक नाम', 'उधारकर्ता नाम', 'आवेदक नाम',
                'intended recipient', 'customer', 'name', 'details', 'sold to'
            ],
            'bank_name': [
                'hdfc bank', 'icici bank', 'axis bank', 'state bank of india', 'bank of baroda',
                'बैंक का नाम', 'जारीकर्ता बैंक',
                'bank name', 'bank', 'hypothecation', 'hypothecation bank', 'hypothecation bank name'
            ],
            'ifsc': [
                'ifsc', 'ifsc code', 'bank ifsc', 'branch ifsc',
                'आईएफएससी', 'आईएफएससी कोड'
            ],
            'bank_branch_region': [
                'regional office', 'region', 'ro', 'zonal office', 'branch region',
                'क्षेत्र', 'क्षेत्रीय कार्यालय',
                'branch name', 'branch'
            ]
        }
        
        self.regexes = {
            'amount': r'(?:₹|Rs\.?|INR)?\s?([0-9,]{3,12}(?:\.\d{1,2})?)',
            'interest': r'(\d{1,2}(?:\.\d{1,2})?)(?:\s?(?:%|percent|p\.a))?',
            'tenure': r'(\d{1,3})(?:\s?(months?|yrs?|years?))?',
            'ifsc': r'([A-Z]{4}0[A-Z0-9]{6})'
        }

    def get_document_type(self) -> str:
        return "disbursement_order"

    def _fuzzy_match(self, text: str, keyword: str, threshold=0.78) -> bool:
        """Stage 2: Returns True if fuzzy match ratio >= threshold."""
        if not text or not keyword:
            return False
        
        kw_lower = keyword.lower()
        if kw_lower in text.lower():
            return True
            
        words = text.lower().split()
        kw_len = len(keyword.split())
        
        if kw_len == 0 or len(words) < kw_len:
            return False
            
        for i in range(len(words) - kw_len + 1):
            window = ' '.join(words[i:i+kw_len])
            if SequenceMatcher(None, window, kw_lower).ratio() >= threshold:
                return True
        return False

    def _extract_field_proximity(self, text: str, field_name: str, regex_pattern: str, is_amount: bool = False) -> Tuple[Optional[str], float]:
        """Multi-stage retrieval using line-by-line proximity to keywords."""
        lines = text.split('\n')
        best_val = None
        max_conf = 0.0
        
        for keyword in self.keywords[field_name]:
            for i, line in enumerate(lines):
                found_kw = False
                if keyword.lower() in line.lower() or self._fuzzy_match(line, keyword):
                    found_kw = True
                    
                if found_kw:
                    search_lines = [line]
                    if i + 1 < len(lines):
                        search_lines.append(lines[i + 1])
                    if i + 2 < len(lines):
                        search_lines.append(lines[i + 2])
                        
                    for j, search_line in enumerate(search_lines):
                        # Avoid searching backwards into the keyword itself if on the same line
                        # But for simplicity, we regex search the line
                        matches = list(re.finditer(regex_pattern, search_line, re.IGNORECASE))
                        if matches:
                            # If we are on the exact same line as the keyword, we prefer numbers that appear AFTER the keyword
                            for match in matches:
                                val_str = match.group(1) if match.lastindex else match.group(0)
                                if field_name == 'tenure' and match.lastindex and match.lastindex >= 2:
                                    val_str = f"{match.group(1)} {match.group(2)}"
                                
                                conf = 0.95 if j == 0 else (0.85 if j == 1 else 0.75)
                                
                                # Ambiguity handling for amounts
                                if is_amount:
                                    sanc_ctx = any(k.lower() in search_line.lower() or self._fuzzy_match(search_line, k) for k in self.keywords['loan_amount'])
                                    disb_ctx = any(k.lower() in search_line.lower() or self._fuzzy_match(search_line, k) for k in self.keywords['disbursed_amount'])
                                    
                                    if sanc_ctx and disb_ctx:
                                        conf = 0.0
                                    elif field_name == 'loan_amount' and disb_ctx and not sanc_ctx:
                                        conf = 0.0
                                    elif field_name == 'disbursed_amount' and sanc_ctx and not disb_ctx:
                                        conf = 0.0
                                        
                                if conf > max_conf:
                                    max_conf = conf
                                    best_val = val_str

        return best_val, max_conf

    def _extract_name_heuristic(self, text: str, field_name: str) -> Tuple[Optional[str], float]:
        """Extract name (Bank or Customer) implicitly if regex not suitable."""
        lines = text.split('\n')
        best_name = None
        max_conf = 0.0
        
        # For Bank Name, check against known list directly in text
        if field_name == 'bank_name':
            max_len = 0
            for b in self.keywords['bank_name']:
                if b not in ('बैंक का नाम', 'जारीकर्ता बैंक'): # Ignore the generic hindi descriptors for this check
                    if b.lower() in text.lower() and len(b) > max_len:
                        best_name = b.title()
                        max_conf = 0.95
                        max_len = len(b)
            if best_name:
                return best_name, max_conf

        for kws in self.keywords[field_name]: # e.g., 'customer name'
            for idx, line in enumerate(lines):
                if self._fuzzy_match(line, kws):
                    # Usually name is adjacent or on same line
                    parts = re.split(kws, line, flags=re.IGNORECASE)
                    if len(parts) > 1 and len(parts[-1].strip()) > 3:
                        name_cand = re.sub(r'^[:\-\s]+', '', parts[-1]).strip()
                        if name_cand:
                            return name_cand, 0.85
                    
                    # check next line
                    if idx + 1 < len(lines):
                        next_line = lines[idx+1].strip()
                        if len(next_line) > 3 and not re.search(r'\d', next_line):
                            return next_line, 0.80
                            
        return best_name, max_conf

    def _normalize_currency(self, val_str: Optional[str]) -> Optional[float]:
        if not val_str: return None
        cleaned = re.sub(r'[^\d.]', '', val_str)
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _normalize_percentage(self, val_str: Optional[str]) -> Optional[float]:
        if not val_str: return None
        cleaned = re.sub(r'[^\d.]', '', val_str)
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _normalize_tenure(self, val_str: Optional[str]) -> Optional[int]:
        if not val_str: return None
        match = re.search(self.regexes['tenure'], val_str, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            unit = 'months'
            if match.lastindex and match.lastindex >= 2 and match.group(2):
                unit = match.group(2).lower()
            if unit.startswith('y'):
                return val * 12
            return val
        return None

    def extract_fields(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Extract structured fields using keyword proximity and regex."""
        text = ocr_result.full_text
        
        extracted = {}
        confidences = {}
        
        # 1. Loan Amount
        val, conf = self._extract_field_proximity(text, 'loan_amount', self.regexes['amount'], is_amount=True)
        extracted['loan_amount'] = self._normalize_currency(val)
        confidences['loan_amount'] = conf if extracted['loan_amount'] is not None else 0.0
        
        # 2. Disbursed Amount
        val, conf = self._extract_field_proximity(text, 'disbursed_amount', self.regexes['amount'], is_amount=True)
        extracted['disbursed_amount'] = self._normalize_currency(val)
        confidences['disbursed_amount'] = conf if extracted['disbursed_amount'] is not None else 0.0

        # 3. Rate of Interest
        val, conf = self._extract_field_proximity(text, 'rate_of_interest', self.regexes['interest'])
        extracted['rate_of_interest'] = self._normalize_percentage(val)
        confidences['rate_of_interest'] = conf if extracted['rate_of_interest'] is not None else 0.0

        # 4. Tenure
        val, conf = self._extract_field_proximity(text, 'tenure', self.regexes['tenure'])
        extracted['tenure_months'] = self._normalize_tenure(val)
        confidences['tenure_months'] = conf if extracted['tenure_months'] is not None else 0.0

        # 5. IFSC
        val, conf = self._extract_field_proximity(text, 'ifsc', self.regexes['ifsc'])
        extracted['ifsc'] = val.upper() if val else None
        confidences['ifsc'] = conf if extracted['ifsc'] else 0.0

        # 6. Customer Name
        val, conf = self._extract_name_heuristic(text, 'customer_name')
        extracted['customer_name'] = val.title() if val else None
        confidences['customer_name'] = conf if extracted['customer_name'] else 0.0

        # 7. Bank Name
        val, conf = self._extract_name_heuristic(text, 'bank_name')
        extracted['bank_name'] = val
        confidences['bank_name'] = conf if val else 0.0

        # 8. Bank Branch Region (Bank of Baroda specific)
        extracted['bank_branch_region'] = None
        confidences['bank_branch_region'] = 0.0
        if extracted['bank_name'] and 'baroda' in extracted['bank_name'].lower():
            r_val, r_conf = self._extract_name_heuristic(text, 'bank_branch_region')
            if r_val:
                extracted['bank_branch_region'] = r_val
                confidences['bank_branch_region'] = r_conf

        # Save confidences mentally to class scope if needed, or structured payload
        # Pipeline expects standard dict, so we will use validation to reflect rules
        self._last_confidences = confidences
        return extracted

    def validate_fields(self, fields: Dict[str, Any]) -> Dict:
        """Validate logic returning validation dict."""
        valid = True
        reasons = []

        if fields.get('tenure_months') is not None and fields['tenure_months'] < 3:
            valid = False
            reasons.append("Tenure < 3 months is invalid")

        if fields.get('rate_of_interest') is not None and fields['rate_of_interest'] > 40:
            valid = False
            reasons.append("Interest rate > 40% is invalid")

        return {'is_valid': valid, 'reasons': reasons}

    def validate_layout(self, ocr_result: OCRResult, image_shape: tuple) -> Dict:
        # Layout validation is bypassed in Semantic Extractor
        return {'is_valid': True, 'score': 1.0, 'reasons': ["Layout validation bypassed for SEFM"]}

    def check_consistency(self, fields: Dict[str, Any]) -> Dict:
        valid = True
        reasons = []
        
        la = fields.get('loan_amount')
        da = fields.get('disbursed_amount')
        
        if la and da and da > la:
            valid = False # or flag
            reasons.append("Disbursed amount > Loan amount")
            
        return {'is_valid': valid, 'reasons': reasons}
