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
        
        # --------------- LABEL KEYWORDS (for proximity search) ---------------
        # These are used to LOCATE a field on a line; they are NOT returned as values.
        # Ordered: most-specific first → least-specific last.
        self.keywords = {
            'loan_amount': [
                'sanctioned amount', 'loan amount', 'approved amount',
                'facility amount', 'credit limit', 'loan value', 'principal amount',
                'ऋण राशि', 'स्वीकृत राशि', 'स्वीकृत ऋण', 'मूलधन',
                'credit facility', 'loan sanction', 'term loan amount',
                'amount loan',
            ],
            'disbursed_amount': [
                'disbursed amount', 'net disbursement amount', 'net disbursement',
                'amount released', 'amount credited', 'payout amount',
                'released amount', 'disbursement amount',
                'वितरित राशि', 'जारी राशि', 'खाते में जमा राशि', 'भुगतान राशि',
                'net proceeds', 'disbursement value', 'finance amount',
                'total payable', 'amount payable',
                'loan disbursement', 'case disbursed',
            ],
            'rate_of_interest': [
                'interest rate', 'rate of interest', 'roi',
                'applicable interest', 'lending rate',
                'floating rate', 'fixed rate',
                'ब्याज दर', 'वार्षिक ब्याज', 'लागू ब्याज',
            ],
            'tenure': [
                'tenure months', 'tenure in months', 'period in months',
                'tenure', 'loan period', 'repayment period', 'duration',
                'ऋण अवधि', 'पुनर्भुगतान अवधि', 'अवधि',
                'emi tenure', 'loan tenure',
            ],
            'customer_name': [
                'name of customer', 'borrower name', 'applicant name',
                'customer name', 'account holder', 'sold to',
                'ग्राहक नाम', 'उधारकर्ता नाम', 'आवेदक नाम',
                'intended recipient',
            ],
            'bank_name': [
                'hypothecation bank name', 'hypothecation bank',
                'bank name', 'financier name', 'lender name',
                'बैंक का नाम', 'जारीकर्ता बैंक',
                'hypothecation',
            ],
            'ifsc': [
                'ifsc code', 'ifsc', 'bank ifsc', 'branch ifsc',
                'आईएफएससी', 'आईएफएससी कोड',
            ],
            'bank_branch_region': [
                'regional office', 'branch name', 'branch region',
                'zonal office', 'region', 'loan branch',
                'क्षेत्र', 'क्षेत्रीय कार्यालय',
            ],
        }
        
        # --------------- KNOWN BANK NAMES (for direct text matching) ---------------
        # These are returned as the extracted bank_name value.
        self.known_banks = [
            'hdfc bank', 'icici bank', 'axis bank', 'state bank of india',
            'bank of baroda', 'kotak mahindra bank', 'kotak mahindra',
            'yes bank', 'idbi bank', 'punjab national bank', 'pnb',
            'canara bank', 'union bank of india', 'union bank',
            'indian overseas bank', 'federal bank', 'bandhan bank',
            'indusind bank', 'idfc first bank', 'rbl bank',
            'south indian bank', 'karur vysya bank',
            'bank of india', 'central bank of india', 'uco bank',
            'indian bank', 'bank of maharashtra',
            'au small finance bank', 'equitas small finance bank',
            'jana small finance bank', 'ujjivan small finance bank',
            'hero fincorp', 'bajaj finance', 'tata capital',
            'mahindra finance', 'shriram finance', 'muthoot finance',
            'muthoot capital', 'cholamandalam', 'manappuram finance',
            'l&t finance', 'sundaram finance',
            'punjab & sind bank', 'punjab and sind bank',
            'esaf bank', 'piramal finance', 'fullerton india',
            'hinduja leyland', 'hdb financial', 'iifl finance',
        ]
        
        # --------------- CUSTOMER NAME BLOCKLIST ---------------
        # Tokens/patterns that should never appear in a valid person name.
        self._name_blocklist_tokens = {
            'a/c', 'no.', 'no', 'rs', 'rs.', 'inr', '₹', 'amount', 'loan',
            'emi', 'rate', 'bank', 'branch', 'ifsc', 'neft', 'rtgs',
            'hypothecation', 'disbursement', 'sanction', 'tenure',
            'maruti', 'hyundai', 'honda', 'toyota', 'tata', 'kia',
            'mahindra', 'suzuki', 'brezza', 'creta', 'verna', 'swift',
            'alto', 'sedan', 'suv', 'hatchback', 'cng', 'petrol', 'diesel',
            'lxi', 'vxi', 'zxi', 'vdi', 'zdi', 'limited', 'pvt', 'ltd',
            'invoice', 'vehicle', 'receipt', 'document', 'order',
            'favour', 'following', 'disbursed', 'behalf', 'subject',
            'location', 'address', 'please', 'note', 'authorized',
            'attached', 'reference', 'kindly', 'delete', 'mail',
            'proprietary', 'revoked', 'applicant', 'type', 'intended',
            'recipient', 'confidential', 'contain', 'may',
        }
        
        # Blocklist for bank_name fallback to reject non-bank text
        self._bank_blocklist_tokens = {
            'code', 'branch', 'security', 'payment', 'charges', 'deposit',
            'deposited', 'hypothecatee', 'cap', 'blue', 'book', 'delayed',
            'insurance', 'cover', 'note', 'clause', 'certificate',
        }
        
        self.regexes = {
            'amount': r'(?:₹|Rs\.?|INR)?\s?([0-9,]{3,12}(?:\.\d{1,2})?)',
            'interest': r'(\d{1,2}(?:\.\d{1,2})?)(?:\s?(?:%|percent|p\.a))?',
            'tenure': r'(\d{1,3})(?:\s?(months?|yrs?|years?))?',
            'ifsc': r'([A-Z]{4}0[A-Z0-9]{6})',
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

    def _keyword_end_position(self, line: str, keyword: str) -> int:
        """Return the character index where the keyword ends in the line, or -1."""
        idx = line.lower().find(keyword.lower())
        if idx >= 0:
            return idx + len(keyword)
        return -1

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
                        
                    # Find where keyword ends on the first line for ordering preference
                    kw_end = self._keyword_end_position(line, keyword)
                        
                    for j, search_line in enumerate(search_lines):
                        matches = list(re.finditer(regex_pattern, search_line, re.IGNORECASE))
                        if matches:
                            for match in matches:
                                val_str = match.group(1) if match.lastindex else match.group(0)
                                if field_name == 'tenure' and match.lastindex and match.lastindex >= 2:
                                    val_str = f"{match.group(1)} {match.group(2)}"
                                
                                # Base confidence by line distance
                                conf = 0.95 if j == 0 else (0.85 if j == 1 else 0.75)
                                
                                # Prefer values AFTER the keyword on the same line
                                if j == 0 and kw_end >= 0 and match.start() < kw_end:
                                    conf *= 0.6  # Penalize values before keyword
                                
                                # TENURE RANGE GUARD: reject implausible at extraction time
                                if field_name == 'tenure':
                                    try:
                                        raw_num = int(match.group(1))
                                        has_unit = match.lastindex and match.lastindex >= 2 and match.group(2)
                                        unit = match.group(2).lower() if has_unit else ''
                                        if unit.startswith('y'):
                                            effective_months = raw_num * 12
                                        else:
                                            effective_months = raw_num
                                        if effective_months < 3 or effective_months > 120:
                                            conf = 0.0  # Skip implausible tenure
                                        # Boost confidence if explicit unit present
                                        if has_unit:
                                            conf *= 1.1
                                    except (ValueError, IndexError):
                                        pass
                                
                                # Ambiguity handling for amounts
                                if is_amount:
                                    sanc_ctx = any(k.lower() in search_line.lower() for k in self.keywords['loan_amount'][:6])
                                    disb_ctx = any(k.lower() in search_line.lower() for k in self.keywords['disbursed_amount'][:6])
                                    
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

    def _is_valid_person_name(self, name: str) -> bool:
        """Validate that extracted text looks like a person's name."""
        if not name or len(name.strip()) < 3:
            return False
        name = name.strip()
        # Reject names starting with punctuation (leading dots, colons, dashes)
        if name[0] in '.:-;,/|':
            return False
        # Reject overly long strings (likely paragraphs)
        if len(name) > 50:
            return False
        # Reject names containing digits
        if re.search(r'\d', name):
            return False
        # Reject names with fewer than 2 words (too ambiguous for SEFM)
        # Allow Mr./Mrs./Ms. prefixed single names
        words = [w for w in name.split() if len(w) > 1]
        if len(words) < 2:
            return False
        # Reject names containing too many special characters
        special_count = sum(1 for c in name if c in '₹@#$%^&*()[]{}|<>/\\')
        if special_count > 1:
            return False
        # Check blocklist tokens
        name_lower = name.lower()
        for token in self._name_blocklist_tokens:
            if token in name_lower.split() or token in name_lower:
                # Allow partial match only for very short tokens if they are substrings within a word
                # e.g., 'no' in 'Noor' should be OK, but 'no' as a standalone word is not
                if len(token) <= 2:
                    if token in name_lower.split():
                        return False
                else:
                    if token in name_lower:
                        return False
        return True

    def _extract_name_title_scan(self, text: str) -> Tuple[Optional[str], float]:
        """Pre-scan for customer names using Mr./Mrs./Ms./Shri/Smt title patterns."""
        lines = text.split('\n')
        title_pat = re.compile(
            r'(?:Mr\.?|Mrs\.?|Ms\.?|Shri\.?|Smt\.?)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)',
            re.IGNORECASE
        )
        for line in lines:
            m = title_pat.search(line)
            if m:
                candidate = m.group(0).strip()
                # Validate the name part (after title)
                name_part = m.group(1).strip()
                if self._is_valid_person_name(name_part):
                    return candidate, 0.90
        return None, 0.0

    def _extract_name_heuristic(self, text: str, field_name: str) -> Tuple[Optional[str], float]:
        """Extract name (Bank or Customer) implicitly if regex not suitable."""
        lines = text.split('\n')
        best_name = None
        max_conf = 0.0
        
        # For Bank Name, check against known bank list directly in text
        if field_name == 'bank_name':
            max_len = 0
            for b in self.known_banks:
                if b.lower() in text.lower() and len(b) > max_len:
                    best_name = b.title()
                    max_conf = 0.95
                    max_len = len(b)
            if best_name:
                return best_name, max_conf
        
        # For Customer Name, first try title-based pre-scan (Mr./Mrs./Ms.)
        if field_name == 'customer_name':
            title_name, title_conf = self._extract_name_title_scan(text)
            if title_name:
                return title_name, title_conf

        for kws in self.keywords[field_name]:
            for idx, line in enumerate(lines):
                if self._fuzzy_match(line, kws):
                    # Usually name is adjacent or on same line
                    parts = re.split(re.escape(kws), line, flags=re.IGNORECASE)
                    if len(parts) > 1 and len(parts[-1].strip()) > 3:
                        name_cand = re.sub(r'^[:\-,;\s]+', '', parts[-1]).strip()
                        # Truncate at common delimiters
                        name_cand = re.split(r'[,;|]', name_cand)[0].strip()
                        if field_name == 'customer_name':
                            if self._is_valid_person_name(name_cand):
                                return name_cand, 0.85
                        elif field_name == 'bank_name':
                            # For bank names extracted by label proximity, validate against known banks
                            name_lower = name_cand.lower()
                            for kb in self.known_banks:
                                if kb in name_lower:
                                    return kb.title(), 0.85
                            # If not a known bank, return with lower confidence only if it passes blocklist
                            if (len(name_cand) > 5 and
                                not any(t in name_cand.lower() for t in self._bank_blocklist_tokens) and
                                not any(t in name_cand.lower() for t in ['hypothecation', 'a/c', 'no.'])):
                                return name_cand, 0.55
                        else:
                            return name_cand, 0.85
                    
                    # check next line
                    if idx + 1 < len(lines):
                        next_line = lines[idx+1].strip()
                        if len(next_line) > 3 and not re.search(r'\d', next_line):
                            if field_name == 'customer_name':
                                if self._is_valid_person_name(next_line):
                                    return next_line, 0.80
                            elif field_name == 'bank_name':
                                # Validate next-line bank candidates against blocklist too
                                if (len(next_line) > 5 and
                                    not any(t in next_line.lower() for t in self._bank_blocklist_tokens)):
                                    return next_line, 0.70
                            else:
                                return next_line, 0.80
                            
        return best_name, max_conf

    def _validate_amount(self, amount: Optional[float]) -> Optional[float]:
        """Reject implausible loan/disbursed amounts."""
        if amount is None:
            return None
        # Too small: catches year numbers (2025, 2026), page numbers, small digits
        if amount < 1000:
            return None
        # Too large: catches concatenated date+account number strings
        if amount > 1_000_000_000:
            return None
        return amount

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
        
        # 1. Loan Amount (with amount guard rails)
        val, conf = self._extract_field_proximity(text, 'loan_amount', self.regexes['amount'], is_amount=True)
        raw_amount = self._normalize_currency(val)
        extracted['loan_amount'] = self._validate_amount(raw_amount)
        confidences['loan_amount'] = conf if extracted['loan_amount'] is not None else 0.0
        
        # 2. Disbursed Amount (with amount guard rails)
        val, conf = self._extract_field_proximity(text, 'disbursed_amount', self.regexes['amount'], is_amount=True)
        raw_amount = self._normalize_currency(val)
        extracted['disbursed_amount'] = self._validate_amount(raw_amount)
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

        # 8. Bank Branch Region (for any bank, not just BoB)
        extracted['bank_branch_region'] = None
        confidences['bank_branch_region'] = 0.0
        if extracted['bank_name']:
            r_val, r_conf = self._extract_name_heuristic(text, 'bank_branch_region')
            if r_val:
                extracted['bank_branch_region'] = r_val
                confidences['bank_branch_region'] = r_conf

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
