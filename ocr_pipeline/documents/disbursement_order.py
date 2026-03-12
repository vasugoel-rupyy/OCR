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
    branch_id: Optional[FieldValue] = None

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
                'amount loan', 'facility value', 'baroda car loan', 'total loan amount',
                'section amt', 'loan amountlimit', 'amount of facility',
                'amount of loan', 'approval loan amount', 'loan amount',
                'term loan', 'amount in words', 'rupees',
            ],
            'disbursed_amount': [
                'disbursed amount', 'net disbursement amount', 'net disbursement',
                'amount released', 'amount credited', 'payout amount',
                'released amount', 'disbursement amount',
                'वितरित राशि', 'जारी राशि', 'खाते में जमा राशि', 'भुगतान राशि',
                'net proceeds', 'disbursement value', 'finance amount',
                'total payable', 'amount payable',
                'loan disbursement', 'case disbursed', 'net finance',
                'loan disbursement amount', 'disbursement amount',
                'net disb', 'net disbursement amt', 'disbursement amt',
                'net loan amount', 'net disbh',
            ],
            'rate_of_interest': [
                'interest rate', 'rate of interest', 'roi',
                'applicable interest', 'lending rate',
                'floating rate', 'fixed rate',
                'ब्याज दर', 'वार्षिक ब्याज', 'लागू ब्याज', 'interest',
            ],
            'tenure': [
                'tenure months', 'tenure in months', 'period in months',
                'tenure', 'loan period', 'repayment period', 'duration',
                'ऋण अवधि', 'पुनर्भुगतान अवधि', 'अवधि',
                'emi tenure', 'loan tenure', 'months', 'period',
            ],
            'customer_name': [
                'customer name', 'name of customer', 'name of applicant', 'applicant name',
                'name ot customer', 'customer name :', 'borrower name', 'ref name',
                'name', 'borrower.name', 'borrower', 'name of the customer', 'to.', 'to :',
                'ग्राहक नाम', 'उधारकर्ता नाम', 'आवेदक नाम',
                'intended recipient', 'name ot customer', 'customer_name',
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
            'branch_id': [
                'branch id', 'branch code', 'br code', 'branch sol id', 'br. code',
                'branch name & code', 'sol id', 'branch 1d',
                'branch id-', 'branch_id', 'solid',
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
            'bank of baroda', 'baroda', 'bob',
            'ambit finvest', 'jana small finance bank', 'manappuram finance',
            'federal bank', 'bandhan bank', 'south indian bank',
            'shriram finance', 'cholamandalam', 'hinduja leyland',
            'canara bank', 'uco bank', 'indian overseas bank',
            'Bank Of Baroda', 'Baroda', 'BoB', 'Lead Acknowledgement Receipt', 'Baroda Car Loan',
            'Indian Overseas Bank', 'Bandhan Bank', 'Ambit Finvest', 'Jana Small Finance Bank',
            'Poonawala Housing Finance', 'Ambit', 'Jana SFB', 'Manappuram Finance', 'Yes Bank',
            'Au Small Finance Bank', 'Poonawalla Fincorp', 'TVS Credit', 'ITI Finance',
            'Kogta Financial', 'Equitas Small Finance Bank', 'Bajaj Finance', 'South Indian Bank',
            'Muthoot Capital', 'Punjab & Sind Bank', 'Punjab And Sind Bank', 'Bank Of India',
            'Canara Bank', 'Federal Bank', 'Icici Bank', 'Tata Capital', 'Mahindra Finance',
            'Union Bank', 'Sundaram Finance', 'Hero Fincorp', 'Suneet Finman', 'TrillionLoans',
            'Liquiloans', 'Lead Acknowledgement Receipt'
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
            'costing', 'purchased', 'model', 'asset', 'changed', 'the',
            'and', 'agrees', 'sufficient', 'funds', 'notify', 'notified', 'information',
            'strictly', 'prohibited', 'opinions', 'intended', 'address', 'exclusive',
            'confidential', 'privileged', 'attachment', 'original', 'integrity',
            'security', 'guaranteed', 'individual', 'recipient', 'unauthorized',
            'reading', 'distribution', 'copying', 'opinion', 'intended', 'recipient',
            'expressly', 'authorized', 'hereby', 'notified', 'any', 'use', 'files',
            'documents', 'attached', 'transmitted', 'exclusive', 'addressed', 'you',
            'are', 'not', 'individual', 'intended', 'recipient', 'sales', 'manager',
            'sourcing', 'channel', 'application', 'status', 'payout', 'subvention',
            'is', 'sanctioned', 'kindly', 'provide', 'registration', 'ownership',
            'hypothecation', 'social', 'media', 'follow', 'us', 'dear', 'sir', 'madam',
            'ms', 'to', 'borrower', 'branch', 'name', 'region', 'location', 'raichur',
            'main', 'kalaburag', 'office', 'centre', 'center',
            'shall', 'made',
            'available', 'confirm', 'email', 'originated', 'caution',
        }
        
        # Blocklist for bank_name fallback to reject non-bank text
        self._bank_blocklist_tokens = {
            'code', 'branch', 'security', 'payment', 'charges', 'deposit',
            'deposited', 'hypothecatee', 'cap', 'blue', 'book', 'delayed',
            'insurance', 'cover', 'note', 'clause', 'certificate',
            'purchased', 'costing', 'model', 'asset', 'customer', 'changed',
            'sign number', 'signnumber', 'sign pumbera',
        }
        
        self.regexes = {
            # Common currency prefixes often prepended without space in OCR
            'amount': r'(?:₹|Rs\.?|INR|rs|inr|amt)\.?\s*[:\-\s]*([0-9,.]{3,15}(?:\.\d{1,2})?)',
            'interest': r'(\d{1,2}(?:\.\d{1,2})?)(?:\s?(?:%|percent|p\.a))?',
            'tenure': r'(\d{1,3})(?:\s?(months?|yrs?|years?))?',
            'ifsc': r'([A-Z]{4}0[A-Z0-9]{6})',
            'branch_id': r'\b(\d{4,6})\b',
            'text_amount': r'(?:Rupees?|Rs\.?)\s?([\w\s()-]+?)\s?only',
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
                    # Search up to 4 lines downstream for the value
                    search_lines = [line]
                    for offset in range(1, 5):
                        if i + offset < len(lines):
                            search_lines.append(lines[i + offset])
                        
                    # Find where keyword ends on the first line for ordering preference
                    kw_end = self._keyword_end_position(line, keyword)
                        
                    for j, search_line in enumerate(search_lines):
                        matches = list(re.finditer(regex_pattern, search_line, re.IGNORECASE))
                        if matches:
                            for match in matches:
                                val_str = match.group(1) if match.lastindex else match.group(0)
                                if field_name == 'tenure' and match.lastindex and match.lastindex >= 2:
                                    val_str = f"{match.group(1)} {match.group(2)}"
                                
                                # Base confidence by line distance (refined for vertical layouts)
                                dist_scores = [1.0, 0.9, 0.8, 0.7, 0.6]
                                conf = dist_scores[j] if j < len(dist_scores) else 0.5
                                
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
                                    # Include newly added keywords in context check
                                    sanc_kws = self.keywords['loan_amount'][:8] + ['baroda car loan', 'loan amount']
                                    disb_kws = self.keywords['disbursed_amount'][:8] + ['loan disbursement amount', 'disbursement amount']
                                    
                                    sanc_ctx = any(k.lower() in search_line.lower() for k in sanc_kws)
                                    disb_ctx = any(k.lower() in search_line.lower() for k in disb_kws)
                                    
                                    if sanc_ctx and disb_ctx:
                                        conf = 0.0
                                    elif field_name == 'loan_amount' and disb_ctx and not sanc_ctx:
                                        conf = 0.0
                                    elif field_name == 'disbursed_amount' and sanc_ctx and not disb_ctx:
                                        conf = 0.0
                                        
                                if conf > max_conf:
                                    # VALIDATION: Reject alphanumeric ID strings or dates
                                    # Extract just digits for date check
                                    digits_only = re.sub(r'\D', '', val_str)
                                    if len(digits_only) == 8:
                                        # Likely a date DDMMYYYY or YYYYMMDD
                                        conf *= 0.1
                                    
                                    if len(val_str) < 4 and not is_amount: # Tiny fragments
                                        conf *= 0.5
                                        
                                    # If the match is part of a word with letters, reject it
                                    start, end = match.span()
                                    if start > 0 and search_line[start-1].isalnum():
                                        continue
                                    if end < len(search_line) and search_line[end].isalnum():
                                        continue
                                        
                                    # PREFERENCE: Boost confidence for numbers with commas/periods in Indian format
                                    if ',' in val_str or ('.' in val_str and val_str.count('.') >= 1 and len(val_str) > 5):
                                        conf *= 1.1
                                    
                                    # REJECTION: Reject obviously non-amount values (e.g., single digits, short codes)
                                    # This part was malformed, ensuring it's correctly placed within the `if conf > max_conf` block
                                    if conf > max_conf:
                                        max_conf = conf
                                        best_val = val_str
        return best_val, max_conf

    def _is_valid_person_name(self, name: str) -> bool:
        """Validate that extracted text looks like a person's name."""
        if not name or len(name.strip()) < 3:
            return False
        name = name.strip()
        # Reject names starting with punctuation
        if name[0] in '.:-;,/|':
            return False
        # Reject names containing digits
        if re.search(r'\d', name):
            return False
        # Reject names with fewer than 2 words (too ambiguous for SEFM)
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
                if len(token) <= 2:
                    if token in name_lower.split(): return False
                else:
                    if token in name_lower: return False
        return True

    def _get_amount_candidates(self, text: str) -> List[Dict[str, Any]]:
        """Find all strings that look like currency amounts."""
        candidates = []
        # Match common currency patterns, including dotted Indian format: 21.00.000
        # Also catch Rs.123 and similar
        pattern = r'(?:₹|Rs\.?|INR|rs|inr|amt|amount)\.?\s*[:\-\s]*([0-9,.]{3,15}(?:\.\d{1,2})?)'
        # Also simple numbers that are big enough to be amounts
        pattern_simple = r'\b(\d{1,3}(?:[.,]\d{2,3})*(?:[.,]\d{2}))\b|\b([1-9]\d{4,7})\b'
        
        lines = text.split('\n')
        line_offsets = []
        current_offset = 0
        for line in lines:
            line_offsets.append(current_offset)
            current_offset += len(line) + 1

        all_matches = list(re.finditer(pattern, text, re.IGNORECASE))
        all_matches.extend(list(re.finditer(pattern_simple, text)))

        seen_positions = set()

        for m in all_matches:
            # Determine which group captured the number
            val_str = None
            for g in range(1, m.lastindex + 1 if m.lastindex else 2):
                if m.group(g):
                    val_str = m.group(g)
                    break
            
            if not val_str or m.start() in seen_positions:
                continue
            seen_positions.add(m.start())

            # Find which line this match is on
            pos = m.start()
            line_idx = 0
            for i, offset in enumerate(line_offsets):
                if pos >= offset:
                    line_idx = i
                else: break
            
            num_val = self._normalize_currency(val_str)
            if num_val and self._validate_amount(num_val):
                # Penalty for potential dates (8 digits) or phone numbers (10 digits starting with 6-9)
                digits_only = re.sub(r'\D', '', val_str)
                penalty = 1.0
                if len(digits_only) == 8: penalty = 0.4
                if len(digits_only) == 10 and digits_only[0] in '6789': penalty = 0.2
                
                candidates.append({
                    'val': num_val,
                    'raw': val_str,
                    'line_idx': line_idx,
                    'pos': pos,
                    'penalty': penalty
                })
        return candidates

    def _score_candidates(self, candidates: List[Dict[str, Any]], field_name: str, lines: List[str]) -> Tuple[Optional[Any], float]:
        """Score candidates based on proximity to field keywords."""
        best_val = None
        best_score = 0.0
        
        field_kws = self.keywords.get(field_name, [])
        if not field_kws: return None, 0.0

        for cand in candidates:
            cand_score = 0.0
            
            # Proximity check: scan lines around the candidate
            # Expanded range to 4 lines for multi-page/messy vertical layouts
            start_row = max(0, cand['line_idx'] - 4)
            end_row = min(len(lines), cand['line_idx'] + 4)
            
            for r in range(start_row, end_row + 1):
                if r >= len(lines): continue
                line = lines[r].lower()
                
                for kw in field_kws:
                    if kw in line:
                        dist = abs(r - cand['line_idx'])
                        # More forgiving decay for vertical proximity
                        weight = 1.0 - (dist * 0.15) 
                        # Boost if on same line and keyword is before amount
                        if dist == 0:
                            kw_pos = line.find(kw)
                            # Estimate relative position of amount in line
                            if kw_pos < line.find(cand['raw'].lower()):
                                weight *= 1.3
                            else:
                                weight *= 0.4
                                
                        cand_score = max(cand_score, weight)
                
            if cand_score > best_score:
                best_score = cand_score
                best_val = cand['val']
                
        return best_val, best_score

    def _extract_name_title_scan(self, text: str) -> Tuple[Optional[str], float]:
        lines = text.split('\n')
        title_pat = re.compile(r'(?:Mr\.?|Mrs\.?|Ms\.?|Shri\.?|Smt\.?)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)', re.IGNORECASE)
        for line in lines:
            m = title_pat.search(line)
            if m:
                candidate = m.group(0).strip()
                name_part = m.group(1).strip()
                if self._is_valid_person_name(name_part): return candidate, 0.90
        return None, 0.0

    def _extract_name_heuristic(self, text: str, field_name: str) -> Tuple[Optional[str], float]:
        lines = text.split('\n')
        if field_name == 'bank_name':
            max_len = 0
            best_bank = None
            for b in self.known_banks:
                if b.lower() in text.lower() and len(b) > max_len:
                    best_bank = b.title()
                    max_len = len(b)
            if best_bank: return best_bank, 0.95
        if field_name == 'customer_name':
            title_name, title_conf = self._extract_name_title_scan(text)
            if title_name: return title_name, title_conf
        for kws in self.keywords[field_name]:
            for idx, line in enumerate(lines):
                if self._fuzzy_match(line, kws):
                    parts = re.split(re.escape(kws), line, flags=re.IGNORECASE)
                    if len(parts) > 1 and len(parts[-1].strip()) > 3:
                        name_cand = re.sub(r'^[:\-,;\s]+', '', parts[-1]).strip()
                        name_cand = re.split(r'[,;|]', name_cand)[0].strip()
                        if field_name == 'customer_name' and self._is_valid_person_name(name_cand): return name_cand, 0.85
                    for offset in range(1, 5):
                        if idx + offset < len(lines):
                            next_line = lines[idx+offset].strip()
                            if len(next_line) > 3 and not re.search(r'\d', next_line):
                                if field_name == 'customer_name' and self._is_valid_person_name(next_line): 
                                    return next_line, 0.85 - (offset * 0.05)
        return None, 0.0

    def _validate_amount(self, amount: Optional[float]) -> Optional[float]:
        if amount is None: return None
        if amount < 25000: return None
        if amount > 1_000_000_000: return None
        return amount

    def _normalize_currency(self, val_str: Optional[str]) -> Optional[float]:
        if not val_str: return None
        if re.search(r'[a-zA-Z]', val_str) and any(kw in val_str.lower() for kw in ['lakh', 'thousand', 'crore']):
            return self._parse_text_amount(val_str)
        separators = re.findall(r'[.,]', val_str)
        if len(separators) > 1:
            last_sep_match = list(re.finditer(r'[.,]', val_str))[-1]
            last_sep_pos = last_sep_match.start()
            suffix = val_str[last_sep_pos+1:].strip()
            if re.fullmatch(r'\d{1,2}', suffix):
                main_part = val_str[:last_sep_pos]
                cleaned_main = re.sub(r'[^\d]', '', main_part)
                cleaned = f"{cleaned_main}.{suffix}"
            else:
                cleaned = re.sub(r'[^\d]', '', val_str)
        else:
            cleaned = re.sub(r'[^\d.]', '', val_str)
        try:
            val = float(cleaned)
            if val > 10000000 and str(val).endswith('00.0'): val = val / 100.0
            return val
        except ValueError: return None

    def _parse_text_amount(self, text_string: str) -> Optional[float]:
        if not text_string: return None
        units = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
        multipliers = {'hundred': 100, 'thousand': 1_000, 'lakh': 100_000, 'lakhs': 100_000, 'lac': 100_000, 'lacs': 100_000, 'crore': 10_000_000, 'crores': 10_000_000}
        text = re.sub(r'rupees?|only|and|[.,/\-()]|[\(s\)]', ' ', text_string.lower())
        words = text.split(); total = 0; current_segment = 0
        for word in words:
            if word in units: current_segment += units[word]
            elif word in multipliers:
                if current_segment == 0: current_segment = 1
                total += current_segment * multipliers[word]; current_segment = 0
            elif word.isdigit(): current_segment += int(word)
        total += current_segment
        return float(total) if total > 25000 else None

    def _normalize_percentage(self, val_str: Optional[str]) -> Optional[float]:
        if not val_str: return None
        cleaned = re.sub(r'[^\d.]', '', val_str)
        try: return float(cleaned)
        except ValueError: return None

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
        """Extract fields using a scoring-based candidate system."""
        text = ocr_result.full_text
        lines = text.split('\n')
        
        extracted = {}
        confidences = {}
        
        # 1. Bank Name Heuristic (Prioritized)
        bank_val, bank_conf = self._extract_name_heuristic(text, 'bank_name')
        extracted['bank_name'] = bank_val
        confidences['bank_name'] = bank_conf

        # 2. Get Amount Candidates
        amt_candidates = self._get_amount_candidates(text)
        
        # 3. Loan Amount scoring
        val, conf = self._score_candidates(amt_candidates, 'loan_amount', lines)
        if val is None or conf < 0.3:
            # Fallback for text amounts
            text_val, text_conf = self._extract_field_proximity(text, 'loan_amount', self.regexes['text_amount'], is_amount=True)
            if text_val:
                f_val = self._normalize_currency(text_val)
                if f_val:
                    val, conf = f_val, text_conf
        
        extracted['loan_amount'] = val
        confidences['loan_amount'] = conf

        # 4. Disbursed Amount scoring
        val, conf = self._score_candidates(amt_candidates, 'disbursed_amount', lines)
        # Avoid picking the same candidate for both if they are identical and confidences are close
        if extracted['loan_amount'] == val and len(amt_candidates) > 1:
            # Try second best
            remaining = [c for c in amt_candidates if c['val'] != val]
            if remaining:
                val2, conf2 = self._score_candidates(remaining, 'disbursed_amount', lines)
                if conf2 > 0.3:
                    val, conf = val2, conf2

        extracted['disbursed_amount'] = val
        confidences['disbursed_amount'] = conf

        # 5. Customer Name (using refined heuristics)
        val, conf = self._extract_name_heuristic(text, 'customer_name')
        extracted['customer_name'] = val.title() if val else None
        confidences['customer_name'] = conf

        # 6. Branch ID & ROI (standard proximity for now as they are less ambiguous)
        val, conf = self._extract_field_proximity(text, 'branch_id', self.regexes['branch_id'])
        extracted['branch_id'] = val
        confidences['branch_id'] = conf

        val, conf = self._extract_field_proximity(text, 'rate_of_interest', self.regexes['interest'])
        extracted['rate_of_interest'] = self._normalize_percentage(val)
        confidences['rate_of_interest'] = conf

        val, conf = self._extract_field_proximity(text, 'tenure', self.regexes['tenure'])
        extracted['tenure_months'] = self._normalize_tenure(val)
        confidences['tenure_months'] = conf

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
