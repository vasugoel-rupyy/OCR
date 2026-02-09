"""PAN card specific field extraction and validation."""

import re
from typing import Dict, Optional, List
from pydantic import Field
from ..ocr.models import OCRResult
from ..validation.normalization import TokenNormalizer
from .base import BaseDocument, FieldValue


class PanDocument(BaseDocument):
    """Structured PAN document model."""
    pan_number: FieldValue
    name: FieldValue
    father_name: FieldValue
    date_of_birth: FieldValue
    signature_present: Optional[bool] = False
    
    def get_document_type(self) -> str:
        return "pan"



class PANExtractor:
    """Specialized extractor for PAN (Permanent Account Number) cards."""
    
    def __init__(self):
        """Initialize PAN extractor."""
        # Known PAN card keywords
        self.pan_keywords = [
            'income tax', 'permanent account number', 'pan',
            'govt. of india', 'government of india', 'income tax department',
            'आयकर विभाग', 'स्थायी खाता संख्या'
        ]
    
    def extract_fields(self, ocr_result: OCRResult) -> Dict[str, any]:
        """Extract all fields from PAN card.
        
        Args:
            ocr_result: OCR result object
            
        Returns:
            Dictionary of extracted fields
        """
        text = ocr_result.full_text
        fields = {}
        
        # Extract PAN number (most important)
        pan_number = self._extract_pan_number(text, ocr_result)
        if pan_number:
            fields['pan_number'] = pan_number
            fields['id_number'] = pan_number  # Alias for compatibility
        
        # Extract name
        name = self._extract_name(text, ocr_result)
        if name:
            fields['name'] = name
        
        # Extract father's name
        father_name = self._extract_father_name(text, ocr_result)
        if father_name:
            fields['father_name'] = father_name
        
        # Extract DOB
        dob = self._extract_dob(text)
        if dob:
            fields['date_of_birth'] = dob
            
        # Check visuals (Proxy via text keywords)
        if self._check_signature(text):
            fields['signature_present'] = True
        
        return fields
    
    
    def _extract_pan_number(self, text: str, ocr_result: OCRResult) -> Optional[str]:
        """Extract 10-character PAN number with multiple strategies and fuzzy correction.
        
        PAN Format: ABCDE1234F
        """
        text_upper = text.upper()
        
        # Strategy 1: Standard PAN pattern
        pattern1 = r'\b([A-Z]{5}[0-9]{4}[A-Z]{1})\b'
        matches = re.findall(pattern1, text_upper)
        unique_pans = set()
        for match in matches:
            if self._validate_pan(match):
                unique_pans.add(match)
        
        # Hard Reject: Multiple distinctive PANs
        if len(unique_pans) > 1:
            return None # Trigger rejection due to ambiguity
            
        if len(unique_pans) == 1:
            return list(unique_pans)[0]
        
        # Strategy 2: Fuzzy Extraction with Substitution
        # Look for any 10-character alphanumeric sequence that LOOKS like a PAN
        # We look for 5 mostly-alpha, 4 mostly-numeric, 1 mostly-alpha
        
        # Get all 10-char tokens (simple split)
        tokens = re.split(r'[\s.,:;-]+', text_upper)
        candidates = [t for t in tokens if len(t) == 10]
        
        # Also look for combined tokens (e.g. ABCDE 1234F) in original words list
        if ocr_result.words:
            # Combine adjacent words to form 10 chars
             for i in range(len(ocr_result.words) - 1):
                combined = ocr_result.words[i].text.upper() + ocr_result.words[i+1].text.upper()
                combined = re.sub(r'[^A-Z0-9]', '', combined)
                if len(combined) == 10:
                    candidates.append(combined)
                    
        for candle in candidates:
            corrected = self._fuzzy_correct_pan(candle)
            if corrected and self._validate_pan(corrected):
                return corrected
                
        # Strategy 3: Loose Regex Search on full text
        # Look for 5 letters followed by something that looks like numbers
        loose_pattern = r'([A-Z]{5})([0-9IOZS]{4})([A-Z0-9])'
        matches = re.findall(loose_pattern, text_upper)
        for groups in matches:
            raw_pan = "".join(groups)
            corrected = self._fuzzy_correct_pan(raw_pan)
            if corrected and self._validate_pan(corrected):
                return corrected
        
        return None
        
    def _fuzzy_correct_pan(self, text: str) -> Optional[str]:
        """Attempt to fix common OCR errors in PAN string.
        
        Logic:
           First 5 chars: Expect Letters (0->O, 1->I, 5->S, 8->B)
           Next 4 chars: Expect Digits (O->0, I->1, S->5, B->8)
           Last char: Expect Letter
        """
        if len(text) != 10:
            return None
            
        # Substitution maps
        to_alpha = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '6': 'G'}
        to_digit = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'Z': '2', 'A': '4'} # A->4 common? sometimes.
        
        chars = list(text)
        
        # Correct first 5 (Alpha)
        for i in range(5):
            if not chars[i].isalpha():
                if chars[i] in to_alpha:
                    chars[i] = to_alpha[chars[i]]
                else:
                    return None # Cannot fix
                    
        # Correct next 4 (Numeric)
        for i in range(5, 9):
            if not chars[i].isdigit():
                if chars[i] in to_digit:
                    chars[i] = to_digit[chars[i]]
                else:
                    return None # Cannot fix
                    
        # Correct last 1 (Alpha)
        if not chars[9].isalpha():
             if chars[9] in to_alpha:
                 chars[9] = to_alpha[chars[9]]
             # Note: Last char is check digit, hard to fix if completely wrong type
        
        corrected_pan = "".join(chars)
        return corrected_pan
    
    def _validate_pan(self, pan: str) -> bool:
        """Validate PAN number format.
        
        Args:
            pan: PAN number string
            
        Returns:
            True if valid format
        """
        # Must be exactly 10 characters
        if len(pan) != 10:
            return False
        
        # Check format: 5 letters + 4 digits + 1 letter
        if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', pan):
            return False
        
        # 4th character should be P for individual, C for company, etc.
        # Valid 4th characters: A, B, C, F, G, H, L, J, P, T
        valid_fourth_chars = ['A', 'B', 'C', 'F', 'G', 'H', 'L', 'J', 'P', 'T']
        if pan[3] not in valid_fourth_chars:
            return False
        
        return True
    
    def _extract_name(self, text: str, ocr_result: OCRResult) -> Optional[str]:
        """Extract person's name from PAN card.
        
        Args:
            text: Full OCR text
            ocr_result: OCR result
            
        Returns:
            Name or None
        """
        # Strategy 1: Look for name after "Name" keyword
        name_patterns = [
            r'(?:name|नाम)\s*:?\s*([A-Z][A-Z\s]{3,50})',
            r'([A-Z][A-Z\s]+(?:[A-Z][A-Z\s]+)+)',  # Multiple capitalized words
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                raw_name = match.group(1).strip()
                # Clean up
                name = re.sub(r'\s+', ' ', raw_name).strip()
                
                # Filter out false positives
                if self._is_valid_name(name):
                    return name
        
        # Strategy 2: Look in specific lines (name usually after PAN number)
        if ocr_result.lines and len(ocr_result.lines) > 2:
            for i, line in enumerate(ocr_result.lines):
                text_line = line.text.strip()
                # Look for all-caps names (PAN cards typically have names in caps)
                if re.match(r'^[A-Z][A-Z\s]{5,}$', text_line):
                    if self._is_valid_name(text_line):
                        return text_line
        
        return None
    
    def _extract_father_name(self, text: str, ocr_result: OCRResult) -> Optional[str]:
        """Extract father's name from PAN card.
        
        Args:
            text: Full OCR text
            ocr_result: OCR result
            
        Returns:
            Father's name or None
        """
        # Look for father's name after keywords
        father_patterns = [
            r"(?:father'?s?\s+name|पिता का नाम)\s*:?\s*([A-Z][A-Z\s]{3,50})",
        ]
        
        for pattern in father_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw_name = match.group(1).strip()
                name = re.sub(r'\s+', ' ', raw_name).strip()
                
                if self._is_valid_name(name):
                    return name
        
        return None
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if extracted text looks like a valid name.
        
        Args:
            name: Potential name string
            
        Returns:
            True if looks like a name
        """
        # Filter out common false positives
        invalid_keywords = [
            'income', 'tax', 'department', 'government', 'india', 'permanent',
            'account', 'number', 'signature', 'date', 'birth', 'father'
        ]
        
        name_lower = name.lower()
        for keyword in invalid_keywords:
            if keyword in name_lower:
                return False
        
        # Must have at least 2 words for full name
        words = name.split()
        if len(words) < 2:
            return False
        
        # Each word should be mostly alphabetic
        for word in words:
            if not word.isalpha() or len(word) < 2:
                return False
        
        # Name should not be too long
        if len(name) > 50:
            return False
        
        return True
    
    def _extract_dob(self, text: str) -> Optional[str]:
        """Extract Date of Birth from PAN card.
        
        Args:
            text: Full OCR text
            
        Returns:
            DOB or None
        """
        # Common formats: DD/MM/YYYY, DD-MM-YYYY
        dob_patterns = [
            r'(?:dob|date\s+of\s+birth|जन्म\s+तिथि)\s*:?\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})',
            r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})',  # Any date format
        ]
        
        for pattern in dob_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                if self._is_valid_date(date_str):
                    return date_str
        
        return None
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Check if date string is valid.
        
        Args:
            date_str: Date string
            
        Returns:
            True if valid date format
        """
        # Basic validation - check format
        if not re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', date_str):
            return False
        
        parts = re.split(r'[/-]', date_str)
        if len(parts) != 3:
            return False
        
        try:
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            
            # Basic range checks
            if not (1 <= day <= 31):
                return False
            if not (1 <= month <= 12):
                return False
            if year < 100:  # 2-digit year
                year += 1900 if year > 50 else 2000
            if not (1900 <= year <= 2024):
                return False
            
            return True
        except ValueError:
            return False

    def _check_signature(self, text: str) -> bool:
        """Check for signature keywords."""
        return bool(re.search(r'(?:signature|sign|hastakshar|हस्ताक्षर)', text, re.IGNORECASE))
