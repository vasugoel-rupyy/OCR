"""Aadhaar card specific field extraction with template matching."""

import re
import cv2
import numpy as np

from typing import Dict, Optional, List, Tuple
from pydantic import Field
from ..ocr.models import OCRResult
from ..validation.normalization import TokenNormalizer
from .base import BaseDocument, FieldValue


class AadhaarDocument(BaseDocument):
    """Structured Aadhaar document model."""
    aadhaar_number: FieldValue
    name: FieldValue
    date_of_birth: Optional[FieldValue] = None
    year_of_birth: Optional[FieldValue] = None
    gender: FieldValue
    address: Optional[FieldValue] = None
    pin_code: Optional[FieldValue] = None
    vid: Optional[FieldValue] = None
    enrollment_id: Optional[FieldValue] = None
    
    def get_document_type(self) -> str:
        return "aadhaar"



class AadhaarExtractor:
    """Specialized extractor for Aadhaar cards."""
    
    def __init__(self):
        """Initialize Aadhaar extractor."""
        # Known Aadhaar patterns
        self.aadhaar_keywords = [
            'aadhaar', 'आधार', 'uidai', 'government of india',
            'भारत सरकार', 'unique identification'
        ]
    
    def extract_fields(self, ocr_result: OCRResult) -> Dict[str, any]:
        """Extract all fields from Aadhaar card.
        
        Args:
            ocr_result: OCR result object
            
        Returns:
            Dictionary of extracted fields
        """
        text = ocr_result.full_text
        fields = {}
        
        # Extract Aadhaar number (most important)
        aadhaar_number = self._extract_aadhaar_number(text, ocr_result)
        if aadhaar_number:
            fields['aadhaar_number'] = aadhaar_number
            # fields['id_number'] = aadhaar_number  # Alias
        
        # Extract VID if present
        vid = self._extract_vid(text)
        if vid:
            fields['vid'] = vid
        
        # Extract name (both English and Hindi)
        name = self._extract_name(text, ocr_result)
        if name:
            fields['name'] = name
        
        # Extract DOB
        dob = self._extract_dob(text)
        if dob:
            fields['date_of_birth'] = dob
            # fields['dob'] = dob  # Alias
        
        # Extract Gender
        gender = self._extract_gender(text)
        if gender:
            fields['gender'] = gender
            
        # Extract PIN Code
        pin = self._extract_pin_code(text)
        if pin:
            fields['pin_code'] = pin
            
        # Extract Enrollment ID
        eid = self._extract_enrollment_id(text)
        if eid:
            fields['enrollment_id'] = eid
            
        # Extract Address (Composite)
        address = self._extract_address(text, ocr_result)
        if address:
            fields['address'] = address
            
        
        return fields
    
    
    def _extract_aadhaar_number(self, text: str, ocr_result: OCRResult) -> Optional[str]:
        """Extract 12-digit Aadhaar number with multiple strategies.
        
        Args:
            text: Full OCR text
            ocr_result: OCR result with word-level data
            
        Returns:
            Aadhaar number or None
        """
        # Strategy 1: Look for 12-digit number with spaces, hyphens, or DOTS
        pattern1 = r'\b(\d{4})[\s.-]+(\d{4})[\s.-]+(\d{4})\b'
        matches = re.findall(pattern1, text)
        for match in matches:
            aadhaar = ''.join(match)
            if self._validate_aadhaar(aadhaar):
                return aadhaar
        
        # Strategy 2: Look for 12 consecutive digits
        pattern2 = r'\b(\d{12})\b'
        matches = re.findall(pattern2, text)
        for match in matches:
            if self._validate_aadhaar(match):
                return match
                
        # Strategy 3: Look for spaced digits (e.g. 4 8 2 8 ...)
        if ocr_result.words:
            aadhaar = self._extract_from_words(ocr_result.words)
            if aadhaar:
                return aadhaar
        
        # Strategy 4: Look for numbers near "Aadhaar" keyword
        aadhaar_pattern = r'(?:aadhaar|आधार).*?(\d{4}[\s.-]*\d{4}[\s.-]*\d{4})'
        match = re.search(aadhaar_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            aadhaar = re.sub(r'[\s.-]+', '', match.group(1))
            if self._validate_aadhaar(aadhaar):
                return aadhaar
        
        return None
    
    def _extract_from_words(self, words: List) -> Optional[str]:
        """Extract Aadhaar from word-level OCR data.
        
        Args:
            words: List of WordData objects
            
        Returns:
            Aadhaar number or None
        """
        # Look for sequence of 3 4-digit numbers
        digit_words = []
        for word in words:
            # Clean punctuation that might stick to numbers (e.g. "4828-")
            cleaned = re.sub(r'[^\d]', '', word.text)
            if len(cleaned) == 4:
                digit_words.append(cleaned)
        
        # Check consecutive sequences
        for i in range(len(digit_words) - 2):
            aadhaar = digit_words[i] + digit_words[i+1] + digit_words[i+2]
            if self._validate_aadhaar(aadhaar):
                return aadhaar
        
        return None

    def _validate_aadhaar(self, number: str) -> bool:
        """Validate Aadhaar number format.
        
        Args:
            number: Aadhaar number string
            
        Returns:
            True if valid format
        """
        # Normalize and validate
        number = TokenNormalizer.convert_devanagari_to_arabic(number)
        
        # Must be exactly 12 digits
        if not number.isdigit() or len(number) != 12:
            return False
        
        # First digit cannot be 0 or 1
        if number[0] in ['0', '1']:
            return False
        
        # Basic Verhoeff algorithm check (simplified)
        # In production, implement full Verhoeff validation
        return True

    def _extract_vid(self, text: str) -> Optional[str]:
        """Extract VID (Virtual ID) if present.
        
        Args:
            text: Full OCR text
            
        Returns:
            VID or None
        """
        # VID is 16 digits
        pattern = r'(?:vid|virtual\s+id).*?(\d{4}\s*\d{4}\s*\d{4}\s*\d{4})'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            vid = re.sub(r'\s+', '', match.group(1))
            if vid.isdigit() and len(vid) == 16:
                return vid
        
        return None

    def _extract_name(self, text: str, ocr_result: OCRResult) -> Optional[str]:
        """Extract person's name with multiple strategies.
        
        Args:
            text: Full OCR text
            ocr_result: OCR result
            
        Returns:
            Name or None
        """
        # Strategy 1: Keyword-based match (High precision)
        # Search for name specifically after "Name" label
        keyword_pattern = r'(?:name|नाम|नामा|नाभ)\s*[:.-]?\s*([A-Za-z \t]{3,50})'
        for match in re.finditer(keyword_pattern, text, re.IGNORECASE | re.UNICODE):
            name = re.sub(r'[@:.,]', ' ', match.group(1))
            name = re.sub(r'\s+', ' ', name).strip()
            if self._is_valid_name(name):
                return name

        # Strategy 2: Positional merging of top lines (Handles skew/splits)
        # Name is usually line 2 or 3 in standard Aadhaar formats
        if ocr_result.lines:
            # Check lines 1 to 4 (skipping Govt of India header at line 0)
            for i in range(1, min(len(ocr_result.lines), 5)):
                line = ocr_result.lines[i]
                text_line = line.text.strip()
                name_cand = re.sub(r'([a-z])([A-Z])', r'\1 \2', text_line)
                name_cand = re.sub(r'\s+', ' ', name_cand).strip()
                
                if self._is_valid_name(name_cand):
                    # Check if next line is a continuation of the name (skew case)
                    if i + 1 < len(ocr_result.lines):
                        next_line_text = ocr_result.lines[i+1].text.strip()
                        next_name_part = re.sub(r'([a-z])([A-Z])', r'\1 \2', next_line_text)
                        next_name_part = re.sub(r'\s+', ' ', next_name_part).strip()
                        
                        # Merge if next line is also a valid name part and doesn't look like a new field
                        if (self._is_valid_name(next_name_part) and 
                            not any(kw in next_line_text.lower() for kw in ['dob', 'address', 'pata', 'birth', 'gender'])):
                            return f"{name_cand} {next_name_part}"
                    
                    return name_cand

        # Strategy 3: Global pattern match for capitalized words (Last resort)
        # Useful if keywords are missing and position is non-standard
        global_pattern = r'([A-Z][A-Za-z]{2,}(?:[ \t@:.,]+[A-Z][A-Za-z]{1,})*)'
        candidates = []
        for match in re.finditer(global_pattern, text):
            name = re.sub(r'[@:.,]', ' ', match.group(0))
            name = re.sub(r'\s+', ' ', name).strip()
            if self._is_valid_name(name):
                candidates.append((name, match.start()))
        
        if candidates:
            # Pick the earliest occurrence in the document
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]
        
        return None
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if extracted text looks like a valid name.
        
        Args:
            name: Potential name string
            
        Returns:
            True if looks like a name
        """
        # Filter out common false positives and metadata
        invalid_keywords = [
            'government', 'india', 'aadhaar', 'male', 'female',
            'address', 'date', 'birth', 'dob', 'yob', 'pata',
            'unique', 'identification', 'authority', 'enrollment',
            's/o', 'd/o', 'w/o', 'c/o', 'care of', 'son of', 'daughter of', 'wife of',
            'vid', 'help', 'email', 'www.', 'website', 'download',
            'uidai', 'govt', 'भारत', 'सरकार', 'आबकारी', 'विभाग', 'nrc', 'pib',
            'name', 'नाम', 'नामा', 'नाभ'
        ]
        
        name_lower = name.lower()
        # Clean name from common parentage prefixes if they got caught
        for prefix in ['s/o', 'd/o', 'w/o', 'c/o', 'son of', 'daughter of']:
            if name_lower.startswith(prefix):
                return False

        for keyword in invalid_keywords:
            if keyword in name_lower:
                return False
        
        # Minimum total length for a name
        if len(name) < 3:
            return False

        words = name.split()
        if len(words) == 0:
            return False
            
        # Each word should be mostly alphabetic
        for word in words:
            # Allow names with dots like "A.K. Sharma" or "Anjali."
            word_clean = re.sub(r'[.]', '', word)
            if not word_clean.isalpha() and len(word_clean) > 0:
                return False
            if len(word_clean) < 2 and len(words) == 1:
                return False # Single letter names are rare/invalid
        
        return True
    
    def _extract_dob(self, text: str) -> Optional[str]:
        """Extract Date of Birth."""
        text = TokenNormalizer.convert_devanagari_to_arabic(text)
        
        # Pattern: DOB : DD/MM/YYYY or YOB : YYYY
        # Also simple date search
        dob_pattern = r'(?:dob|date\s+of\s+birth|yob|year\s+of\s+birth)\s*[:.-]?\s*(\d{2}/\d{2}/\d{4}|\d{4})'
        
        # Robust patterns handling OCR noise and compound labels
        dob_patterns = [
            # Matches "DOB : DD/MM/YYYY", "Date of Birth / DOB : DD/MM/YYYY"
            r'(?:dob|d0b|date\s+of\s+birth|जन्म\s+तिथि|dop)(?:[/\s\w]*)\s*[:.-]?\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})',
            r'(?:dob|d0b|date\s+of\s+birth|जन्म\s+तिथि|dop)(?:[/\s\w]*)\s*[:.-]?\s*(\d{8})',
            r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})',  # Any date format fall back
        ]
        
        for pattern in dob_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # If 8 digits check if it's DDMMYYYY
                if len(date_str) == 8 and date_str.isdigit():
                     # Insert separators for validation
                     date_str = f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"
                
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
    
    def _extract_gender(self, text: str) -> Optional[str]:
        """Extract gender.
        
        Args:
            text: Full OCR text
            
        Returns:
            Gender or None
        """
        text_lower = text.lower()
        
        if 'male' in text_lower and 'female' not in text_lower:
            return 'Male'
        elif 'female' in text_lower:
            return 'Female'
        elif 'पुरुष' in text:
            return 'Male'
        elif 'महिला' in text:
            return 'Female'
        
        return None
    
    def _extract_address(self, text: str, ocr_result: OCRResult) -> Optional[str]:
        """Extract address.
        
        Args:
            text: Full OCR text
            ocr_result: OCR result
            
        Returns:
            Address or None
        """
        # Look for address keywords
        address_pattern = r'(?:address|पता)\s*:?\s*(.{20,200})'
        match = re.search(address_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            address = match.group(1).strip()
            # Clean up
            address = re.sub(r'\s+', ' ', address)
            return address[:200]  # Limit length
        
        # Alternative: Look in bottom half of document
        if ocr_result.lines and len(ocr_result.lines) > 5:
            # Address usually in bottom half
            bottom_lines = ocr_result.lines[len(ocr_result.lines)//2:]
            address_parts = []
            # Patterns that indicate a line is NOT part of an address
            non_address_re = re.compile(
                r'\b(VID|MALE|FEMALE|TRANSGENDER|aadhaar|आधार|पुरुष|महिला)\b'
                r'|\d{4}\s*\d{4}\s*\d{4}'   # Aadhaar number pattern
                r'|\d{16}',                   # VID pattern
                re.IGNORECASE
            )
            for line in bottom_lines:
                text_line = line.text.strip()
                # Skip short lines, pure-digit lines, and non-address lines
                if len(text_line) > 10 and not text_line.isdigit() and not non_address_re.search(text_line):
                    address_parts.append(text_line)
            
            if address_parts:
                return ' '.join(address_parts[:3])  # Take first 3 lines
        
        return None
    
    def _extract_pin_code(self, text: str) -> Optional[str]:
        """Extract 6-digit PIN code."""
        # Normalize
        text = TokenNormalizer.convert_devanagari_to_arabic(text)
        
        # Look for 6 digit number, often at end of address or near keywords
        # Strict: \b\d{6}\b
        matches = re.findall(r'\b(\d{6})\b', text)
        for match in matches:
            # Basic PIN validation (India PIN starts with 1-9)
            if match[0] != '0':
                return match
        return None

    def _extract_enrollment_id(self, text: str) -> Optional[str]:
        """Extract Enrollment ID (EID). Format: 1234/12345/12345"""
        text = TokenNormalizer.convert_devanagari_to_arabic(text)
        match = re.search(r'\b(\d{4}/\d{5}/\d{5})\b', text)
        if match:
            return match.group(1)
        return None
        

    def _extract_gender(self, text: str) -> Optional[str]:
        """Extract gender."""
        # English
        if re.search(r'\bMALE\b', text, re.IGNORECASE):
            return "Male"
        if re.search(r'\bFEMALE\b', text, re.IGNORECASE):
            return "Female"
        if re.search(r'\bTRANSGENDER\b', text, re.IGNORECASE):
            return "Other"
            
        # Hindi (Purush/Mahila)
        if re.search(r'पुरुष', text):
            return "Male"
        if re.search(r'महिला', text):
            return "Female"
            
        return None
        
        """Extract address (simplified)."""
        # ... existing logic or improved ...
        # For now, simplistic capture of text block ?
        # Address usually is a large block.
        # Check for "Address:" keyword
        match = re.search(r'(?:address|pata)\s*[:.-]\s*(.+?)(?:\d{6}|$)', text, re.IGNORECASE | re.DOTALL)
        if match:
            return re.sub(r'\s+', ' ', match.group(1)).strip()
        return None
