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
        
        # Extract Aadhaar number
        aadhaar_number = self._extract_aadhaar_number(text, ocr_result)
        if aadhaar_number:
            fields['aadhaar_number'] = aadhaar_number
        
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
        pattern1 = r'\b(\d{4})[\s.-]+(\d{4})[\s.-]+(\d{4})\b'
        matches = re.findall(pattern1, text)
        for match in matches:
            aadhaar = ''.join(match)
            if self._validate_aadhaar(aadhaar):
                return aadhaar
        
        pattern2 = r'\b(\d{12})\b'
        matches = re.findall(pattern2, text)
        for match in matches:
            if self._validate_aadhaar(match):
                return match
                
        if ocr_result.words:
            aadhaar = self._extract_from_words(ocr_result.words)
            if aadhaar:
                return aadhaar
        
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
        digit_words = []
        for word in words:
            cleaned = re.sub(r'[^\d]', '', word.text)
            if len(cleaned) == 4:
                digit_words.append(cleaned)
        
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
        number = TokenNormalizer.convert_devanagari_to_arabic(number)
        
        if not number.isdigit() or len(number) != 12:
            return False
        
        if number[0] in ['0', '1']:
            return False
        
        return True

    def _extract_vid(self, text: str) -> Optional[str]:
        """Extract VID (Virtual ID) if present.
        
        Args:
            text: Full OCR text
            
        Returns:
            VID or None
        """
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
        keyword_pattern = r'(?:name|नाम|नामा|नाभ)\s*[:.-]?\s*([A-Za-z \t]{3,50})'
        for match in re.finditer(keyword_pattern, text, re.IGNORECASE | re.UNICODE):
            name = re.sub(r'[@:.,]', ' ', match.group(1))
            name = re.sub(r'\s+', ' ', name).strip()
            if self._is_valid_name(name):
                return name

        if ocr_result.lines:
            for i in range(1, min(len(ocr_result.lines), 5)):
                line = ocr_result.lines[i]
                text_line = line.text.strip()
                name_cand = re.sub(r'([a-z])([A-Z])', r'\1 \2', text_line)
                name_cand = re.sub(r'\s+', ' ', name_cand).strip()
                
                if self._is_valid_name(name_cand):
                    if i + 1 < len(ocr_result.lines):
                        next_line_text = ocr_result.lines[i+1].text.strip()
                        next_name_part = re.sub(r'([a-z])([A-Z])', r'\1 \2', next_line_text)
                        next_name_part = re.sub(r'\s+', ' ', next_name_part).strip()
                        
                        if (self._is_valid_name(next_name_part) and 
                            not any(kw in next_line_text.lower() for kw in ['dob', 'address', 'pata', 'birth', 'gender'])):
                            return f"{name_cand} {next_name_part}"
                    
                    return name_cand

        global_pattern = r'([A-Z][A-Za-z]{2,}(?:[ \t@:.,]+[A-Z][A-Za-z]{1,})*)'
        candidates = []
        for match in re.finditer(global_pattern, text):
            name = re.sub(r'[@:.,]', ' ', match.group(0))
            name = re.sub(r'\s+', ' ', name).strip()
            if self._is_valid_name(name):
                candidates.append((name, match.start()))
        
        if candidates:
            return candidates[0][0]
        
        return None
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if extracted text looks like a valid name.
        
        Args:
            name: Potential name string
            
        Returns:
            True if looks like a name
        """
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
        for prefix in ['s/o', 'd/o', 'w/o', 'c/o', 'son of', 'daughter of']:
            if name_lower.startswith(prefix):
                return False

        for keyword in invalid_keywords:
            if keyword in name_lower:
                return False
        
        if len(name) < 3:
            return False

        words = name.split()
        if len(words) == 0:
            return False
            
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
        
        dob_pattern = r'(?:dob|date\s+of\s+birth|yob|year\s+of\s+birth)\s*[:.-]?\s*(\d{2}/\d{2}/\d{4}|\d{4})'
        
        dob_patterns = [
            r'(?:dob|d0b|date\s+of\s+birth|जन्म\s+तिथि|dop)(?:[/\s\w]*)\s*[:.-]?\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})',
            r'(?:dob|d0b|date\s+of\s+birth|जन्म\s+तिथि|dop)(?:[/\s\w]*)\s*[:.-]?\s*(\d{8})',
            r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})',
        ]
        
        for pattern in dob_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                if len(date_str) == 8 and date_str.isdigit():
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
        if not re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', date_str):
            return False
        
        parts = re.split(r'[/-]', date_str)
        if len(parts) != 3:
            return False
        
        try:
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            
            if not (1 <= day <= 31):
                return False
            if not (1 <= month <= 12):
                return False
            if year < 100:
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
        address_pattern = r'(?:address|पता)\s*:?\s*(.{20,200})'
        match = re.search(address_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            address = match.group(1).strip()
            address = re.sub(r'\s+', ' ', address)
            return address[:200]
        
        if ocr_result.lines and len(ocr_result.lines) > 5:
            bottom_lines = ocr_result.lines[len(ocr_result.lines)//2:]
            address_parts = []
            non_address_re = re.compile(
                r'\b(VID|MALE|FEMALE|TRANSGENDER|aadhaar|आधार|पुरुष|महिला)\b'
                r'|\d{4}\s*\d{4}\s*\d{4}'
                r'|\d{16}',
                re.IGNORECASE
            )
            for line in bottom_lines:
                text_line = line.text.strip()
                if len(text_line) > 10 and not text_line.isdigit() and not non_address_re.search(text_line):
                    address_parts.append(text_line)
            
            if address_parts:
                return ' '.join(address_parts[:3])
        
        return None
    
    def _extract_pin_code(self, text: str) -> Optional[str]:
        """Extract 6-digit PIN code."""
        text = TokenNormalizer.convert_devanagari_to_arabic(text)
        
        matches = re.findall(r'\b(\d{6})\b', text)
        for match in matches:
            if match[0] != '0':
                return match
        return None

    def _extract_enrollment_id(self, text: str) -> Optional[str]:
        text = TokenNormalizer.convert_devanagari_to_arabic(text)
        match = re.search(r'\b(\d{4}/\d{5}/\d{5})\b', text)
        if match:
            return match.group(1)
        return None
        

    def _extract_gender(self, text: str) -> Optional[str]:
        if re.search(r'\bMALE\b', text, re.IGNORECASE):
            return "Male"
        if re.search(r'\bFEMALE\b', text, re.IGNORECASE):
            return "Female"
        if re.search(r'\bTRANSGENDER\b', text, re.IGNORECASE):
            return "Other"
            
        if re.search(r'पुरुष', text):
            return "Male"
        if re.search(r'महिला', text):
            return "Female"
            
        return None
        
