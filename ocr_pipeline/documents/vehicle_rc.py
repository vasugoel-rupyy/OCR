"""Vehicle Registration Certificate (RC) specific field extraction and validation."""

import re
from typing import Dict, Optional, List
from pydantic import Field
from ..ocr.models import OCRResult
from ..validation.normalization import TokenNormalizer
from .base import BaseDocument, FieldValue


class RcDocument(BaseDocument):
    """Structured Vehicle RC document model."""
    registration_number: FieldValue
    owner_name: FieldValue
    vehicle_make_model: Optional[FieldValue] = None
    engine_number: FieldValue
    chassis_number: FieldValue
    registration_date: FieldValue
    vehicle_class: Optional[FieldValue] = None
    fuel_type: Optional[FieldValue] = None
    seating_capacity: Optional[FieldValue] = None
    wheelbase: Optional[FieldValue] = None
    unladen_weight: Optional[FieldValue] = None
    vehicle_color: Optional[FieldValue] = None
    hypothecation: Optional[FieldValue] = None
    fitness_validity_date: Optional[FieldValue] = None
    insurance_validity_date: Optional[FieldValue] = None
    manufacturing_date: Optional[FieldValue] = None
    
    def get_document_type(self) -> str:
        return "vehicle_rc"



class VehicleRCExtractor:
    """Specialized extractor for Vehicle Registration Certificates (RC)."""
    
    def __init__(self):
        """Initialize Vehicle RC extractor."""
        # Known RC keywords
        self.rc_keywords = [
            'registration certificate', 'vehicle', 'registration number',
            'engine no', 'chassis no', 'owner', 'registering authority',
            'रजिस्ट्रेशन', 'वाहन', 'इंजन', 'चेसिस'
        ]
        
        # Indian state codes for vehicle registration
        self.state_codes = [
            'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN', 'GA',
            'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML',
            'MN', 'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TN',
            'TR', 'TS', 'UK', 'UP', 'WB'
        ]
    
    def extract_fields(self, ocr_result: OCRResult) -> Dict[str, any]:
        """Extract all fields from Vehicle RC.
        
        Args:
            ocr_result: OCR result object
            
        Returns:
            Dictionary of extracted fields
        """
        text = ocr_result.full_text
        fields = {}
        
        # Extract registration number (most important)
        reg_number = self._extract_registration_number(text, ocr_result)
        if reg_number:
            fields['registration_number'] = reg_number
            # fields['id_number'] = reg_number  # Alias for compatibility
        else:
            # Hard Reject signal (will be handled by scorer if field is None)
            pass
        
        # Extract owner name
        owner_name = self._extract_owner_name(text, ocr_result)
        if owner_name:
            fields['owner_name'] = owner_name
            fields['name'] = owner_name  # Alias
        
        # Extract vehicle details
        make_model = self._extract_make_model(text)
        if make_model:
            fields['vehicle_make_model'] = make_model
        
        # Extract engine number
        engine_no = self._extract_engine_number(text)
        if engine_no:
            fields['engine_number'] = engine_no
        
        # Extract chassis number
        chassis_no = self._extract_chassis_number(text)
        if chassis_no:
            fields['chassis_number'] = chassis_no
        
        # Extract registration date
        reg_date = self._extract_registration_date(text)
        if reg_date:
            fields['registration_date'] = reg_date
        
        # Extract vehicle class
        vehicle_class = self._extract_vehicle_class(text)
        if vehicle_class:
            fields['vehicle_class'] = vehicle_class
            
        # Extract fuel type
        fuel = self._extract_fuel_type(text)
        if fuel:
            fields['fuel_type'] = fuel
            
        # Extract seating capacity
        seating = self._extract_seating_capacity(text)
        if seating:
            fields['seating_capacity'] = seating
            
        # Extract wheelbase
        wheelbase = self._extract_generic_params(text, ['wheel', 'base', 'wb'], r'(\d{4})')
        if wheelbase:
            fields['wheelbase'] = wheelbase
            
        # Extract unladen weight
        weight = self._extract_generic_params(text, ['unladen', 'ulw', 'wt'], r'(\d{3,5})')
        if weight:
            fields['unladen_weight'] = weight
            
        # Extract color
        color = self._extract_generic_params(text, ['colour', 'color'], r'([A-Z]{3,10})')
        if color:
            fields['vehicle_color'] = color
            
        # Extract hypothecation
        hypothecation = self._extract_hypothecation(text)
        if hypothecation:
            fields['hypothecation'] = hypothecation
            
        # Extract validity dates
        fitness = self._extract_fitness_date(text)
        if fitness:
             fields['fitness_validity_date'] = fitness
             
        insurance = self._extract_insurance_date(text)
        if insurance:
             fields['insurance_validity_date'] = insurance
             
        mfg = self._extract_mfg_date(text)
        if mfg:
             fields['manufacturing_date'] = mfg
            
        return fields
    
    def _extract_registration_number(self, text: str, ocr_result: OCRResult) -> Optional[str]:
        """Extract vehicle registration number.
        
        Strict mode: Returns None if multiple conflicting registration numbers are found.
        """
        # Collect all candidates
        candidates = set()
        
        # Strategy 1: Standard format with hyphens or spaces
        pattern1 = r'\b([A-Z]{2})\s*[-]?\s*(\d{2})\s*[-]?\s*([A-Z]{1,2})\s*[-]?\s*(\d{4})\b'
        matches = re.findall(pattern1, text.upper())
        
        for match in matches:
            reg_num = ''.join(match)
            if self._validate_registration_number(reg_num):
                 candidates.add(f"{match[0]}-{match[1]}-{match[2]}-{match[3]}")
        
        # Strategy 2: Continuous format (no separators)
        pattern2 = r'\b([A-Z]{2}\d{2}[A-Z]{1,2}\d{4})\b'
        matches = re.findall(pattern2, text.upper())
        for match in matches:
            if self._validate_registration_number(match):
                # Standardize format
                state = match[:2]
                rto = match[2:4]
                series_end = 4
                while series_end < len(match) and match[series_end].isalpha():
                    series_end += 1
                series = match[4:series_end]
                number = match[series_end:]
                candidates.add(f"{state}-{rto}-{series}-{number}")
        
        # Strict enforcement
        if len(candidates) > 1:
            return None # Reject due to ambiguity
        if len(candidates) == 1:
            return list(candidates)[0]
            
        return None
    
    def _extract_from_words(self, words: List) -> Optional[str]:
        """Extract registration number from word-level OCR data.
        
        Args:
            words: List of WordData objects
            
        Returns:
            Registration number or None
        """
        # Look for sequences matching registration pattern
        for word in words:
            cleaned = word.text.upper().strip()
            cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
            
            # Check if it matches registration pattern
            if re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', cleaned):
                if self._validate_registration_number(cleaned):
                    # Format with hyphens
                    state = cleaned[:2]
                    rto = cleaned[2:4]
                    series_end = 4
                    while series_end < len(cleaned) and cleaned[series_end].isalpha():
                        series_end += 1
                    series = cleaned[4:series_end]
                    number = cleaned[series_end:]
                    return f"{state}-{rto}-{series}-{number}"
        
        return None
    
    def _validate_registration_number(self, reg_num: str) -> bool:
        """Validate registration number format.
        
        Args:
            reg_num: Registration number string (without hyphens)
            
        Returns:
            True if valid format
        """
        # Remove any separators
        reg_num = re.sub(r'[\s-]+', '', reg_num)
        
        # Check basic format
        if not re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', reg_num):
            return False
        
        # Validate state code
        state_code = reg_num[:2]
        if state_code not in self.state_codes:
            return False
        
        # RTO code should be 01-99
        try:
            rto_code = int(reg_num[2:4])
            if not (1 <= rto_code <= 99):
                return False
        except ValueError:
            return False
        
        return True
    
    def _extract_owner_name(self, text: str, ocr_result: OCRResult) -> Optional[str]:
        """Extract owner name from RC.
        
        Args:
            text: Full OCR text
            ocr_result: OCR result
            
        Returns:
            Owner name or None
        """
        # Look for owner name after keywords
        owner_patterns = [
            r'(?:owner|owner\'?s?\s+name|registered\s+owner)\s*:?\s*([A-Z][A-Za-z\s]{3,50})',
            r'(?:name|नाम)\s*:?\s*([A-Z][A-Za-z\s]{3,50})',
        ]
        
        for pattern in owner_patterns:
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
            'registration', 'certificate', 'vehicle', 'engine', 'chassis',
            'authority', 'date', 'class', 'model', 'make'
        ]
        
        name_lower = name.lower()
        for keyword in invalid_keywords:
            if keyword in name_lower:
                return False
        
        # Must have at least 2 words
        words = name.split()
        if len(words) < 2:
            return False
        
        # Each word should be mostly alphabetic
        for word in words:
            if not word.isalpha() or len(word) < 2:
                return False
        
        return True
    
    def _extract_make_model(self, text: str) -> Optional[str]:
        """Extract vehicle make and model.
        
        Args:
            text: Full OCR text
            
        Returns:
            Make and model or None
        """
        # Look for make/model after keywords
        make_patterns = [
            r'(?:make|maker|manufacturer)\s*:?\s*([A-Za-z0-9\s]{3,30})',
            r'(?:model)\s*:?\s*([A-Za-z0-9\s]{3,30})',
        ]
        
        for pattern in make_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                make_model = match.group(1).strip()
                make_model = re.sub(r'\s+', ' ', make_model)
                if len(make_model) >= 3:
                    return make_model
        
        return None
    
    def _extract_engine_number(self, text: str) -> Optional[str]:
        """Extract engine number.
        
        Args:
            text: Full OCR text
            
        Returns:
            Engine number or None
        """
        # Look for engine number after keywords
        engine_patterns = [
            r'(?:engine\s+(?:no|number)|e\s*no)\s*:?\s*([A-Z0-9]{6,20})',
        ]
        
        for pattern in engine_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                engine_no = match.group(1).strip()
                # Engine numbers are typically alphanumeric, 6-20 characters
                if re.match(r'^[A-Z0-9]{6,20}$', engine_no.upper()):
                    return engine_no.upper()
        
        return None
    
    def _extract_chassis_number(self, text: str) -> Optional[str]:
        """Extract chassis/VIN number.
        
        Args:
            text: Full OCR text
            
        Returns:
            Chassis number or None
        """
        # Look for chassis number after keywords
        chassis_patterns = [
            r'(?:chassis\s+(?:no|number)|c\s*no|vin)\s*:?\s*([A-Z0-9]{10,20})',
        ]
        
        for pattern in chassis_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                chassis_no = match.group(1).strip()
                # Chassis numbers are typically alphanumeric, 10-20 characters
                if re.match(r'^[A-Z0-9]{10,20}$', chassis_no.upper()):
                    return chassis_no.upper()
        
        return None
    
    def _extract_registration_date(self, text: str) -> Optional[str]:
        """Extract registration date.
        
        Args:
            text: Full OCR text
            
        Returns:
            Registration date or None
        """
        # Look for registration date
        date_patterns = [
            r'(?:registration\s+date|reg\s*date|date\s+of\s+registration)\s*:?\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})',
            r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})',  # Any date format
        ]
        
        for pattern in date_patterns:
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
            if not (1950 <= year <= 2024):
                return False
            
            return True
        except ValueError:
            return False
    
    def _extract_fuel_type(self, text: str) -> Optional[str]:
        """Extract fuel type."""
        fuel_types = ['PETROL', 'DIESEL', 'CNG', 'LPG', 'ELECTRIC', 'HYBRID', 'PETRO']
        
        # Strategy 1: Look for pattern
        match = re.search(r'(?:fuel|propulsion)\s*:?\s*([A-Za-z]+)', text, re.IGNORECASE)
        if match:
            val = match.group(1).upper()
            if any(f in val for f in fuel_types):
                return val
        
        # Strategy 2: Scan for fuel keywords directly
        for f in fuel_types:
            if re.search(r'\b' + f + r'\b', text.upper()):
                return f
        return None

    def _extract_seating_capacity(self, text: str) -> Optional[str]:
        """Extract seating capacity."""
        match = re.search(r'(?:seating|cap|seat)\s*(?:cap)?\s*[:.]?\s*(\d{1,2})', text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
        
    def _extract_generic_params(self, text: str, keywords: List[str], value_pattern: str) -> Optional[str]:
        """Generic extraction for key-value pairs."""
        # Join keywords with |
        kw_regex = '|'.join(keywords)
        pattern = r'(?:' + kw_regex + r')\s*[:.-]?\s*' + value_pattern
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
             return match.group(1)
        return None

    def _extract_vehicle_class(self, text: str) -> Optional[str]:
        """Extract vehicle class (e.g., MCWG, LMV, HMV).
        
        Args:
            text: Full OCR text
            
        Returns:
            Vehicle class or None
        """
        # Common vehicle classes in India
        vehicle_classes = [
            'MCWG', 'MCWOG', 'LMV', 'LMV-NT', 'HMV', 'HTV', 'MGV', 'LGV',
            'PSV', 'HPMV', 'HGMV', 'TRANS'
        ]
        
        # Look for vehicle class
        class_pattern = r'(?:vehicle\s+class|class)\s*:?\s*([A-Z-]{2,10})'
        match = re.search(class_pattern, text, re.IGNORECASE)
        if match:
            vehicle_class = match.group(1).upper()
            if vehicle_class in vehicle_classes:
                return vehicle_class
        
        # Direct search for known classes
        for vc in vehicle_classes:
            if re.search(r'\b' + vc + r'\b', text.upper()):
                return vc
        
        return None
        
    def _extract_hypothecation(self, text: str) -> Optional[str]:
        """Extract financing bank (Hypothecation)."""
        # Search for "HPA" or "Hypothecated"
        match = re.search(r'(?:hypothecation|hypothecated|financed|hpa|hp)\s*(?:by|to|with)?\s*[:.-]?\s*([A-Z0-9\s.,&]+)', text, re.IGNORECASE)
        if match:
             val = match.group(1).strip()
             if len(val) > 3: return val
        return None
        
    def _extract_fitness_date(self, text: str) -> Optional[str]:
        """Extract Fitness Valid Until."""
        match = re.search(r'(?:fitness|fit)\s*(?:valid|upto)?\s*[:.-]?\s*(\d{2}[/.-]\d{2}[/.-]\d{4})', text, re.IGNORECASE)
        if match: return TokenNormalizer.normalize_date(match.group(1))
        return None

    def _extract_insurance_date(self, text: str) -> Optional[str]:
        """Extract Insurance Valid Until."""
        match = re.search(r'(?:insurance|ins)\s*(?:valid|upto)?\s*[:.-]?\s*(\d{2}[/.-]\d{2}[/.-]\d{4})', text, re.IGNORECASE)
        if match: return TokenNormalizer.normalize_date(match.group(1))
        return None

    def _extract_mfg_date(self, text: str) -> Optional[str]:
        """Extract Mfg Date (Month/Year)."""
        match = re.search(r'(?:mfg|manufacturing)\s*(?:date)?\s*[:.-]?\s*(\d{2}[/.-]\d{4}|\d{4})', text, re.IGNORECASE)
        if match: return match.group(1) # Often just MM/YYYY, leave as is? normalize?
        return None
