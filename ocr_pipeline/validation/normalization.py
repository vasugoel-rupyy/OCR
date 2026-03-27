import re
from typing import List, Optional

class TokenNormalizer:
    
    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        return " ".join(text.split())

    @staticmethod
    def normalize_numeric_field(text: str) -> str:
        if not text:
            return ""
        
        replacements = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1', 'i': '1',
            'S': '5', 's': '5',
            'B': '8',
            'G': '6',
            'Z': '2', 'z': '2'
        }
        
        normalized = text
        for char, repl in replacements.items():
            normalized = normalized.replace(char, repl)
            
        normalized = re.sub(r'[^0-9.,-]', '', normalized)
        
        return normalized

    @staticmethod
    def convert_devanagari_to_arabic(text: str) -> str:
        if not text:
            return ""
        devanagari_digits = str.maketrans("०१२३४५६७८९", "0123456789")
        return text.translate(devanagari_digits)

    @staticmethod
    def normalize_date(date_str: str) -> Optional[str]:
        if not date_str:
            return None
            
        clean_date = re.sub(r'[^\d/\-\.]', '', date_str)
        
        dmy_match = re.match(r'^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$', clean_date)
        if dmy_match:
            d, m, y = dmy_match.groups()
            return f"{int(d):02d}/{int(m):02d}/{y}"
            
        ymd_match = re.match(r'^(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})$', clean_date)
        if ymd_match:
            y, m, d = ymd_match.groups()
            return f"{int(d):02d}/{int(m):02d}/{y}"
            
        return None

    @staticmethod
    def standardize_date(date_str: str) -> Optional[str]:
        if not date_str:
            return None
            
        clean_date = re.sub(r'[./]', '-', date_str)
        
        try:
            from datetime import datetime
            from dateutil import parser
            dt = parser.parse(clean_date)
            return dt.strftime('%Y-%m-%d')
        except:
            return None
