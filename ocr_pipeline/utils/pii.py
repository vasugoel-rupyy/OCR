import re

def mask_aadhaar(text: str) -> str:
    """
    Mask Aadhaar numbers: Keep only last 4 digits.
    Example: 1234 5678 9012 -> XXXX XXXX 9012
    """
    def _repl(match):
        digits = re.sub(r'[\s-]', '', match.group(0))
        if len(digits) == 12:
            return f"XXXX XXXX {digits[-4:]}"
        return match.group(0)

    # Match 12-digit numbers with optional spaces/hyphens
    return re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', _repl, text)

def mask_pan(text: str) -> str:
    """
    Mask PAN numbers: Keep only digits and last letter (partially).
    Actually, standard is masking first 5 and last character.
    Example: ABCDE1234F -> XXXXX1234X
    """
    def _repl(match):
        pan = match.group(0)
        if len(pan) == 10:
            return f"XXXXX{pan[5:9]}X"
        return pan

    # PAN format: 5 letters, 4 digits, 1 letter
    return re.sub(r'\b[A-Z]{5}\d{4}[A-Z]\b', _repl, text)

def mask_pii(text: str) -> str:
    """
    Apply all PII masking rules to text.
    """
    if not text:
        return text
    
    text = mask_aadhaar(text)
    text = mask_pan(text)
    return text
