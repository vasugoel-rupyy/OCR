import pytest
from ocr_pipeline.ocr.models import OCRResult, WordData, LineData
from ocr_pipeline.documents.aadhaar import AadhaarExtractor

@pytest.fixture
def extractor():
    return AadhaarExtractor()

def create_ocr_result(full_text, words_data):
    words = []
    for i, (text, conf) in enumerate(words_data):
        words.append(WordData(text, conf, (0, i*10, 10, 10), i, 0))
    
    # Mocking lines - one word per line for simplicity in some tests, or actual lines
    lines = []
    # For simplicity, we'll just use the full_text for matching and few words
    return OCRResult(
        full_text=full_text,
        mean_confidence=95.0,
        words=words,
        lines=[],
        total_words=len(words)
    )

def test_single_word_name(extractor):
    full_text = "Government of India\nAnjali\nDOB: 01/01/2000"
    words = [("Government", 99), ("of", 99), ("India", 99), ("Anjali", 95), ("DOB", 99)]
    ocr_result = create_ocr_result(full_text, words)
    fields = extractor.extract_fields(ocr_result)
    assert fields.get('name') == 'Anjali'

def test_all_caps_single_word_name(extractor):
    full_text = "GOVERNMENT OF INDIA\nANJALI\nDOB: 01/01/2000"
    words = [("GOVERNMENT", 99), ("OF", 99), ("INDIA", 99), ("ANJALI", 95), ("DOB", 99)]
    ocr_result = create_ocr_result(full_text, words)
    fields = extractor.extract_fields(ocr_result)
    assert fields.get('name') == 'ANJALI'

def test_two_word_name(extractor):
    full_text = "Government of India\nMukesh Kumar\nDOB: 01/01/1980"
    words = [("Government", 99), ("of", 99), ("India", 99), ("Mukesh", 95), ("Kumar", 95)]
    ocr_result = create_ocr_result(full_text, words)
    fields = extractor.extract_fields(ocr_result)
    assert fields.get('name') == 'Mukesh Kumar'

def test_name_with_keyword(extractor):
    full_text = "UIDAI\nName: Anjali Kumari\nDOB: 01/01/2000"
    words = [("UIDAI", 99), ("Name:", 99), ("Anjali", 95), ("Kumari", 95)]
    ocr_result = create_ocr_result(full_text, words)
    fields = extractor.extract_fields(ocr_result)
    assert fields.get('name') == 'Anjali Kumari'

def test_prevent_parentage_name(extractor):
    # Here Anjali is the name, Mukesh Kumar is the father's name (S/O)
    full_text = "Government of India\nAnjali\nDOB: 01/01/2000\nFemale\nAddress:\nS/O: Mukesh Kumar"
    words = [("Anjali", 95), ("S/O:", 99), ("Mukesh", 99), ("Kumar", 99)]
    ocr_result = create_ocr_result(full_text, words)
    fields = extractor.extract_fields(ocr_result)
    # It should pick Anjali because it's earlier than Mukesh Kumar and Mukesh Kumar is preceded by S/O
    assert fields.get('name') == 'Anjali'

def test_invalid_keyword_filtering(extractor):
    full_text = "Government of India\nUnique Identification Authority\nAnjali"
    words = [("Government", 99), ("India", 99), ("Unique", 99), ("Anjali", 95)]
    ocr_result = create_ocr_result(full_text, words)
    fields = extractor.extract_fields(ocr_result)
    assert fields.get('name') == 'Anjali'

def test_split_name_due_to_skew(extractor):
    # Name is split across two lines
    full_text = "Government of India\nAnjali\nKumari\nDOB: 01/01/2000"
    words = [("Anjali", 95), ("Kumari", 95)]
    lines = [
        LineData("Government of India", 99, (0,0,10,10)),
        LineData("Anjali", 95, (0,10,10,10)),
        LineData("Kumari", 95, (0,20,10,10)),
        LineData("DOB: 01/01/2000", 99, (0,30,10,10))
    ]
    ocr_result = OCRResult(
        full_text=full_text,
        mean_confidence=95.0,
        words=[], # Words not used by Strategy 2
        lines=lines,
        total_words=4
    )
    fields = extractor.extract_fields(ocr_result)
    assert fields.get('name') == 'Anjali Kumari'
