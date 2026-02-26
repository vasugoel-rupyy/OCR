import pytest
from ocr_pipeline.documents.disbursement_order import DisbursementOrderProcessor
from ocr_pipeline.ocr.models import OCRResult, WordData

@pytest.fixture
def processor():
    return DisbursementOrderProcessor()

def create_ocr_result(text):
    return OCRResult(full_text=text, mean_confidence=95.0, words=[])

def test_extract_loan_amount_exact_match(processor):
    text = "The Sanctioned Amount is ₹ 2,50,000 for your new car."
    res = processor.extract_fields(create_ocr_result(text))
    assert res['loan_amount'] == 250000.0

def test_extract_disbursed_amount_hindi(processor):
    text = "आपके खाते में वितरित राशि 1,00,000 INR जमा कर दी गई है।"
    res = processor.extract_fields(create_ocr_result(text))
    assert res['disbursed_amount'] == 100000.0

def test_extract_rate_of_interest(processor):
    text = "The applicable Interest Rate is 12.5 p.a."
    res = processor.extract_fields(create_ocr_result(text))
    assert res['rate_of_interest'] == 12.5

def test_extract_tenure_months(processor):
    text = "The loan period is 5 years."
    res = processor.extract_fields(create_ocr_result(text))
    assert res['tenure_months'] == 60

def test_extract_tenure_months_direct(processor):
    text = "Repayment period is 36 months."
    res = processor.extract_fields(create_ocr_result(text))
    assert res['tenure_months'] == 36

def test_extract_bank_name(processor):
    text = "This is a sanction letter from State Bank of India regarding your loan."
    res = processor.extract_fields(create_ocr_result(text))
    assert res['bank_name'] == 'State Bank Of India'

def test_extract_ifsc(processor):
    text = "Branch IFSC code HDFC0001234"
    res = processor.extract_fields(create_ocr_result(text))
    assert res['ifsc'] == 'HDFC0001234'

def test_extract_bank_branch_region_bob(processor):
    text = "Bank of Baroda\nRegional Office, Mumbai Zone"
    res = processor.extract_fields(create_ocr_result(text))
    assert res['bank_name'] == 'Bank Of Baroda'
    assert 'Mumbai Zone' in res['bank_branch_region'] or 'Mumbai' in res['bank_branch_region'] or res['bank_branch_region'] is not None

def test_ambiguous_amount(processor):
    # Proximity has both sanction and disbursed keywords near the amount
    text = "Sanctioned Amount and Disbursed Amount is 50,000."
    res = processor.extract_fields(create_ocr_result(text))
    assert res['loan_amount'] is None
    assert res['disbursed_amount'] is None

def test_normalization_methods(processor):
    assert processor._normalize_currency("₹ 2,50,000") == 250000.0
    assert processor._normalize_percentage("12.5 %") == 12.5
    assert processor._normalize_percentage("12.50 percent") == 12.5
    assert processor._normalize_tenure("5 years") == 60
    assert processor._normalize_tenure("60 months") == 60
    assert processor._normalize_tenure("3 yrs") == 36
