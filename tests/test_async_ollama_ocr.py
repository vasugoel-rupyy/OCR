import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import io
import os

from ocr_pipeline.api.server import app
from ocr_pipeline.api.models import OCRResponse

client = TestClient(app)

@pytest.fixture
def mock_pipeline_core():
    with patch('ocr_pipeline.api.server._run_pipeline_core', new_callable=AsyncMock) as mock:
        response = OCRResponse(
            status="success",
            document_type="disbursement_order",
            decision="REVIEW",
            confidence_score=0.85,
            reason="-",
            extracted_fields={"loan_amount": 1000},
            processing_time=1.5,
            extraction_method="ocr",
            raw_ocr_text="DUMMY RAW TEXT"
        )
        mock.return_value = response
        yield mock

@pytest.fixture
def mock_ollama_extractor():
    with patch('ocr_pipeline.api.server.OllamaExtractor.extract_disbursement_order', new_callable=AsyncMock) as mock:
        mock.return_value = {
            "extracted_fields": {
                "loan_amount": {"value": 50000, "confidence": 0.95},
                "disbursed_amount": {"value": 49000, "confidence": 0.95}
            },
            "decision": "APPROVED"
        }
        yield mock

@pytest.fixture
def mock_httpx_post():
    with patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock:
        yield mock

def test_process_file_with_webhook(mock_pipeline_core, mock_httpx_post):
    file_content = b"dummy image data"
    file_obj = io.BytesIO(file_content)
    
    # 1. API Call
    response = client.post(
        "/ocr/process_file",
        files={"file": ("test.jpg", file_obj, "image/jpeg")},
        data={"document_type": "disbursement_order", "webhook_url": "http://test-webhook.com"}
    )
    
    # 2. Check sync response
    assert response.status_code == 202
    assert response.json()["status"] == "processing"
    
    # 3. Verify pipeline was called
    mock_pipeline_core.assert_called_once()
    
    # 4. Verify Webhook post was called
    mock_httpx_post.assert_called_once()
    
    # Check what was sent to the webhook
    args, kwargs = mock_httpx_post.call_args
    assert args[0] == "http://test-webhook.com"
    
    payload = kwargs.get('json', {})
    assert payload['status'] == 'success'
    assert payload['extraction_method'] == 'ocr'
    assert payload['decision'] == 'REVIEW'
    assert payload['extracted_fields']['loan_amount'] == 1000
