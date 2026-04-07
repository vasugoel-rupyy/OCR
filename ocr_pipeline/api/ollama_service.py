import os
import re
import json
import logging
import httpx
import time
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger("ocr_pipeline.api.ollama")

DEFAULT_OLLAMA_URLS = [
    os.getenv("OLLAMA_URL"),
    "http://ollama-service:11434/api/generate",
    "http://host.docker.internal:11434/api/generate",
    "http://10.50.135.162:11434/api/generate",
    "http://192.168.65.1:11434/api/generate",
    "http://192.168.65.2:11434/api/generate",
    "http://172.17.0.1:11434/api/generate",
    "http://172.18.0.1:11434/api/generate",
    "http://172.19.0.1:11434/api/generate",
    "http://localhost:11434/api/generate"
]

def _fix_ollama_url(url: str) -> str:
    """Helper to ensure Ollama URL includes /api/generate if only base URL provided."""
    if not url: return url
    # If it ends with a port or just a hostname, append the endpoint
    # e.g. http://ollama-service:11434 -> http://ollama-service:11434/api/generate
    if not url.endswith("/api/generate") and not url.endswith("/api/chat"):
        if url.endswith("/"):
            url = url + "api/generate"
        else:
            url = url + "/api/generate"
    return url

OLLAMA_URLS = [_fix_ollama_url(url) for url in DEFAULT_OLLAMA_URLS if url]

DISBURSEMENT_ORDER_PROMPT = """You are an expert document parsing engine.

Your task is to convert noisy OCR logs of emails / disbursement orders into a STRICT structured JSON format.

-----------------------------------
INPUT
-----------------------------------
You will receive OCR text that may contain:
- Logging noise (timestamps, INFO logs, pipeline text)
- Email headers (From, To, Subject, Forwarded message)
- Duplicate or repeated content
- Broken formatting
- Random symbols or spacing issues

You MUST:
1. Ignore all logging noise
2. Reconstruct the actual document content
3. Extract only meaningful financial/disbursement information

-----------------------------------
DOCUMENT TYPE DETECTION
-----------------------------------
If the text contains:
- "Disbursement Details"
- Loan details (Loan A/C, EMI, Tenure, Rate, etc.)

Then:
document_type = "DISBURSEMENT_ORDER"

-----------------------------------
FIELD EXTRACTION RULES
-----------------------------------

Map fields EXACTLY as follows:

loan_amount:
  Extract from: "Gross Loan Amt"

disbursed_amount:
  Extract from: "Net Loan Disb Amt"

rate_of_interest:
  Extract from: "Rate"

tenure_months:
  Extract from: "Tenure (M)"

customer_name:
  Extract from: "Customer Name"

bank_name:
  If NOT explicitly present → null

ifsc:
  If NOT present → null

bank_branch_region:
  If NOT present → null

branch_id:
  If NOT present → null

-----------------------------------
CLEANING RULES
-----------------------------------
- Remove commas, extra spaces
- Convert numbers to plain numeric (no symbols)
- Keep strings clean and trimmed
- Normalize percentages (e.g., "15%" → "15")

-----------------------------------
CONFIDENCE SCORING
-----------------------------------
- 0.9+ → exact match (clear label + value)
- 0.7–0.9 → minor OCR noise
- 0.5–0.7 → inferred
- <0.5 → weak / uncertain

-----------------------------------
DECISION LOGIC
-----------------------------------
decision:
- "APPROVED" → if all critical fields present
- "REVIEW" → if some fields missing OR multiple distinct images/documents detected
- "REJECTED" → if unusable or blank

Critical fields:
- customer_name
- loan_amount
- disbursed_amount
- rate_of_interest
- tenure_months

-----------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
-----------------------------------
Return ONLY valid JSON. No explanation.

{"extracted_fields": {
    "loan_amount": { "value": "", "confidence": 0 },
    "disbursed_amount": { "value": "", "confidence": 0 },
    "rate_of_interest": { "value": "", "confidence": 0 },
    "tenure_months": { "value": "", "confidence": 0 },
    "customer_name": { "value": "", "confidence": 0 },
    "bank_name": { "value": "", "confidence": 0 },
    "ifsc": { "value": "", "confidence": 0 },
    "bank_branch_region": { "value": "", "confidence": 0 },
    "branch_id": { "value": "", "confidence": 0 }
  }}

-----------------------------------
IMPORTANT CONSTRAINTS
-----------------------------------
- DO NOT hallucinate missing fields
- DO NOT invent bank details
- DO NOT output text outside JSON
- ALWAYS include all fields (use null if missing)
- Be deterministic

-----------------------------------
INPUT OCR TEXT:
{raw_text}
"""

class OllamaExtractor:
    # Circuit Breaker State
    _failure_count = 0
    _last_failure_time = 0.0
    _state = "CLOSED"  # CLOSED, OPEN
    _MAX_FAILURES = 3
    _RECOVERY_TIMEOUT = 60.0  # seconds

    @classmethod
    def _check_circuit(cls) -> bool:
        """Returns True if the circuit is CLOSED (allowed) or HALF-OPEN (recovery test)."""
        if cls._state == "OPEN":
            if time.time() - cls._last_failure_time > cls._RECOVERY_TIMEOUT:
                logger.warning("Circuit Breaker entering HALF-OPEN state (testing recovery).")
                return True
            return False
        return True

    @classmethod
    def _record_success(cls):
        cls._failure_count = 0
        cls._state = "CLOSED"

    @classmethod
    def _record_failure(cls):
        cls._failure_count += 1
        cls._last_failure_time = time.time()
        if cls._failure_count >= cls._MAX_FAILURES:
            logger.error(f"Circuit Breaker TRIP to OPEN state after {cls._failure_count} failures.")
            cls._state = "OPEN"

    @staticmethod
    async def extract_disbursement_order(raw_text: str) -> Dict[str, Any]:
        if not OllamaExtractor._check_circuit():
            logger.warning("Circuit Breaker is OPEN. Failing fast for Ollama request.")
            return {
                "error": "circuit_breaker_open",
                "decision": "REVIEW",
                "details": "LLM service is currently unavailable (Circuit Breaker OPEN)"
            }
        
        prompt = DISBURSEMENT_ORDER_PROMPT.replace("{raw_text}", raw_text)
        
        payload = {
            "model": os.getenv("OLLAMA_MODEL", "qwen2.5:1.5B"),
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            response = None
            last_err = None
            
            timeout_config = httpx.Timeout(300.0, connect=2.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                for url in OLLAMA_URLS:
                    try:
                        logger.debug(f"Trying Ollama at {url}")
                        response = await client.post(url, json=payload)
                        response.raise_for_status()
                        break
                    except httpx.HTTPStatusError as e:
                        err_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                        logger.error(f"Ollama API Error at {url}: {err_msg}")
                        last_err = Exception(err_msg)
                        break
                    except httpx.ConnectError as e:
                        last_err = e
                        continue
                        
                if response is None or response.status_code != 200:
                    raise Exception(f"Failed to connect to any Ollama URL. Last error: {str(last_err)}")
                    
                result_json = response.json()
                
            generated_text = result_json.get("response", "").strip()
            
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', generated_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                generated_text = json_match.group(1).strip()
            else:
                start_idx = generated_text.find("{")
                end_idx = generated_text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    generated_text = generated_text[start_idx:end_idx+1]
                
            extracted_data = json.loads(generated_text)
            OllamaExtractor._record_success()
            return extracted_data
            
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse JSON from Ollama. Raw text: {generated_text}")
            return {
                "error": "json_parse_error",
                "details": str(json_err),
                "raw_response": generated_text
            }
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            OllamaExtractor._record_failure()
            return {
                "error": "ollama_api_error",
                "details": str(e)
            }

    @staticmethod
    def extract_disbursement_order_sync(raw_text: str) -> Dict[str, Any]:
        if not OllamaExtractor._check_circuit():
            logger.warning("Circuit Breaker is OPEN (sync). Failing fast.")
            return {
                "error": "circuit_breaker_open",
                "decision": "REVIEW",
                "details": "LLM service unavailable"
            }
        
        prompt = DISBURSEMENT_ORDER_PROMPT.replace("{raw_text}", raw_text)
        
        payload = {
            "model": os.getenv("OLLAMA_MODEL", "qwen2.5:1.5B"),
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            response = None
            last_err = None
            
            for url in OLLAMA_URLS:
                try:
                    logger.debug(f"Trying Ollama at {url} (sync)")
                    response = requests.post(url, json=payload, timeout=(2.0, 300.0))
                    response.raise_for_status()
                    break
                except requests.HTTPError as e:
                        err_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                        logger.error(f"Ollama API Error at {url}: {err_msg}")
                        last_err = Exception(err_msg)
                        break
                except requests.ConnectionError as e:
                    last_err = e
                    continue
                    
            if response is None or response.status_code != 200:
                raise Exception(f"Failed to connect to any Ollama URL. Last error: {str(last_err)}")
                
            result_json = response.json()
                
            generated_text = result_json.get("response", "").strip()
            
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', generated_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                generated_text = json_match.group(1).strip()
            else:
                start_idx = generated_text.find("{")
                end_idx = generated_text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    generated_text = generated_text[start_idx:end_idx+1]
                
            extracted_data = json.loads(generated_text)
            return extracted_data
            
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse JSON from Ollama. Raw text: {generated_text}")
            return {
                "error": "json_parse_error",
                "details": str(json_err),
                "raw_response": generated_text
            }
        except Exception as e:
            logger.error(f"Error calling Ollama API synchronously: {str(e)}")
            OllamaExtractor._record_failure()
            return {
                "error": "ollama_api_error",
                "details": str(e)
            }
