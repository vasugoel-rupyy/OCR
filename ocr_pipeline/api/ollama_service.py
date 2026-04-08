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
  Look for: "Gross Loan Amt", "Sanctioned Amt", "Total Loan", or large currency values near "Car Loan" or "Term Loan". Default to null if not found.
  
disbursed_amount:
  Look for: "Net Loan Disb Amt", "Disbursement Amount", "Payment to dealer", or values near "Disbursement". Default to null if not found.

rate_of_interest:
  Look for: "Rate", "ROI", "Interest Rate", or percentages (e.g., 8.75%). Normalize to numeric. Default to null if not found.

tenure_months:
  Look for: "Tenure", "Period", "Months", or "60", "84", etc. near maturity. Default to null if not found.

customer_name:
  Look for: "Customer Name", "Name of Borrower", or names starting with MR/MRS/MS. Default to null if not found.

bank_name:
  Extract the entity name (e.g., "Bank of Baroda", "HDFC BANK"). Default to null if not found.

ifsc:
  Extract the 11-digit alphanumeric IFSC code. Default to null if not found.

bank_branch_region:
  Look for: Branch address, locality, or city labels.
  SPECIAL CASE (Bank of Baroda): Usually found on the 2nd/3rd line after the bank name. Default to null if not found.

branch_id:
  Look for: Alpha-numeric branch codes or IDs.
  SPECIAL CASE (Bank of Baroda): Usually found on the 4th/5th line after the bank name. Default to null if not found.

-----------------------------------
CLEANING RULES
-----------------------------------
- Remove commas, extra spaces, currency symbols (₹, Rs, /-)
- Convert numbers to plain numeric strings
- Keep strings clean and trimmed
- Normalize percentages (e.g., "15%" → "15")

-----------------------------------
CONFIDENCE SCORING
-----------------------------------
- 0.9+ → exact match (clear label + value)
- 0.7–0.9 → minor OCR noise
- 0.5–0.7 → inferred from context/relative position
- <0.5 → weak / uncertain

-----------------------------------
DECISION LOGIC
-----------------------------------
decision:
- "APPROVED" → if all critical fields present and high confidence
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
    _failure_count = 0
    _last_failure_time = 0.0
    _state = "CLOSED"   
    _MAX_FAILURES = 3
    _RECOVERY_TIMEOUT = 60.0 

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
