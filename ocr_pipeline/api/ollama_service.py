import os
import re
import json
import logging
import httpx
import requests
from typing import Dict, Any

logger = logging.getLogger("ocr_pipeline.api.ollama")

# Possible URLs to reach Ollama from within a Docker container or natively
DEFAULT_OLLAMA_URLS = [
    os.getenv("OLLAMA_URL"),
    "http://ollama-service:11434/api/generate", # Hits the new docker-compose sidecar!
    "http://host.docker.internal:11434/api/generate",
    "http://10.50.135.162:11434/api/generate",
    "http://192.168.65.1:11434/api/generate",
    "http://192.168.65.2:11434/api/generate",
    "http://172.17.0.1:11434/api/generate",
    "http://172.18.0.1:11434/api/generate",
    "http://172.19.0.1:11434/api/generate",
    "http://localhost:11434/api/generate"
]
OLLAMA_URLS = [url for url in DEFAULT_OLLAMA_URLS if url]

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
- "REVIEW" → if some fields missing
- "REJECTED" → if unusable

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
    @staticmethod
    async def extract_disbursement_order(raw_text: str) -> Dict[str, Any]:
        """Extracts fields from a Disbursement Order using the Qwen model via Ollama."""
        prompt = DISBURSEMENT_ORDER_PROMPT.replace("{raw_text}", raw_text)
        
        payload = {
            "model": os.getenv("OLLAMA_MODEL", "qwen3.5:0.8B"),
            "prompt": prompt,
            "stream": False,
            "format": "json"  # Hints ollama to return JSON
        }
        
        try:
            # Try to find a working URL
            response = None
            last_err = None
            
            # Use 2.0s connect timeout, 300s read timeout
            timeout_config = httpx.Timeout(300.0, connect=2.0)
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                for url in OLLAMA_URLS:
                    try:
                        logger.debug(f"Trying Ollama at {url}")
                        response = await client.post(url, json=payload)
                        response.raise_for_status()
                        break # Success
                    except httpx.HTTPStatusError as e:
                        err_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                        logger.error(f"Ollama API Error at {url}: {err_msg}")
                        last_err = Exception(err_msg)
                        break # Stop trying other URLs if we actually connected but got an API error
                    except httpx.ConnectError as e:
                        last_err = e
                        continue
                        
                if response is None or response.status_code != 200:
                    raise Exception(f"Failed to connect to any Ollama URL. Last error: {str(last_err)}")
                    
                result_json = response.json()
                
            generated_text = result_json.get("response", "").strip()
            
            # Extract JSON block
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', generated_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                generated_text = json_match.group(1).strip()
            else:
                # Fallback: find first { and last }
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
            logger.error(f"Error calling Ollama API: {str(e)}")
            return {
                "error": "ollama_api_error",
                "details": str(e)
            }

    @staticmethod
    def extract_disbursement_order_sync(raw_text: str) -> Dict[str, Any]:
        """Extracts fields from a Disbursement Order using the Qwen model synchronously."""
        prompt = DISBURSEMENT_ORDER_PROMPT.replace("{raw_text}", raw_text)
        
        payload = {
            "model": os.getenv("OLLAMA_MODEL", "qwen3.5:0.8B"),
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
                    # 2s connect timeout, 300s read timeout
                    response = requests.post(url, json=payload, timeout=(2.0, 300.0))
                    response.raise_for_status()
                    break # Success
                except requests.HTTPError as e:
                        err_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                        logger.error(f"Ollama API Error at {url}: {err_msg}")
                        last_err = Exception(err_msg)
                        break # Stop trying other URLs if we actually connected but got an API error
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
            return {
                "error": "ollama_api_error",
                "details": str(e)
            }
