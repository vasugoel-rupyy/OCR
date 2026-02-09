# OCR Pipeline

A high-performance, self-hosted OCR system designed to extract structured data from Indian Identity Documents (Aadhaar, PAN, Vehicle RC). It utilizes a multi-layered approach involving image preprocessing, document detection, dual-pass OCR, and a 10-component validation system.

## Features

### 🧠 Core Orchestration
- **Structured Data Extraction**: Returns strictly typed Pydantic models for Aadhaar, PAN, and RC documents.
- **Robust Fallback OCR**: Uses advanced preprocessing and full-text OCR when template matching is disabled or fails.
- **Multilingual Support**: Supports English and Hindi/Devanagari text extraction with specialized numeral normalization.
- **Document Classification**: Automatically identifies document types based on keyword frequency and spatial patterns.

### 📉 10-Component Scoring Model
The system explains its decisions via a weighted confidence score [0-1] based on:
1.  **Image Quality**: Blur, brightness, and contrast checks.
2.  **OCR Confidence**: Character-level metadata from PaddleOCR.
3.  **Regex Pattern Match**: Verification against strict document formats.
4.  **Fuzzy Matching**: Anchor word detection (e.g., "Father's Name").
5.  **Layout Analysis**: Verification of physical field locations.
6.  **Key-Value Pair Proximity**: Spatial relationship between keys and values.
7.  **Cross-Field Consistency**: Logical checks between related fields.
8.  **Schema Compliance**: Ensuring all mandatory fields are found.
9.  **Token Distribution**: Analysis of numeric vs. alphabetic ratios.
10. **Spatial Compactness**: Prevents cross-region mixing of fields.

### 🖼️ Image Intelligence
- **Quality Gate**: Rejects blurry or poorly lit images before heavy processing.
- **Auto-Deskewing**: Hough line transform to correct rotated documents.
- **ID Enhancer**: Specialized filters to sharpen small fonts and improve contrast.

## Requirements

- Python 3.8+ (Python 3.13+ supported with compatibility shims)
- See `requirements.txt` for full dependency list

## Quick Start
<... same installation instructions ...>

## API Response Format

The API now returns a `structured_document` field containing the fully typed extraction result.

```json
{
  "document_path": "/tmp/tmpxyz.jpg",
  "document_type": "aadhaar",
  "decision": "accept",
  "confidence": {
    "final_score": 0.92,
    "ocr_confidence_score": 0.95,
    "regex_score": 1.0,
    ...
  },
  "structured_document": {
    "aadhaar_number": {
      "value": "1234 5678 9101",
      "confidence": 0.98
    },
    "name": {
      "value": "Jane Doe",
      "confidence": 0.95
    },
    "date_of_birth": {
      "value": "01/01/1980",
      "confidence": 0.99
    },
    "gender": {
      "value": "Female",
      "confidence": 0.99
    },
    "address": {
      "value": "123 Main St, Delhi",
      "confidence": 0.85
    },
    "overall_confidence": 0.92,
    "decision": "accept"
  },
  "processing_time": 2.45
}
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'imghdr'** (Python 3.13+)
   - This is automatically handled by the compatibility shim. Ensure you're importing `ocr_pipeline` before any paddleocr imports.

2. **Import errors after refactoring**
   - Make sure you're using the new import paths: `from ocr_pipeline import OCRPipeline`

3. **Configuration file not found**
   - Ensure `config.yaml` is in the project root directory.

## License

MIT License

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- Tests are added for new features
- Documentation is updated accordingly
