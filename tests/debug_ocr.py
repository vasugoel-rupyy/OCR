import sys
from ocr_pipeline.core.pipeline import OCRPipeline

pipeline = OCRPipeline()
result = pipeline.process_document(sys.argv[1], document_type='disbursement_order')

print("--- FULL TEXT ---")
print(result.full_text)
print("--- EXTRACTED FIELDS ---")
for k, v in result.extracted_fields.items():
    print(f"{k}: {v}")

