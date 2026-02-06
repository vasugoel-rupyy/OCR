import sys
import os

# Add project root to path
sys.path.append('/home/vasugoel/OCR')

try:
    print("Attempting to import OCRPipeline from core...")
    from ocr_pipeline.core.pipeline import OCRPipeline
    print("SUCCESS: OCRPipeline imported.")
    
    print("Attempting to import TemplatePipeline...")
    from ocr_pipeline.templates.pipeline import TemplatePipeline
    print("SUCCESS: TemplatePipeline imported.")
    
    print("All modules imported successfully.")
    
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"SYNTAX/RUNTIME ERROR: {e}")
    sys.exit(1)
