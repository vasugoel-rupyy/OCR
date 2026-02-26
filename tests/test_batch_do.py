import os
import sys
import json
from pathlib import Path

from ocr_pipeline.core.pipeline import OCRPipeline

def convert_to_serializable(obj):
    if isinstance(obj, float):
        return obj
    if hasattr(obj, '__dict__'):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    return str(obj) if obj is not None else None

def process_directory(dir_path: str, output_file: str):
    print(f"Initializing OCR Pipeline...")
    pipeline = OCRPipeline()
    
    # Collect all supported files (PDFs, Images)
    valid_exts = {'.pdf', '.jpg', '.jpeg', '.png'}
    files_to_process = []
    
    for root, _, files in os.walk(dir_path):
        for file in files:
            if Path(file).suffix.lower() in valid_exts:
                files_to_process.append(os.path.join(root, file))
                
    print(f"Found {len(files_to_process)} files to process.")
    
    results = []
    
    for i, file_path in enumerate(files_to_process):
        print(f"[{i+1}/{len(files_to_process)}] Processing {file_path}...")
        try:
            # We enforce disbursement_order type for this specific test
            result = pipeline.process_document(file_path, document_type='disbursement_order')
            
            # Extract only the relevant structured data for clean reporting
            doc_data = {}
            if result.structured_document:
                # Iterate through Pydantic fields
                for field_name, field_val in result.structured_document:
                     if hasattr(field_val, 'confidence'): # FieldValue explicitly
                         doc_data[field_name] = field_val.value
                         doc_data[f"{field_name}_confidence"] = field_val.confidence
                     elif hasattr(field_val, 'value'): # Emums like Decision
                         doc_data[field_name] = field_val.value
                     else:
                         doc_data[field_name] = field_val # Simple string/float fields
                         
            decision_val = result.decision
            if hasattr(decision_val, 'value'):
                decision_val = decision_val.value
                
            report = {
                'file_path': file_path,
                'decision': decision_val,
                'overall_confidence': getattr(result.structured_document, 'overall_confidence', 0.0),
                'processing_time_sec': result.processing_time,
                'extracted_fields': doc_data,
                'error': result.error
            }
            results.append(report)
            print(f"  Decision: {report['decision']}, Score: {report['overall_confidence']:.2f}")
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            results.append({
                'file_path': file_path,
                'error': str(e)
            })

    # Save to JSON
    out_path = Path(output_file)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert_to_serializable)
        
    print(f"\nProcessing complete. Saved {len(results)} results to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tests/test_batch_do.py <input_dir> <output_json>")
        sys.exit(1)
        
    input_directory = sys.argv[1]
    output_json = sys.argv[2]
    
    process_directory(input_directory, output_json)
