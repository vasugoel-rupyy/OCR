import sys
import re
from pathlib import Path
from collections import Counter
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ocr_pipeline.core.pipeline import OCRPipeline

# Suppress debug logs from pipeline to keep terminal clean
logging.getLogger('ocr_pipeline').setLevel(logging.WARNING)

def get_ngrams(words, n):
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def main(input_dir):
    input_path = Path(input_dir)
    pipeline = OCRPipeline()
    
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    
    # Basic stop words to filter out
    stop_words = {
        'the', 'and', 'to', 'of', 'in', 'for', 'a', 'on', 'is', 'that', 'by', 
        'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 
        'at', 'as', 'your', 'have', 'new', 'no', 'rs', 'inr', 'date', 'rupees', 
        'only', 'all', 'any', 'we', 'our', 'us', 'an', 'has', 'been', 'which',
        'will', 'can', 'if', 'their', 'there', 'was', 'were', 'would', 'should'
    }
    
    files = list(input_path.rglob('*.*'))
    valid_exts = {'.jpg', '.jpeg', '.png', '.pdf'}
    files = [f for f in files if f.suffix.lower() in valid_exts]
    
    print(f"Found {len(files)} valid image/PDF files to process in {input_dir}.")
    
    for idx, fpath in enumerate(files):
        print(f"[{idx+1}/{len(files)}] Extracting text from: {fpath.name}")
        try:
            # Run the document extraction through our pipeline
            result = pipeline.process_document(str(fpath), document_type='disbursement_order')
            text = result.full_text
            
            if not text:
                continue
                
            # Clean text: lowercase, remove digits and punctuation while preserving Unicode words and marks
            import string
            text = text.lower()
            to_remove = string.punctuation + string.digits + "“”‘’«»„"
            text = text.translate(str.maketrans(to_remove, ' ' * len(to_remove)))
            
            # Tokenize and filter
            words = text.split()
            words = [w for w in words if len(w) > 2 and w not in stop_words]
            
            unigrams.update(words)
            bigrams.update(get_ngrams(words, 2))
            trigrams.update(get_ngrams(words, 3))
            
        except Exception as e:
            print(f"  -> Error processing {fpath.name}: {e}")
            
    # Output the results
    print("\n" + "="*50)
    print("MOST FREQUENT UNIGRAMS")
    print("="*50)
    for word, count in unigrams.most_common(30):
        print(f"{count:4d} | {word}")
        
    print("\n" + "="*50)
    print("MOST FREQUENT BIGRAMS")
    print("="*50)
    for word, count in bigrams.most_common(30):
        print(f"{count:4d} | {word}")
        
    print("\n" + "="*50)
    print("MOST FREQUENT TRIGRAMS")
    print("="*50)
    for word, count in trigrams.most_common(30):
        print(f"{count:4d} | {word}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_frequent_words.py <input_dir>")
        sys.exit(1)
    main(sys.argv[1])
