#!/bin/bash
set -e

echo "Applying PaddleOCR patches..."

# Hardcode the PaddleOCR installation directory to avoid importing PaddleOCR
# (importing PaddleOCR before the patch would cause a segfault)
PADDLEOCR_DIR="/usr/local/lib/python3.11/site-packages/paddleocr"

if [ ! -d "$PADDLEOCR_DIR" ]; then
    echo "Error: PaddleOCR not found at $PADDLEOCR_DIR"
    exit 1
fi

echo "Found PaddleOCR at: $PADDLEOCR_DIR"

# Apply the ONNX fix patch
UTILITY_FILE="$PADDLEOCR_DIR/tools/infer/utility.py"

if [ ! -f "$UTILITY_FILE" ]; then
    echo "Error: utility.py not found at $UTILITY_FILE"
    exit 1
fi

echo "Applying ONNX fix patch to $UTILITY_FILE..."

# Apply patch (cd to the directory first so the patch paths work)
cd "$(dirname "$UTILITY_FILE")"
patch -p0 < /app/patches/paddleocr_onnx_fix.patch

echo "Patches applied successfully!"
