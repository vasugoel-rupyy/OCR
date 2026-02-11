"""PDF processing utilities for converting PDFs to images for OCR."""

import logging
from pathlib import Path
from typing import List, Union, Optional
import tempfile
import os

try:
    from pdf2image import convert_from_path, convert_from_bytes
    from PIL import Image
    import numpy as np
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

logger = logging.getLogger(__name__)


def is_pdf_supported() -> bool:
    """Check if PDF processing is available.
    
    Returns:
        True if pdf2image is installed
    """
    return PDF_SUPPORT


def is_pdf(file_path: Union[str, Path]) -> bool:
    """Check if a file is a PDF based on extension and magic bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is a PDF
    """
    file_path = Path(file_path)
    
    # Check extension
    if file_path.suffix.lower() == '.pdf':
        return True
    
    # Check magic bytes (PDF files start with %PDF)
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception as e:
        logger.warning(f"Could not read file header: {e}")
        return False


def convert_pdf_to_images(
    pdf_path: Union[str, Path],
    dpi: int = 300,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None
) -> List:
    """Convert PDF pages to PIL images.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (default 300 for good OCR quality)
        first_page: First page to convert (1-indexed, None = first page)
        last_page: Last page to convert (1-indexed, None = last page)
        
    Returns:
        List of PIL Image objects
        
    Raises:
        ImportError: If pdf2image is not installed
        Exception: If PDF conversion fails
    """
    if not PDF_SUPPORT:
        raise ImportError(
            "pdf2image is required for PDF processing. "
            "Install it with: pip install pdf2image"
        )
    
    try:
        logger.info(f"Converting PDF to images: {pdf_path} (DPI: {dpi})")
        
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
            fmt='jpeg'
        )
        
        logger.info(f"Successfully converted {len(images)} page(s) from PDF")
        return images
        
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}", exc_info=True)
        raise


def convert_pdf_bytes_to_images(
    pdf_bytes: bytes,
    dpi: int = 300,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None
) -> List:
    """Convert PDF bytes to PIL images.
    
    Args:
        pdf_bytes: PDF file as bytes
        dpi: Resolution for conversion (default 300 for good OCR quality)
        first_page: First page to convert (1-indexed, None = first page)
        last_page: Last page to convert (1-indexed, None = last page)
        
    Returns:
        List of PIL Image objects
        
    Raises:
        ImportError: If pdf2image is not installed
        Exception: If PDF conversion fails
    """
    if not PDF_SUPPORT:
        raise ImportError(
            "pdf2image is required for PDF processing. "
            "Install it with: pip install pdf2image"
        )
    
    try:
        logger.info(f"Converting PDF bytes to images (DPI: {dpi})")
        
        images = convert_from_bytes(
            pdf_bytes,
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
            fmt='jpeg'
        )
        
        logger.info(f"Successfully converted {len(images)} page(s) from PDF bytes")
        return images
        
    except Exception as e:
        logger.error(f"Failed to convert PDF bytes to images: {e}", exc_info=True)
        raise


def pdf_to_image_file(
    pdf_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    page: int = 1
) -> str:
    """Convert a single PDF page to an image file.
    
    Args:
        pdf_path: Path to PDF file
        output_path: Path for output image (if None, uses temp file)
        dpi: Resolution for conversion
        page: Page number to convert (1-indexed)
        
    Returns:
        Path to the output image file
        
    Raises:
        ImportError: If pdf2image is not installed
        Exception: If PDF conversion fails
    """
    images = convert_pdf_to_images(pdf_path, dpi=dpi, first_page=page, last_page=page)
    
    if not images:
        raise ValueError(f"No images extracted from PDF page {page}")
    
    image = images[0]
    
    # Use temp file if no output path specified
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
    
    output_path = Path(output_path)
    image.save(str(output_path), 'JPEG', quality=95)
    
    logger.info(f"Saved PDF page {page} to {output_path}")
    return str(output_path)
