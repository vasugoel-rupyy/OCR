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
    return PDF_SUPPORT


def is_pdf(file_path: Union[str, Path]) -> bool:
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.pdf':
        return True
    
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
    images = convert_pdf_to_images(pdf_path, dpi=dpi, first_page=page, last_page=page)
    
    if not images:
        raise ValueError(f"No images extracted from PDF page {page}")
    
    image = images[0]
    
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
    
    output_path = Path(output_path)
    image.save(str(output_path), 'JPEG', quality=95)
    
    logger.info(f"Saved PDF page {page} to {output_path}")
    return str(output_path)
