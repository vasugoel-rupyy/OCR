"""Utility functions for OCR pipeline."""

import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import cv2
import numpy as np
from PIL import Image


def load_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Configured logger instance
    """
    if config is None:
        config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'ocr_pipeline.log',
            'max_bytes': 10485760,
            'backup_count': 5
        }
    
    # Create logger
    logger = logging.getLogger('ocr_pipeline')
    logger.setLevel(getattr(logging, config.get('level', 'INFO')))
    
    # Avoid adding duplicate handlers if already configured
    if logger.hasHandlers():
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(config.get('format'))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if 'file' in config:
        file_handler = logging.handlers.RotatingFileHandler(
            config['file'],
            maxBytes=config.get('max_bytes', 10485760),
            backupCount=config.get('backup_count', 5)
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(config.get('format'))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array in BGR format
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """Save image to file.
    
    Args:
        image: Image as numpy array
        output_path: Path to save image
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format.
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        Image as numpy array in BGR format
    """
    # Convert PIL to RGB numpy array
    rgb_image = np.array(pil_image.convert('RGB'))
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL format.
    
    Args:
        cv2_image: Image as numpy array in BGR format
        
    Returns:
        PIL Image object
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert to PIL
    pil_image = Image.fromarray(rgb_image)
    return pil_image


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to [0, 1] range.
    
    Args:
        value: Value to normalize
        min_val: Minimum expected value
        max_val: Maximum expected value
        
    Returns:
        Normalized value in [0, 1]
    """
    if max_val == min_val:
        return 1.0
    
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def weighted_average(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Calculate weighted average of scores.
    
    Args:
        scores: Dictionary of score names to values
        weights: Dictionary of score names to weights
        
    Returns:
        Weighted average score
    """
    if not scores:
        return 0.0
    
    total_weight = 0.0
    weighted_sum = 0.0
    
    for key, score in scores.items():
        weight = weights.get(key, 1.0)
        weighted_sum += score * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return weighted_sum / total_weight


def get_image_dimensions(image: np.ndarray) -> tuple:
    """Get image dimensions.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Tuple of (height, width, channels)
    """
    if len(image.shape) == 2:
        return (*image.shape, 1)
    return image.shape


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Ensure image is in grayscale format.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def ensure_color(image: np.ndarray) -> np.ndarray:
    """Ensure image is in color (BGR) format.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Color image in BGR format
    """
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def clean_text(text: str) -> str:
    """Clean OCR text by removing noise and symbols.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned text
    """
    import re
    # Remove common OCR noise patterns
    text = re.sub(r'[।॥|]+', '', text)  # Remove Devanagari danda and pipes
    text = re.sub(r'\s+[-–—]\s+', ' ', text)  # Remove stray dashes
    text = re.sub(r'[^\w\s\u0900-\u097F.,/:()\-]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    return text
