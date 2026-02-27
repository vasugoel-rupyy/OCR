"""Utility functions for OCR pipeline."""

import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import cv2
import numpy as np


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
