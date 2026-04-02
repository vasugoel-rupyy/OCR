import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import cv2
import numpy as np
from .config_models import AppConfig


def load_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        # Return default config as dict if file is missing
        return AppConfig().model_dump()
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f) or {}
    
    # Use Pydantic to validate and return as a dictionary for backward compatibility
    return AppConfig(**config_data).model_dump()


def setup_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    if config is None:
        config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'ocr_pipeline.log',
            'max_bytes': 10485760,
            'backup_count': 5
        }
    
    logger = logging.getLogger('ocr_pipeline')
    logger.setLevel(getattr(logging, config.get('level', 'INFO')))
    
    if logger.hasHandlers():
        return logger
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(config.get('format'))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
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
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def clean_text(text: str) -> str:
    import re
    text = re.sub(r'[।॥|]+', '', text)
    text = re.sub(r'\s+[-–—]\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0900-\u097F.,/:()\-]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
