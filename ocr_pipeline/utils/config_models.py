from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Tuple, Union, Any
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class QualityWeights(BaseModel):
    blur: float = Field(0.3, ge=0, le=1)
    brightness: float = Field(0.2, ge=0, le=1)
    resolution: float = Field(0.2, ge=0, le=1)
    contrast: float = Field(0.2, ge=0, le=1)
    glare: float = Field(0.1, ge=0, le=1)

class QualityConfig(BaseModel):
    min_resolution: int = Field(200, ge=72)
    min_blur_score: float = Field(50.0, ge=0)
    min_brightness: int = Field(20, ge=0, le=255)
    max_brightness: int = Field(240, ge=0, le=255)
    min_contrast_ratio: float = Field(0.2, ge=0, le=1)
    min_edge_density: float = Field(0.005, ge=0)
    max_glare_ratio: float = Field(0.05, ge=0, le=1)
    glare_threshold: int = Field(253, ge=0, le=255)
    weights: QualityWeights = Field(default_factory=QualityWeights)

class BusinessRulesConfig(BaseModel):
    enabled: bool = True
    date_validation: Dict[str, Union[bool, int]] = Field(default_factory=lambda: {"reject_future_dates": True, "max_age_days": 365})

class PreprocessingConfig(BaseModel):
    enable_skew_correction: bool = True
    max_skew_angle: int = 45
    enable_perspective_correction: bool = True
    enable_illumination_normalization: bool = True
    enable_noise_removal: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    median_blur_ksize: int = 3
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75

class OCRConfig(BaseModel):
    paddle_ocr: Dict[str, Union[bool, str]] = Field(default_factory=lambda: {"use_angle_cls": True, "lang": "en", "use_gpu": False, "show_log": False})
    min_word_confidence: int = Field(60, ge=0, le=100)
    min_words_detected: int = 5
    stopwords: List[str] = Field(default_factory=list)
    numeric_token_weight: float = 1.5
    alpha_token_weight: float = 1.0
    stopword_weight: float = 0.3

class ScoringWeights(BaseModel):
    image_quality: float = 0.10
    ocr_confidence: float = 0.15
    regex_match: float = 0.10
    fuzzy_match: float = 0.10
    layout_validity: float = 0.10
    kv_match: float = 0.10
    consistency: float = 0.10
    schema_completeness: float = 0.15
    distribution: float = 0.05
    spatial_compactness: float = 0.05

class DecisionConfig(BaseModel):
    accept_threshold: float = Field(0.85, ge=0, le=1)
    review_threshold: float = Field(0.60, ge=0, le=1)
    reject_threshold: float = Field(0.60, ge=0, le=1)
    hard_reject: Dict[str, Union[bool, float]] = Field(default_factory=dict)

class LoggingConfig(BaseModel):
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "ocr_pipeline.log"
    max_bytes: int = 10485760
    backup_count: int = 5

class AppConfig(BaseModel):
    quality: QualityConfig = Field(default_factory=QualityConfig)
    business_rules: BusinessRulesConfig = Field(default_factory=BusinessRulesConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    scoring: Dict[str, Any] = Field(default_factory=dict)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    batch: Dict[str, Any] = Field(default_factory=dict)
    
    # Generic catch-all for existing sections not fully modeled yet
    semantic: Dict[str, Any] = Field(default_factory=dict)
    layout: Dict[str, Any] = Field(default_factory=dict)
    anchors: Dict[str, Any] = Field(default_factory=dict)
    distribution: Dict[str, Any] = Field(default_factory=dict)
    consistency: Dict[str, Any] = Field(default_factory=dict)
