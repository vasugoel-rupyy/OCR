"""Multi-stage confidence scoring module (Approach 6 - Enhanced)."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class FieldConfidence:
    """Confidence score for a single field."""
    field_name: str
    ocr_confidence: float
    semantic_validity: float
    positional_validity: float
    composite_score: float


@dataclass
class DocumentConfidence:
    """Overall document confidence scores (10-component model with spatial validation)."""
    image_quality_score: float
    ocr_confidence_score: float
    regex_score: float
    fuzzy_score: float
    layout_score: float
    kv_score: float
    consistency_score: float
    schema_score: float
    distribution_score: float
    spatial_compactness_score: float  # NEW: Spatial validation score
    final_score: float
    field_confidences: Dict[str, FieldConfidence] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'image_quality_score': self.image_quality_score,
            'ocr_confidence_score': self.ocr_confidence_score,
            'regex_score': self.regex_score,
            'fuzzy_score': self.fuzzy_score,
            'layout_score': self.layout_score,
            'kv_score': self.kv_score,
            'consistency_score': self.consistency_score,
            'schema_score': self.schema_score,
            'distribution_score': self.distribution_score,
            'spatial_compactness_score': self.spatial_compactness_score,
            'final_score': self.final_score,
            'field_confidences': {
                name: {
                    'field_name': fc.field_name,
                    'ocr_confidence': fc.ocr_confidence,
                    'semantic_validity': fc.semantic_validity,
                    'positional_validity': fc.positional_validity,
                    'composite_score': fc.composite_score
                }
                for name, fc in self.field_confidences.items()
            }
        }


class ConfidenceScorer:
    """Calculates multi-stage confidence scores using 9-component model."""
    
    def __init__(self, config: Dict):
        """Initialize confidence scorer.
        
        Args:
            config: Scoring configuration
        """
        self.config = config
        
        # Stage weights
        weights = config.get('weights', {})
        self.w_image = weights.get('image_quality', 0.10)
        self.w_ocr = weights.get('ocr_confidence', 0.15)
        self.w_regex = weights.get('regex_match', 0.10)
        self.w_fuzzy = weights.get('fuzzy_match', 0.10)
        self.w_layout = weights.get('layout_validity', 0.10)
        self.w_kv = weights.get('kv_match', 0.10)
        self.w_consistency = weights.get('consistency', 0.10)
        self.w_schema = weights.get('schema_completeness', 0.15)
        self.w_distribution = weights.get('distribution', 0.05)
        self.w_spatial = weights.get('spatial_compactness', 0.05)
        
        # Field weights
        self.field_weights = config.get('field_weights', {})
    
    def calculate_document_confidence(self,
                                     image_quality_score: float,
                                     ocr_confidence_score: float,
                                     regex_score: float,
                                     fuzzy_score: float,
                                     layout_score: float,
                                     kv_score: float,
                                     consistency_score: float,
                                     schema_score: float,
                                     distribution_score: float,
                                     spatial_compactness_score: float = 1.0,
                                     field_scores: Optional[Dict[str, FieldConfidence]] = None) -> DocumentConfidence:
        """Calculate overall document confidence.
        
        Args:
            image_quality_score: Image quality score [0, 1]
            ocr_confidence_score: OCR confidence score [0, 1]
            regex_score: Regex match score [0, 1]
            fuzzy_score: Fuzzy anchor match score [0, 1]
            layout_score: Layout validation score [0, 1]
            kv_score: Key-Value pair match score [0, 1]
            consistency_score: Consistency validation score [0, 1]
            schema_score: Schema completeness score [0, 1]
            distribution_score: Token distribution score [0, 1]
            spatial_compactness_score: Spatial compactness score [0, 1]
            field_scores: Optional field-level confidence scores
            
        Returns:
            DocumentConfidence object
        """
        # Rebalanced weights prioritizing extraction completeness (schema_score)
        # Weights should sum to 1.0
        w_image = 0.05
        w_ocr = 0.10
        w_regex = 0.15
        w_fuzzy = 0.10
        w_layout = 0.10
        w_kv = 0.10
        w_consistency = 0.10
        w_schema = 0.35  # Increased from 0.15
        w_distribution = 0.02
        w_spatial = 0.03

        final_score = (
            w_image * image_quality_score +
            w_ocr * ocr_confidence_score +
            w_regex * regex_score +
            w_fuzzy * fuzzy_score +
            w_layout * layout_score +
            w_kv * kv_score +
            w_consistency * consistency_score +
            w_schema * schema_score +
            w_distribution * distribution_score +
            w_spatial * spatial_compactness_score
        )
        
        # Aggressive damping: penalize heavily if schema_score is low
        # multiplier is 0.05 if schema_score is 0, and ramps to 1.0 if schema_score is 1.0
        multiplier = 0.05 + 0.95 * schema_score
        final_score *= multiplier
        
        # If schema_score is 0, cap the score at 0.10 regardless of other signals
        if schema_score == 0:
            final_score = min(final_score, 0.10)
        
        final_score = max(0.0, min(1.0, final_score))
        
        return DocumentConfidence(
            image_quality_score=image_quality_score,
            ocr_confidence_score=ocr_confidence_score,
            regex_score=regex_score,
            fuzzy_score=fuzzy_score,
            layout_score=layout_score,
            kv_score=kv_score,
            consistency_score=consistency_score,
            schema_score=schema_score,
            distribution_score=distribution_score,
            spatial_compactness_score=spatial_compactness_score,
            final_score=final_score,
            field_confidences=field_scores or {}
        )
    
    def calculate_field_confidence(self,
                                   field_name: str,
                                   ocr_confidence: float,
                                   semantic_valid: bool,
                                   positional_valid: bool = True) -> FieldConfidence:
        """Calculate confidence for a single field.
        
        Args:
            field_name: Name of the field
            ocr_confidence: OCR confidence for this field [0, 100]
            semantic_valid: Whether field passed semantic validation
            positional_valid: Whether field is in expected position
            
        Returns:
            FieldConfidence object
        """
        # Normalize OCR confidence to [0, 1]
        ocr_score = ocr_confidence / 100.0
        
        # Convert boolean validations to scores
        semantic_score = 1.0 if semantic_valid else 0.0
        positional_score = 1.0 if positional_valid else 0.5  # Partial credit
        
        # Calculate composite score
        composite = (
            0.4 * ocr_score +
            0.4 * semantic_score +
            0.2 * positional_score
        )
        
        return FieldConfidence(
            field_name=field_name,
            ocr_confidence=ocr_score,
            semantic_validity=semantic_score,
            positional_validity=positional_score,
            composite_score=composite
        )
