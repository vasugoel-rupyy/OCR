from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class FieldConfidence:
    field_name: str
    ocr_confidence: float
    semantic_validity: float
    positional_validity: float
    composite_score: float


@dataclass
class DocumentConfidence:
    image_quality_score: float
    ocr_confidence_score: float
    regex_score: float
    fuzzy_score: float
    layout_score: float
    kv_score: float
    consistency_score: float
    schema_score: float
    distribution_score: float
    spatial_compactness_score: float
    final_score: float
    field_confidences: Dict[str, FieldConfidence] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
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
    
    def __init__(self, config: Dict):
        self.config = config
        
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
        w_image = 0.05
        w_ocr = 0.10
        w_regex = 0.15
        w_fuzzy = 0.10
        w_layout = 0.10
        w_kv = 0.10
        w_consistency = 0.10
        w_schema = 0.35
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
        
        multiplier = 0.05 + 0.95 * schema_score
        final_score *= multiplier
        
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
        ocr_score = ocr_confidence / 100.0
        
        semantic_score = 1.0 if semantic_valid else 0.0
        positional_score = 1.0 if positional_valid else 0.5
        
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
