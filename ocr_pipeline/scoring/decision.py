from typing import Dict, List
from enum import Enum
from dataclasses import dataclass
from .confidence import DocumentConfidence


class Decision(Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


@dataclass
class DecisionResult:
    decision: Decision
    confidence_score: float
    reasons: List[str]
    hard_rejection: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'decision': self.decision.value,
            'confidence_score': self.confidence_score,
            'reasons': self.reasons,
            'hard_rejection': self.hard_rejection
        }


class DecisionEngine:
    
    def __init__(self, config: Dict):
        self.config = config
        self.accept_threshold = config.get('accept_threshold', 0.85)
        self.review_threshold = config.get('review_threshold', 0.60)
        
        hard_reject_config = config.get('hard_reject', {})
        self.reject_no_text = hard_reject_config.get('no_text_detected', True)
        self.reject_quality_fail = hard_reject_config.get('quality_gate_failed', True)
        self.reject_missing_mandatory = hard_reject_config.get('mandatory_field_missing', True)
        self.max_non_alphanumeric = hard_reject_config.get('excessive_non_alphanumeric', 0.7)
        self.reject_multiple_documents = hard_reject_config.get('multiple_documents_detected', True)
        self.reject_conflicting_schemas = hard_reject_config.get('conflicting_schemas', True)
    
    def make_decision(self,
                     document_confidence: DocumentConfidence,
                     quality_passed: bool,
                     text_detected: bool,
                     mandatory_fields_present: bool,
                     non_alphanumeric_ratio: float = 0.0,
                     multi_document_detected: bool = False,
                     conflicting_schemas: bool = False,
                     business_rule_failures: List[str] = None) -> DecisionResult:
        reasons = []
        if business_rule_failures:
            reasons.extend(business_rule_failures)
        hard_rejection = False
        
        if self.reject_no_text and not text_detected:
            reasons.append("No text detected in image")
            hard_rejection = True
            return DecisionResult(
                decision=Decision.REJECT,
                confidence_score=0.0,
                reasons=reasons,
                hard_rejection=True
            )
        
        if self.reject_quality_fail and not quality_passed:
            reasons.append("Image failed quality gate")
            if document_confidence.final_score >= self.accept_threshold:
                reasons.append(f"Hard REJECT downgraded to REVIEW due to high confidence score ({document_confidence.final_score:.2f})")
                return DecisionResult(
                    decision=Decision.REVIEW,
                    confidence_score=document_confidence.final_score,
                    reasons=reasons,
                    hard_rejection=False
                )
            
            hard_rejection = True
            return DecisionResult(
                decision=Decision.REJECT,
                confidence_score=document_confidence.final_score,
                reasons=reasons,
                hard_rejection=True
            )
        
        if self.reject_missing_mandatory and not mandatory_fields_present:
            reasons.append("Mandatory fields are missing")
            hard_rejection = True
            return DecisionResult(
                decision=Decision.REJECT,
                confidence_score=document_confidence.final_score,
                reasons=reasons,
                hard_rejection=True
            )
        
        if non_alphanumeric_ratio > self.max_non_alphanumeric:
            reasons.append(f"Excessive non-alphanumeric content ({non_alphanumeric_ratio:.1%})")
            hard_rejection = True
            return DecisionResult(
                decision=Decision.REJECT,
                confidence_score=document_confidence.final_score,
                reasons=reasons,
                hard_rejection=True
            )
        
        if self.reject_conflicting_schemas and conflicting_schemas:
            reasons.append("Conflicting document schemas detected (multiple documents in image)")
            return DecisionResult(
                decision=Decision.REVIEW,
                confidence_score=document_confidence.final_score,
                reasons=reasons,
                hard_rejection=False
            )
        
        if self.reject_multiple_documents and multi_document_detected:
            reasons.append("Multiple documents detected without clear best candidate")
            return DecisionResult(
                decision=Decision.REVIEW,
                confidence_score=document_confidence.final_score,
                reasons=reasons,
                hard_rejection=False
            )
        
        score = document_confidence.final_score
        
        if score >= self.accept_threshold:
            if business_rule_failures:
                decision = Decision.REVIEW
                reasons.append("Downgraded to REVIEW due to business rule failures")
            else:
                decision = Decision.ACCEPT
                reasons.append(f"Confidence score {score:.2f} exceeds accept threshold {self.accept_threshold}")
            
            if document_confidence.image_quality_score > 0.8:
                reasons.append("High image quality")
            if document_confidence.schema_score > 0.9:
                reasons.append("Strong schema completeness")
            if document_confidence.regex_score > 0.9:
                reasons.append("Strong regex matches")
            if document_confidence.consistency_score > 0.9:
                reasons.append("Excellent consistency checks")
        
        elif score >= self.review_threshold:
            decision = Decision.REVIEW
            reasons.append(f"Confidence score {score:.2f} in review range [{self.review_threshold}, {self.accept_threshold})")
            
            if document_confidence.image_quality_score < 0.6:
                reasons.append("Low image quality")
            if document_confidence.ocr_confidence_score < 0.7:
                reasons.append("Low OCR confidence")
            if document_confidence.schema_score < 0.7:
                reasons.append("Incomplete schema")
            if document_confidence.regex_score < 0.7:
                reasons.append("Weak regex matches")
            if document_confidence.layout_score < 0.5:
                reasons.append("Poor layout match")
            if document_confidence.consistency_score < 0.7:
                reasons.append("Consistency issues detected")
        
        else:
            decision = Decision.REJECT
            reasons.append(f"Confidence score {score:.2f} below review threshold {self.review_threshold}")
            
            if document_confidence.image_quality_score < 0.5:
                reasons.append("Very poor image quality")
            if document_confidence.ocr_confidence_score < 0.5:
                reasons.append("Very low OCR confidence")
            if document_confidence.schema_score < 0.5:
                reasons.append("Failed schema validation")
            if document_confidence.regex_score < 0.5:
                reasons.append("Failed regex validation")
            if document_confidence.layout_score < 0.3:
                reasons.append("Layout does not match expected template")
            if document_confidence.consistency_score < 0.5:
                reasons.append("Failed consistency checks")
        
        return DecisionResult(
            decision=decision,
            confidence_score=score,
            reasons=reasons,
            hard_rejection=hard_rejection
        )
    
    def get_threshold_info(self) -> Dict:
        return {
            'accept_threshold': self.accept_threshold,
            'review_threshold': self.review_threshold,
            'reject_threshold': self.review_threshold,
            'ranges': {
                'accept': f'>= {self.accept_threshold}',
                'review': f'[{self.review_threshold}, {self.accept_threshold})',
                'reject': f'< {self.review_threshold}'
            }
        }
