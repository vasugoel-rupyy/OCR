"""Main OCR pipeline orchestrator.

Coordinates all stages: quality gate → preprocessing → OCR → validation → scoring → decision.
"""

import logging
from pathlib import Path
from typing import Dict, Union, Optional, List
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils import load_config, setup_logging, load_image, clean_text
from ..quality import ImageQualityAssessor, QualityMetrics
from ..preprocessing import PreprocessingPipeline
from ..preprocessing.id_enhancer import IDDocumentEnhancer
from ..validation.normalization import TokenNormalizer
from ..validation.anchors import AnchorValidator
from ..validation.distribution import DistributionAnalyzer
from ..validation.key_value import KeyValueExtractor
from ..ocr import PaddleOCREngine, OCRResult
from ..documents import AadhaarExtractor, PANExtractor, VehicleRCExtractor, BaseDocumentProcessor
from ..scoring import ConfidenceScorer, DecisionEngine, DocumentConfidence, DecisionResult, Decision
from .classification import DocumentClassifier
from ..segmentation import SegmentationPipeline, Region
from ..validation.spatial_validator import SpatialValidator
from ..validation.business_rules import BusinessRuleValidator
from ..templates.pipeline import TemplatePipeline


@dataclass
class PipelineResult:
    """Complete pipeline result for a document."""
    document_path: str
    document_type: str
    decision: str
    confidence: DocumentConfidence
    decision_result: DecisionResult
    extracted_fields: Dict
    quality_metrics: Dict # Changed to Dict
    ocr_stats: Dict # Changed from ocr_result to ocr_stats dict
    full_text: str = "" # Added
    processing_time: float = 0.0
    error: Optional[str] = None
    regions_detected: int = 1  # NEW: Number of regions detected
    region_selected: Optional[Dict] = None  # NEW: Selected region info
    multi_document_flag: bool = False  # NEW: Multiple documents detected
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'document_path': self.document_path,
            'document_type': self.document_type,
            'decision': self.decision,
            'confidence_scores': self.confidence.to_dict(),
            'decision_details': self.decision_result.to_dict(),
            'extracted_fields': self.extracted_fields,
            'quality_metrics': self.quality_metrics, # Already a dict
            'ocr_stats': self.ocr_stats,
            'full_text': self.full_text,
            'processing_time': self.processing_time,
            'error': self.error,
            'regions_detected': self.regions_detected,
            'region_selected': self.region_selected,
            'multi_document_flag': self.multi_document_flag
        }



class OCRPipeline:
    """Main OCR pipeline for document processing."""
    
    
    # Critical field weights for schema scoring
    FIELD_WEIGHTS = {
        'aadhaar': {'aadhaar_number': 0.4, 'name': 0.3, 'date_of_birth': 0.3},
        'pan': {'pan_number': 0.5, 'name': 0.25, 'date_of_birth': 0.25},
        'vehicle_rc': {'registration_number': 0.4, 'owner_name': 0.2, 'engine_number': 0.2, 'chassis_number': 0.2}
    }

    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        """Initialize OCR pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config.get('logging', {}))
        self.logger.info("Initializing OCR Pipeline")
        
        # Initialize components
        self.quality_assessor = ImageQualityAssessor(self.config.get('quality', {}))
        self.preprocessing_pipeline = PreprocessingPipeline(self.config.get('preprocessing', {}))
        self.ocr_engine = PaddleOCREngine(self.config.get('ocr', {}))
        self.confidence_scorer = ConfidenceScorer(self.config.get('scoring', {}))
        self.decision_engine = DecisionEngine(self.config.get('decision', {}))
        self.classifier = DocumentClassifier()
        
        # Initialize validation components
        self.anchor_validator = AnchorValidator(self.config)
        self.distribution_analyzer = DistributionAnalyzer(self.config)
        self.kv_extractor = KeyValueExtractor(self.config)
        self.token_normalizer = TokenNormalizer()
        
        # Initialize document extractors
        self.aadhaar_extractor = AadhaarExtractor()
        self.pan_extractor = PANExtractor()
        self.vehicle_rc_extractor = VehicleRCExtractor()
        
        # Initialize segmentation pipeline
        self.segmentation_pipeline = SegmentationPipeline(self.config.get('segmentation', {}))
        self.spatial_validator = SpatialValidator(self.config)
        self.business_rule_validator = BusinessRuleValidator(self.config.get('business_rules', {}))
        
        # Initialize Template Pipeline (New Hybrid Mode)
        self.template_pipeline = TemplatePipeline(self.config)
        
        self.logger.info("OCR Pipeline initialized successfully")
    
    def process_document(self,
                        image_path: Union[str, Path],
                        document_type: str = 'auto',
                        save_intermediates: bool = False) -> PipelineResult:
        """Process a single document through the complete pipeline.
        
        Args:
            image_path: Path to document image
            document_type: Type of document ('aadhaar', 'pan', 'vehicle_rc', or 'auto')
            save_intermediates: Whether to save intermediate processing results
            
        Returns:
            PipelineResult object
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Processing document: {image_path} (type: {document_type})")
        
        # Initialize confidence tracking
        template_conf = 0.0
        
        try:
            # Load image
            image = load_image(image_path)
            image_shape = image.shape[:2]
            
            # Stage 1: Image Quality Gate
            self.logger.debug("Stage 1: Image Quality Assessment")
            quality_metrics = self.quality_assessor.assess(image)
            
            if not quality_metrics.passed:
                self.logger.warning(f"Image failed quality gate: {quality_metrics.failure_reasons}")
            
            # Stage 1.5: Document Detection & Segmentation (NEW)
            self.logger.debug("Stage 1.5: Document Detection & Segmentation")
            
            # Run lightweight OCR for text clustering (if enabled)
            ocr_boxes_for_clustering = None
            if self.segmentation_pipeline.use_text_clustering:
                try:
                    # Quick OCR pass just to get bounding boxes
                    quick_ocr = self.ocr_engine.extract_text(image)
                    if hasattr(quick_ocr, 'boxes') and quick_ocr.boxes:
                        ocr_boxes_for_clustering = quick_ocr.boxes
                except Exception as e:
                    self.logger.warning(f"Failed to get OCR boxes for clustering: {e}")
            
            # Detect document regions
            detected_regions = self.segmentation_pipeline.detect_regions(image, ocr_boxes_for_clustering)
            num_regions = len(detected_regions)
            
            self.logger.info(f"Detected {num_regions} document region(s)")
            
            # Determine if multiple documents detected
            multi_document_flag = num_regions > 1
            conflicting_schemas = False
            
            # Stage 2: Preprocessing & Correction
            self.logger.debug("Stage 2: Image Preprocessing")
            
            # Select region to process
            if num_regions == 1:
                # Single region (could be full image or single detected document)
                selected_region = detected_regions[0]
                region_image = selected_region.image
                self.logger.debug(f"Processing single region (method: {selected_region.detection_method})")
            else:
                # Multiple regions detected - select best one
                # Sort by confidence and select highest
                selected_region = detected_regions[0]  # Already sorted by confidence
                region_image = selected_region.image
                self.logger.info(f"Multiple regions detected, selected region with confidence {selected_region.confidence:.3f}")
                
                # Check if there are multiple high-confidence regions (ambiguous case)
                high_conf_regions = [r for r in detected_regions if r.confidence > 0.7]
                if len(high_conf_regions) > 1:
                    self.logger.warning(f"Multiple high-confidence regions detected ({len(high_conf_regions)})")
                    multi_document_flag = True
            
            # Store region info for result
            region_info = selected_region.to_dict() if selected_region else None
            
            # Preprocess the selected region
            preprocessing_result = self.preprocessing_pipeline.process(region_image, save_intermediates)
            processed_image = preprocessing_result['processed_image']

            
            # Stage 2.5: Template-Based Extraction (Hybrid Path)
            self.logger.debug("Stage 2.5: Template-Based Extraction")
            template_name, template_data, template_conf = self.template_pipeline.process(processed_image)
            
            if template_name and template_conf > 0.8 and template_data:
                self.logger.info(f"Template '{template_name}' matched with high confidence ({template_conf:.2f}). Using template result.")
                
                # Construct result from template data
                # We skip the rest of the expensive heavy pipeline
                processing_time = time.time() - start_time
                
                # Create a synthetic confidence object (mostly high scores)
                synth_confidence = DocumentConfidence(
                    final_score=template_conf,
                    ocr_confidence=0.95, # Region OCR is usually cleaner
                    regex_match_score=1.0,
                    fuzzy_match_score=1.0,
                    layout_score=1.0,
                    kv_pair_score=1.0,
                    consistency_score=1.0, 
                    schema_compliance_score=1.0,
                    distribution_score=1.0,
                    spatial_compactness_score=1.0,
                    image_quality_score=quality_metrics.composite_score
                )
                
                # Create Decision
                decision_result = DecisionResult(
                    decision=Decision.ACCEPT,
                    confidence_score=template_conf,
                    reasons=[f"Matched template: {template_name}"],
                    hard_rejection=False
                )
                
                return PipelineResult(
                    document_path=str(image_path),
                    document_type=template_name.split('_')[0] if template_name else "auto", 
                    decision=decision_result.decision.value,
                    confidence=synth_confidence,
                    decision_result=decision_result,
                    extracted_fields=template_data,
                    quality_metrics=quality_metrics.to_dict(),
                    ocr_stats={
                        'method': 'template', 
                        'regions_count': len(template_data),
                        'template_score': template_conf,
                        'fallback_score': None
                    },
                    full_text="[Template Extraction - Full Text Not Assembled]", 
                    processing_time=processing_time,
                    error=None,
                    regions_detected=num_regions,
                    region_selected=region_info,
                    multi_document_flag=multi_document_flag
                )
                
            self.logger.info("Template matching low confidence or failed. Falling back to Full-Document OCR.")

            # Stage 3: OCR
            self.logger.debug("Stage 3: OCR Extraction")
            
            # Run standard OCR first (needed for classification and general text)
            ocr_result = self.ocr_engine.extract_text(processed_image)
            
            # Auto-classify if needed
            if document_type == 'auto':
                self.logger.info("Auto-detecting document type...")
                
                # If standard pass failed to find text, it might be a challenging ID document
                # Try running the ID enhancement pass blindly to see if we find anything
                # Initial classification attempt
                document_type_candidate, scores = self.classifier.classify_with_scores(ocr_result.full_text)
                max_score = max(scores.values()) if scores else 0
                self.logger.info(f"Initial classification scores: {scores}")

                # If standard pass failed to find text OR no clear classification signal
                if ocr_result.total_words == 0 or max_score == 0:
                    self.logger.info("Weak or no classification signal. Trying ID enhancement...")
                    
                    if hasattr(self.preprocessing_pipeline, 'corrector'):
                        deskewed_image = self.preprocessing_pipeline.corrector.correct_skew(image)
                    else:
                        deskewed_image = image
                        
                    id_enhancer = IDDocumentEnhancer()
                    enhanced_image = id_enhancer.enhance_for_ocr(deskewed_image)
                    ocr_result_enh = self.ocr_engine.extract_text(enhanced_image)
                    
                    if ocr_result_enh.total_words > 0:
                        ocr_result = ocr_result_enh  # Use this for classification
                        self.logger.info(f"Enhancement found {ocr_result.total_words} words. Re-classifying.")
                        
                        # Re-classify
                        document_type_candidate, scores = self.classifier.classify_with_scores(ocr_result.full_text)
                        max_score = max(scores.values()) if scores else 0
                        self.logger.info(f"Re-classification scores: {scores}")

                document_type = document_type_candidate
                self.logger.info(f"Detected document type: {document_type} (score: {max_score})")
            
            primary_ocr_result = ocr_result
            
            # ID-Specific Enhanced OCR Pass (Dual Pass)
            # For all Indian ID documents, run enhanced pass for better accuracy
            if document_type in ['aadhaar', 'pan', 'vehicle_rc']:
                self.logger.info(f"Running enhanced pass for {document_type} document")
                
                # Pass 1 was already done above (OCRResult) on standard processed image
                # Pass 2: Enhanced image
                # Use robust skew correction for the base of enhancement
                if hasattr(self.preprocessing_pipeline, 'corrector'):
                    deskewed_image = self.preprocessing_pipeline.corrector.correct_skew(image)
                else:
                    deskewed_image = image
                
                id_enhancer = IDDocumentEnhancer()
                enhanced_image = id_enhancer.enhance_for_ocr(deskewed_image)
                ocr_result_enh = self.ocr_engine.extract_text(enhanced_image)
                
                # Use the pass with more detected words as primary for confidence stats
                if ocr_result_enh.total_words > ocr_result.total_words:
                    primary_ocr_result = ocr_result_enh
                
                # Note: We keep both results available for extractor logic
            else:
                 # Unknown type - single pass
                 ocr_result_enh = None
                 primary_ocr_result = ocr_result
            
            # Check if text was detected
            
            # Check if text was detected
            text_detected = primary_ocr_result.total_words > 0
            if not text_detected:
                self.logger.warning("No text detected in document")
            
            # Calculate OCR confidence score
            ocr_confidence_score = self.ocr_engine.calculate_ocr_confidence_score(primary_ocr_result)
            
            # Stage 4: Document-Specific Processing
            self.logger.debug("Stage 4: Document-Specific Processing")
            
            # Get appropriate extractor
            extractor = self._get_extractor(document_type)
            
            # Extract from standard pass first
            extracted_fields = extractor.extract_fields(ocr_result)
            
            # For all document types, try enhanced pass if available and fields are missing
            if ocr_result_enh:
                fields_enh = extractor.extract_fields(ocr_result_enh)
                
                # Merge fields from enhanced pass if missing in standard
                # Priority fields that benefit from enhancement
                if document_type == 'aadhaar':
                    priority_fields = ['aadhaar_number', 'name', 'date_of_birth', 'gender', 'address']
                    # Special handling for aadhaar_number alias
                    if 'aadhaar_number' in fields_enh and 'aadhaar_number' not in extracted_fields:
                        extracted_fields['aadhaar_number'] = fields_enh['aadhaar_number']
                        extracted_fields['id_number'] = fields_enh['aadhaar_number']
                elif document_type == 'pan':
                    priority_fields = ['pan_number', 'name', 'father_name', 'date_of_birth']
                elif document_type == 'vehicle_rc':
                    priority_fields = ['registration_number', 'owner_name', 'engine_number', 'chassis_number']
                else:
                    priority_fields = []
                
                # Merge priority fields
                for key in priority_fields:
                    if key not in extracted_fields and key in fields_enh:
                        extracted_fields[key] = fields_enh[key]
            
            # Update main ocr_result to be the primary one for reporting
            ocr_result = primary_ocr_result
            
            # Check if mandatory fields are present
            required_fields = self._get_required_fields(document_type)
            mandatory_fields_present = all(field in extracted_fields for field in required_fields)
            
            # Simple validation scores (since we don't have processor methods)
            # Semantic score based on field extraction success
            semantic_score = len(extracted_fields) / max(len(required_fields), 1) if required_fields else 1.0
            
            # Layout score (simplified - based on OCR confidence)
            layout_score = ocr_confidence_score
            
            # Consistency score (simplified - all fields present = high score)
            consistency_score = 1.0 if mandatory_fields_present else 0.5
            
            # Stage 5: Advanced Validation Layer
            self.logger.debug("Stage 5: Post-OCR Validation")
            
            # 5.1 Token Normalization (in-place or separate? keeping raw text for now, using norm helper)
            # 5.2 Regex Score (Redundant with Schema Score in weighted model, setting to 1.0 or same as schema)
            # We'll set it to be consistent with Schema Score later
            regex_score = 1.0 
            
            
            # 5.3 Fuzzy Anchors (use 'aadhaar' for all ID types as fallback)
            anchor_doc_type = document_type if document_type in ['aadhaar', 'pan', 'vehicle_rc'] else 'aadhaar'
            fuzzy_score, anchor_details = self.anchor_validator.validate_anchors(ocr_result.full_text, anchor_doc_type)
            
            # 5.4 Layout (already done)
            
            # 5.5 KV Proximity
            kv_doc_type = document_type if document_type in ['aadhaar', 'pan', 'vehicle_rc'] else 'aadhaar'
            kv_score = self.kv_extractor.validate_kv_pairs(ocr_result, kv_doc_type)
            
            # 5.6 Consistency (already done)
            
            # 5.7 Distribution
            dist_doc_type = document_type if document_type in ['aadhaar', 'pan', 'vehicle_rc'] else 'aadhaar'
            distribution_score, dist_metrics = self.distribution_analyzer.analyze(ocr_result.full_text, dist_doc_type)
            
            # 5.8 Schema Completeness (Weighted)
            schema_score = self._calculate_weighted_schema_score(extracted_fields, document_type)
            regex_score = schema_score # align regex score with schema score for consistency
            
            # 5.9 Spatial Compactness (NEW)
            spatial_score = 1.0  # Default
            if hasattr(ocr_result, 'boxes') and ocr_result.boxes and hasattr(ocr_result, 'texts'):
                try:
                    spatial_score, spatial_details = self.spatial_validator.validate_field_compactness(
                        extracted_fields,
                        ocr_result.boxes,
                        ocr_result.texts
                    )
                    self.logger.debug(f"Spatial validation score: {spatial_score:.3f}")
                    
                    # Check for conflicting schemas
                    if spatial_details.get('num_clusters', 1) > 1:
                        conflicting_schemas = True
                        self.logger.warning("Multiple spatial clusters detected - possible conflicting schemas")
                except Exception as e:
                    self.logger.warning(f"Spatial validation failed: {e}")
                    spatial_score = 1.0
            
            # 5.10 Business Rules
            business_doc_type = document_type if document_type in ['aadhaar', 'pan', 'vehicle_rc'] else 'aadhaar'
            business_valid, business_reasons = self.business_rule_validator.validate(extracted_fields, business_doc_type)
            if not business_valid:
                self.logger.info(f"Business rule validation failed: {business_reasons}")

            # Explicit check for critical fields to provide detailed rejection reasons
            if document_type in self.FIELD_WEIGHTS:
                weights = self.FIELD_WEIGHTS[document_type]
                missing_critical = []
                for field, weight in weights.items():
                    # fields with high weight (>= 0.25) are considered critical enough to mention
                    if weight >= 0.25 and field not in extracted_fields:
                        missing_critical.append(field)
                
                if missing_critical:
                    missing_str = ", ".join(missing_critical)
                    business_reasons.append(f"Missing critical field(s): {missing_str}")
                    # Also set mandatory_fields_present to False if critical fields are missing
                    # This ensures DecisionEngine sees it as a data completeness failure
                    mandatory_fields_present = False

            # Stage 6: Multi-Stage Confidence Scoring
            self.logger.debug("Stage 6: Confidence Scoring")
            document_confidence = self.confidence_scorer.calculate_document_confidence(
                image_quality_score=quality_metrics.composite_score,
                ocr_confidence_score=ocr_confidence_score,
                regex_score=regex_score,
                fuzzy_score=fuzzy_score,
                layout_score=layout_score,
                kv_score=kv_score,
                consistency_score=consistency_score,
                schema_score=schema_score,
                distribution_score=distribution_score,
                spatial_compactness_score=spatial_score
            )
            
            # Calculate non-alphanumeric ratio
            non_alphanumeric_ratio = self._calculate_non_alphanumeric_ratio(ocr_result.full_text)
            
            # Stage 7: Decision
            self.logger.debug("Stage 7: Decision Making")
            decision_result = self.decision_engine.make_decision(
                document_confidence=document_confidence,
                quality_passed=quality_metrics.passed,
                text_detected=text_detected,
                mandatory_fields_present=mandatory_fields_present,
                non_alphanumeric_ratio=non_alphanumeric_ratio,
                multi_document_detected=multi_document_flag,
                conflicting_schemas=conflicting_schemas,
                business_rule_failures=business_reasons
            )
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"Document processed: {decision_result.decision.value} "
                f"(score: {document_confidence.final_score:.3f}, time: {processing_time:.2f}s)"
            )
            
            return PipelineResult(
                document_path=str(image_path),
                document_type=document_type,
                decision=decision_result.decision.value,
                confidence=document_confidence,
                decision_result=decision_result,
                extracted_fields=extracted_fields,
                quality_metrics=quality_metrics.to_dict(),
                ocr_stats={
                    **primary_ocr_result.get_stats(),
                    'method': 'fallback_ocr',
                    'template_score': template_conf,
                    'fallback_score': document_confidence.final_score
                },
                full_text=primary_ocr_result.full_text,
                processing_time=processing_time,
                error=None,
                regions_detected=num_regions,
                region_selected=region_info,
                multi_document_flag=multi_document_flag
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing document {image_path}: {str(e)}", exc_info=True)
            
            # Return error result
            return PipelineResult(
                document_path=str(image_path),
                document_type=document_type,
                decision='error',
                confidence=DocumentConfidence(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # Added spatial_compactness_score
                decision_result=DecisionResult(
                    decision=Decision.REJECT,
                    confidence_score=0.0,
                    reasons=[f"Processing error: {str(e)}"],
                    hard_rejection=True
                ),
                extracted_fields={},
                quality_metrics={'error': str(e)}, # Manual dict
                ocr_stats={'error': str(e)}, # Manual dict
                full_text="", # Manual empty
                processing_time=processing_time,
                error=str(e),
                regions_detected=0,
                region_selected=None,
                multi_document_flag=False
            )
    
    def process_batch(self,
                     image_paths: List[Union[str, Path]],
                     document_type: str = 'invoice',
                     max_workers: Optional[int] = None) -> List[PipelineResult]:
        """Process multiple documents in parallel.
        
        Args:
            image_paths: List of paths to document images
            document_type: Type of documents
            max_workers: Maximum number of parallel workers (default from config)
            
        Returns:
            List of PipelineResult objects
        """
        if max_workers is None:
            max_workers = self.config.get('batch', {}).get('max_workers', 4)
        
        self.logger.info(f"Processing batch of {len(image_paths)} documents with {max_workers} workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_document, path, document_type): path
                for path in image_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing {path}: {str(e)}")
                    results.append(PipelineResult(
                        document_path=str(path),
                        document_type=document_type,
                        decision='error',
                        confidence=DocumentConfidence(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        decision_result=DecisionResult(
                            decision=Decision.REJECT,
                            confidence_score=0.0,
                            reasons=[f"Batch processing error: {str(e)}"],
                            hard_rejection=True
                        ),
                        extracted_fields={},
                        quality_metrics=QualityMetrics(0, 0, 0, 0, 0, 0, False, [str(e)]),
                        ocr_result=OCRResult('', 0.0),
                        error=str(e)
                    ))
        
        self.logger.info(f"Batch processing complete: {len(results)} documents processed")
        
        return results
    
    def _get_extractor(self, document_type: str):
        """Get appropriate document extractor.
        
        Args:
            document_type: Type of document
            
        Returns:
            Document extractor instance
        """
        if document_type == 'aadhaar':
            return self.aadhaar_extractor
        elif document_type == 'pan':
            return self.pan_extractor
        elif document_type == 'vehicle_rc':
            return self.vehicle_rc_extractor
        else:
            # Default to aadhaar for unknown types
            self.logger.warning(f"Unknown document type: {document_type}, defaulting to aadhaar extractor")
            return self.aadhaar_extractor
    
    def _calculate_weighted_schema_score(self, extracted_fields: Dict[str, any], document_type: str) -> float:
        """Calculate schema score based on field weights."""
        if document_type not in self.FIELD_WEIGHTS:
            # Fallback to unweighted count
            required = self._get_required_fields(document_type)
            if not required: return 1.0
            found = sum(1 for f in required if f in extracted_fields)
            return found / len(required)
            
        weights = self.FIELD_WEIGHTS[document_type]
        score = 0.0
        total_weight = 0.0
        
        for field, weight in weights.items():
            total_weight += weight
            if field in extracted_fields:
                score += weight
                
        # Handle optional/extra fields contribution? 
        # For strict schema, we focus on critical fields.
        # Use simple count for remaining weight if total_weight < 1.0?
        # Assuming weights sum to ~1.0. If not, normalize.
        
        if total_weight > 0:
            return score / total_weight
        return 0.0

    def _get_required_fields(self, document_type: str) -> List[str]:
        """Get required fields for document type.
        
        Args:
            document_type: Type of document
            
        Returns:
            List of required field names
        """
        required_fields_map = {
            'aadhaar': ['aadhaar_number', 'name', 'date_of_birth'],
            'pan': ['pan_number', 'name', 'date_of_birth'],
            'vehicle_rc': ['registration_number', 'owner_name']
        }
        
        return required_fields_map.get(document_type, ['id_number', 'name'])

    
    def _calculate_non_alphanumeric_ratio(self, text: str) -> float:
        """Calculate ratio of non-alphanumeric characters.
        
        Args:
            text: Input text
            
        Returns:
            Ratio of non-alphanumeric characters
        """
        if not text:
            return 0.0
        
        alphanumeric_count = sum(c.isalnum() or c.isspace() for c in text)
        total_count = len(text)
        
        return 1.0 - (alphanumeric_count / total_count)


def main():
    """Main entry point for command-line usage."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='OCR Pipeline for Indian Documents (Aadhaar, PAN, Vehicle RC)')
    parser.add_argument('image_path', help='Path to document image')
    parser.add_argument('--type', choices=['aadhaar', 'pan', 'vehicle_rc', 'auto'], default='auto',
                       help='Document type (default: auto)')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--show-text', action='store_true', help='Print full OCR text to stdout')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = OCRPipeline(args.config)
    
    # Process document
    try:
        result = pipeline.process_document(args.image_path, document_type=args.type)
        
        # Print full text if requested
        if args.show_text and hasattr(result, 'full_text') and result.full_text:
            print("\n" + "="*50)
            print("EXTRACTED TEXT")
            print("="*50)
            print(result.full_text)
            print("="*50 + "\n")
            
        print(json.dumps(result.to_dict(), indent=2))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResult saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
