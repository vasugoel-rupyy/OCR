import logging
import re
from pathlib import Path
from typing import Dict, Union, Optional, List, Callable
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

from ..utils import load_config, setup_logging, load_image, clean_text, is_pdf, is_pdf_supported, convert_pdf_to_images
from ..quality import ImageQualityAssessor, QualityMetrics
from ..preprocessing import PreprocessingPipeline
from ..preprocessing.id_enhancer import IDDocumentEnhancer
from ..validation.normalization import TokenNormalizer
from ..validation.anchors import AnchorValidator
from ..validation.distribution import DistributionAnalyzer
from ..validation.key_value import KeyValueExtractor
from ..ocr import PaddleOCREngine, OCRResult
from ..documents import AadhaarExtractor, PANExtractor, VehicleRCExtractor, BaseDocumentProcessor
from ..documents.base import BaseDocument
from ..documents.aadhaar import AadhaarDocument
from ..documents.pan import PanDocument
from ..documents.vehicle_rc import RcDocument
from ..builders.document_builder import DocumentBuilder
from ..scoring import ConfidenceScorer, DecisionEngine, DocumentConfidence, DecisionResult, Decision
from ..scoring.confidence import FieldConfidence
from .classification import DocumentClassifier
from ..segmentation import SegmentationPipeline, Region, BoundingBox
from ..validation.spatial_validator import SpatialValidator
from ..validation.business_rules import BusinessRuleValidator
from ..utils.pii import mask_pii


@dataclass
class PipelineResult:
    document_path: str
    document_type: str
    decision: str
    confidence: DocumentConfidence
    decision_result: DecisionResult
    extracted_fields: Dict
    quality_metrics: Dict
    ocr_stats: Dict
    full_text: str = ""
    processing_time: float = 0.0
    error: Optional[str] = None
    regions_detected: int = 1
    region_selected: Optional[Dict] = None
    multi_document_flag: bool = False
    structured_document: Optional[BaseDocument] = None

    
    def to_dict(self) -> Dict:
        return {
            'document_path': self.document_path,
            'document_type': self.document_type,
            'decision': self.decision,
            'confidence_scores': self.confidence.to_dict(),
            'decision_details': self.decision_result.to_dict(),
            'extracted_fields': self.extracted_fields,
            'quality_metrics': self.quality_metrics,
            'ocr_stats': self.ocr_stats,
            'full_text': self.full_text,
            'processing_time': self.processing_time,
            'error': self.error,
            'regions_detected': self.regions_detected,
            'region_selected': self.region_selected,
            'multi_document_flag': self.multi_document_flag,
            'structured_document': self.structured_document.model_dump(mode='json') if self.structured_document else None

        }



class OCRPipeline:
    
    
    FIELD_WEIGHTS = {
        'aadhaar': {'aadhaar_number': 0.4, 'name': 0.3, 'date_of_birth': 0.3},
        'pan': {'pan_number': 0.5, 'name': 0.25, 'date_of_birth': 0.25},
        'vehicle_rc': {'registration_number': 0.4, 'owner_name': 0.2, 'engine_number': 0.2, 'chassis_number': 0.2},
        'disbursement_order': {
            'loan_amount': 0.25,
            'disbursed_amount': 0.25,
            'rate_of_interest': 0.15,
            'tenure_months': 0.15,
            'customer_name': 0.10,
            'bank_name': 0.10,
        },
    }

    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        self.config = load_config(config_path)
        
        self.logger = setup_logging(self.config.get('logging', {}))
        self.logger.info("Initializing OCR Pipeline")
        
        self.quality_assessor = ImageQualityAssessor(self.config.get('quality', {}))
        self.preprocessing_pipeline = PreprocessingPipeline(self.config.get('preprocessing', {}))
        self.ocr_engine = PaddleOCREngine(self.config.get('ocr', {}))
        self.confidence_scorer = ConfidenceScorer(self.config.get('scoring', {}))
        self.decision_engine = DecisionEngine(self.config.get('decision', {}))
        self.classifier = DocumentClassifier()
        
        self.anchor_validator = AnchorValidator(self.config)
        self.distribution_analyzer = DistributionAnalyzer(self.config)
        self.kv_extractor = KeyValueExtractor(self.config)
        self.token_normalizer = TokenNormalizer()
        
        self.aadhaar_extractor = AadhaarExtractor()
        self.pan_extractor = PANExtractor()
        self.vehicle_rc_extractor = VehicleRCExtractor()
        
        
        self.segmentation_pipeline = SegmentationPipeline(self.config.get('segmentation', {}))
        self.spatial_validator = SpatialValidator(self.config)
        self.business_rule_validator = BusinessRuleValidator(self.config.get('business_rules', {}))
        
        self.logger.info("OCR Pipeline initialized successfully")
    
    def process_document(self,
                        image_path: Union[str, Path],
                        document_type: str = 'auto',
                        save_intermediates: bool = False) -> PipelineResult:
        import time
        start_time = time.time()
        
        document_type = document_type.lower().strip()
        if document_type in ['aadhar', 'adhara', 'adhar']:
            document_type = 'aadhaar'
        elif document_type in ['rc', 'vehicle', 'car_rc']:
            document_type = 'vehicle_rc'
        elif document_type in ['do', 'disbursement']:
            document_type = 'disbursement_order'
        
        self.logger.info(f"Processing document: {image_path} (type: {document_type})")
        
        template_conf = 0.0
        
        try:
            pil_images = []
            if is_pdf(image_path):
                if not is_pdf_supported():
                    raise ValueError(f"pdf2image is not installed to process PDF: {image_path}")
                pil_images = convert_pdf_to_images(image_path)
            else:
                img = load_image(image_path)
                from PIL import Image
                pil_images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))]

            if not pil_images:
                raise ValueError(f"No pages/images found in: {image_path}")
            
            master_ocr_result = None
            master_quality_metrics = None
            primary_image = None
            
            for idx, pil_image in enumerate(pil_images):
                page_num = idx + 1
                self.logger.info(f"Processing page {page_num}/{len(pil_images)}")
                
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                if primary_image is None:
                    primary_image = image
                
                stage_start = time.time()
                quality_metrics = self.quality_assessor.assess(image)
                
                if master_quality_metrics is None:
                    master_quality_metrics = quality_metrics
                else:
                    if not quality_metrics.passed:
                        master_quality_metrics.passed = False
                        master_quality_metrics.failure_reasons.extend(
                            [f"Page {page_num}: {r}" for r in quality_metrics.failure_reasons]
                        )
                
                stage_start = time.time()
                
                current_preprocessing_config = self.config.get('preprocessing', {}).copy()
                if document_type == 'disbursement_order':
                    if current_preprocessing_config.get('enable_perspective_correction'):
                        self.logger.info("Disabling perspective correction for Disbursement Order to ensure full-page coverage")
                        current_preprocessing_config['enable_perspective_correction'] = False
                
                orig_enable_perspective = self.preprocessing_pipeline.corrector.enable_perspective
                if document_type == 'disbursement_order':
                    self.preprocessing_pipeline.corrector.enable_perspective = False
                
                try:
                    preprocessing_result = self.preprocessing_pipeline.process(image, save_intermediates)
                finally:
                    self.preprocessing_pipeline.corrector.enable_perspective = orig_enable_perspective
                    
                processed_image = preprocessing_result['processed_image']
                
                stage_start = time.time()
                page_ocr_result = self.ocr_engine.extract_text(processed_image)
                
                if master_ocr_result is None:
                    master_ocr_result = page_ocr_result
                else:
                    master_ocr_result.append_page(page_ocr_result, page_num)
            
            ocr_result = master_ocr_result
            quality_metrics = master_quality_metrics
            image = primary_image

            self.logger.info("--- STAGE 3: FINAL OCR TEXT (MASKED) ---")
            self.logger.info(f"\n{mask_pii(ocr_result.full_text)}")
            self.logger.info("--- END OF OCR TEXT ---")
            
            h, w = image.shape[:2]
            selected_region = Region(
                bbox=BoundingBox(0, 0, w, h),
                image=image.copy(),
                confidence=1.0,
                detection_method='full_image',
                area_ratio=1.0
            )
            detected_regions = [selected_region]
            num_regions = 1
            region_info = selected_region.to_dict()
            
            multi_document_flag = len(pil_images) > 1
            conflicting_schemas = False
            
            if document_type == 'auto':
                stage_start = time.time()
                self.logger.info("Auto-detecting document type...")
                
                document_type_candidate, scores = self.classifier.classify_with_scores(ocr_result.full_text)
                max_score = max(scores.values()) if scores else 0
                self.logger.info(f"Initial classification scores: {scores}")

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
                        ocr_result = ocr_result_enh
                        self.logger.info(f"Enhancement found {ocr_result.total_words} words. Re-classifying.")
                        
                        document_type_candidate, scores = self.classifier.classify_with_scores(ocr_result.full_text)
                        max_score = max(scores.values()) if scores else 0
                        self.logger.info(f"Re-classification scores: {scores}")

                document_type = document_type_candidate
                
                if document_type == 'disbursement_order':
                    do_conf = scores.get('disbursement_order_confidence', 0.0)
                    if do_conf < 0.45:
                        self.logger.warning(f"Low confidence DO ({do_conf:.2f}), sending to manual review")

                stage_time = (time.time() - stage_start) * 1000
                self.logger.info(f"Stage 4: Document Classification - completed in {stage_time:.0f}ms")
                self.logger.info(f"Detected document type: {document_type} (score: {max_score})")
            
            primary_ocr_result = ocr_result
            ocr_result_enh = None
            
            if document_type in ['aadhaar', 'pan', 'vehicle_rc']:
                stage_start_eval = time.time()
                
                extractor = self._get_extractor(document_type)
                extracted_fields = extractor.extract_fields(ocr_result)
                
                ocr_confidence_score = self.ocr_engine.calculate_ocr_confidence_score(ocr_result)
                
                required_fields = self._get_required_fields(document_type)
                mandatory_fields_present = all(field in extracted_fields for field in required_fields)
                
                if ocr_confidence_score > 0.8 and mandatory_fields_present:
                    self.logger.info(f"High confidence ({ocr_confidence_score:.2f}) and all mandatory fields found. Skipping enhanced pass.")
                else:
                    self.logger.info(f"Running enhanced pass for {document_type} (Score: {ocr_confidence_score:.2f}, Mandatory: {mandatory_fields_present})")
                    
                    if hasattr(self.preprocessing_pipeline, 'corrector'):
                        deskewed_image = self.preprocessing_pipeline.corrector.correct_skew(image)
                    else:
                        deskewed_image = image
                    
                    id_enhancer = IDDocumentEnhancer()
                    enhanced_image = id_enhancer.enhance_for_ocr(deskewed_image)
                    ocr_result_enh = self.ocr_engine.extract_text(enhanced_image)
                    
                    if ocr_result_enh.total_words > ocr_result.total_words:
                        primary_ocr_result = ocr_result_enh
                
                stage_eval_time = (time.time() - stage_start_eval) * 1000
                self.logger.info(f"Adaptive OCR Strategy - completed in {stage_eval_time:.0f}ms")
            else:
                primary_ocr_result = ocr_result
            
            text_detected = primary_ocr_result.total_words > 0
            if not text_detected:
                self.logger.warning("No text detected in document")
            
            ocr_confidence_score = self.ocr_engine.calculate_ocr_confidence_score(primary_ocr_result)
            
            stage_start = time.time()
            self.logger.debug("Stage 5: Field Extraction")
            
            if document_type == 'disbursement_order':
                import sys
                from pathlib import Path
                root_dir = str(Path(__file__).parent.parent.parent)
                if root_dir not in sys.path: sys.path.insert(0, root_dir)
                from ocr_pipeline.api.ollama_service import OllamaExtractor
                
                self.logger.info("Using Ollama Qwen for DO extraction...")
                ollama_result = OllamaExtractor.extract_disbursement_order_sync(ocr_result.full_text)
                
                extracted_fields = {}
                ollama_decision = "REVIEW"
                if "extracted_fields" in ollama_result:
                    for k, v in ollama_result["extracted_fields"].items():
                        extracted_fields[k] = v.get("value") if isinstance(v, dict) else v
                    ollama_decision = ollama_result.get("decision", "REVIEW")
                else:
                    extracted_fields = ollama_result
            else:
                extractor = self._get_extractor(document_type)
                extracted_fields = extractor.extract_fields(ocr_result)
            
            if ocr_result_enh:
                fields_enh = extractor.extract_fields(ocr_result_enh)
                
                if document_type == 'aadhaar':
                    priority_fields = ['aadhaar_number', 'name', 'date_of_birth', 'gender', 'address']
                    if 'aadhaar_number' in fields_enh:
                        extracted_fields['aadhaar_number'] = fields_enh['aadhaar_number']
                        extracted_fields['id_number'] = fields_enh['aadhaar_number']
                elif document_type == 'pan':
                    priority_fields = ['pan_number', 'name', 'father_name', 'date_of_birth']
                elif document_type == 'vehicle_rc':
                    priority_fields = ['registration_number', 'owner_name', 'engine_number', 'chassis_number']
                else:
                    priority_fields = []
                
                for key in priority_fields:
                    if key in fields_enh:
                        extracted_fields[key] = fields_enh[key]
                
                for key, value in fields_enh.items():
                    if key not in extracted_fields:
                        extracted_fields[key] = value
            
            ocr_result = primary_ocr_result
            stage_time = (time.time() - stage_start) * 1000
            self.logger.info(f"Stage 5: Field Extraction - completed in {stage_time:.0f}ms (found {len(extracted_fields)} fields)")
            
            required_fields = self._get_required_fields(document_type)
            mandatory_fields_present = all(
                field in extracted_fields and extracted_fields[field] is not None 
                for field in required_fields
            )
            
            non_null_extracted = sum(1 for v in extracted_fields.values() if v is not None)
            semantic_score = non_null_extracted / max(len(required_fields), 1) if required_fields else 1.0
            
            layout_score = ocr_confidence_score
            consistency_score = 1.0 if mandatory_fields_present else 0.5
            
            stage_start = time.time()
            self.logger.debug("Stage 6: Post-OCR Validation & Fuzzy Matching")
            
            if False: 
                pass
            else:
                regex_score = 1.0 
                
                valid_doc_types = ['aadhaar', 'pan', 'vehicle_rc', 'disbursement_order']
                anchor_doc_type = document_type if document_type in valid_doc_types else 'aadhaar'
                fuzzy_score, anchor_details = self.anchor_validator.validate_anchors(ocr_result.full_text, anchor_doc_type)
                
                kv_doc_type = document_type if document_type in valid_doc_types else 'aadhaar'
                kv_score = self.kv_extractor.validate_kv_pairs(ocr_result, kv_doc_type)
                
                dist_doc_type = document_type if document_type in valid_doc_types else 'aadhaar'
                distribution_score, dist_metrics = self.distribution_analyzer.analyze(ocr_result.full_text, dist_doc_type)
                
                schema_score = self._calculate_weighted_schema_score(extracted_fields, document_type)
                regex_score = schema_score
                
                spatial_score = 1.0
                if (hasattr(ocr_result, 'words') and ocr_result.words):
                    try:
                        boxes = [w.bbox for w in ocr_result.words]
                        texts = [w.text for w in ocr_result.words]
                        
                        spatial_score, spatial_details = self.spatial_validator.validate_field_compactness(
                            extracted_fields,
                            boxes,
                            texts
                        )
                        self.logger.debug(f"Spatial validation score: {spatial_score:.3f}")
                        
                        if spatial_details.get('num_clusters', 1) > 1:
                            conflicting_schemas = True
                            self.logger.warning("Multiple spatial clusters detected - possible conflicting schemas")
                    except Exception as e:
                        self.logger.warning(f"Spatial validation failed: {e}")
                        spatial_score = 1.0
                
                business_doc_type = document_type if document_type in valid_doc_types else 'aadhaar'
                business_valid, business_reasons = self.business_rule_validator.validate(extracted_fields, business_doc_type)
                if not business_valid:
                    self.logger.info(f"Business rule validation failed: {business_reasons}")
    
                if document_type in self.FIELD_WEIGHTS:
                    weights = self.FIELD_WEIGHTS[document_type]
                    missing_critical = []
                    for field, weight in weights.items():
                        if weight >= 0.15 and field not in extracted_fields: 
                            missing_critical.append(field)
                    
                    if missing_critical:
                        missing_str = ", ".join(missing_critical)
                        business_reasons.append(f"Missing critical field(s): {missing_str}")
                        mandatory_fields_present = False

            stage_time = (time.time() - stage_start) * 1000
            self.logger.info(f"Stage 6: Validation & Fuzzy Matching - completed in {stage_time:.0f}ms")
            
            stage_start = time.time()
            self.logger.debug("Stage 7: Confidence Scoring")
            field_scores = {}
            semantic_validators = self._get_semantic_validators(document_type)
            for field_name, field_value in extracted_fields.items():
                if not isinstance(field_value, str):
                    continue
                field_ocr_conf = self._compute_field_ocr_confidence(
                    field_value, ocr_result, ocr_confidence_score
                )
                validator = semantic_validators.get(field_name)
                semantic_valid = validator(field_value) if validator else True
                
                field_scores[field_name] = self.confidence_scorer.calculate_field_confidence(
                    field_name=field_name,
                    ocr_confidence=field_ocr_conf * 100.0,
                    semantic_valid=semantic_valid,
                    positional_valid=True
                )
            
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
                spatial_compactness_score=spatial_score,
                field_scores=field_scores
            )
            stage_time = (time.time() - stage_start) * 1000
            self.logger.info(f"Stage 7: Confidence Scoring - completed in {stage_time:.0f}ms (score: {document_confidence.final_score:.3f})")
            
            non_alphanumeric_ratio = self._calculate_non_alphanumeric_ratio(ocr_result.full_text)
            
            stage_start = time.time()
            self.logger.debug("Stage 8: Decision Making")
            
            effective_quality_passed = quality_metrics.passed
            if not effective_quality_passed:
                if document_confidence.schema_score >= 0.60:
                    self.logger.info(f"High extraction completeness ({document_confidence.schema_score:.2f}) - bypassing hard quality reject")
                    effective_quality_passed = True
            
            decision_result = self.decision_engine.make_decision(
                document_confidence=document_confidence,
                quality_passed=effective_quality_passed,
                text_detected=text_detected,
                mandatory_fields_present=mandatory_fields_present,
                non_alphanumeric_ratio=non_alphanumeric_ratio,
                multi_document_detected=multi_document_flag,
                conflicting_schemas=conflicting_schemas,
                business_rule_failures=business_reasons
            )
            stage_time = (time.time() - stage_start) * 1000
            self.logger.info(f"Stage 8: Decision Making - completed in {stage_time:.0f}ms (decision: {decision_result.decision.value})")
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"Document processed: {decision_result.decision.value} "
                f"(score: {document_confidence.final_score:.3f}, time: {processing_time:.2f}s)"
            )
            
            field_confs = {
                k: fs.composite_score for k, fs in field_scores.items()
            }
            structured_doc = None
            
            if document_type == 'aadhaar':
                structured_doc = DocumentBuilder.build_aadhaar(extracted_fields, field_confs)
            elif document_type == 'pan':
                structured_doc = DocumentBuilder.build_pan(extracted_fields, field_confs)
            elif document_type == 'vehicle_rc':
                structured_doc = DocumentBuilder.build_rc(extracted_fields, field_confs)
            elif document_type == 'disbursement_order':
                structured_doc = DocumentBuilder.build_disbursement_order(extracted_fields, field_confs)
            
            if structured_doc:
                structured_doc.overall_confidence = document_confidence.final_score
                try:
                    structured_doc.decision = Decision(decision_result.decision.value)
                except:
                    structured_doc.decision = Decision.REVIEW
                
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
                    'method': 'ocr'
                },
                full_text=primary_ocr_result.full_text,
                processing_time=processing_time,
                error=None,
                regions_detected=num_regions,
                region_selected=region_info,
                multi_document_flag=multi_document_flag,
                structured_document=structured_doc
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing document {image_path}: {str(e)}", exc_info=True)
            
            return PipelineResult(
                document_path=str(image_path),
                document_type=document_type,
                decision='error',
                confidence=DocumentConfidence(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                decision_result=DecisionResult(
                    decision=Decision.REJECT,
                    confidence_score=0.0,
                    reasons=[f"Processing error: {str(e)}"],
                    hard_rejection=True
                ),
                extracted_fields={},
                quality_metrics={'error': str(e)},
                ocr_stats={'error': str(e)},
                full_text="",
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
        if max_workers is None:
            max_workers = self.config.get('batch', {}).get('max_workers', 4)
        
        self.logger.info(f"Processing batch of {len(image_paths)} documents with {max_workers} workers")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.process_document, path, document_type): path
                for path in image_paths
            }
            
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
                        quality_metrics={'error': str(e)},
                        ocr_stats={'error': str(e)},
                        full_text="",
                        error=str(e)
                    ))
        
        self.logger.info(f"Batch processing complete: {len(results)} documents processed")
        
        return results
    
    def _get_extractor(self, document_type: str):
        if document_type == 'aadhaar':
            return self.aadhaar_extractor
        elif document_type == 'pan':
            return self.pan_extractor
        elif document_type == 'vehicle_rc':
            return self.vehicle_rc_extractor
        else:
            self.logger.warning(f"Unknown document type: {document_type}, defaulting to aadhaar extractor")
            return self.aadhaar_extractor
    
    def _calculate_weighted_schema_score(self, extracted_fields: Dict[str, any], document_type: str) -> float:
        if document_type not in self.FIELD_WEIGHTS:
            required = self._get_required_fields(document_type)
            if not required: return 1.0
            found = sum(1 for f in required if f in extracted_fields and extracted_fields[f] is not None)
            return found / len(required)
            
        weights = self.FIELD_WEIGHTS[document_type]
        score = 0.0
        total_weight = 0.0
        
        for field, weight in weights.items():
            total_weight += weight
            if field in extracted_fields and extracted_fields[field] is not None:
                score += weight
                
        if total_weight > 0:
            return score / total_weight
        return 0.0

    def _get_required_fields(self, document_type: str) -> List[str]:
        required_fields_map = {
            'aadhaar': ['aadhaar_number', 'name', 'date_of_birth'],
            'pan': ['pan_number', 'name', 'date_of_birth'],
            'vehicle_rc': ['registration_number', 'owner_name'],
            'disbursement_order': ['customer_name', 'loan_amount', 'disbursed_amount', 'tenure_months', 'rate_of_interest']
        }
        
        return required_fields_map.get(document_type, ['id_number', 'name'])

    def _calculate_non_alphanumeric_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        
        alphanumeric_count = sum(c.isalnum() or c.isspace() for c in text)
        total_count = len(text)
        
        return 1.0 - (alphanumeric_count / total_count)

    def _compute_field_ocr_confidence(self, field_value: str, ocr_result, fallback: float) -> float:
        if not field_value or not ocr_result.words:
            return fallback
        
        field_lower = field_value.lower().replace(' ', '')
        matched_confidences = []
        
        for word in ocr_result.words:
            word_text = word.text.strip().lower()
            if not word_text:
                continue
            if word_text in field_lower or field_lower in word_text:
                matched_confidences.append(word.confidence / 100.0 if word.confidence > 1.0 else word.confidence)
        
        if matched_confidences:
            return sum(matched_confidences) / len(matched_confidences)
        return fallback
    
    def _get_semantic_validators(self, document_type: str) -> Dict[str, Callable[[str], bool]]:
        validators: Dict[str, Callable[[str], bool]] = {}
        
        if document_type == 'aadhaar':
            validators['aadhaar_number'] = lambda v: bool(
                re.match(r'^[2-9]\d{11}$', re.sub(r'[\s.-]+', '', v))
            )
            validators['date_of_birth'] = lambda v: bool(
                re.search(r'\d{2}[/.-]\d{2}[/.-]\d{4}', v)
            )
            validators['gender'] = lambda v: v.upper() in ('MALE', 'FEMALE', 'TRANSGENDER', 'पुरुष', 'महिला')
            validators['pin_code'] = lambda v: bool(re.match(r'^\d{6}$', v.strip()))
            validators['vid'] = lambda v: bool(re.match(r'^\d{16}$', re.sub(r'\s+', '', v)))
            validators['address'] = lambda v: (
                len(v) >= 20
                and sum(c.isalpha() or c in ' ,.-/' for c in v) / max(len(v), 1) > 0.5
                and not re.search(r'\b(VID|MALE|FEMALE|पुरुष|महिला)\b', v, re.IGNORECASE)
                and sum(c.isdigit() for c in v) / max(len(v), 1) < 0.5
            )
            
        elif document_type == 'pan':
            validators['pan_number'] = lambda v: bool(
                re.match(r'^[A-Z]{5}\d{4}[A-Z]$', v.strip().upper())
            )
            validators['date_of_birth'] = lambda v: bool(
                re.search(r'\d{2}[/.-]\d{2}[/.-]\d{4}', v)
            )
            
        elif document_type == 'vehicle_rc':
            validators['registration_number'] = lambda v: bool(
                re.match(r'^[A-Z]{2}\s*\d{1,2}\s*[A-Z]{0,3}\s*\d{1,4}$', v.strip().upper())
            )
            validators['engine_number'] = lambda v: len(v.strip()) >= 5
            validators['chassis_number'] = lambda v: len(v.strip()) >= 5
            validators['registration_date'] = lambda v: bool(
                re.search(r'\d{2}[/.-]\d{2}[/.-]\d{4}', v)
            )
            validators['fuel_type'] = lambda v: v.upper() in (
                'PETROL', 'DIESEL', 'CNG', 'LPG', 'ELECTRIC', 'HYBRID'
            )
        elif document_type == 'disbursement_order':
            def _safe_float(v):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
            def _safe_int(v):
                try:
                    return int(float(v))
                except (ValueError, TypeError):
                    return None

            validators['loan_amount'] = lambda v: (
                _safe_float(v) is not None and 1000 <= _safe_float(v) <= 1_000_000_000
            )
            validators['disbursed_amount'] = lambda v: (
                _safe_float(v) is not None and 1000 <= _safe_float(v) <= 1_000_000_000
            )
            validators['rate_of_interest'] = lambda v: (
                _safe_float(v) is not None and 0.1 <= _safe_float(v) <= 40
            )
            validators['tenure_months'] = lambda v: (
                _safe_int(v) is not None and 3 <= _safe_int(v) <= 120
            )
            validators['customer_name'] = lambda v: (
                len(v) >= 3 and not any(
                    t in v.lower() for t in ['please', 'note', 'authorized', 'mail', 'delete', 'location', 'revoked']
                )
            )
            validators['bank_name'] = lambda v: len(v) >= 3
            validators['ifsc'] = lambda v: bool(re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', v.strip().upper()))
        
        return validators

def main():
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
    
    pipeline = OCRPipeline(args.config)
    
    try:
        result = pipeline.process_document(args.image_path, document_type=args.type)
        
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
