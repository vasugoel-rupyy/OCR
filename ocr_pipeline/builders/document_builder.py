from typing import Dict, Any, Type, Optional
from ..documents.base import BaseDocument, FieldValue, Decision
from ..documents.aadhaar import AadhaarDocument
from ..documents.pan import PanDocument
from ..documents.vehicle_rc import RcDocument
from ..documents.disbursement_order import DisbursementOrderDocument

class DocumentBuilder:
    
    @staticmethod
    def build_aadhaar(raw_data: Dict[str, Any], confidence_scores: Dict[str, float] = None) -> AadhaarDocument:
        if confidence_scores is None:
            confidence_scores = {}
            
        fields = {}
        fields['aadhaar_number'] = DocumentBuilder._create_field(raw_data.get('aadhaar_number'), confidence_scores.get('aadhaar_number', 0.9))
        fields['name'] = DocumentBuilder._create_field(raw_data.get('name'), confidence_scores.get('name', 0.8))
        fields['gender'] = DocumentBuilder._create_field(raw_data.get('gender'), confidence_scores.get('gender', 0.8))
        fields['date_of_birth'] = DocumentBuilder._create_field(raw_data.get('date_of_birth'), confidence_scores.get('date_of_birth', 0.8))
        fields['year_of_birth'] = DocumentBuilder._create_field(raw_data.get('year_of_birth'), confidence_scores.get('year_of_birth', 0.8))
        fields['address'] = DocumentBuilder._create_field(raw_data.get('address'), confidence_scores.get('address', 0.6))
        fields['pin_code'] = DocumentBuilder._create_field(raw_data.get('pin_code'), confidence_scores.get('pin_code', 0.8))
        fields['vid'] = DocumentBuilder._create_field(raw_data.get('vid'), confidence_scores.get('vid', 0.9))
        fields['enrollment_id'] = DocumentBuilder._create_field(raw_data.get('enrollment_id'), confidence_scores.get('enrollment_id', 0.9))
        
        doc = AadhaarDocument(**fields)
        doc.overall_confidence = DocumentBuilder._calculate_overall_confidence(doc)
        doc.decision = DocumentBuilder._make_decision(doc)
        
        return doc

    @staticmethod
    def build_pan(raw_data: Dict[str, Any], confidence_scores: Dict[str, float] = None) -> PanDocument:
        if confidence_scores is None:
            confidence_scores = {}
            
        fields = {}
        fields['pan_number'] = DocumentBuilder._create_field(raw_data.get('pan_number'), confidence_scores.get('pan_number', 0.95))
        fields['name'] = DocumentBuilder._create_field(raw_data.get('name'), confidence_scores.get('name', 0.8))
        fields['father_name'] = DocumentBuilder._create_field(raw_data.get('father_name'), confidence_scores.get('father_name', 0.7))
        fields['date_of_birth'] = DocumentBuilder._create_field(raw_data.get('date_of_birth'), confidence_scores.get('date_of_birth', 0.85))
        fields['signature_present'] = raw_data.get('signature_present', False)
        
        doc = PanDocument(**fields)
        doc.overall_confidence = DocumentBuilder._calculate_overall_confidence(doc)
        doc.decision = DocumentBuilder._make_decision(doc)
        
        return doc

    @staticmethod
    def build_rc(raw_data: Dict[str, Any], confidence_scores: Dict[str, float] = None) -> RcDocument:
        if confidence_scores is None:
            confidence_scores = {}
            
        fields = {}
        fields['registration_number'] = DocumentBuilder._create_field(raw_data.get('registration_number'), confidence_scores.get('registration_number', 0.95))
        fields['owner_name'] = DocumentBuilder._create_field(raw_data.get('owner_name'), confidence_scores.get('owner_name', 0.8))
        fields['engine_number'] = DocumentBuilder._create_field(raw_data.get('engine_number'), confidence_scores.get('engine_number', 0.8))
        fields['chassis_number'] = DocumentBuilder._create_field(raw_data.get('chassis_number'), confidence_scores.get('chassis_number', 0.8))
        fields['registration_date'] = DocumentBuilder._create_field(raw_data.get('registration_date'), confidence_scores.get('registration_date', 0.8))
        fields['vehicle_make_model'] = DocumentBuilder._create_field(raw_data.get('vehicle_make_model'), confidence_scores.get('vehicle_make_model', 0.7))
        fields['vehicle_class'] = DocumentBuilder._create_field(raw_data.get('vehicle_class'), confidence_scores.get('vehicle_class', 0.8))
        fields['fuel_type'] = DocumentBuilder._create_field(raw_data.get('fuel_type'), confidence_scores.get('fuel_type', 0.85))
        fields['seating_capacity'] = DocumentBuilder._create_field(raw_data.get('seating_capacity'), confidence_scores.get('seating_capacity', 0.8))
        fields['wheelbase'] = DocumentBuilder._create_field(raw_data.get('wheelbase'), confidence_scores.get('wheelbase', 0.7))
        fields['unladen_weight'] = DocumentBuilder._create_field(raw_data.get('unladen_weight'), confidence_scores.get('unladen_weight', 0.7))
        fields['vehicle_color'] = DocumentBuilder._create_field(raw_data.get('vehicle_color'), confidence_scores.get('vehicle_color', 0.7))
        fields['hypothecation'] = DocumentBuilder._create_field(raw_data.get('hypothecation'), confidence_scores.get('hypothecation', 0.6))
        fields['fitness_validity_date'] = DocumentBuilder._create_field(raw_data.get('fitness_validity_date'), confidence_scores.get('fitness_validity_date', 0.7))
        fields['insurance_validity_date'] = DocumentBuilder._create_field(raw_data.get('insurance_validity_date'), confidence_scores.get('insurance_validity_date', 0.7))
        fields['manufacturing_date'] = DocumentBuilder._create_field(raw_data.get('manufacturing_date'), confidence_scores.get('manufacturing_date', 0.7))
        
        doc = RcDocument(**fields)
        doc.overall_confidence = DocumentBuilder._calculate_overall_confidence(doc)
        doc.decision = DocumentBuilder._make_decision(doc)
        
        return doc

    @staticmethod
    def build_disbursement_order(raw_data: Dict[str, Any], confidence_scores: Dict[str, float] = None) -> DisbursementOrderDocument:
        if confidence_scores is None:
            confidence_scores = {}
            
        fields = {}
        fields['loan_amount'] = DocumentBuilder._create_field(raw_data.get('loan_amount'), confidence_scores.get('loan_amount', 0.8))
        fields['disbursed_amount'] = DocumentBuilder._create_field(raw_data.get('disbursed_amount'), confidence_scores.get('disbursed_amount', 0.8))
        fields['rate_of_interest'] = DocumentBuilder._create_field(raw_data.get('rate_of_interest'), confidence_scores.get('rate_of_interest', 0.8))
        fields['tenure_months'] = DocumentBuilder._create_field(raw_data.get('tenure_months'), confidence_scores.get('tenure_months', 0.8))
        fields['customer_name'] = DocumentBuilder._create_field(raw_data.get('customer_name'), confidence_scores.get('customer_name', 0.8))
        fields['bank_name'] = DocumentBuilder._create_field(raw_data.get('bank_name'), confidence_scores.get('bank_name', 0.8))
        fields['ifsc'] = DocumentBuilder._create_field(raw_data.get('ifsc'), confidence_scores.get('ifsc', 0.8))
        fields['bank_branch_region'] = DocumentBuilder._create_field(raw_data.get('bank_branch_region'), confidence_scores.get('bank_branch_region', 0.8))
        fields['branch_id'] = DocumentBuilder._create_field(raw_data.get('branch_id'), confidence_scores.get('branch_id', 0.8))
        
        doc = DisbursementOrderDocument(**fields)
        doc.overall_confidence = DocumentBuilder._calculate_overall_confidence(doc)
        doc.decision = DocumentBuilder._make_decision(doc)
        
        return doc
        
    @staticmethod
    def _create_field(value: Optional[str], confidence: float) -> FieldValue:
        if not value:
            return FieldValue(value=None, confidence=0.0)
        return FieldValue(value=str(value), confidence=confidence)

    @staticmethod
    def _calculate_overall_confidence(doc: BaseDocument) -> float:
        scores = []
        total_fields = 0
        for field_name, field_value in doc:
             if field_name in ['template_used', 'template_confidence', 'overall_confidence', 'decision', 'raw_extraction']:
                 continue
             
             total_fields += 1
             if isinstance(field_value, FieldValue) and field_value.value:
                 scores.append(field_value.confidence)
             else:
                 scores.append(0.0)
        
        if total_fields == 0:
            return 0.0
            
        return sum(scores) / total_fields

    @staticmethod
    def _make_decision(doc: BaseDocument) -> Decision:
        critical_missing = False
        
        if isinstance(doc, AadhaarDocument):
            if not doc.aadhaar_number.value: critical_missing = True
            
        elif isinstance(doc, PanDocument):
            if not doc.pan_number.value: critical_missing = True
            
        elif isinstance(doc, RcDocument):
             if not doc.registration_number.value: critical_missing = True

        elif isinstance(doc, DisbursementOrderDocument):
             if not doc.loan_amount.value: critical_missing = True
        
        if critical_missing:
            return Decision.REJECT
            
        if doc.overall_confidence > 0.85:
            return Decision.ACCEPT
        elif doc.overall_confidence > 0.6:
            return Decision.REVIEW
        else:
            return Decision.REJECT
