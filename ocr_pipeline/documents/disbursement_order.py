"""Disbursement Order (DO) document definition models.

This file previously housed extensive Regex processors, but the extraction engine 
is now natively driven by LLMs. Only the data structure (Pydantic model) remains.
"""

from typing import Optional
from .base import BaseDocument, FieldValue

class DisbursementOrderDocument(BaseDocument):
    """Pydantic model for Disbursement Order."""
    loan_amount: Optional[FieldValue] = None
    disbursed_amount: Optional[FieldValue] = None
    rate_of_interest: Optional[FieldValue] = None
    tenure_months: Optional[FieldValue] = None
    customer_name: Optional[FieldValue] = None
    bank_name: Optional[FieldValue] = None
    ifsc: Optional[FieldValue] = None
    bank_branch_region: Optional[FieldValue] = None
    branch_id: Optional[FieldValue] = None
