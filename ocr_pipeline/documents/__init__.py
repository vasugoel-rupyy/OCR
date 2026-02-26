"""Documents module."""

from .base import BaseDocumentProcessor
from .aadhaar import AadhaarExtractor
from .pan import PANExtractor
from .vehicle_rc import VehicleRCExtractor
from .disbursement_order import DisbursementOrderProcessor

__all__ = ['BaseDocumentProcessor', 'AadhaarExtractor', 'PANExtractor', 'VehicleRCExtractor', 'DisbursementOrderProcessor']


