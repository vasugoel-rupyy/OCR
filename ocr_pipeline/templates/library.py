from .definitions import Template, Region

# ----------------- AADHAAR TEMPLATES -----------------

# Standard Long Aadhaar (Bottom Part / Cutout)
AADHAAR_LONG_EAADHAAR = Template(
    name="aadhaar_long_oa",
    document_type="aadhaar",
    width_height_ratio=1.5, # Approx ratio for the bottom cut-out card part
    anchor_keywords=["aadhaar", "unique", "identification", "authority"],
    regions=[
        Region(
            name="aadhaar_number",
            coords=(0.30, 0.65, 0.90, 0.85), # Very rough estimate, needs calibration
            type="digits",
            processing_hints=["digits_only"]
        ),
        Region(
            name="name",
            coords=(0.30, 0.25, 0.90, 0.40),
            type="text"
        ),
        Region(
            name="dob",
            coords=(0.30, 0.40, 0.90, 0.50),
            type="date",
            processing_hints=["allow_special_chars"]
        ),
        Region(
            name="gender",
            coords=(0.30, 0.50, 0.90, 0.60),
            type="text"
        )
    ]
)

# Short Paper Aadhaar (Cut-out from e-Aadhaar, non-PVC)
AADHAAR_SHORT_PAPER = Template(
    name="aadhaar_short_paper",
    document_type="aadhaar",
    width_height_ratio=1.45, # Slightly squarer than PVC often, or varies by cutting. Setting range logic in matcher handles diff.
    anchor_keywords=["government", "india", "aadhaar", "dob"],
    regions=[
        Region(
            name="aadhaar_number",
            coords=(0.25, 0.75, 0.75, 0.95), # Bottom center
            type="digits",
            processing_hints=["digits_only"]
        ),
        Region(
            name="name",
            coords=(0.25, 0.20, 0.75, 0.35), # Below emblem/header
            type="text"
        ),
        Region(
            name="dob",
            coords=(0.25, 0.35, 0.75, 0.48),
            type="date"
        ),
        Region(
            name="gender",
            coords=(0.25, 0.48, 0.75, 0.58),
            type="text"
        )
    ]
)

# PVC Aadhaar Card (Front)
AADHAAR_PVC_FRONT = Template(
    name="aadhaar_pvc_front",
    document_type="aadhaar",
    width_height_ratio=1.58, # ISO ID-1 size: 85.60 x 53.98 mm -> ~1.58
    anchor_keywords=["government", "india", "uidai"],
    regions=[
        Region(
            name="aadhaar_number",
            coords=(0.30, 0.68, 0.85, 0.82), # Bottom right-ish
            type="digits",
            processing_hints=["digits_only"]
        ),
        Region(
            name="name",
            coords=(0.30, 0.20, 0.85, 0.35),
            type="text"
        ),
        Region(
            name="dob",
            coords=(0.30, 0.35, 0.85, 0.45),
            type="date"
        ),
        Region(
            name="gender",
            coords=(0.30, 0.45, 0.85, 0.55),
            type="text"
        )
    ]
)

# PVC Aadhaar Card (Back)
AADHAAR_PVC_BACK = Template(
    name="aadhaar_pvc_back",
    document_type="aadhaar",
    width_height_ratio=1.58,
    anchor_keywords=["address", "uidai.gov.in"],
    regions=[
        Region(
            name="address",
            coords=(0.05, 0.15, 0.70, 0.60), 
            type="text"
        ),
        Region(
            name="pin_code",
            coords=(0.50, 0.50, 0.70, 0.65), # Often at end of address
            required=False
        )
    ]
)

# ----------------- PAN PATTERNS -----------------

PAN_CARD_FRONT = Template(
    name="pan_card_front",
    document_type="pan",
    width_height_ratio=1.58,
    anchor_keywords=["income", "tax", "department", "govt", "india", "permanent", "account"],
    regions=[
        Region(
            name="pan_number", # The critical field, usually in middle
            coords=(0.30, 0.55, 0.70, 0.70), 
            type="text",
            processing_hints=["uppercase", "alphanumeric"]
        ),
        Region(
            name="name",
            coords=(0.05, 0.25, 0.95, 0.38),
            type="text",
            processing_hints=["uppercase"]
        ),
        Region(
            name="father_name",
            coords=(0.05, 0.40, 0.95, 0.52),
            type="text",
            processing_hints=["uppercase"]
        ),
        Region(
            name="dob",
            coords=(0.05, 0.78, 0.40, 0.88), # Bottom left usually
            type="date",
            processing_hints=["digits_only", "slash_separator"]
        )
    ]
)


TEMPLATE_LIBRARY = [
    AADHAAR_LONG_EAADHAAR,
    AADHAAR_SHORT_PAPER,
    AADHAAR_PVC_FRONT,
    AADHAAR_PVC_BACK,
    PAN_CARD_FRONT
]
