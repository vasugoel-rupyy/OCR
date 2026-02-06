from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Region:
    """Defines a region of interest in a document template."""
    name: str
    coords: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in percentage (0.0 to 1.0)
    required: bool = True
    type: str = "text"  # text, digits, date, mixed
    processing_hints: List[str] = field(default_factory=list)  # e.g. ["digits_only", "uppercase"]

    def to_absolute(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert percentage coordinates to absolute pixels."""
        x1, y1, x2, y2 = self.coords
        return (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height)
        )

@dataclass
class Template:
    """Defines a document layout template."""
    name: str
    document_type: str  # aadhaar, pan, vehicle_rc
    regions: List[Region]
    width_height_ratio: float  # Expected aspect ratio (width/height)
    anchor_keywords: List[str] = field(default_factory=list)  # Keywords expected to be found
    threshold_score: float = 0.6  # Default matching threshold
