"""Data models for OCR results."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class WordData:
    """Data for a single OCR word."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    line_num: int
    word_num: int
    is_numeric: bool = False
    is_stopword: bool = False


@dataclass
class LineData:
    """Data for a single OCR line."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    words: List[WordData] = field(default_factory=list)


@dataclass
class OCRResult:
    """Complete OCR result for a document."""
    full_text: str
    mean_confidence: float
    words: List[WordData] = field(default_factory=list)
    lines: List[LineData] = field(default_factory=list)
    total_words: int = 0
    low_confidence_words: int = 0
    numeric_words: int = 0
    
    def get_stats(self) -> dict:
        """Get OCR statistics."""
        return {
            'total_words': self.total_words,
            'mean_confidence': self.mean_confidence,
            'low_confidence_words': self.low_confidence_words,
            'numeric_words': self.numeric_words
        }

    def get_words_by_confidence(self, min_confidence: float) -> List[WordData]:
        """Get words above a confidence threshold.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of words above threshold
        """
        return [w for w in self.words if w.confidence >= min_confidence]
    
    def get_numeric_words(self) -> List[WordData]:
        """Get all numeric words.
        
        Returns:
            List of numeric words
        """
        return [w for w in self.words if w.is_numeric]
    
    def get_text_by_region(self, region: Tuple[int, int, int, int]) -> str:
        """Get text within a specific region."""
        x, y, w, h = region
        words_in_region = []
        
        for word in self.words:
            wx, wy, ww, wh = word.bbox
            # Check if word center is in region
            word_center_x = wx + ww // 2
            word_center_y = wy + wh // 2
            
            if (x <= word_center_x <= x + w and 
                y <= word_center_y <= y + h):
                words_in_region.append(word.text)
        
        return ' '.join(words_in_region)

    def append_page(self, other: 'OCRResult', page_num: int) -> None:
        """Append results from another page.
        
        Args:
           other: OCRResult from the next page
           page_num: Page number being appended
        """
        if not other:
            return
            
        # Append text with separator
        separator = f"\n--- Page {page_num} ---\n"
        if self.full_text:
            self.full_text += separator
        self.full_text += other.full_text
        
        # Append words and lines
        # NOTE: We keep coordinates as-is (per-page) since processors 
        # usually rely on proximity and keywords rather than global coordinates
        self.words.extend(other.words)
        self.lines.extend(other.lines)
        
        # Update statistics
        total_prev = self.total_words
        total_new = other.total_words
        
        if total_prev + total_new > 0:
            self.mean_confidence = (
                self.mean_confidence * total_prev + 
                other.mean_confidence * total_new
            ) / (total_prev + total_new)
            
        self.total_words += total_new
        self.low_confidence_words += other.low_confidence_words
        self.numeric_words += other.numeric_words
