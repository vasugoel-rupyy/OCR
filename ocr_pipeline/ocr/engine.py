"""OCR engine integration using PaddleOCR."""

import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import Dict, List, Set, Tuple
import re
from .models import OCRResult, WordData, LineData

class PaddleOCREngine:
    """PaddleOCR engine wrapper with confidence extraction."""
    
    def __init__(self, config: Dict):
        """Initialize PaddleOCR engine.
        
        Args:
            config: OCR configuration dictionary
        """
        self.config = config
        paddle_config = config.get('paddle_ocr', {})
        
        # Initialize PaddleOCR with ONNX Runtime backend
        # use_angle_cls=True allows detecting rotated text
        # use_onnx=True bypasses native PaddlePaddle inference which segfaults on macOS
        self.ocr = PaddleOCR(
            use_angle_cls=paddle_config.get('use_angle_cls', True),
            lang=paddle_config.get('lang', 'en'),
            use_gpu=paddle_config.get('use_gpu', False),
            show_log=paddle_config.get('show_log', False),
            use_onnx=True
        )
        
        self.min_word_confidence = config.get('min_word_confidence', 50)
        self.min_words_detected = config.get('min_words_detected', 5)
        
        # Load stopwords
        self.stopwords = set(config.get('stopwords', []))
        
        # Confidence weights
        self.numeric_weight = config.get('numeric_token_weight', 1.5)
        self.alpha_weight = config.get('alpha_token_weight', 1.0)
        self.stopword_weight = config.get('stopword_weight', 0.3)
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extract text from image using PaddleOCR.
        
        Args:
            image: Input image (preprocessed)
            
        Returns:
            OCRResult object with detailed information
        """
        # PaddleOCR expects RGB image usually, but handles BGR from cv2 fine.
        # Ensure it's numpy array
        
        result = self.ocr.ocr(image, cls=True)
        
        words = []
        lines = []
        full_text_lines = []
        
        # Handle case where no text is found
        if not result or result[0] is None:
            return OCRResult(
                full_text="",
                mean_confidence=0.0,
                words=[],
                lines=[],
                total_words=0,
                low_confidence_words=0,
                numeric_words=0
            )

        # Iterate over detected lines
        # PaddleOCR result structure: [ [ [ [x1,y1], [x2,y2], ... ], (text, confidence) ], ... ]
        for line_idx, line_res in enumerate(result[0]):
            box, (text, confidence) = line_res
            text = text.strip()
            confidence = float(confidence) * 100  # Convert 0-1 to 0-100 scale to match Tesseract
            
            if not text:
                continue
                
            # Convert box points to x, y, w, h
            # box is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] (roughly)
            box = np.array(box).astype(np.int32)
            x_min = np.min(box[:, 0])
            y_min = np.min(box[:, 1])
            x_max = np.max(box[:, 0])
            y_max = np.max(box[:, 1])
            w = x_max - x_min
            h = y_max - y_min
            line_bbox = (int(x_min), int(y_min), int(w), int(h))
            
            full_text_lines.append(text)
            
            # Split line into words
            line_words_tokens = text.split()
            
            # Estimate word bounding boxes
            # We'll split the line width proportionally to word length (plus spaces)
            total_chars = len(text)
            if total_chars == 0:
                continue
                
            avg_char_width = w / total_chars
            current_x = x_min
            
            line_word_objects = []
            
            for word_idx, token in enumerate(line_words_tokens):
                token_len = len(token)
                word_w = int(token_len * avg_char_width)
                
                # Check for numeric and stopword
                is_numeric = self._is_numeric(token)
                is_stopword = token.lower() in self.stopwords
                
                word_data = WordData(
                    text=token,
                    confidence=confidence, # Assign line confidence to words
                    bbox=(int(current_x), int(y_min), int(word_w), int(h)),
                    line_num=line_idx + 1,
                    word_num=word_idx + 1,
                    is_numeric=is_numeric,
                    is_stopword=is_stopword
                )
                words.append(word_data)
                line_word_objects.append(word_data)
                
                # Advance x (word width + space)
                current_x += word_w + avg_char_width
            
            # Create LineData
            line_data = LineData(
                text=text,
                confidence=confidence,
                bbox=line_bbox,
                words=line_word_objects
            )
            lines.append(line_data)

        # Calculate statistics
        full_text = '\n'.join(full_text_lines)
        
        if words:
            mean_confidence = self._calculate_weighted_confidence(words)
            low_confidence_words = sum(1 for w in words if w.confidence < self.min_word_confidence)
            numeric_words = sum(1 for w in words if w.is_numeric)
        else:
            mean_confidence = 0.0
            low_confidence_words = 0
            numeric_words = 0
            
        return OCRResult(
            full_text=full_text,
            mean_confidence=mean_confidence,
            words=words,
            lines=lines,
            total_words=len(words),
            low_confidence_words=low_confidence_words,
            numeric_words=numeric_words
        )

    def calculate_ocr_confidence_score(self, ocr_result: OCRResult) -> float:
        """Calculate normalized OCR confidence score.
        
        Args:
            ocr_result: OCR result object
            
        Returns:
            Confidence score normalized to [0, 1]
        """
        if ocr_result.total_words == 0:
            return 0.0
        
        # Check minimum words threshold
        if ocr_result.total_words < self.min_words_detected:
            return 0.0
        
        # Penalize high percentage of low confidence words
        low_conf_ratio = ocr_result.low_confidence_words / ocr_result.total_words
        if low_conf_ratio > 0.4:
            return 0.0
        
        # Normalize mean confidence to [0, 1]
        normalized_confidence = ocr_result.mean_confidence / 100.0
        
        # Bonus for numeric content (important for invoices)
        numeric_ratio = ocr_result.numeric_words / ocr_result.total_words
        numeric_bonus = min(0.1, numeric_ratio * 0.2)
        
        final_score = min(1.0, normalized_confidence + numeric_bonus)
        
        return final_score
    
    def _calculate_weighted_confidence(self, words: List[WordData]) -> float:
        """Calculate weighted average confidence.
        
        Args:
            words: List of word data
            
        Returns:
            Weighted mean confidence
        """
        if not words:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for word in words:
            # Determine weight based on word type
            if word.is_stopword:
                weight = self.stopword_weight
            elif word.is_numeric:
                weight = self.numeric_weight
            else:
                weight = self.alpha_weight
            
            weighted_sum += word.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _is_numeric(self, text: str) -> bool:
        """Check if text is primarily numeric.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains numbers
        """
        # Remove common separators
        cleaned = re.sub(r'[,.\s$€£¥]', '', text)
        
        # Check if majority is digits
        if not cleaned:
            return False
        
        digit_count = sum(c.isdigit() for c in cleaned)
        return digit_count / len(cleaned) > 0.5


def extract_text_from_image(image: np.ndarray, config: Dict) -> OCRResult:
    """Convenience function to extract text from image.
    
    Args:
        image: Input image
        config: OCR configuration
        
    Returns:
        OCRResult object
    """
    engine = PaddleOCREngine(config)
    return engine.extract_text(image)
