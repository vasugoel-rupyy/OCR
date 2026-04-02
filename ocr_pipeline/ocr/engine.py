import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import Dict, List, Set, Tuple
import re
from .models import OCRResult, WordData, LineData

class PaddleOCREngine:
    
    def __init__(self, config: Dict):
        self.config = config
        paddle_config = config.get('paddle_ocr', {})
        
        import platform
        use_onnx = platform.system() == 'Darwin'
        
        self.ocr = PaddleOCR(
            use_angle_cls=paddle_config.get('use_angle_cls', True),
            lang='en',
            use_gpu=paddle_config.get('use_gpu', False),
            show_log=paddle_config.get('show_log', False),
            use_onnx=use_onnx
        )
        
        self.min_word_confidence = config.get('min_word_confidence', 50)
        self.min_words_detected = config.get('min_words_detected', 5)
        self.stopwords = set(config.get('stopwords', []))
        self.numeric_weight = config.get('numeric_token_weight', 1.5)
        self.alpha_weight = config.get('alpha_token_weight', 1.0)
        self.stopword_weight = config.get('stopword_weight', 0.3)
    def extract_text(self, image: np.ndarray) -> OCRResult:
        result = self.ocr.ocr(image, cls=True)
        
        words = []
        lines = []
        full_text_lines = []
        
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

        for line_idx, line_res in enumerate(result[0]):
            box, (text, confidence) = line_res
            text = text.strip()
            confidence = float(confidence) * 100
            
            if not text:
                continue
                
            box = np.array(box).astype(np.int32)
            x_min = np.min(box[:, 0])
            y_min = np.min(box[:, 1])
            x_max = np.max(box[:, 0])
            y_max = np.max(box[:, 1])
            w = x_max - x_min
            h = y_max - y_min
            line_bbox = (int(x_min), int(y_min), int(w), int(h))
            
            full_text_lines.append(text)
            
            line_words_tokens = text.split()
            
            total_chars = len(text)
            if total_chars == 0:
                continue
                
            avg_char_width = w / total_chars
            current_x = x_min
            
            line_word_objects = []
            
            for word_idx, token in enumerate(line_words_tokens):
                token_len = len(token)
                word_w = int(token_len * avg_char_width)
                
                is_numeric = self._is_numeric(token)
                is_stopword = token.lower() in self.stopwords
                
                word_data = WordData(
                    text=token,
                    confidence=confidence,
                    bbox=(int(current_x), int(y_min), int(word_w), int(h)),
                    line_num=line_idx + 1,
                    word_num=word_idx + 1,
                    is_numeric=is_numeric,
                    is_stopword=is_stopword
                )
                words.append(word_data)
                line_word_objects.append(word_data)
                
                current_x += word_w + avg_char_width
            
            line_data = LineData(
                text=text,
                confidence=confidence,
                bbox=line_bbox,
                words=line_word_objects
            )
            lines.append(line_data)

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
        if ocr_result.total_words == 0:
            return 0.0
        
        if ocr_result.total_words < self.min_words_detected:
            return 0.0
        
        low_conf_ratio = ocr_result.low_confidence_words / ocr_result.total_words
        if low_conf_ratio > 0.4:
            return 0.0
        
        normalized_confidence = ocr_result.mean_confidence / 100.0
        
        numeric_ratio = ocr_result.numeric_words / ocr_result.total_words
        numeric_bonus = min(0.1, numeric_ratio * 0.2)
        
        final_score = min(1.0, normalized_confidence + numeric_bonus)
        
        return final_score
    
    def _calculate_weighted_confidence(self, words: List[WordData]) -> float:
        if not words:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for word in words:
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
        cleaned = re.sub(r'[,.\s$€£¥]', '', text)
        
        if not cleaned:
            return False
        
        digit_count = sum(c.isdigit() for c in cleaned)
        return digit_count / len(cleaned) > 0.5


def extract_text_from_image(image: np.ndarray, config: Dict) -> OCRResult:
    engine = PaddleOCREngine(config)
    return engine.extract_text(image)
