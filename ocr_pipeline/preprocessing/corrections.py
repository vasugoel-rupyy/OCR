import cv2
import numpy as np
from typing import Optional, Tuple
from scipy import ndimage


class ImageCorrector:
    
    def __init__(self, config: dict):
        self.config = config
        self.enable_skew = config.get('enable_skew_correction', True)
        self.max_skew_angle = config.get('max_skew_angle', 45)
        self.enable_perspective = config.get('enable_perspective_correction', True)
        self.enable_illumination = config.get('enable_illumination_normalization', True)
        self.enable_noise_removal = config.get('enable_noise_removal', True)
        
        self.clahe_clip_limit = config.get('clahe_clip_limit', 2.0)
        self.clahe_tile_grid_size = tuple(config.get('clahe_tile_grid_size', [8, 8]))
        
        self.median_blur_ksize = config.get('median_blur_ksize', 3)
        self.bilateral_d = config.get('bilateral_d', 9)
        self.bilateral_sigma_color = config.get('bilateral_sigma_color', 75)
        self.bilateral_sigma_space = config.get('bilateral_sigma_space', 75)
    
    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        if not self.enable_skew:
            return image
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return image
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)
        
        median_angle = np.median(angles)
        
        if abs(median_angle) > 0.5 and abs(median_angle) < self.max_skew_angle:
            rotated = self._rotate_image(image, median_angle)
            return rotated
        
        return image
    
    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        if not self.enable_perspective:
            return image
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:
            return self._four_point_transform(image, approx.reshape(4, 2))
        
        return image
    
    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        if not self.enable_illumination:
            return image
        
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )
        
        if len(image.shape) == 2:
            return clahe.apply(image)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        if not self.enable_noise_removal:
            return image
        
        denoised = cv2.medianBlur(image, self.median_blur_ksize)
        
        denoised = cv2.bilateralFilter(
            denoised,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space
        )
        
        return denoised
    
    def apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        return binary
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def _four_point_transform(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        rect = self._order_points(points)
        (tl, tr, br, bl) = rect
        
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(rect, dst)
        
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        
        return warped
    
    def _order_points(self, points: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        
        return rect
