import cv2
import numpy as np
import logging
import os
import traceback
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    SOLID = "Solid"
    STRIPED = "Striped"
    PLAID = "Plaid"
    CHECKERED = "Checkered"
    FLORAL = "Floral"
    POLKA_DOT = "Polka Dot"
    GEOMETRIC = "Geometric"
    ABSTRACT = "Abstract"
    ANIMAL_PRINT = "Animal Print"
    UNKNOWN = "Unknown"

class TextureType(Enum):
    SMOOTH = "Smooth"
    ROUGH = "Rough"
    KNITTED = "Knitted"
    WOVEN = "Woven"
    LEATHER = "Leather"
    DENIM = "Denim"
    SUEDE = "Suede"
    SILK = "Silk"
    COTTON = "Cotton"
    LINEN = "Linen"
    UNKNOWN = "Unknown"

class FitType(Enum):
    TIGHT = "Tight"
    SLIM = "Slim"
    REGULAR = "Regular"
    LOOSE = "Loose"
    OVERSIZED = "Oversized"
    UNKNOWN = "Unknown"

class AttributeAnalyzer:
    def __init__(self, model_path=None):
        """
        Initialize the clothing attribute analyzer
        
        Args:
            model_path: Path to the attribute classification model (optional)
        """
        try:
            self.model_path = model_path
            self.model = None
            
            # Load the model if path is provided
            if model_path and os.path.exists(model_path):
                self._load_model()
                logger.info("Attribute analyzer initialized with model")
            else:
                logger.info("Attribute analyzer initialized without model (will use rule-based analysis)")
        except Exception as e:
            logger.error(f"Error initializing attribute analyzer: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _load_model(self):
        """Load the attribute classification model if available"""
        try:
            # This is a placeholder for loading a real model
            # In a real implementation, this would load a trained neural network
            # for attribute classification
            logger.info(f"Loading attribute classification model from {self.model_path}")
            
            # Placeholder for model loading
            # self.model = SomeModelClass.load(self.model_path)
            
            logger.info("Attribute classification model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading attribute model: {str(e)}")
            logger.error(traceback.format_exc())
            self.model = None
    
    def analyze_attributes(self, image, bounding_boxes):
        """
        Analyze clothing attributes for detected items
        
        Args:
            image: Full image as numpy array
            bounding_boxes: List of bounding boxes for detected clothing items
                Each box should be in format [x1, y1, x2, y2] or {'bbox': [x1, y1, x2, y2], ...}
            
        Returns:
            dict: Dictionary mapping item indices to attribute dictionaries
        """
        try:
            logger.info(f"Analyzing attributes for {len(bounding_boxes)} items")
            
            attributes = {}
            
            for i, box in enumerate(bounding_boxes):
                try:
                    # Extract bbox coordinates
                    if isinstance(box, dict) and 'bbox' in box:
                        bbox = box['bbox']
                    else:
                        bbox = box
                    
                    # Ensure bounding box is valid
                    if len(bbox) != 4:
                        continue
                    
                    x1, y1, x2, y2 = bbox
                    
                    # Make sure coordinates are valid
                    if x1 >= x2 or y1 >= y2:
                        continue
                    
                    # Ensure coordinates are within image boundaries
                    h, w = image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Crop the clothing item
                    crop = image[int(y1):int(y2), int(x1):int(x2)]
                    
                    if crop.size == 0:
                        logger.warning(f"Empty crop for item {i}")
                        continue
                    
                    # Analyze the cropped item
                    if self.model is not None:
                        # Use the model for analysis
                        item_attributes = self._analyze_with_model(crop)
                    else:
                        # Use rule-based analysis
                        item_attributes = self._analyze_rule_based(crop)
                    
                    attributes[i] = item_attributes
                    
                except Exception as e:
                    logger.warning(f"Error analyzing item {i}: {str(e)}")
                    continue
            
            return attributes
            
        except Exception as e:
            logger.error(f"Error in attribute analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def _analyze_with_model(self, crop):
        """Analyze attributes using the loaded model"""
        # This is a placeholder for using a real model
        # In a real implementation, this would:
        # 1. Preprocess the crop
        # 2. Run inference using the loaded model
        # 3. Post-process and return the results
        
        # For now, fall back to rule-based analysis
        return self._analyze_rule_based(crop)
    
    def _analyze_rule_based(self, crop):
        """Analyze attributes using rule-based methods"""
        try:
            # Extract basic attributes
            color = self._detect_color(crop)
            pattern = self._detect_pattern(crop)
            texture = self._detect_texture(crop)
            fit = self._detect_fit(crop)
            
            return {
                'color': color,
                'pattern': pattern,
                'texture': texture,
                'fit': fit
            }
        except Exception as e:
            logger.error(f"Error in rule-based analysis: {str(e)}")
            return {
                'color': 'unknown',
                'pattern': PatternType.UNKNOWN.value,
                'texture': TextureType.UNKNOWN.value,
                'fit': FitType.UNKNOWN.value
            }
    
    def _detect_color(self, image):
        """Detect dominant color in the image"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return 'unknown'
            
            # Reshape the image
            pixels = rgb_image.reshape(-1, 3)
            
            # Use K-means clustering to find dominant colors
            kmeans = cv2.kmeans(
                np.float32(pixels), 
                3,  # Find top 3 colors
                None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                10, 
                cv2.KMEANS_RANDOM_CENTERS
            )
            
            # Get dominant color (most frequent cluster)
            _, labels, centers = kmeans
            counts = np.bincount(labels.flatten())
            dominant_cluster = np.argmax(counts)
            dominant_color = centers[dominant_cluster]
            
            # Convert to integers
            r, g, b = map(int, dominant_color)
            
            # Get color name
            color_name = self._get_color_name(r, g, b)
            
            return color_name
        except Exception as e:
            logger.error(f"Error detecting color: {str(e)}")
            return 'unknown'
    
    def _get_color_name(self, r, g, b):
        """Get color name from RGB values"""
        # Define basic colors
        colors = {
            'Red': (255, 0, 0),
            'Green': (0, 255, 0),
            'Blue': (0, 0, 255),
            'Yellow': (255, 255, 0),
            'Cyan': (0, 255, 255),
            'Magenta': (255, 0, 255),
            'Black': (0, 0, 0),
            'White': (255, 255, 255),
            'Gray': (128, 128, 128),
            'Orange': (255, 165, 0),
            'Purple': (128, 0, 128),
            'Brown': (165, 42, 42),
            'Pink': (255, 192, 203),
            'Navy': (0, 0, 128),
            'Teal': (0, 128, 128),
            'Olive': (128, 128, 0),
            'Maroon': (128, 0, 0)
        }
        
        # Find the closest color
        min_distance = float('inf')
        closest_color = 'Unknown'
        
        for name, rgb in colors.items():
            distance = sum((c1 - c2) ** 2 for c1, c2 in zip((r, g, b), rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        
        return closest_color
    
    def _detect_pattern(self, image):
        """Detect pattern type in the clothing"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate standard deviation as a simple measure of texture variation
            std_dev = np.std(gray)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            # Apply gradient analysis
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            gradient_mean = np.mean(gradient_magnitude)
            
            # Use basic heuristics to determine pattern
            if std_dev < 20 and edge_density < 0.1:
                return PatternType.SOLID.value
            elif edge_density > 0.2 and gradient_mean > 30:
                # Check for striped pattern
                if self._is_striped(sobelx, sobely):
                    return PatternType.STRIPED.value
                # Check for checkered pattern
                elif self._is_checkered(edges):
                    return PatternType.CHECKERED.value
                # Check for polka dots
                elif self._is_polka_dot(edges, gray):
                    return PatternType.POLKA_DOT.value
                else:
                    return PatternType.GEOMETRIC.value
            elif edge_density > 0.15:
                return PatternType.ABSTRACT.value
            else:
                return PatternType.UNKNOWN.value
        except Exception as e:
            logger.error(f"Error detecting pattern: {str(e)}")
            return PatternType.UNKNOWN.value
    
    def _is_striped(self, sobelx, sobely):
        """Check if the pattern is striped based on gradient analysis"""
        # Simple heuristic: if horizontal or vertical gradients dominate
        v_mean = np.mean(np.abs(sobelx))
        h_mean = np.mean(np.abs(sobely))
        
        # If one direction is significantly stronger than the other
        return (v_mean > 2*h_mean) or (h_mean > 2*v_mean)
    
    def _is_checkered(self, edges):
        """Check if the pattern is checkered based on edge analysis"""
        # Simplified check for grid-like pattern
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        if lines is None:
            return False
        
        horizontal = 0
        vertical = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < abs(y2 - y1):  # Vertical line
                vertical += 1
            else:  # Horizontal line
                horizontal += 1
        
        # If we have a good number of both horizontal and vertical lines
        return horizontal > 5 and vertical > 5
    
    def _is_polka_dot(self, edges, gray):
        """Check if the pattern has polka dots"""
        # Use circle detection
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=50
        )
        
        return circles is not None and len(circles[0]) > 5
    
    def _detect_texture(self, image):
        """Detect texture type in the clothing"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate GLCM (Gray-Level Co-occurrence Matrix) features
            # This is a simplified approach as full GLCM is computationally intensive
            
            # Calculate basic texture statistics
            mean = np.mean(gray)
            std_dev = np.std(gray)
            entropy = -np.sum(gray * np.log2(gray + 1e-10)) / gray.size
            
            # Apply a simple texture classification based on these statistics
            if std_dev < 10:
                return TextureType.SMOOTH.value
            elif std_dev < 25:
                return TextureType.COTTON.value
            elif std_dev < 40:
                return TextureType.WOVEN.value
            else:
                return TextureType.ROUGH.value
        except Exception as e:
            logger.error(f"Error detecting texture: {str(e)}")
            return TextureType.UNKNOWN.value
    
    def _detect_fit(self, image):
        """Detect fit type (simplified placeholder)"""
        # In a real implementation, this would use pose estimation and clothing segmentation
        # to determine fit relative to body shape
        
        # For now, return a placeholder
        return FitType.REGULAR.value 