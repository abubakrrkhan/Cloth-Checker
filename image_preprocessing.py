import cv2
import numpy as np
from skimage import exposure, restoration
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import math
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Production-grade image preprocessing pipeline for clothing detection and analysis.
    Handles various real-world image conditions like poor lighting, noise, and occlusions.
    """
    
    def __init__(self, config=None):
        """
        Initialize the image preprocessor with optional configuration.
        
        Args:
            config (dict, optional): Configuration parameters for the preprocessor.
        """
        self.config = {
            # Default configuration
            'target_size': (512, 512),
            'normalization': 'imagenet',  # Options: 'imagenet', 'centered', 'minmax'
            'use_segmentation': True,
            'use_person_detection': True,
            'use_garment_detection': True,
            'use_clahe': True,
            'use_denoising': True,
            'clahe_clip_limit': 3.0,
            'clahe_tile_grid_size': (8, 8),
            'detect_brightness': True,
            'brightness_threshold': 120,  # Average brightness threshold
            'detect_contrast': True,
            'contrast_threshold': 50,  # Std dev threshold for contrast
            'detect_blur': True,
            'blur_threshold': 100,  # Laplacian variance threshold
            'min_person_confidence': 0.5,
            'min_garment_confidence': 0.3,
            'use_perspective_correction': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Update config with user provided values
        if config:
            self.config.update(config)
            
        # Initialize models
        self._init_models()
        
        logger.info(f"Image preprocessor initialized with device: {self.config['device']}")
    
    def _init_models(self):
        """Initialize detection and segmentation models"""
        try:
            # Load person detection model
            if self.config['use_person_detection']:
                logger.info("Loading person detection model...")
                model_path = os.path.join('models', 'yolov8n.pt')
                if os.path.exists(model_path):
                    self.person_model = YOLO(model_path)
                else:
                    self.person_model = YOLO('yolov8n.pt')  # Will download from internet
                logger.info("Person detection model loaded")
            
            # Load garment segmentation model
            if self.config['use_garment_detection']:
                logger.info("Loading garment detection model...")
                # Here we'd typically load a specialized clothing segmentation model
                # For this example, we'll use YOLOv8 but in practice you might want a dedicated model
                try:
                    model_path = os.path.join('models', 'clothing_detector.pt')
                    if os.path.exists(model_path):
                        self.garment_model = YOLO(model_path)
                    else:
                        # Fallback to person detection if no specialized model is available
                        self.garment_model = self.person_model
                        logger.warning("Specialized clothing model not found, using person model as fallback")
                except Exception as e:
                    logger.error(f"Error loading garment model: {e}")
                    self.config['use_garment_detection'] = False
                    
            # Create normalization transforms
            self._create_transforms()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _create_transforms(self):
        """Create image transforms for normalization"""
        if self.config['normalization'] == 'imagenet':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.config['normalization'] == 'centered':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif self.config['normalization'] == 'minmax':
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
    
    def process(self, image):
        """
        Process an image through the full preprocessing pipeline.
        
        Args:
            image: Input image as numpy array (BGR) or file path
            
        Returns:
            dict: Processed data including normalized image and metadata
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image not found: {image}")
                image = cv2.imread(image)
                if image is None:
                    raise ValueError(f"Failed to load image: {image}")
            
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
                
            # Store original for reference
            original_image = rgb_image.copy()
            
            # Analyze image quality
            quality_issues = self.analyze_image_quality(rgb_image)
            logger.info(f"Image quality issues: {quality_issues}")
            
            # Create a metadata dictionary
            metadata = {
                'original_size': rgb_image.shape[:2],
                'quality_issues': quality_issues,
                'preprocessing_steps': []
            }
            
            # Resize for better processing (keep original ratio)
            img_resized = self.resize_image(rgb_image, self.config['target_size'])
            metadata['preprocessing_steps'].append('resize')
            
            # Apply enhancements based on image quality
            enhanced_image = self.enhance_image(img_resized, quality_issues)
            metadata['preprocessing_steps'].append('enhance')
            
            # Detect person in the image
            person_bbox = None
            if self.config['use_person_detection']:
                person_detections = self.detect_persons(enhanced_image)
                metadata['person_detections'] = person_detections
                
                if person_detections:
                    # Get the person with the largest area
                    largest_person = max(person_detections, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))
                    person_bbox = largest_person['bbox']
                    
                    # Crop to person
                    try:
                        x1, y1, x2, y2 = person_bbox
                        # Add padding
                        h, w = enhanced_image.shape[:2]
                        pad = 20  # pixels
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x2 + pad)
                        y2 = min(h, y2 + pad)
                        enhanced_image = enhanced_image[int(y1):int(y2), int(x1):int(x2)]
                        metadata['preprocessing_steps'].append('person_crop')
                    except Exception as e:
                        logger.warning(f"Error cropping to person: {e}")
            
            # Detect garments
            if self.config['use_garment_detection']:
                garment_detections = self.detect_garments(enhanced_image)
                metadata['garment_detections'] = garment_detections
            
            # Apply perspective correction if needed
            if self.config['use_perspective_correction'] and 'skewed_perspective' in quality_issues:
                try:
                    enhanced_image = self.correct_perspective(enhanced_image)
                    metadata['preprocessing_steps'].append('perspective_correction')
                except Exception as e:
                    logger.warning(f"Error in perspective correction: {e}")
            
            # Final resize to model's expected input dimensions
            processed_image = self.resize_image(enhanced_image, self.config['target_size'])
            
            # Normalize the image according to model requirements
            normalized_image = self.normalize_image(processed_image)
            metadata['preprocessing_steps'].append('normalize')
            
            # Create a result dictionary
            result = {
                'original': original_image,
                'processed': processed_image,
                'normalized': normalized_image,
                'metadata': metadata
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in image processing pipeline: {e}")
            # Return a simplified result with the original image
            return {
                'original': image if isinstance(image, np.ndarray) else None,
                'processed': None,
                'normalized': None,
                'metadata': {'error': str(e)}
            }
    
    def analyze_image_quality(self, image):
        """
        Analyze image quality and identify potential issues.
        
        Args:
            image: Input RGB image
            
        Returns:
            list: List of detected quality issues
        """
        issues = []
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Check image size
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            issues.append('low_resolution')
        
        # Check for low brightness
        if self.config['detect_brightness']:
            avg_brightness = np.mean(gray)
            if avg_brightness < self.config['brightness_threshold']:
                issues.append('low_brightness')
            elif avg_brightness > 240:
                issues.append('high_brightness')
        
        # Check for low contrast
        if self.config['detect_contrast']:
            contrast = np.std(gray)
            if contrast < self.config['contrast_threshold']:
                issues.append('low_contrast')
        
        # Check for blur
        if self.config['detect_blur']:
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < self.config['blur_threshold']:
                issues.append('blurry')
        
        # Check for skewed perspective (simple heuristic)
        # A more robust implementation would use line detection
        try:
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is not None:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if x2 - x1 == 0:  # Vertical line
                        angle = 90
                    else:
                        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    angles.append(angle)
                
                # Check if angles vary significantly from horizontal and vertical
                horizontal_angles = [a for a in angles if abs(a) < 30 or abs(a) > 150]
                vertical_angles = [a for a in angles if abs(a - 90) < 30 or abs(a + 90) < 30]
                
                if len(horizontal_angles) > 0 and len(vertical_angles) > 0:
                    h_deviation = np.std(horizontal_angles)
                    v_deviation = np.std(vertical_angles)
                    if h_deviation > 5 or v_deviation > 5:
                        issues.append('skewed_perspective')
        except Exception as e:
            logger.warning(f"Error checking perspective: {e}")
        
        return issues

    def enhance_image(self, image, quality_issues):
        """
        Apply enhancements based on detected quality issues.
        
        Args:
            image: Input RGB image
            quality_issues: List of quality issues
            
        Returns:
            Enhanced RGB image
        """
        enhanced = image.copy()
        
        # Convert to PIL Image for some enhancements
        pil_img = Image.fromarray(enhanced)
        
        # Apply brightness correction
        if 'low_brightness' in quality_issues:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1.5)  # Increase brightness by 50%
        elif 'high_brightness' in quality_issues:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(0.7)  # Reduce brightness by 30%
        
        # Apply contrast enhancement
        if 'low_contrast' in quality_issues:
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.5)  # Increase contrast by 50%
            
        # Convert back to numpy array
        enhanced = np.array(pil_img)
        
        # Apply adaptive histogram equalization
        if self.config['use_clahe'] and ('low_contrast' in quality_issues or 'low_brightness' in quality_issues):
            clahe = cv2.createCLAHE(
                clipLimit=self.config['clahe_clip_limit'],
                tileGridSize=self.config['clahe_tile_grid_size']
            )
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            lab_planes = cv2.split(lab)
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Apply denoising for blur
        if self.config['use_denoising'] and 'blurry' in quality_issues:
            # Use fastNlMeansDenoisingColored for color images
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced, None, 10, 10, 7, 21
            )
            
            # Apply gentle sharpening
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced

    def resize_image(self, image, target_size, keep_aspect_ratio=True):
        """
        Resize image while optionally maintaining aspect ratio.
        
        Args:
            image: Input image
            target_size: Target size as tuple (width, height)
            keep_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        if keep_aspect_ratio:
            # Calculate new dimensions preserving aspect ratio
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize the image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create a black canvas of target size
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Place the resized image in the center of the canvas
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        else:
            # Resize directly to target size
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image):
        """
        Normalize image according to model requirements.
        
        Args:
            image: Input RGB image
            
        Returns:
            Normalized image tensor
        """
        # Convert to PIL image for torchvision transforms
        pil_image = Image.fromarray(image)
        
        # Apply normalization
        return self.transform(pil_image)
    
    def detect_persons(self, image):
        """
        Detect persons in the image.
        
        Args:
            image: Input RGB image
            
        Returns:
            list: List of detected persons with bounding boxes and confidence scores
        """
        if not self.config['use_person_detection']:
            return []
            
        try:
            # Convert to BGR for YOLO
            results = self.person_model(image, classes=[0])  # 0 is the class index for person
            
            persons = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    conf = float(box.conf)
                    if conf >= self.config['min_person_confidence']:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        persons.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        })
            
            return persons
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            return []
    
    def detect_garments(self, image):
        """
        Detect garments in the image.
        
        Args:
            image: Input RGB image
            
        Returns:
            list: List of detected garments with bounding boxes, classes, and confidence scores
        """
        if not self.config['use_garment_detection']:
            return []
            
        try:
            results = self.garment_model(image)
            
            garments = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    conf = float(box.conf)
                    cls = int(box.cls)
                    cls_name = self.garment_model.names[cls]
                    
                    # Filter clothing related classes
                    clothing_classes = ["person", "tie", "backpack", "umbrella", "handbag", 
                                    "suitcase", "clothing", "jacket", "shirt", "pants", 
                                    "dress", "shoes", "hat", "skirt", "suit"]
                    
                    if conf >= self.config['min_garment_confidence'] and cls_name.lower() in [c.lower() for c in clothing_classes]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        garments.append({
                            'bbox': [x1, y1, x2, y2],
                            'class': cls_name,
                            'confidence': conf
                        })
            
            return garments
        except Exception as e:
            logger.error(f"Error in garment detection: {e}")
            return []
    
    def correct_perspective(self, image):
        """
        Apply perspective correction to the image.
        
        Args:
            image: Input RGB image
            
        Returns:
            Perspective-corrected image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) < 2:
            # Not enough lines detected for perspective correction
            return image
        
        # Extract main horizontal and vertical lines
        h_lines = []
        v_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):  # Horizontal line
                h_lines.append(line[0])
            else:  # Vertical line
                v_lines.append(line[0])
        
        # If we don't have enough horizontal or vertical lines, return original
        if len(h_lines) < 2 or len(v_lines) < 2:
            return image
        
        # Sort horizontal lines by y-coordinate and vertical lines by x-coordinate
        h_lines.sort(key=lambda x: (x[1] + x[3]) / 2)
        v_lines.sort(key=lambda x: (x[0] + x[2]) / 2)
        
        # Get top-left, top-right, bottom-left, and bottom-right corners
        try:
            # Estimate corners based on intersections of extreme lines
            top_left = self._line_intersection(v_lines[0], h_lines[0])
            top_right = self._line_intersection(v_lines[-1], h_lines[0])
            bottom_left = self._line_intersection(v_lines[0], h_lines[-1])
            bottom_right = self._line_intersection(v_lines[-1], h_lines[-1])
            
            # Check if corners are valid
            corners = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)
            if np.any(corners < 0) or np.any(corners > np.array(image.shape[1::-1])):
                # Invalid corners, return original image
                return image
            
            # Define the destination points (rectangle)
            h, w = image.shape[:2]
            dst_points = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
            
            # Calculate perspective transform matrix
            M = cv2.getPerspectiveTransform(corners, dst_points)
            
            # Apply perspective transformation
            corrected = cv2.warpPerspective(image, M, (w, h))
            
            return corrected
        except Exception as e:
            logger.warning(f"Error in perspective correction calculation: {e}")
            return image
    
    def _line_intersection(self, line1, line2):
        """Calculate the intersection point of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Line1 represented as a1x + b1y = c1
        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1
        
        # Line2 represented as a2x + b2y = c2
        a2 = y4 - y3
        b2 = x3 - x4
        c2 = a2 * x3 + b2 * y3
        
        determinant = a1 * b2 - a2 * b1
        
        if determinant == 0:
            # Lines are parallel
            return (x1 + x3) / 2, (y1 + y3) / 2  # Return midpoint as fallback
        
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        
        return x, y


def preprocess_batch(images, config=None):
    """
    Process a batch of images using the image preprocessing pipeline.
    
    Args:
        images: List of images or file paths
        config: Configuration for the preprocessor
        
    Returns:
        dict: Processed batch data
    """
    preprocessor = ImagePreprocessor(config)
    
    results = []
    for img in images:
        results.append(preprocessor.process(img))
    
    # Combine all normalized images into a batch tensor
    normalized_batch = torch.stack([r['normalized'] for r in results if r['normalized'] is not None])
    
    return {
        'batch': normalized_batch,
        'results': results
    } 