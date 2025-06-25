import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import requests
import json
import datetime
from dotenv import load_dotenv

from image_preprocessing import ImagePreprocessor, preprocess_batch
from model_wrapper import ClothingDetectionModel, load_model

# Load environment variables
load_dotenv()

# Get backend URL from environment or use default
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5000/api/clothing-analysis')

def display_results(image, results):
    """Display the original image with bounding boxes and detected attributes"""
    img = image.copy()
    
    # Draw bounding boxes for clothing items
    for i, detection in enumerate(results['detections']):
        x1, y1, x2, y2 = detection['bbox']
        category = detection['category']
        confidence = detection['confidence']
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{category} ({confidence:.2f})"
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add attributes if available
        if i in results['attributes']:
            attrs = results['attributes'][i]
            color = attrs.get('color', 'unknown')
            pattern = attrs.get('pattern', 'unknown')
            attr_text = f"Color: {color}, Pattern: {pattern}"
            cv2.putText(img, attr_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Clothing Detection Results')
    plt.show()

def send_to_backend(results, image_path=None):
    """Send analysis results to backend server"""
    try:
        # Prepare payload
        payload = {
            'timestamp': datetime.datetime.now().isoformat(),
            'image_name': os.path.basename(image_path) if image_path else 'unknown',
            'analysis_data': {
                'clothing_items': results['detections'],
                'attributes': results['attributes'],
                'preprocessing_metadata': results.get('preprocessing_metadata', {})
            },
            'skin_analysis': {
                'skin_tone': 'not_analyzed',  # Placeholder for skin tone analysis
                'undertone': 'not_analyzed'   # Placeholder for undertone analysis
            },
            'recommendations': {
                'outfit_recommendations': [],  # Placeholder for outfit recommendations
                'color_recommendations': []    # Placeholder for color recommendations
            }
        }
        
        # Make POST request to backend
        headers = {'Content-Type': 'application/json'}
        response = requests.post(BACKEND_URL, json=payload, headers=headers)
        
        # Check response status
        if response.status_code == 200:
            print(f"Results successfully sent to backend server. Response: {response.json()}")
            return True
        else:
            print(f"Error sending results to backend: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        print(f"Exception occurred when sending results to backend: {str(e)}")
        return False

def process_image(image_path, display=True, send_to_server=True):
    """Process a single image and optionally display results"""
    # Load model
    model = load_model(model_type='yolo')
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    
    # Run prediction
    results = model.predict(image)
    
    # Display results
    if display and 'error' not in results:
        display_results(image, results)
    
    # Print detection summary
    print(f"\nDetected {len(results['detections'])} clothing items:")
    for i, detection in enumerate(results['detections']):
        category = detection['category']
        confidence = detection['confidence']
        print(f"  Item {i+1}: {category} (confidence: {confidence:.2f})")
        
        # Print attributes if available
        if i in results['attributes']:
            attrs = results['attributes'][i]
            print(f"    Color: {attrs.get('color', 'unknown')}")
            print(f"    Pattern: {attrs.get('pattern', 'unknown')}")
    
    # Print preprocessing metadata
    if 'preprocessing_metadata' in results:
        meta = results['preprocessing_metadata']
        print("\nPreprocessing Info:")
        if 'quality_issues' in meta:
            print(f"  Quality issues: {', '.join(meta['quality_issues']) if meta['quality_issues'] else 'None'}")
        if 'preprocessing_steps' in meta:
            print(f"  Applied steps: {', '.join(meta['preprocessing_steps'])}")
    
    # Send results to backend server
    if send_to_server and 'error' not in results:
        send_to_backend(results, image_path)
    
    return results

def process_directory(directory_path, output_path=None, send_to_server=True):
    """Process all images in a directory"""
    # Load model
    model = load_model(model_type='yolo')
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} not found")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    for file in os.listdir(directory_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(directory_path, file))
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    
    print(f"Found {len(image_files)} images. Processing batch...")
    
    # Process images in batch
    batch_results = model.predict_batch(image_files)
    
    # Create output directory if specified
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Process and save results
    for i, (image_path, results) in enumerate(zip(image_files, batch_results)):
        print(f"\nProcessing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        if 'error' in results:
            print(f"  Error: {results['error']}")
            continue
        
        # Load image
        image = cv2.imread(image_path)
        
        # Draw results on image
        img_with_results = image.copy()
        for j, detection in enumerate(results['detections']):
            x1, y1, x2, y2 = detection['bbox']
            category = detection['category']
            confidence = detection['confidence']
            
            # Draw rectangle
            cv2.rectangle(img_with_results, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{category} ({confidence:.2f})"
            cv2.putText(img_with_results, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Print detection summary
        print(f"  Detected {len(results['detections'])} clothing items")
        
        # Save annotated image if output path is specified
        if output_path:
            output_file = os.path.join(output_path, f"result_{os.path.basename(image_path)}")
            cv2.imwrite(output_file, img_with_results)
            print(f"  Saved results to {output_file}")
        
        # Send results to backend server
        if send_to_server:
            send_to_backend(results, image_path)
    
    print("\nBatch processing complete!")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Clothing Detection Demo')
    parser.add_argument('--input', type=str, required=True, help='Input image file or directory')
    parser.add_argument('--output', type=str, help='Output directory for batch processing')
    parser.add_argument('--no-display', action='store_true', help='Disable display for single image processing')
    parser.add_argument('--no-server', action='store_true', help='Disable sending results to backend server')
    args = parser.parse_args()
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process single image
        process_image(args.input, display=not args.no_display, send_to_server=not args.no_server)
    elif os.path.isdir(args.input):
        # Process directory
        process_directory(args.input, args.output, send_to_server=not args.no_server)
    else:
        print(f"Error: Input '{args.input}' is not a valid file or directory")

if __name__ == "__main__":
    main() 