#!/usr/bin/env python3

import os
import sys
import chardet

def detect_encoding(file_path):
    """
    Detect the encoding of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Detected encoding or 'utf-8' if detection fails
    """
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            print(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            return encoding
    except Exception as e:
        print(f"Error detecting encoding: {str(e)}")
        return 'utf-8'  # default to utf-8

def convert_file_encoding(file_path, target_encoding='utf-8'):
    """
    Convert a file to target encoding after detecting its current encoding.
    
    Args:
        file_path: Path to the file to convert
        target_encoding: Target file encoding
    """
    source_encoding = detect_encoding(file_path)
    if source_encoding == target_encoding:
        print(f"File {file_path} is already in {target_encoding} encoding. Skipping.")
        return True
        
    print(f"Converting {file_path} from {source_encoding} to {target_encoding}...")
    
    try:
        # Read content with source encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            
        # Try to decode with detected encoding
        try:
            content = raw_data.decode(source_encoding)
        except UnicodeDecodeError:
            # If that fails, try with utf-16-le (common Windows encoding)
            print(f"Failed to decode with {source_encoding}, trying utf-16-le...")
            content = raw_data.decode('utf-16-le')
            
        # Write content with target encoding
        with open(file_path, 'w', encoding=target_encoding) as file:
            file.write(content)
            
        print(f"Successfully converted {file_path}")
        return True
    except Exception as e:
        print(f"Error converting {file_path}: {str(e)}")
        
        # As a fallback for index.html, try a direct approach
        if os.path.basename(file_path) == 'index.html':
            try:
                print(f"Trying direct approach for index.html...")
                with open(file_path, 'r', encoding='utf-16') as file:
                    content = file.read()
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                print(f"Direct approach succeeded for {file_path}")
                return True
            except Exception as e2:
                print(f"Direct approach failed: {str(e2)}")
                
        return False

def main():
    # Convert all HTML files in the templates directory
    templates_dir = 'templates'
    
    if not os.path.exists(templates_dir):
        print(f"Error: Directory {templates_dir} does not exist")
        return
    
    # Install chardet if not already installed
    try:
        import chardet
    except ImportError:
        print("Installing chardet package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
        import chardet
    
    # Convert all HTML files
    for filename in os.listdir(templates_dir):
        if filename.endswith('.html'):
            file_path = os.path.join(templates_dir, filename)
            convert_file_encoding(file_path, 'utf-8')

if __name__ == "__main__":
    main() 