"""
Image processor for PNG, JPG files
"""
import os
from typing import Dict
from PIL import Image
import io


class ImageProcessor:
    """Process image files"""
    
    @staticmethod
    def process_image(file_path: str) -> Dict:
        """
        Process image file and prepare it for analysis
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with image info and data
        """
        try:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Open and validate image
            image = Image.open(file_path)
            
            # Get image metadata
            width, height = image.size
            format_type = image.format
            mode = image.mode
            
            # Convert image to bytes for API calls
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=format_type)
            img_byte_arr = img_byte_arr.getvalue()
            
            return {
                "file_name": file_name,
                "file_type": file_ext[1:],
                "content": file_path,  # Store path for later use
                "content_type": "image",
                "metadata": {
                    "width": width,
                    "height": height,
                    "format": format_type,
                    "mode": mode
                },
                "image_data": img_byte_arr
            }
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    @staticmethod
    def extract_text_from_image(file_path: str) -> str:
        """
        Extract text from image using OCR (optional feature)
        Requires tesseract to be installed
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Extracted text
        """
        try:
            import pytesseract
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except ImportError:
            return "[OCR not available - tesseract not installed]"
        except Exception as e:
            return f"[Error extracting text from image: {str(e)}]"
