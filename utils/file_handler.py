"""
File handling utilities
"""
import os
import shutil
from typing import List, Dict
from config import (
    UPLOAD_FOLDER, 
    ALLOWED_TEXT_EXTENSIONS, 
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_MEDIA_EXTENSIONS
)


class FileHandler:
    """Handle file operations"""
    
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """
        Determine file type category
        
        Args:
            file_path: Path to file
            
        Returns:
            File type category: 'text', 'image', 'media', or 'unknown'
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ALLOWED_TEXT_EXTENSIONS:
            return 'text'
        elif ext in ALLOWED_IMAGE_EXTENSIONS:
            return 'image'
        elif ext in ALLOWED_MEDIA_EXTENSIONS:
            return 'media'
        else:
            return 'unknown'
    
    @staticmethod
    def save_uploaded_file(uploaded_file, upload_folder: str = UPLOAD_FOLDER) -> str:
        """
        Save uploaded file to disk
        
        Args:
            uploaded_file: Streamlit uploaded file object
            upload_folder: Folder to save file
            
        Returns:
            Path to saved file
        """
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, uploaded_file.name)
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    @staticmethod
    def cleanup_folder(folder_path: str):
        """
        Remove all files in a folder
        
        Args:
            folder_path: Path to folder
        """
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
    @staticmethod
    def is_valid_file(filename: str) -> bool:
        """
        Check if file has valid extension
        
        Args:
            filename: Name of file
            
        Returns:
            True if valid, False otherwise
        """
        ext = os.path.splitext(filename)[1].lower()
        all_extensions = ALLOWED_TEXT_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS | ALLOWED_MEDIA_EXTENSIONS
        return ext in all_extensions
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """
        Get file size in MB
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in MB
        """
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
