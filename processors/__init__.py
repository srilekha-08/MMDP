"""
Processors package for handling different file types
"""
from .text_processor import TextProcessor
from .image_processor import ImageProcessor
from .media_processor import MediaProcessor

__all__ = ['TextProcessor', 'ImageProcessor', 'MediaProcessor']
