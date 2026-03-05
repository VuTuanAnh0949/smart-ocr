"""
OCR Application Package
=======================

A comprehensive OCR (Optical Character Recognition) application with support for multiple engines,
web interface, CLI, and question-answering capabilities.

Main Components:
- core: Core OCR functionality and engines
- ui: User interface components (web and CLI)
- models: Model management and AI components
- utils: Utility functions and helpers
- config: Configuration management
"""

__version__ = "2.0.0"
__author__ = "Vũ Tuấn Anh"
__email__ = "vutuananh0949@gmail.com"
__github__ = "https://github.com/VuTuanAnh0949"

# Import main classes and functions for easy access
from .core.ocr_engine import OCREngine
from .core.image_processor import ImageProcessor
from .models.model_manager import ModelManager
from .config.settings import Settings

__all__ = [
    "OCREngine",
    "ImageProcessor", 
    "ModelManager",
    "Settings"
]
