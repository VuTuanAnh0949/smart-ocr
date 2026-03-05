#!/usr/bin/env python
"""
Script to check and fix common issues before running the OCR application
"""

import sys
import subprocess
import importlib.util
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required!")
        return False
    elif version.major == 3 and version.minor >= 12:
        print("⚠️  Warning: Python 3.12+ may have compatibility issues")
        print("   Recommended: Python 3.9, 3.10, or 3.11")
    else:
        print("✅ Python version is compatible")
    
    return True

def check_module(module_name, package_name=None):
    """Check if a module is installed"""
    if package_name is None:
        package_name = module_name
    
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ {module_name} not installed")
        return False, package_name
    else:
        print(f"✅ {module_name} installed")
        return True, None

def install_package(package_name):
    """Install a package using pip"""
    print(f"📦 Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def main():
    print("=" * 60)
    print("🔍 OCR APPLICATION - DEPENDENCY CHECKER")
    print("=" * 60)
    print()
    
    # Check Python version
    print("📌 Checking Python version...")
    if not check_python_version():
        return
    print()
    
    # Required packages
    required_packages = [
        ("streamlit", "streamlit"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sentence_transformers", "sentence-transformers"),
    ]
    
    # Optional packages
    optional_packages = [
        ("pytesseract", "pytesseract"),
        ("easyocr", "easyocr"),
        ("paddleocr", "paddleocr"),
        ("langdetect", "langdetect"),
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    print("📌 Checking required packages...")
    for module_name, package_name in required_packages:
        is_installed, pkg = check_module(module_name, package_name)
        if not is_installed:
            missing_required.append(pkg)
    print()
    
    # Check optional packages
    print("📌 Checking optional packages...")
    for module_name, package_name in optional_packages:
        is_installed, pkg = check_module(module_name, package_name)
        if not is_installed:
            missing_optional.append(pkg)
    print()
    
    # Install missing packages
    if missing_required:
        print("=" * 60)
        print("⚠️  MISSING REQUIRED PACKAGES")
        print("=" * 60)
        for pkg in missing_required:
            print(f"  - {pkg}")
        print()
        
        response = input("Do you want to install missing required packages? (y/n): ")
        if response.lower() == 'y':
            for pkg in missing_required:
                install_package(pkg)
        else:
            print("❌ Application may not work without required packages")
            return
    
    if missing_optional:
        print("=" * 60)
        print("ℹ️  MISSING OPTIONAL PACKAGES")
        print("=" * 60)
        for pkg in missing_optional:
            print(f"  - {pkg}")
        print()
        print("These packages are optional but recommended for full functionality.")
        
        response = input("Do you want to install missing optional packages? (y/n): ")
        if response.lower() == 'y':
            for pkg in missing_optional:
                install_package(pkg)
    
    # Final check
    print()
    print("=" * 60)
    print("🎉 DEPENDENCY CHECK COMPLETE")
    print("=" * 60)
    print()
    
    if not missing_required:
        print("✅ All required packages are installed!")
        print()
        print("🚀 You can now run the application:")
        print()
        print("   Option 1: python ocr_app\\streamlit_app.py")
        print("   Option 2: streamlit run ocr_app\\ui\\web_app.py")
        print()
    else:
        print("⚠️  Some packages are still missing. Please install them manually:")
        print()
        print(f"   pip install {' '.join(missing_required)}")
        print()

if __name__ == "__main__":
    main()
