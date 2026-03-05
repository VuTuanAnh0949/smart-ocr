#!/usr/bin/env python
"""
Safe launcher for OCR application with error handling
"""

import sys
import os
import subprocess

def main():
    """Run the OCR application with proper error handling"""
    
    print("=" * 60)
    print("🚀 STARTING OCR APPLICATION")
    print("=" * 60)
    print()
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"   Current version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return 1
    
    # Try to import required modules
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit not installed")
        print()
        print("Installing Streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Set environment variables
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["KERAS_BACKEND"] = "tensorflow"
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different ways to run the app
    app_paths = [
        os.path.join(script_dir, "ocr_app", "ui", "web_app.py"),
        os.path.join(script_dir, "ocr_app", "streamlit_app.py"),
    ]
    
    for app_path in app_paths:
        if os.path.exists(app_path):
            print(f"📂 Found app at: {app_path}")
            print()
            print("Starting Streamlit server...")
            print("=" * 60)
            print()
            
            try:
                # Run streamlit
                cmd = [sys.executable, "-m", "streamlit", "run", app_path]
                subprocess.run(cmd)
                return 0
            except KeyboardInterrupt:
                print()
                print("👋 Application stopped by user")
                return 0
            except Exception as e:
                print(f"❌ Error running application: {e}")
                print()
                print("Trying alternative method...")
                try:
                    # Try running directly
                    cmd = ["streamlit", "run", app_path]
                    subprocess.run(cmd)
                    return 0
                except Exception as e2:
                    print(f"❌ Error: {e2}")
                    return 1
    
    print("❌ Could not find application files")
    return 1

if __name__ == "__main__":
    sys.exit(main())
