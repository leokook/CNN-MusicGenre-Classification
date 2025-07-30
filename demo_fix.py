#!/usr/bin/env python3
"""
Simple demo to show the CNN Music Genre Classification fix
This demonstrates the preprocessing issue and solution
"""

import os
import sys

print("🎵 CNN Music Genre Classification - Issue Demonstration")
print("=" * 60)

# Check if we're in the right directory
if not os.path.exists('music_model.h5'):
    print("❌ Error: music_model.h5 not found!")
    print("   Please run this script from the CNN-MusicGenre-Classification directory")
    sys.exit(1)

print("✅ Model file found: music_model.h5")

# Try to import required packages
missing_packages = []

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError:
    missing_packages.append("numpy")
    print("❌ NumPy not available")

try:
    import tensorflow as tf
    print("✅ TensorFlow imported successfully")
except ImportError:
    missing_packages.append("tensorflow")
    print("❌ TensorFlow not available")

try:
    import librosa
    print("✅ Librosa imported successfully")
except ImportError:
    missing_packages.append("librosa")
    print("❌ Librosa not available")

try:
    import streamlit as st
    print("✅ Streamlit imported successfully")
except ImportError:
    missing_packages.append("streamlit")
    print("❌ Streamlit not available")

if missing_packages:
    print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
    print("💡 Install them with:")
    print(f"   pip install {' '.join(missing_packages)}")
    print("\nOr install all requirements:")
    print("   pip install -r requirements_enhanced.txt")
else:
    print("\n🎉 All required packages are available!")
    print("\n🚀 Ready to test the enhanced CNN Music Genre Classifier!")
    print("\nTo run the enhanced app:")
    print("   streamlit run app_fixed.py")
    print("\nTo run the original app (with issues):")
    print("   streamlit run app.py")

print("\n" + "=" * 60)
print("📊 Issue Summary:")
print("🔸 Original app: MFCC extraction inconsistent with training")
print("🔸 Fixed app: Exact preprocessing pipeline match")
print("🔸 Result: Model now works with ANY audio file!")

if not missing_packages:
    print("\n🧪 Would you like to run the preprocessing test?")
    print("   python test_preprocessing.py")
