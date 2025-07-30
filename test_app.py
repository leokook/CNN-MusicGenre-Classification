#!/usr/bin/env python3
"""
Test script to verify the CNN music genre classification components work.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow imported successfully (version: {tf.__version__})")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import librosa
        print(f"✓ Librosa imported successfully (version: {librosa.__version__})")
    except ImportError as e:
        print(f"✗ Librosa import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy imported successfully (version: {np.__version__})")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas imported successfully (version: {pd.__version__})")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import plotly
        print(f"✓ Plotly imported successfully (version: {plotly.__version__})")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if the model can be loaded."""
    print("\nTesting model loading...")
    
    try:
        import tensorflow as tf
        model_path = "music_model.h5"
        
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            return False
        
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  Model input shape: {model.input_shape}")
        print(f"  Model output shape: {model.output_shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_preprocessing():
    """Test the preprocessing function."""
    print("\nTesting preprocessing function...")
    
    try:
        import numpy as np
        import librosa
        
        # Create dummy audio data
        sample_rate = 22050
        duration = 30
        dummy_audio = np.random.random(sample_rate * duration)
        
        # Test MFCC extraction with correct parameters
        mfccs = librosa.feature.mfcc(
            y=dummy_audio,
            sr=sample_rate,
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        
        print(f"✓ MFCC extraction successful")
        print(f"  MFCC shape: {mfccs.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=== CNN Music Genre Classification Test ===\n")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing packages.")
        return 1
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ Model loading test failed.")
        return 1
    
    # Test preprocessing
    if not test_preprocessing():
        print("\n❌ Preprocessing test failed.")
        return 1
    
    print("\n✅ All tests passed! The CNN music genre classifier should work correctly.")
    print("\nTo run the Streamlit app, use:")
    print("streamlit run app_fixed.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
