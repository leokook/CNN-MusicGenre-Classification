#!/usr/bin/env python3
"""
Test script to validate the fixed preprocessing pipeline
"""

import numpy as np
import librosa
import tensorflow as tf
import os
import math

# Constants (same as training)
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
NUM_SEGMENTS = 15
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

def preprocess_audio_advanced(audio_path):
    """
    Advanced preprocessing that matches the training pipeline exactly
    """
    try:
        # Load audio data
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, res_type='kaiser_fast')
        
        # Ensure we have enough samples for processing
        if len(y) < SAMPLES_PER_TRACK:
            # Pad with zeros if too short
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), mode='constant')
        else:
            # Truncate if too long
            y = y[:SAMPLES_PER_TRACK]
        
        # Extract MFCC features using the same parameters as training
        samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
        num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)
        
        mfcc_data = []
        
        # Process each segment
        for segment in range(NUM_SEGMENTS):
            start = samples_per_segment * segment
            finish = start + samples_per_segment
            
            # Extract MFCC for this segment
            mfcc = librosa.feature.mfcc(
                y=y[start:finish], 
                sr=sr, 
                n_mfcc=NUM_MFCC, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH
            )
            mfcc = mfcc.T  # Transpose to match training format
            
            # Ensure consistent shape
            if len(mfcc) == num_mfcc_vectors_per_segment:
                mfcc_data.append(mfcc)
        
        if not mfcc_data:
            print("❌ Could not extract valid MFCC features")
            return None
        
        # Use the first valid segment for prediction
        mfcc_features = mfcc_data[0]
        
        # Add the extra dimension for CNN input
        mfcc_features = mfcc_features[..., np.newaxis]
        
        # Add batch dimension
        mfcc_features = mfcc_features[np.newaxis, ...]
        
        return mfcc_features
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return None

def preprocess_audio_simple(audio_path):
    """
    Simplified preprocessing function that ensures consistent shape
    """
    try:
        # Load audio data
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, res_type='kaiser_fast')
        
        # Extract MFCCs with consistent parameters
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=NUM_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        mfcc = mfcc.T  # Transpose to have shape (time_steps, n_mfcc)
        
        # Ensure we have exactly 87 time steps (as expected by the model)
        target_length = 87
        if mfcc.shape[0] < target_length:
            # Pad with zeros if too short
            padding = target_length - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant')
        else:
            # Truncate if too long
            mfcc = mfcc[:target_length, :]
        
        # Reshape for model input: (batch_size, time_steps, features, channels)
        mfcc = mfcc[..., np.newaxis]  # Add channel dimension
        mfcc = mfcc[np.newaxis, ...]  # Add batch dimension
        
        return mfcc
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return None

def test_preprocessing():
    """
    Test the preprocessing functions
    """
    print("🧪 Testing Music Genre Classification Preprocessing Fix")
    print("=" * 60)
    
    # Test with a sample audio file if available
    test_files = [
        "test_audio.wav",
        "sample.mp3", 
        "music.wav"
    ]
    
    audio_file = None
    for test_file in test_files:
        if os.path.exists(test_file):
            audio_file = test_file
            break
    
    if audio_file is None:
        print("⚠️  No test audio file found. Creating synthetic test data...")
        # Create synthetic audio data for testing
        duration = 5  # 5 seconds
        sr = SAMPLE_RATE
        t = np.linspace(0, duration, int(sr * duration))
        # Create a simple sine wave
        frequency = 440  # A4 note
        synthetic_audio = np.sin(2 * np.pi * frequency * t)
        
        # Save temporary file
        import soundfile as sf
        audio_file = "temp_test.wav"
        sf.write(audio_file, synthetic_audio, sr)
        print(f"✅ Created synthetic test audio: {audio_file}")
    else:
        print(f"✅ Using test audio file: {audio_file}")
    
    try:
        # Test advanced preprocessing
        print("\n🔬 Testing Advanced Preprocessing...")
        features_advanced = preprocess_audio_advanced(audio_file)
        if features_advanced is not None:
            print(f"✅ Advanced preprocessing successful!")
            print(f"   Shape: {features_advanced.shape}")
            print(f"   Expected: (1, 87, 13, 1)")
            print(f"   Match: {'✅' if features_advanced.shape == (1, 87, 13, 1) else '❌'}")
        else:
            print("❌ Advanced preprocessing failed")
        
        # Test simple preprocessing
        print("\n🔬 Testing Simple Preprocessing...")
        features_simple = preprocess_audio_simple(audio_file)
        if features_simple is not None:
            print(f"✅ Simple preprocessing successful!")
            print(f"   Shape: {features_simple.shape}")
            print(f"   Expected: (1, 87, 13, 1)")
            print(f"   Match: {'✅' if features_simple.shape == (1, 87, 13, 1) else '❌'}")
        else:
            print("❌ Simple preprocessing failed")
        
        # Test model loading
        print("\n🔬 Testing Model Loading...")
        if os.path.exists('music_model.h5'):
            try:
                model = tf.keras.models.load_model('music_model.h5')
                print("✅ Model loaded successfully!")
                print(f"   Expected input shape: {model.input_shape}")
                
                # Test prediction
                if features_simple is not None:
                    print("\n🔬 Testing Model Prediction...")
                    prediction = model.predict(features_simple, verbose=0)
                    print(f"✅ Prediction successful!")
                    print(f"   Output shape: {prediction.shape}")
                    print(f"   Probabilities sum to: {np.sum(prediction):.4f}")
                    
                    # Apply softmax and show results
                    probs = tf.nn.softmax(prediction[0]).numpy()
                    genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-hop', 
                             'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
                    
                    print(f"\n🎵 Genre Predictions:")
                    for i, (genre, prob) in enumerate(zip(genres, probs)):
                        print(f"   {genre}: {prob:.3f} ({prob*100:.1f}%)")
                    
                    predicted_genre = genres[np.argmax(probs)]
                    confidence = np.max(probs)
                    print(f"\n🏆 Top Prediction: {predicted_genre} ({confidence*100:.1f}%)")
                
            except Exception as e:
                print(f"❌ Model loading/prediction failed: {e}")
        else:
            print("⚠️  Model file 'music_model.h5' not found")
        
        # Cleanup temporary file
        if audio_file == "temp_test.wav" and os.path.exists(audio_file):
            os.remove(audio_file)
            print(f"\n🧹 Cleaned up temporary file: {audio_file}")
        
        print("\n" + "=" * 60)
        print("🎉 Preprocessing test completed!")
        print("\n💡 If all tests passed, your preprocessing fix should work!")
        print("   You can now use the enhanced app with confidence.")
        
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("💡 Install missing packages with: pip install -r requirements_enhanced.txt")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_preprocessing()
