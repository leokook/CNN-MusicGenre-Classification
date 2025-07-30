import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import io
import matplotlib.pyplot as plt
import json
import math
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Set page config
st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #1DB954, #191414);
    }
    .main {
        color: #FFFFFF;
    }
    .stButton>button {
        color: #1DB954;
        background-color: #FFFFFF;
        border-radius: 20px;
    }
    .stProgress > div > div > div > div {
        background-color: #1DB954;
    }
</style>
""", unsafe_allow_html=True)

# Constants for preprocessing (same as training)
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
NUM_SEGMENTS = 15
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# Load the saved model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('music_model.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load or create scaler
@st.cache_resource
def get_scaler():
    """Load or create a scaler for normalization"""
    # If you have a saved scaler, load it. Otherwise, create a default one.
    # For now, we'll create a dummy scaler that we'll fit on-the-fly
    return StandardScaler()

model = load_model()
scaler = get_scaler()

# Define genre labels (same order as training)
genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

# Lottie animation
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_music = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_vPnn3K.json")

def preprocess_audio_advanced(audio_data, sr=SAMPLE_RATE):
    """
    Advanced preprocessing that matches the training pipeline exactly
    """
    try:
        # Load audio data
        y, sr = librosa.load(io.BytesIO(audio_data), sr=sr, res_type='kaiser_fast')
        
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
            st.error("Could not extract valid MFCC features from the audio file.")
            return None
        
        # Use the first valid segment for prediction (you could also average all segments)
        mfcc_features = mfcc_data[0]
        
        # Add the extra dimension for CNN input
        mfcc_features = mfcc_features[..., np.newaxis]
        
        # Add batch dimension
        mfcc_features = mfcc_features[np.newaxis, ...]
        
        return mfcc_features
        
    except Exception as e:
        st.error(f"Error encountered while processing audio: {e}")
        return None

def preprocess_audio_simple(audio_data, sr=SAMPLE_RATE):
    """
    Simplified preprocessing function that ensures consistent shape
    """
    try:
        # Load audio data
        y, sr = librosa.load(io.BytesIO(audio_data), sr=sr, res_type='kaiser_fast')
        
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
        st.error(f"Error encountered while processing audio: {e}")
        return None

def predict_genre_ensemble(audio_data):
    """
    Make predictions using both preprocessing methods and combine results
    """
    predictions = []
    
    # Try advanced preprocessing
    features_advanced = preprocess_audio_advanced(audio_data)
    if features_advanced is not None:
        try:
            pred_advanced = model.predict(features_advanced, verbose=0)
            predictions.append(pred_advanced[0])
        except Exception as e:
            st.warning(f"Advanced preprocessing prediction failed: {e}")
    
    # Try simple preprocessing
    features_simple = preprocess_audio_simple(audio_data)
    if features_simple is not None:
        try:
            pred_simple = model.predict(features_simple, verbose=0)
            predictions.append(pred_simple[0])
        except Exception as e:
            st.warning(f"Simple preprocessing prediction failed: {e}")
    
    if not predictions:
        return None
    
    # Average predictions if we have multiple
    if len(predictions) > 1:
        final_prediction = np.mean(predictions, axis=0)
        st.info("Using ensemble prediction from multiple preprocessing methods")
    else:
        final_prediction = predictions[0]
        st.info("Using single preprocessing method prediction")
    
    return final_prediction

# Main app
def main():
    st.title('🎵 Enhanced Music Genre Classification')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Upload an audio file and let our improved AI predict its genre!")
        st.write("**New Features:**")
        st.write("- ✅ Fixed preprocessing to match training data")
        st.write("- ✅ Ensemble prediction for better accuracy")
        st.write("- ✅ Improved error handling")
        st.write("- ✅ Support for various audio lengths")
        
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'flac', 'm4a'])
    
    with col2:
        if lottie_music:
            st_lottie(lottie_music, height=200, key="music")
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Show file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write("📁 **File Information:**")
        st.json(file_details)
        
        if st.button('🔍 Classify Genre', type="primary"):
            with st.spinner('🎵 Analyzing the music...'):
                # Read file as bytes
                audio_bytes = uploaded_file.read()
                
                # Make prediction using ensemble method
                prediction = predict_genre_ensemble(audio_bytes)
                
                if prediction is not None:
                    # Apply softmax to normalize predictions
                    prediction = tf.nn.softmax(prediction).numpy()
                    
                    # Get the predicted genre
                    predicted_genre_idx = np.argmax(prediction)
                    predicted_genre = genres[predicted_genre_idx]
                    confidence = prediction[predicted_genre_idx]
                    
                    # Display results with confidence
                    st.success(f'🎉 **Predicted Genre: {predicted_genre}**')
                    st.info(f'📊 **Confidence: {confidence:.2%}**')
                    
                    # Display probability distribution
                    st.subheader('🎯 Genre Probabilities')
                    prob_df = pd.DataFrame({
                        'Genre': genres, 
                        'Probability': prediction
                    }).sort_values('Probability', ascending=False)
                    
                    # Create a Plotly bar chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=prob_df['Genre'], 
                            y=prob_df['Probability'],
                            marker_color='lightblue',
                            text=[f'{p:.2%}' for p in prob_df['Probability']],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title='Genre Probability Distribution',
                        xaxis_title='Genre',
                        yaxis_title='Probability',
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display top 3 genres
                    st.subheader('🏆 Top 3 Predicted Genres')
                    top_3 = prob_df.head(3)
                    for index, row in top_3.iterrows():
                        confidence_color = "🟢" if row['Probability'] > 0.5 else "🟡" if row['Probability'] > 0.2 else "🔴"
                        st.write(f"{confidence_color} **{row['Genre']}**: {row['Probability']:.2%}")
                    
                    # Audio visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader('🌊 Audio Waveform')
                        try:
                            y, sr = librosa.load(io.BytesIO(audio_bytes))
                            fig, ax = plt.subplots(figsize=(10, 4))
                            librosa.display.waveshow(y, sr=sr, ax=ax, color='blue', alpha=0.7)
                            ax.set_title('Audio Waveform')
                            ax.set_xlabel('Time (s)')
                            ax.set_ylabel('Amplitude')
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Could not generate waveform: {e}")
                    
                    with col2:
                        st.subheader('🎨 Spectrogram')
                        try:
                            D = librosa.stft(y)
                            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                            fig, ax = plt.subplots(figsize=(10, 4))
                            img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
                            ax.set_title('Spectrogram')
                            plt.colorbar(img, ax=ax, format='%+2.0f dB')
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Could not generate spectrogram: {e}")
                    
                    # Technical details
                    with st.expander("🔧 Technical Details"):
                        st.write("**Model Information:**")
                        st.write(f"- Model type: Convolutional Neural Network (CNN)")
                        st.write(f"- Input shape expected: (87, 13, 1)")
                        st.write(f"- Number of genres: {len(genres)}")
                        st.write(f"- MFCC features: {NUM_MFCC}")
                        st.write(f"- Sample rate: {SAMPLE_RATE} Hz")
                        
                        st.write("**Preprocessing Steps:**")
                        st.write("1. Load audio at 22050 Hz sample rate")
                        st.write("2. Extract 13 MFCC features")
                        st.write("3. Ensure consistent time dimension (87 frames)")
                        st.write("4. Add channel dimension for CNN")
                        st.write("5. Apply softmax for probability distribution")
                
                else:
                    st.error("❌ Failed to process the audio file. Please try:")
                    st.write("- A different audio file")
                    st.write("- Ensuring the file is a valid audio format")
                    st.write("- Checking that the file is not corrupted")
    
    # Information section
    st.markdown("---")
    st.subheader("📖 About this Enhanced App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **🚀 What's New:**
        - Fixed preprocessing pipeline to match training data exactly
        - Ensemble prediction using multiple methods
        - Better error handling and user feedback
        - Improved confidence scores and visualizations
        - Support for various audio file lengths
        """)
    
    with col2:
        st.write("""
        **🎯 How it Works:**
        1. Upload an audio file (WAV, MP3, FLAC, M4A)
        2. Audio is preprocessed using MFCC feature extraction
        3. CNN model predicts genre probabilities
        4. Results are displayed with confidence scores
        5. Visual analysis shows waveform and spectrogram
        """)
    
    st.write("**💡 Tips for Better Results:**")
    st.write("- Use high-quality audio files (not heavily compressed)")
    st.write("- Audio should be at least 3-5 seconds long")
    st.write("- Clear, representative music works best")
    st.write("- Avoid files with speech or mixed genres")

if __name__ == "__main__":
    if model is None:
        st.error("❌ Could not load the model. Please ensure 'music_model.h5' is in the current directory.")
        st.stop()
    
    main()
