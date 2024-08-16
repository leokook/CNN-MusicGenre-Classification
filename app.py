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

# Load the saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('music_model.h5')

model = load_model()

# Define genre labels
genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

# Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_music = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_vPnn3K.json")

def preprocess_audio(audio_data, sr=22050):
    try:
        # Load audio data
        y, sr = librosa.load(io.BytesIO(audio_data), sr=sr, res_type='kaiser_fast')
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs = mfccs.T  # Transpose to have shape (time_steps, n_mfcc)
        
        # Pad or truncate to ensure the shape is (87, 13)
        if mfccs.shape[0] < 87:
            padding = 87 - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, padding), (0, 0)), mode='constant')
        else:
            mfccs = mfccs[:87, :]
        
        # Reshape for model input
        mfccs = mfccs[..., np.newaxis]  # Add an extra dimension to match the model's expected input
        mfccs = mfccs[np.newaxis, ...]  # Add batch dimension
    except Exception as e:
        st.error(f"Error encountered while parsing file: {e}")
        return None
    
    return mfccs

# Main app
def main():
    st.title('🎵 Music Genre Classification')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Upload an audio file and let our AI predict its genre!")
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
    
    with col2:
        st_lottie(lottie_music, height=200, key="music")
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button('Classify Genre'):
            with st.spinner('Analyzing the music...'):
                # Read file as bytes
                audio_bytes = uploaded_file.read()
                
                # Preprocess the audio
                preprocessed_audio = preprocess_audio(audio_bytes)
                
                if preprocessed_audio is not None:
                    # Make prediction
                    raw_prediction = model.predict(preprocessed_audio)
                    
                    # Normalize predictions
                    prediction = tf.nn.softmax(raw_prediction).numpy()
                    
                    # Get the predicted genre
                    predicted_genre = genres[np.argmax(prediction)]
                    
                    # Display results
                    st.success(f'🎉 Predicted Genre: {predicted_genre}')
                    
                    # Display probability distribution
                    st.subheader('Genre Probabilities')
                    prob_df = pd.DataFrame({'Genre': genres, 'Probability': prediction[0]})
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Create a Plotly pie chart
                    fig = go.Figure(data=[go.Pie(labels=prob_df['Genre'], values=prob_df['Probability'], hole=.3)])
                    fig.update_layout(title_text='Genre Distribution')
                    st.plotly_chart(fig)
                    
                    # Display top 3 genres
                    st.subheader('Top 3 Predicted Genres')
                    top_3 = prob_df.head(3)
                    for index, row in top_3.iterrows():
                        st.write(f"{row['Genre']}: {row['Probability']:.2%}")
                    
                    # Audio features visualization
                    st.subheader('Audio Waveform')
                    y, sr = librosa.load(io.BytesIO(audio_bytes))
                    fig, ax = plt.subplots()
                    librosa.display.waveshow(y, sr=sr, ax=ax)
                    st.pyplot(fig)
                    
                    # Spectrogram
                    st.subheader('Spectrogram')
                    D = librosa.stft(y)
                    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                    fig, ax = plt.subplots()
                    img = librosa.display.specshow(S_db, x_axis='time', y_axis='hz', ax=ax)
                    fig.colorbar(img, ax=ax, format='%+2.0f dB')
                    st.pyplot(fig)
                else:
                    st.error("Failed to preprocess the audio file. Please try another file.")
    
    # Add some information about the app
    st.markdown("---")
    st.subheader("About this app")
    st.write("""
    This Music Genre Classification app uses a deep learning model to predict the genre of uploaded music files.
    The model has been trained on a diverse dataset of music across various genres.
    
    **How it works:**
    1. Upload an audio file (WAV or MP3)
    2. Click 'Classify Genre'
    3. The app will analyze the audio and predict the most likely genre
    4. You'll see a breakdown of genre probabilities and audio visualizations
    
    **Note:** The accuracy of predictions may vary depending on the complexity and uniqueness of the music.
    """)

if __name__ == "__main__":
    main()