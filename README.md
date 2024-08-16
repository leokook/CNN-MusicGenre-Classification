# 🎵 Music Genre Classification

## Overview

This project is an interactive Streamlit web application that uses machine learning to classify music genres. Users can upload audio files, and the app will predict the genre of the music, providing visualizations and probability distributions for different genres.



## Features

- 🎼 Upload WAV or MP3 audio files
- 🤖 AI-powered genre classification
- 📊 Visual representation of genre probabilities
- 📈 Audio waveform and spectrogram visualization
- 🎨 Sleek, user-friendly interface with animations

## Tech Stack

- Python
- Streamlit
- TensorFlow
- Librosa
- Plotly
- Matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/music-genre-classification.git
   cd music-genre-classification
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have the trained model file `music_model.h5` in the project directory.

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and go to `http://localhost:8501`.

4. Upload an audio file and click "Classify Genre" to see the results!

## How It Works

1. **Audio Preprocessing**: The app uses Librosa to load and process the audio file, extracting Mel-frequency cepstral coefficients (MFCCs) as features.

2. **Genre Classification**: A pre-trained deep learning model (CNN) predicts the genre based on the extracted features.

3. **Visualization**: The app displays the predicted genre, probability distribution, audio waveform, and spectrogram using Plotly and Matplotlib.

## Model Training

The model was trained on the [GTZAN Dataset](http://marsyas.info/downloads/datasets.html), which contains 1000 audio tracks each 30 seconds long, 100 in each of 10 genres.

For details on the model architecture and training process, please refer to `model_training.ipynb` in the repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Librosa](https://librosa.org/) for audio processing
- [Streamlit](https://streamlit.io/) for the web application framework
- [TensorFlow](https://www.tensorflow.org/) for machine learning capabilities
- [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/) for visualizations

## Contact

For any queries or suggestions, please open an issue in this repository.

---

Happy Music Classification! 🎶
