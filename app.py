import tensorflow.keras as keras
import numpy as np
import librosa
import json
import random
import os
from flask import Flask, request, jsonify


MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050*3

# creating a singleton class
class _Emotion_classifier:
    
    model = None
    _mappings = [
        "neutral",
        "happy",
        "sad",
        "angry"
    ]
    _instance = None
    
    def predict(self, file_path):
        
        # extract MFCCs
        features = self.preprocess(file_path) # (#segments, #coefficients)
        
        # convert 2d MFCCs array into 4d array -> (#samples, #segments, #coefficients, #channels)
        features = features[np.newaxis, ..., np.newaxis]
        
        # make prediction
        predictions = self.model.predict(features) # [ [0.1, 0.6, 0.2, 0.0] ]
        print(predictions)
        predicted_index = np.argmax(predictions)
        predicted_keyword =self._mappings[predicted_index]
        
        return predicted_keyword
    
    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=256):
        
        # load audio file
        signal, sr = librosa.load(file_path)
        
        # ensure consistency in the audio file length
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]
            
        # extract features
        
        chroma = librosa.feature.chroma_stft(y=signal,sr=sr,hop_length=hop_length).T
        
        S = librosa.feature.melspectrogram(y=signal, hop_length=hop_length, sr=sr, n_mels=128, fmax=8000)
        log_power_Mel_spectrogram = librosa.feature.mfcc(S=librosa.power_to_db(S)).T
        
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length).T
        
        features = [list(MFCCs[i])+list(chroma[i])+list(log_power_Mel_spectrogram[i])  for i in range(259)]
        features = np.array(features)
        
        return features

def Emotion_classifier():
    
    # ensure that we only have 1 instance of KSS
    if _Emotion_classifier._instance is None:
        _Emotion_classifier._instance = _Emotion_classifier()
        _Emotion_classifier.model = keras.models.load_model(MODEL_PATH)
    return _Emotion_classifier._instance

# instantiate flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():

	# get file from POST request and save it
	audio_file = request.files["file"]
	file_name = str(random.randint(0, 100000))
	audio_file.save(file_name)

	# instantiate keyword spotting service singleton and get prediction
	EC = Emotion_classifier()
	predicted_Emotion = EC.predict(file_name)

	# we don't need the audio file any more - let's delete it!
	os.remove(file_name)

	# send back result as a json file
	result = {"keyword": predicted_Emotion}
	return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)
