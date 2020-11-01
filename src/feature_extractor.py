# Get features from audio clips so that we can use that to train our ML Algo.

import os
import librosa
import soundfile
import numpy as np

from glob import glob
from sklearn.model_selection import train_test_split

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# MFCC, MEL Spectrogram Frequency, Contrast, Tonnetz
def extract_feature(file_name, **kwargs):

    mfcc     = kwargs.get("mfcc")
    chroma   = kwargs.get("chroma")
    mel      = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz  = kwargs.get("tonnetz")
    
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        result = np.array([])
        stft = np.abs(librosa.stft(X))
        
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    
    return result


def load_data(test_size=0.2):
    
    X, y = [], []
    for file in glob("dataset/Actor_*/*.wav"):
        basename = os.path.basename(file)
        emotion  = emotion_dict[basename.split("-")[2]]
        
        features = extract_feature(file, mfcc=True, mel=True)
        X.append(features)
        y.append(emotion)
    
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)