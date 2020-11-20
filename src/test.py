# Load the trained model from the pickle file and use it for inference

import os
import wave
import pickle
import pyaudio
import sounddevice as sd

from scipy.io.wavfile import write
from sys import byteorder
from array import array
from struct import pack
from sklearn.neural_network import MLPClassifier

from feature_extractor import extract_feature

RATE = 16000


def record(time=3):

    myrecording = sd.rec(int(time * RATE), samplerate=RATE, channels=1)
    sd.wait()  # Wait until recording is finished
    write('test.wav', RATE, myrecording)  # Save as WAV file 


def test(time=3):

    model = pickle.load(open("emotion-recognition.model", "rb"))
    print("Please talk")
    filename = "test.wav"
    record(time)

    # extract features and reshape it
    features = extract_feature(filename, mfcc=True, mel=True).reshape(1, -1)
    # predict
    result = model.predict(features)
    # show the result !
    print("result:", result)

    return result[0]



if __name__ == "__main__":

    model = pickle.load(open("emotion-recognition.model", "rb"))
    print("Please talk")
    filename = "test.wav"
    record()

    # extract features and reshape it
    features = extract_feature(filename, mfcc=True, mel=True).reshape(1, -1)
    # predict
    result = model.predict(features)
    print(result)
    # show the result !
    print("result:", result[0])