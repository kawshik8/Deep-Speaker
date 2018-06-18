# % pylab inline
import logging
import os
from glob import glob

import librosa
import numpy as np
#import pandas as pd
import numpy as np
from python_speech_features import fbank, delta

i=0
def aread(filename, sample_rate = 1000):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    global i
    print(i)
    i+=1 
    audio = audio.flatten()
    return audio

train_path = "/Users/apple/projects/pucho_submit/deep speaker/LibriSpeech/train-clean-100"
pattern = "**/*.wav"

files = glob(os.path.join(train_path, pattern), recursive=True)

min_frames = min([len(aread(i)) for i in files])
print(min_frames)

audio,sr = librosa.load("/Users/apple/projects/pucho_submit/deep speaker/LibriSpeech/train-clean-100/19/198/19-198-0000.wav",sr=1000,mono=True)
au = np.array(audio)
print(len(audio),au.shape)
plt.figure(figsize=(12, 4))
librosa.display.waveplot(audio, sr)
plt.show()


audio.flatten()
print(len(audio))

