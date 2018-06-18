
from recurrent_model import recurrent_model
from conv_model import convolutional_model
import sklearn
from glob import glob
import librosa
from python_speech_features import fbank, delta
import os
from keras.models import Model
from keras.layers.core import  Dense
import numpy as np



def aread(filename, sample_rate):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    #global i
    #print(i)
    #i+=1 
    audio = audio.flatten()
    return audio

def load_audio(filename):
    audio = aread(filename,srate)
    #print(len(audio))
    start_sec, end_sec = 0.5,3.5
    start_frame = int(start_sec * srate)
    #end_frame = min(int(end_sec * srate), min_frames)
    end_frame = int(end_sec * srate)
    audio = audio[start_frame:end_frame]
   # print(len(audio),srate,int(end_sec * srate), min_frames)
    #print(end_frame,start_frame)

    if len(audio)<(end_frame-start_frame):
        au = [0]*(end_frame-start_frame)
        for i in range(len(audio)):
            au[i] = audio[i]
        #extra = end_frame - start_frame - len(audio)
        #print(extra)
        #print(audio.shape)
        #au[:audio.shape[0]]=audio
        #print(np.count_nonzero(au))
        audio = np.array(au)
    #print(len(audio),srate,len)

    filter_banks, energies = fbank(audio, samplerate=srate, nfilt=64, winlen=0.025)
    #delta_1 = delta(filter_banks, N=1)
    #delta_2 = delta(delta_1, N=1)
    #print("filterbanks:" + str(len(filter_banks)))

    filter_banks = [(i - np.mean(i)) / np.std(i) for i in filter_banks]
    #delta_1 = [(i - np.mean(i)) / np.std(i) for i in delta_1]
    #delta_2 = [(i - np.mean(i)) / np.std(i) for i in delta_2]

    #print("filterbanks:" + str(len(filter_banks)))
    frames_features = filter_banks #np.hstack([filter_banks, delta_1, delta_2]) #filter_banks

    num_frames = len(frames_features)
    network_inputs = []

    for j in range(24, num_frames - 24):
        frames_slice = frames_features[j - 24:j + 24]
        #print(np.array(frames_slice).shape)
        network_inputs.append(np.reshape(np.array(frames_slice), (32,32, 3)))
    #print(np.array(frames_slice).shape)
    final = np.array(network_inputs)
    #print(final.shape)
    return final

def loadFromList(x_paths,batch_start,limit,labels_to_id):
    x = []
    y = []
    for i in range(batch_start,limit):
        # if i%100==0:
        #     print(i)
        audio = load_audio(x_paths[i])
        last = x_paths[i].split("/")[-1]
        y.append(labels_to_id[last.split("-")[0]])
        x.append(audio)
    x = np.asarray(x)
    y = np.asarray(y)
    return x,y      

# def batchValidationImageLoader(batch_size=64,num_classes=251):
    
#     paths = val_files
#     labels = val_labels
    
#     np.random.seed(42)    
#     np.random.shuffle(img_paths)   
#     np.random.seed(42) 
#     np.random.shuffle(labels)      
    
#     L = len(img_paths)

#     while True:

#         batch_start = 0
#         batch_end = batch_size

#         while batch_start < L:
            
#             limit = min(batch_end, L)
#             x_train_t,y_train_t = loadFromList(paths,batch_start,limit)
#             yield (x_train_T,y_train_t)
#             batch_start += batch_size   
#             batch_end += batch_size

def batchTrainingImageLoader(batch_size=64,num_classes=251):

    paths = train_files
    labels = train_labels
    
    np.random.seed(42)    
    np.random.shuffle(img_paths)   
    np.random.seed(42) 
    np.random.shuffle(labels)      
    
    L = len(img_paths)

    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            x_train_t,y_train_t = loadFromList(paths,batch_start,limit)
            yield (x_train_T,y_train_t)
            batch_start += batch_size   
            batch_end += batch_size

no_of_speakers = 251
srate = 12000
min_frames = 1600 
train_path = "/Users/apple/projects/pucho_submit/deep speaker/LibriSpeech/train-clean-100"
#train_path = "/Users/apple/projects/pucho_submit/deep-speaker/audio/LibriSpeechSamples/"
pattern = "**/*.wav"

train_files = glob(os.path.join(train_path, pattern), recursive=True)
train_labels1 = []

for file in train_files:
    last = file.split("/")[-1]
    train_labels1.append(last.split("-")[0])

labels_to_id= {}
id_to_labels = {}
i=0

for label in np.unique(train_labels1):
    labels_to_id[label] = i
    id_to_labels[i] = label
    i+=1

# for i,label in enumerate(train_labels1):
#     train_labels[i] = labels_to_id[label]

#train_labels = keras.utils.to_categorical(train_labels,np.unique(train_labels))

# val_path = "/Users/apple/projects/pucho_submit/deep speaker/LibriSpeech/dev-clean"
# val_files = glob(os.path.join(val_path, pattern), recursive=True)
# val_labels = []

# for file in val_files:
#     last = file.split("/")[-1]
#     val_labels.append(last.split("-")[0])

x_train,y_train = loadFromList(train_files,labels_to_id)
print(x_train.shape,y_train.shape)
base_model = convolutional_model(64,)
x = base_model.output
x = Dense(251,activation='softmax')(x)

model = Model(base_model.input,x)
print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

model.fit((x_train,y_train), epochs=10, class_weight=class_weights, shuffle=True, validation_split=0.2)

model.save_weights("/Users/apple/projects/pucho_submit/deep speaker/checkpoints/model_softmax.h5")

