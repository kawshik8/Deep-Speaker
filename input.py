import logging
import os
from glob import glob
import keras.backend as K
import librosa
import numpy as np
#import pandas as pd
import numpy as np
from python_speech_features import fbank, delta
from conv_model import convolutional_model
from recurrent_model import recurrent_model
from keras.optimizers import Adam
from natsort import natsorted
import tensorflow as tf

srate = 12000
batch_size = 12
i=0
min_frames = 1600
gru_model = False#True        #T or F to switch between models

train_path = "/Users/apple/projects/pucho_submit/deep speaker/LibriSpeech/train-clean-100"
#train_path = "/Users/apple/projects/pucho_submit/deep-speaker/audio/LibriSpeechSamples/"
pattern = "**/*.wav"
L = len(glob(os.path.join(train_path, pattern), recursive=True))

# def convolutional_model():
    
def get_last_checkpoint(folder="/Users/apple/projects/pucho_submit/deep speaker/checkpoints/"):
    files = glob('{}/*.h5'.format(folder), recursive=True)
    if len(files) == 0:
        return None
    return natsorted(files)[-1]

from triplet_loss import deep_speaker_loss
# def triplet_loss(y_pred,y_true,alpha =0.4):

#     split = 6
#     anchor = y_pred[:,0:split]
#     positive = y_pred[:,split:split*2]
#     negative = y_pred[:,split*2:]
    
#     pos_dist = K.sum(K.square(anchor-positive),axis=1) #K.squeeze(K.batch_dot(anchor, positive, axes=1), axis=1)  #
#     #print(K.shape(pos_dist))
#     #K.squeeze(K.batch_dot(anchor, positive, axes=1), axis=1)
#     #print(K.len(pos_dist))
#     neg_dist = K.sum(K.square(anchor-negative),axis=1) #K.squeeze(K.batch_dot(anchor, negative, axes=1), axis=1)  #K.sum(K.square(anchor-negative),axis=1)

#     basic_loss = pos_dist-neg_dist+alpha
#     # print(K.shape(basic_loss))

#     loss = K.maximum(basic_loss,0.0)
#     # print(K.shape(loss))
#     # print(K.shape(K.sum(loss)))
#     #print(len(loss))
#     return K.mean(loss)

#y_pred = np.random.rand(shape=())
def aread(filename, sample_rate):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    #global i
    #print(i)
    #i+=1 
    audio = audio.flatten()
    return audio

def create_dict():
    train_path = "/Users/apple/projects/pucho_submit/deep speaker/LibriSpeech/train-clean-100"
    #train_path = "/Users/apple/projects/pucho_submit/deep-speaker/audio/LibriSpeechSamples/"
    pattern = "**/*.wav"

    files = glob(os.path.join(train_path, pattern), recursive=True)

    #min_frames = 1965 #min([len(aread(i)) for i in files])#

    #print("Minimum frames:" + min_frames)

    classes = []
    for file in files:
        last = file.split("/")[-1]
        classes.append(last.split("-")[0])
        #print(last,last.split("-")[0])

    spk_uniq = np.unique(classes)

    train_dict = {}
    print("no of files & sample rate " ,len(files), srate)

    for i in range(len(spk_uniq)):
        #print(i)
        train_dict[spk_uniq[i]] = []

    for i in range(len(classes)):
        train_dict[classes[i]].append(files[i])

    #print(train_dict.keys(),len(train_dict.keys()))
    
    # for i in range(len(spk_uniq)):
    #     print(spk_uniq[i],len(train_dict[spk_uniq[i]]))


    #print(files[:5],len(files),type(files),len(classes),len(spk_uniq))
    return train_dict,spk_uniq

# def load_audio_gru(filename):
#     audio = aread(filename,srate)
#     print(len(audio))
#     start_sec, end_sec = 0.5,3.5
#     start_frame = int(start_sec * srate)
#     end_frame = min(int(end_sec * srate), min_frames)
#     audio = audio[start_frame:end_frame]
#     #print(len(audio),srate,int(end_sec * srate), min_frames)
#     #print(end_frame,min_frames)
#     if len(audio)<min_frames:
#         au = np.zeros((min_frames))
#         extra = min_frames - len(audio)
#         #print(extra)
#         #print(audio.shape)
#         au[:audio.shape[0]]=audio
#         #print(np.count_nonzero(au))
#         audio = au
#     print(len(audio),srate,len)

#     filter_banks, energies = fbank(audio, samplerate=srate, nfilt=64, winlen=0.025)
#     #delta_1 = delta(filter_banks, N=1)
#     #delta_2 = delta(delta_1, N=1)
#     #print("filterbanks:" + str(len(filter_banks)))

#     filter_banks = [(i - np.mean(i)) / np.std(i) for i in filter_banks]
#     #delta_1 = [(i - np.mean(i)) / np.std(i) for i in delta_1]
#     #delta_2 = [(i - np.mean(i)) / np.std(i) for i in delta_2]

#     print("filterbanks:" + str(len(filter_banks)))
#     frames_features = filter_banks #np.hstack([filter_banks, delta_1, delta_2]) #

#     num_frames = len(frames_features)
#     network_inputs = []

#     for j in range(6, num_frames - 6):
#         frames_slice = frames_features[j - 6:j + 6]
#         #print(np.array(frames_slice).shape)
#         network_inputs.append(np.reshape(np.array(frames_slice), (16,16, 3)))

#     print(np.array(frames_slice).shape)
#     final = np.array(network_inputs)
#     print(final.shape)
#     return final


def load_audio(filename,recurrent=False):
    audio = aread(filename,srate)
    #print(len(audio))
    start_sec, end_sec = 0.5,3.5
    start_frame = int(start_sec * srate)
    #end_frame = min(int(end_sec * srate), min_frames)
    end_frame = int(end_sec * srate)
    audio = audio[start_frame:end_frame]
    #print(len(audio),srate,int(end_sec * srate), min_frames)
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

    if recurrent is False:
        for j in range(24, num_frames - 24):
            frames_slice = frames_features[j - 24:j + 24]
            #print(np.array(frames_slice).shape)
            network_inputs.append(np.reshape(np.array(frames_slice), (32,32, 3)))
        #print(np.array(frames_slice).shape)
        final = np.array(network_inputs)
        #print(final.shape)
        return final
    
    else:
        for j in range(6, num_frames - 6):
            frames_slice = frames_features[j - 6:j + 6]
            #print(np.array(frames_slice).shape)
            network_inputs.append(np.reshape(np.array(frames_slice), (16,16, 3)))
        #print(np.array(frames_slice).shape)
        final = np.array(network_inputs)
        #print(final.shape)
        return final


#load_audio("/Users/apple/projects/pucho_submit/deep speaker/LibriSpeech/train-clean-100/311/124404/311-124404-0058.wav",True)



def MiniBatch(train_dict,batch_size,spk_uniq,recurrent = False):

    # batch_start = 0
    # batch_end = batch_size

        anchor_batch = []
        positive_batch = []
        negative_batch = []
        for i in range(batch_size):
            two_different_speakers = np.random.choice(spk_uniq, size=2, replace=False)
            anchor_positive_speaker = two_different_speakers[0]
            negative_speaker = two_different_speakers[1]
            anchor_positive_file = np.random.choice(train_dict[anchor_positive_speaker], size=2, replace=False) #libri[libri['speaker_id'] == anchor_positive_speaker].sample(n=2, replace=False)
            
            #if recurrent is False:
            #anchor = [0]*2 #pd.DataFrame(anchor_positive_file[0:1])
            anchor = load_audio(str(anchor_positive_file[0]),recurrent)

        # anchor[1] = 'anchor'

            #positive = [0]*2
            positive = load_audio(str(anchor_positive_file[1]),recurrent)
            #positive[1] = 'positive'
            
            #negative = [0]*2
            negative = load_audio(str(np.random.choice(train_dict[anchor_positive_speaker],size=1)[0]),recurrent)  # libri[libri['speaker_id'] == negative_speaker].sample(n=1)
            #negative[1] = 'negative'

            # else:
            #     anchor = load_audio_gru(str(anchor_positive_file[0]))

            # # anchor[1] = 'anchor'

            #     #positive = [0]*2
            #     positive = load_audio_gru(str(anchor_positive_file[1]))
            #     #positive[1] = 'positive'
                
            #     #negative = [0]*2
            #     negative = load_audio_gru(str(np.random.choice(train_dict[anchor_positive_speaker],size=1)[0]))  # libri[libri['speaker_id'] == negative_speaker].sample(n=1)
            # #negative[1] = 'negative'

            anchor_batch.append(anchor)
            positive_batch.append(positive)
            negative_batch.append(negative)
            #print(anchor_positive_file[0],anchor_positive_file[1],str(np.random.choice(train_dict[anchor_positive_speaker],size=1)[0]))
        anchor_batch = np.array(anchor_batch)
        positive_batch = np.array(positive_batch)
        negative_batch = np.array(negative_batch)
        #print(anchor_batch.shape,positive_batch.shape,negative_batch.shape)
        batch = np.concatenate((anchor_batch, positive_batch, negative_batch), axis=0)    #pd.DataFrame(pd.concat([anchor_batch, positive_batch, negative_batch], axis=0))
        return batch
    
def main():
    train_dict,spk_uniq = create_dict()
    x = MiniBatch(train_dict,batch_size,spk_uniq)
    print(x.shape)
    print(x[0][0].shape,x[0][1])

#main()

if __name__ == '__main__':
    train_dict,spk_uniq = create_dict()
    srate = 12000
    batch_size = 12
    i=0
    min_frames = 1600 #3*srate
    x = MiniBatch(train_dict,batch_size,spk_uniq,gru_model)
    b = x[0]
    num_frames = b.shape[0]
    #global batch_size
    batch_size1 = batch_size*3
    batch_shape = [batch_size1 * num_frames] + list(b.shape[1:])
    print(batch_shape,batch_size,num_frames,x.shape,b.shape)

    #
    if gru_model is True:
        model = recurrent_model(batch_size,num_frames)
    else:
        model = convolutional_model(batch_size=batch_size, num_frames=num_frames)

    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer='adam', loss=deep_speaker_loss)

    print(model.summary())

    grad_step = 0

    last_checkpoint = get_last_checkpoint()

    if last_checkpoint is not None:
        model.load_weights(last_checkpoint)
        grad_steps = int(last_checkpoint.split('_')[-2])
    min_loss = 10
    while True:
        grad_step+=1
        x = MiniBatch(train_dict,batch_size,spk_uniq,gru_model)
        print(x.shape,batch_size1,num_frames)
        x = np.reshape(x, (batch_size1 * num_frames, b.shape[1], b.shape[2], b.shape[3]))
        Y = np.random.uniform(size=(x.shape[0], 1))
        loss = model.train_on_batch(x, Y)
        print("Step",grad_step,loss)
        if loss<min_loss:
            model.save_weights('{0}/model_{1}_{2:.5f}.h5'.format("/Users/apple/projects/pucho_submit/deep speaker/checkpoints", grad_step, loss))
            min_loss = loss

