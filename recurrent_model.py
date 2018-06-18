import keras.backend as K
from keras import layers
from keras import regularizers
from keras.layers import Input,GRU
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.core import Lambda, Dense, RepeatVector
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model

def clipped_relu(inputs):
    return Lambda(lambda y: K.minimum(K.maximum(y, 0), 20))(inputs)

def recurrent_model(batch_size,num_frames):
    inputs = Input(shape=(12,64,1))
    #x = Permute((2,1))(inputs)
    x = Conv2D(64,kernel_size=5,strides=2,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
    x = BatchNormalization()(x)
    x = clipped_relu(x)
    x = Reshape((6,2048))(x)
    #x = RepeatVector(256)(x)
    x = GRU(1024,return_sequences=True)(x)
    x = GRU(1024,return_sequences=True)(x)
    x = GRU(1024,return_sequences=True)(x)
    x = Reshape((6*1024,))(x)
    x = Lambda(lambda y: K.reshape(y, (batch_size*3, num_frames, 6*1024)), name='reshape')(x)
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)
    x = Dense(512)(x)  
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)

    model = Model(inputs,x,name='recurrent')

    print(model.summary())
    return model

#recurrent_model(6,251)
