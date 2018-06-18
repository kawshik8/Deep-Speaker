import keras.backend as K
from keras import layers
from keras import regularizers
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Lambda, Dense, RepeatVector
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def clipped_relu(inputs):
    return Lambda(lambda y: K.minimum(K.maximum(y, 0), 20))(inputs)


def identity_block(input_tensor, kernel_size, filters, stage, block):

    x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001))(input_tensor)
    x = BatchNormalization()(x)
    x = clipped_relu(x)

    x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001))(x)
    x = BatchNormalization()(x)

    x = layers.add([x, input_tensor])
    x = clipped_relu(x)
    return x

def conv_and_res_block(inp, filters, stage):
    conv_name = 'conv{}-s'.format(filters)
    o = Conv2D(filters,
                    kernel_size=5,
                    strides=2,
                    padding='same',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(l=0.00001))(inp)
    o = BatchNormalization()(o)
    o = clipped_relu(o)
    for i in range(3):
        o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
    return o

def cnn_component(inp):
    x_ = conv_and_res_block(inp, 64, stage=1)
    x_ = conv_and_res_block(x_, 128, stage=2)
    x_ = conv_and_res_block(x_, 256, stage=3)
    x_ = conv_and_res_block(x_, 512, stage=4)
    return x_

def convolutional_model(batch_size, num_frames):

    inputs = Input(shape=(32,32,3))  
    x = cnn_component(inputs)
    x = Reshape((2048,))(x)
    x = Lambda(lambda y: K.reshape(y, (batch_size*3, num_frames, 2048)), name='reshape')(x)
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)
    x = Dense(512)(x)  
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
    model = Model(inputs, x, name='convolutional')
    return model