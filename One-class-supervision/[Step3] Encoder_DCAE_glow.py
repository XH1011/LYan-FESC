# pip install tensorflow_addons
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import numpy as np
import math
import os
import random
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array
# from Utils import *
# from Unet_Utils import *
from tensorflow.python.framework import ops

# Global Settings
batch_size=2
num_epochs=1000
learning_rate = 1e-4
img_size=1024
img_channels=1

faults= ['x0','x1','x2','x3','x4','x5','x6']
root = './One-class-supervision/results/CWRU/glow/'
# root = './One-class-supervision/results/Chopper/glow/'
for fault in faults:
    # Build graph
    ops.reset_default_graph()
    # Build encoder
    inputs_=layers.Input(shape=(img_size, img_channels), name="image_input")
    # 2，神经网络
    layers = tf.keras.layers
    # ### Encoder
    conv1 = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)(inputs_)
    maxpool1 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv1)
    conv2 = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool1)
    maxpool2 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv2)
    conv3 = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool2)
    maxpool3 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv3)
    conv4 = layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool3)
    maxpool4 = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv4)
    conv5 = layers.Conv1D(filters=2, kernel_size=5, padding='same', activation=tf.nn.relu)(maxpool4)
    maxpool5 = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(conv5)
    re = tf.reshape(maxpool5, [-1, 128])
    # -----------#
    latent = layers.Dense(units=128)(re)
    # -----------#
    # ---Decoder---#
    x = layers.Dense(units=128, activation=tf.nn.relu)(re)
    x = tf.reshape(x, [-1, 64, 2])
    x = layers.UpSampling1D(1)(x)
    x = layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    x = layers.UpSampling1D(2)(x)
    rx = layers.Conv1D(filters=1, kernel_size=5, padding='same', activation=tf.nn.relu)(x)
    # print(rx.shape, inputs_.shape)
    # print('Built Encoder../')
    # print(image_input.shape, enout.shape, x_out.shape)
    # #Build model
    dcae=keras.Model(inputs_, rx)
    #
    # # Opimizer and loss function
    opt = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8)
    print('Network Summary-->')
    # dcae.summary()

    dir = './One-class-supervision/results/CWRU/DCAE/model_last_999.ckpt'
    # dir = './One-class-supervision/results/Chooper/DCAE/model_last_999.ckpt'
    print('Load weights from ', dir)
    dcae.load_weights(dir)
    new_enout=tf.keras.models.Model(inputs=inputs_,outputs=latent)

#data_test
    file_name = root + fault + '_G_test.pkl'
    # x = LoadData_pickle(file_name)[0]
    x = pickle.load(open(file_name, 'rb'))
    data = tf.reshape(x, shape=[-1, 1024, 1])
    extracted_features = new_enout.predict(data)
    print(extracted_features.shape)
    with open('./One-class-supervision/results/CWRU/DCAE/en_' + fault + '_G_test.pkl', 'wb') as f:
    # with open('./One-class-supervision/results/Chopper/DCAE/en_' + fault + '_G_test.pkl', 'wb') as f:
        pickle.dump(extracted_features, f, pickle.HIGHEST_PROTOCOL)

#data_train
    file_name = root + fault + '_G_train.pkl'
    data = pickle.load(open(file_name, 'rb'))
    # file_name=root+fault+'_train'
    # data=LoadData_pickle(file_name)[0]
    data = tf.reshape(data, shape=[-1, 1024, 1])
    extracted_features = new_enout.predict(data)
    print(extracted_features.shape)
    with open('./One-class-supervision/results/CWRU/DCAE/en_' + fault + '_G_train.pkl', 'wb') as f:
    # with open('./One-class-supervision/results/Chooper/DCAE/en_' + fault + '_G_train.pkl', 'wb') as f:
        pickle.dump(extracted_features, f, pickle.HIGHEST_PROTOCOL)




