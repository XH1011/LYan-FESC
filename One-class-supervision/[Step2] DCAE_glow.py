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

def load_data(file,BATCH_SIZE):
    f = open(file, 'rb')
    x0=pickle.load(f)
    x = tf.data.Dataset.from_tensor_slices(x0)
    x = x.shuffle(5000).batch(BATCH_SIZE)
    return x,x0

# Global Settings
batch_size=2
num_epochs=1000
learning_rate=1e-4
img_size=1024
img_channels=1

#glow_CWRU
file_name='./One-class-supervision/results/CWRU/x0_G_train.pkl'
train_ds,x0=load_data(file=file_name,BATCH_SIZE=32)
##glow_Chopper
# file_name='./One-class-supervision/results/Chopper/x0_G_train.pkl'
# train_ds,x0=load_data(file=file_name,BATCH_SIZE=32)


#inputs
inputs_=layers.Input(shape=(img_size, img_channels), name="image_input")
# 2，神经网络
layers=tf.keras.layers
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
print(rx.shape,inputs_.shape)
print('Built Encoder../')
print(inputs_.shape,latent.shape,rx.shape)
# #Build model
dcae=keras.Model(inputs_, rx)
#
# # Opimizer and loss function
opt=keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-8)
print('Network Summary-->')
dcae.summary()

def train_on_step(images_batch):
    image_batch = tf.reshape(images_batch, shape=[-1, 1024, 1])
    loss=train_loss(image_batch)
    return loss

def train_loss(image_batch):
    with tf.GradientTape() as tape:
        # model
        recon_image = dcae(image_batch, training=True)
        # loss
        diff = recon_image - image_batch
        recon_loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))
        # print('loss:',image_batch.shape,images_batch.shape,recon_image.shape)
        total_loss = recon_loss
    gradients = tape.gradient(total_loss, dcae.trainable_weights)
    opt.apply_gradients(zip(gradients, dcae.trainable_weights))

    return total_loss.numpy()

# --------------->>>Training Phase<<<---------------------------
# Run
loss_list=[]
# total_batch=350
total_batch = int(len(x0) / batch_size)
for epoch in range(num_epochs):
    ave_cost=0
    for images_batch in train_ds:
        loss=train_on_step(images_batch)
        ave_cost+=loss/total_batch
    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(ave_cost))
    loss_list.append(ave_cost)
    if epoch%100==0:
      save_dir = './One-class-supervision/results/CWRU/DCAE/'
      # save_dir = './One-class-supervision/results/Chopper/DCAE/'
      os.makedirs(save_dir, exist_ok=True)
      dcae.save_weights(save_dir+'model_'+str(epoch)+'.ckpt')
print('Optimization Finished')


# Save the model weights (last step)
save_dir = './One-class-supervision/results/CWRU/DCAE/'
# save_dir='./One-class-supervision/results/Chopper/DCAE/'
os.makedirs(save_dir, exist_ok=True)
dcae.save_weights(save_dir+'model_last_'+str(epoch)+'.ckpt')

loss_curve=np.array(loss_list)
np.savetxt('./One-class-supervision/results/CWRU/DCAE/loss.txt',loss_curve)
# np.savetxt('./One-class-supervision/results/Chopper/DCAE/loss.txt',loss_curve)


