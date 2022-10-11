# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:37:28 2021

@author: Master
"""

import numpy as np
import tensorflow as tf
import numpy as np

# MSE loss
def MSE(x,y):
      ms= tf.math.square (tf.subtract(x,y))
      return tf.math.reduce_mean(ms)

# SSIM loss function between two images

def SSIM(x, y):
    return  tf.image.ssim(x, y, 1.0)

# Multimodal SSIM loss function between two (channel first !) images 
def MSSIM(x, y):
    
    return tf.image.ssim_multiscale(x, y, 1.0)

# Loss function defined in the paper for two images compatible with keras
def paper_loss(alpha=0.5, beta=0.3):
    def loss(y_true, y_pred):
        
        return alpha * (1 - SSIM(y_true, y_pred)) + (1 - alpha) * (1 - SSIM(y_true, y_pred)) \
               + beta * MSE(y_true, y_pred)            
    return loss

# Color space conversions

def rgb2gray(img_rgb):
    
    #Transform to RGB image into a grayscale one using weighted method.
    output = np.zeros((img_rgb.shape[0], img_rgb.shape[1],1))
    output[:, :, 0] = 0.3 * img_rgb[:, :, 0] + 0.59 * img_rgb[:, :, 1] + 0.11 * img_rgb[:, :, 2]
    return np.clip(output,0,255)

    #Transform RGB to Luminance 
def rgb2ycc(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    xform = xform.astype(np.float32)
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    ycbcr = ycbcr.astype(np.float32)
    return np.clip(ycbcr,0,255)

    #Transform Luminance to RGB 
def ycc2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    xform = xform.astype(np.float32)
    rgb = im.astype(np.float32)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    rgb = rgb.astype(np.float32)
    return rgb

def InceptionBlock(filters_in, filters_out):
    # Network inception V1 use in  Decoder network
    input_layer = tf.keras.Input(shape=(256,256,filters_in))
    tower_filters = int(filters_out / 4)

    tower_1 = tf.keras.layers.Conv2D(tower_filters, 1, padding='same', activation='relu', data_format='channels_last')(input_layer)

    tower_2 = tf.keras.layers.Conv2D(tower_filters, 1, padding='same', activation='relu', data_format='channels_last')(input_layer)
    tower_2 = tf.keras.layers.Conv2D(tower_filters, 3, padding='same', activation='relu', data_format='channels_last')(tower_2)

    tower_3 = tf.keras.layers.Conv2D(tower_filters, 1, padding='same', activation='relu', data_format='channels_last')(input_layer)
    tower_3 = tf.keras.layers.Conv2D(tower_filters, 5, padding='same', activation='relu', data_format='channels_last')(tower_3)

    tower_4 = tf.keras.layers.MaxPool2D(tower_filters, padding='same', strides=(1, 1), data_format='channels_last')(input_layer)
    tower_4 = tf.keras.layers.Conv2D(tower_filters, 1, padding='same', activation='relu', data_format='channels_last')(tower_4)

    concat = tf.keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=3)

    res_link = tf.keras.layers.Conv2D(filters_out, 1, padding='same', activation='relu', data_format='channels_last')(input_layer)

    output = tf.keras.layers.add([concat, res_link])
    output = tf.keras.layers.Activation('relu')(output)

    model_output = tf.keras.Model([input_layer], output)
    return model_output



