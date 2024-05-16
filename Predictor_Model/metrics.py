# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:42:27 2024

@author: edgar
"""
import tensorflow as tf
from tensorflow.keras import backend as K

# Define your custom metric function (if not already defined)
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)