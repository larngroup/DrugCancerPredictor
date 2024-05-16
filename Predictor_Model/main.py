# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:46:36 2024

@author: edgar
"""

import tensorflow as tf
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
from global_model import exp_mut, split_data
from tools import load_cfg
import numpy as np
import pandas as pd
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
import os

config_file = 'config_model.json' # Configuration file 

def run():
    """Loads data, builds model, trains and evaluates it"""
    # load configuration file
    config = load_cfg(config_file)
    
    if config.exp_type == 'rsem':
        config.inp_dimension = 1954
    else:
        config.inp_dimension = 1428    
    
    file_path = 'grid_results.csv'
    
    
    model = exp_mut(config)
    model.load_data()
    model.load_autoencoders()
    
    n_runs = 1
    metrics = []
    total_train_time = []
    
    for i in range(n_runs):
        model.pre_process_data()
        if config.cross_val == "False":
            model.build()
            model.train()
            model.evaluate()
        
        metric = model.metric
        train_time = model.time_elapsed
        
        metrics.append(metric)
        total_train_time.append(train_time)
        
    mean_metrics = np.mean(metrics, axis = 0)
    mean_time = np.mean(total_train_time)
        
    data={'Data_type': config.exp_type, 'Layers encoder': str(model.layers_enc_units),
                              'layers decoder': str(config.layers_dec_units), 'Batch_size': config.batch_size,
                              'Units': config.n_units , 'Dropout': config.dropout ,'RNN': config.rnn, 'Remove outliers': config.remove_outliers, 
                              'Normalize': config.normalize, 'Norm_type': config.norm_type,
                              'Epochs': config.n_epochs, 'Runs': n_runs ,'mse': round(mean_metrics[0],4), 'rmse': round(mean_metrics[1],4),
                              'r^2': round(mean_metrics[2],4), 'mse_denorm': round(mean_metrics[3],4), 'rmse_denorm': round(mean_metrics[4],4),
                              'r^2_denorm': round(mean_metrics[5],4),'train_time': round(mean_time,2)}     
    
    results = pd.DataFrame(data)
    
    results.to_csv(file_path)       
if __name__ == '__main__':
    run()