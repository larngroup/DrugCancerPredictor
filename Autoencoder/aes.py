# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 01:35:23 2024

@author: edgar
"""

#functions to r_square and rmse metrics
def r_square(y_true, y_pred):
    import tensorflow.keras.backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def rmse(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


import math
import tensorflow 
import os
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Autoencoder:
    
    """Class for Gene Expression Profile and Mutation Autoencoder Model"""
    
    def __init__(self, config):
        # Initialize variables
        self.exp_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        self.dataset = None
        self.config = config
        self.data_train = None
        self.data_test = None

    def load_data(self, data_type):
        """Function to load gene expression or mutation data"""
        print("Loading dataset...")
        if data_type == 'expression_rsem':
            print("Loading expression_rsem data from:", self.config['exp_file_rsem'])
            self.dataset = pd.read_pickle(self.config['exp_file_rsem'])
        elif data_type == 'mutation_rsem':
            print("Loading mutation_rsem data from:", self.config['mut_file_rsem'])
            self.dataset = pd.read_pickle(self.config['mut_file_rsem'])
        elif data_type == 'expression_tpm':
            print("Loading expression_tpm from:", self.config['exp_file_tpm'])
            self.dataset = pd.read_pickle(self.config['exp_file_tpm'])
        elif data_type == 'mutation_tpm':
            print("Loading mutation_tpm data from:", self.config['mut_file_tpm'])
            self.dataset = pd.read_pickle(self.config['mut_file_tpm'])    

    def pre_process_data(self):
        """Function to split datasets"""
        if self.config.get('cross_validation', False):
            # Splitting data into train, test, and validation sets with cross-validation
            self.data_train, self.data_test = train_test_split(self.dataset, test_size=0.20, random_state=55)
            self.data_test, self.data_val = train_test_split(self.data_test, test_size=0.50, random_state=55)
        else:
            # Splitting data into train and test sets with KFold cross-validation
            data_train, self.data_test = train_test_split(self.dataset, test_size=0.15, random_state=55)
            cross_validation_split = KFold(n_splits=5, shuffle=True, random_state=42)
            self.data_cv = list(cross_validation_split.split(data_train))
            # Assuming data_val is set here
        
    def build(self):
        """Function to build the autoencoder architecture"""
        print("Building the model...")
    
        inp_dimension = self.config.get('inp_dimension')  
        layers_enc_units = self.config.get('layers_enc_units')
        layers_dec_units = self.config.get('layers_dec_units')
        latent_dim = self.config.get('latent_dim')
        dropout = self.config.get('dropout')
    
        input_data = tensorflow.keras.layers.Input(shape=(inp_dimension,))
        
        # Building encoder layers
        for i, layer in enumerate(layers_enc_units):
            encoder = tensorflow.keras.layers.Dense(layer, activation="relu")(input_data if i == 0 else encoder)
            encoder = tensorflow.keras.layers.BatchNormalization()(encoder)
            encoder = tensorflow.keras.layers.Dropout(dropout)(encoder) 
    
        latent_encoding = tensorflow.keras.layers.Dense(latent_dim)(encoder)
    
        self.encoder_model = tensorflow.keras.Model(input_data, latent_encoding)
        
        decoder_input = tensorflow.keras.layers.Input(shape=(latent_dim,))
        
        # Building decoder layers
        for i, layer in enumerate(layers_dec_units):
            decoder = tensorflow.keras.layers.Dense(layer, activation="relu")(decoder_input if i == 0 else decoder)
            decoder = tensorflow.keras.layers.BatchNormalization()(decoder)
            decoder = tensorflow.keras.layers.Dropout(dropout)(decoder) 
        
        decoder_output = tensorflow.keras.layers.Dense(inp_dimension, activation='linear')(decoder)
        
        self.decoder_model = tensorflow.keras.Model(decoder_input, decoder_output)
        
        encoded = self.encoder_model(input_data)
        decoded = self.decoder_model(encoded)
        
        self.autoencoder = tensorflow.keras.models.Model(input_data, decoded)
        
       
    def train(self):
        """Compiling and training the model"""
        optimizer_name = self.config.get('optimizer', 'adam')
    
        if optimizer_name == 'adam':
            opt = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
    
        es = EarlyStopping(verbose=1, patience=200, restore_best_weights=True)
        reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6)
        callbacks_list = [es, reduce_lr]
    
        self.autoencoder.compile(loss="mse", optimizer=opt, experimental_run_tf_function=False, metrics=[rmse, r_square])
        self.autoencoder.summary()
    
        # Ensure data_val is set before calling the train method
        self.autoencoder.fit(self.data_train, self.data_train, epochs=self.config.get('n_epochs', 100),
                             batch_size=self.config.get('batch_size', 64), validation_data=(self.data_val, self.data_val),
                             callbacks=callbacks_list)
        
    def evaluate(self):
        """Predicting results for the test dataset"""
        test_set = self.data_test
        preds_test = self.autoencoder.predict(test_set)
        
        self.mse = round(mean_squared_error(test_set, preds_test, squared = True),4)
        self.rmse = math.sqrt(self.mse)
        self.mae = round(mean_absolute_error(test_set, preds_test),4)
        self.r2 = round(r2_score(test_set, preds_test),4)

        print('\nRMSE test: ', self.rmse)
        print('\nMSE test: ', self.mse)
        
        return self.rmse, self.mse, self.mae, self.r2
        

    def save(self, data_type):
        """Saving trained models"""
        models = ['Encoder','Decoder', 'Autoencoder']
        
        dirs = os.path.join('../saved_autoencoders_vectors',data_type+'//')
            
       	try:
            if not os.path.exists(dirs):
                os.makedirs(dirs)

       	except Exception as err:
       		print('Creating directories error: {}'.format(err))
       		exit(-1)
               
        for model in models:
            if model == 'Encoder':
                model_name = self.encoder_model
            elif model == 'Decoder':
                model_name = self.decoder_model
            elif model == 'Autoencoder':
                model_name = self.autoencoder
                
            model_name.save(dirs+model+"_"+data_type+".h5")
            print("\n" + model + " successfully saved to disk")