# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:36:16 2024

@author: edgar
"""

import csv
import numpy as np
import json
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from bunch import Bunch


def load_cfg(path):
    with open(path, 'r') as config_file:
        config_dict = json.load(config_file)
        config = Bunch(config_dict)
    return config
           

def tokenize(smiles, tokens, len_threshold):
    #Transforms the SMILES strings into a list of tokens
    
    tokenized = []  
    for smile in smiles:
        i = 0
        token = []
        while (i < len(smile)):
            for j in range(len(tokens)):
                symbol = tokens[j]
                if symbol == smile[i:i + len(symbol)]:
                    token.append(symbol)
                    i += len(symbol)
                    break
                
        while len(token) < len_threshold:
            token.append(' ')
        tokenized.append(token)
        
    return tokenized


def token(mol):
    #Returns the list of tokens in a molecule  
    
    tokens=[]
    iao='1'
    i=0
    while i < len(mol):
        x=mol[i]
        if x=='[':
            iao=x
        elif iao[-1]!=']' and iao[0]=='[':
            iao=iao+x     
            if iao[-1]==']':
                tokens.append(iao)
                iao='1'
        elif i<=len(mol)-2 and x=='B' and mol[i+1]=='r':
            tokens.append(x+mol[i+1])
            i+=1
        elif i<=len(mol)-2 and x=='C' and mol[i+1]=='l':
            tokens.append(x+mol[i+1])
            i+=1
        else:
            tokens.append(x)
        i+=1
    return tokens


def padding(molecules,lenfeatures):  
    #Adds padding to all molecules
    
    padMolecules = []
    for mol in molecules:
        molecule=token(mol)
        if len(molecule) <= lenfeatures:
            dif = lenfeatures-len(molecule)   
            for i in range(dif):
                molecule.append(' ')
            padMolecules.append(molecule)
    return padMolecules  
  
             
def smiles2idx(smiles,tokenDict):
   #Transforms each SMILES token to the correspondent integer, according to the token-integer dictionary.
    
   newSmiles =  np.zeros((len(smiles), len(smiles[0])))
   for i in range(0,len(smiles)):
        for j in range(0,len(smiles[i])):
            
            newSmiles[i,j] = tokenDict[smiles[i][j]]
            
   return newSmiles


def cv_split(data, config):
    train_val_smiles = data[0]
    train_val_labels = data[1]
    
    cross_validation_split = KFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    data_cv = list(cross_validation_split.split(train_val_smiles, train_val_labels))
    return data_cv


def normalize(y_train, y_val, y_test, norm_type):
    #Returns data with normalized values. 
    print(y_test)

    if norm_type == 'percentile':
        q1_train = np.percentile(y_train, 5)
        q3_train = np.percentile(y_train, 95)
        
        aux = [q1_train, q3_train]

        y_train = (y_train - q1_train) / (q3_train - q1_train)
        y_val = (y_val - q1_train) / (q3_train - q1_train)
        y_test  = (y_test - q1_train) / (q3_train- q1_train)

    elif norm_type == 'min_max':
        y_min = np.min(y_train)
        y_max = np.max(y_train)
        
        aux = [y_min, y_max]

        y_train = (y_train - y_min) / (y_max - y_min)
        y_val = (y_val - y_min) / (y_max - y_min)
        y_test  = (y_test - y_min) / (y_max - y_min)

    elif norm_type == 'mean_std': 
        y_mean= np.mean(y_train)
        y_std = np.std(y_train)

        aux = [y_mean, y_std]

        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
        y_test  = (y_test - y_mean) / y_std
    
    print(y_test)
   
    return aux, y_train, y_val, y_test

def denormalization(predictions, aux, norm_type):
    print(predictions)
    if norm_type == 'percentile':
        q1_train = aux[0]
        q3_train = aux[1]

        predictions = (q3_train - q1_train) * predictions + q1_train
    
    elif norm_type == 'min_max':
        y_min = aux[0]
        y_max = aux[1]

        predictions = (y_max - y_min) * predictions + y_min
    
    elif norm_type == 'mean_std':
        y_mean = aux[0]
        y_std = aux[1]

        predictions = (y_std) * predictions + y_mean

    print(predictions)

  
    return predictions

def pred_scatter_plot(real_values, pred_values, title, xlabel, ylabel, data_type):
    fig, ax = plt.subplots()
    ax.scatter(real_values, pred_values, c='tab:blue',
               alpha=0.6, edgecolors='black')
    ax.plot(real_values, real_values, 'k--', lw=4)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    plt.title(title)
    
    if data_type == 'norm':
        plt.xlim([-1, 2])
        plt.ylim([-1, 2])
    else:
        plt.xlim([-10, 12])
        plt.ylim([-10, 12])
    plt.show()
