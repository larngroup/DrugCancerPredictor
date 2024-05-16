# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 02:30:29 2024

@author: edgar
"""
# RSEM DATA - AUTOENCODER MODEL

import json
from aes import Autoencoder

def load_config(config_file_path):
    """Load configurations from a JSON file."""
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    return config

def train_autoencoder(config, data_type):
    """Train an autoencoder model based on the provided configuration and data type."""
    try:
        model = Autoencoder(config)
        model.load_data(data_type)
        model.pre_process_data()
        model.build()
        model.train()
        rmse, mse, mae, r2 = model.evaluate()
        model.save(data_type)
        return rmse, mse, mae, r2
    except Exception as e:
        print(f"An error occurred during training with data type '{data_type}': {str(e)}")
        return None, None, None, None  # Return default values if an error occurs

def main():
    # Load configurations
    config_file_path = 'config_autoencoder_rsem.json'
    config = load_config(config_file_path)

    # Train autoencoder model with 'expression' data in rsem format
    rmse_exp_rsem, mse_exp_rsem, mae_exp_rsem, r2_exp_rsem = train_autoencoder(config, 'expression_rsem')

    if rmse_exp_rsem is not None:
        print("Training completed successfully with 'expression' data in RSEM Format.")
        print("\nTesting expression dataset with RSEM format:")
        print(f"RMSE: {rmse_exp_rsem}, MSE: {mse_exp_rsem}, MAE: {mae_exp_rsem}, R^2: {r2_exp_rsem}")
    else:
        print("Training failed with 'expression' data. Please check logs for details.")

    # Train autoencoder model with 'mutation' data in rsem format
    rmse_mut_rsem, mse_mut_rsem, mae_mut_rsem, r2_mut_rsem = train_autoencoder(config, 'mutation_rsem')

    if rmse_mut_rsem is not None:
        print("Training completed successfully with 'expression' data in TPM Format.")
        print("\nTesting expression dataset with RSEM format:")
        print(f"RMSE: {rmse_mut_rsem}, MSE: {mse_mut_rsem}, MAE: {mae_mut_rsem}, R^2: {r2_mut_rsem}")
    else:
        print("Training failed with 'mutation' data. Please check logs for details.")

if __name__ == "__main__":
    main()