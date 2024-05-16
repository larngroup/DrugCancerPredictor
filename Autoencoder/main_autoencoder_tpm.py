# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:45:18 2024

@author: edgar
"""

# TPM DATA - AUTOENCODER MODEL

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
    config_file_path = 'config_autoencoder_tpm.json'
    config = load_config(config_file_path)

    # Train autoencoder model with 'expression' data in rsme format
    rmse_exp_tpm, mse_exp_tpm, mae_exp_tpm, r2_exp_tpm = train_autoencoder(config, 'expression_tpm')

    if rmse_exp_tpm is not None:
        print("Training completed successfully with 'expression' data in RSME Format.")
        print("\nTesting expression dataset with TPM format:")
        print(f"RMSE: {rmse_exp_tpm}, MSE: {mse_exp_tpm}, MAE: {mae_exp_tpm}, R^2: {r2_exp_tpm}")
    else:
        print("Training failed with 'expression' data. Please check logs for details.")

    # Train autoencoder model with 'expression' data in tpm format
    rmse_mut_tpm, mse_mut_tpm, mae_mut_tpm, r2_mut_tpm = train_autoencoder(config, 'mutation_tpm')

    if rmse_mut_tpm is not None:
        print("Training completed successfully with 'expression' data in TPM Format.")
        print("\nTesting mutation dataset with TPM format:")
        print(f"RMSE: {rmse_mut_tpm}, MSE: {mse_mut_tpm}, MAE: {mae_mut_tpm}, R^2: {r2_mut_tpm}")
    else:
        print("Training failed with 'mutation' data. Please check logs for details.")

if __name__ == "__main__":
    main()