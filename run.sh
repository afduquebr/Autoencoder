#!/bin/bash

# Source Conda environment
source /atlas/tools/anaconda/anaconda3/etc/profile.d/conda.sh

# Define the name of the Conda environment
conda_env="/AtlasDisk/home2/duquebran/Autoencoder/.venv"

# Activate the Conda environment
conda activate $conda_env

# Go to directory
cd /AtlasDisk/home2/duquebran/Autoencoder/

# Run your Python script
python train.py
python test.py

# Deactivate the Conda environment
conda deactivate
