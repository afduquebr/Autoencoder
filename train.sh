#!/bin/bash

# Source Conda environment
source /atlas/tools/anaconda/anaconda3/etc/profile.d/conda.sh

# Define the name of the Conda environment
conda_env="/AtlasDisk/home2/duquebran/Autoencoder/.venv"

# Activate the Conda environment
if ! conda activate $conda_env; then
    echo "Error: Failed to activate Conda environment."
    exit 1
fi

# Go to directory
cd /AtlasDisk/home2/duquebran/Autoencoder/ || exit

# Define variables for training
path="server"
dataset="sig1"
anomaly="0.005"

# Run Training Python script
if ! python train.py -p $path -d $dataset -a $anomaly; then
    echo "Error: Failed to run Python script."
    exit 1
fi

# Update Git repository automatically
if ! git add -A; then
    echo "Error: Failed to add files to Git."
    exit 1
fi

if ! git commit -m "Training with $dataset insertion and percentage $anomaly"; then
    echo "Error: Failed to commit changes to Git."
    exit 1
fi

if ! git push; then
    echo "Error: Failed to push changes to Git repository."
    exit 1
fi

# Deactivate the Conda environment
if ! conda deactivate; then
    echo "Error: Failed to deactivate Conda environment."
    exit 1
fi

echo "Script executed successfully."
exit 0
