#!/bin/bash

# Source and activate environment
# echo "Activating virtual environment."
# if ! source .venv/bin/activate; then
#     echo "Error: Failed to activate environment."
#     exit 1
# fi

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

# Define variables for testing
path="server"
dataset="sig1"
anomaly="0.05"

# Run Testing Python script
echo "Running test."
if ! python test.py -p $path -d $dataset -a $anomaly; then
    echo "Error: Failed to run Testing Python script."
    exit 1
fi

# Run Histogram Python script
# echo "Plotting histograms."
# if ! python hist.py -p $path -d $dataset -a $anomaly; then
#     echo "Error: Failed to run Histogram Python script."
#     exit 1
# fi

# Update Git repository automatically
if ! git add .; then
    echo "Error: Failed to add files to Git."
    exit 1
fi

if ! git commit -m "Testing with $dataset insertion and percentage $anomaly"; then
    echo "Error: Failed to commit changes to Git."
    exit 1
fi

if ! git push; then
    echo "Error: Failed to push changes to Git repository."
    exit 1
fi

# Deactivate the environment
echo "deactivating virtual environment."
if ! deactivate; then
    echo "Error: Failed to deactivate environment."
    exit 1
fi

echo "Script executed successfully."
exit 0
