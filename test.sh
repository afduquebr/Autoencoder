#!/bin/bash

# Source and activate environment
echo "Activating virtual environment."
if ! source .venv/bin/activate; then
    echo "Error: Failed to activate environment."
    exit 1
fi

# Define variables for training
scale="standard"
middle_dim="84"
latent_dim="14" 

# Run Testing Python script
echo "Running test."
if ! python test.py -p local -s $scale -m $middle_dim -l $latent_dim; then
    echo "Error: Failed to run Testing Python script."
    exit 1
fi

# Run Histogram Python script
echo "Plotting histograms."
if ! python hist.py -p local -s $scale -m $middle_dim -l $latent_dim; then
    echo "Error: Failed to run Histogram Python script."
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
