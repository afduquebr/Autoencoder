#######################################################################################################
"""
Created on Jul 05 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import os
import numpy as np
import pandas as pd
import pyBumpHunter as bh

import torch

from autoencoder import AutoEncoder, loss
from preprocessing import Preprocessor
from main import parse_args

####################################### GPU or CPU running ###########################################

# Determine whether to use a GPU or CPU for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#######################################################################################################
####################################### Data Preprocessing ###########################################

# Parse arguments to get signal and anomaly percentage
signal, pct = parse_args()

# Initialize data preprocessing
preprocessing = Preprocessor()

# Retrieve scaled datasets (sample, background, and signal)
sample_scaled, bkg_scaled, sig_scaled = preprocessing.get_scaled_data()

# Retrieve weights and mass distributions
weights_sample = preprocessing.get_weights()
mjj_sample, mjj_bkg, mjj_sig = preprocessing.get_mass()
labels = preprocessing.get_labels()

# Calculate the original Signal-to-Background ratio
sbr = labels[labels == 1].size / labels[labels == 0].size
print(f'Original S/B = {100 * sbr:.3f}%')

#######################################################################################################
######################################## Data Preparation ############################################

# Convert scaled sample data to torch tensors and move to the appropriate device
data = torch.from_numpy(sample_scaled[:100000].values).float().to(device)
mjj = mjj_sample[:100000]
labels = labels[:100000]

#######################################################################################################
############################################# Analysis ################################################

# Define the latent space dimension (embedding)
input_dim = preprocessing.selection.size

# Load the pre-trained AutoEncoder model
model = AutoEncoder(input_dim=input_dim).to(device)
model.load_state_dict(torch.load(f"models/{signal}/parameters_{(int(pct * 10) % 100):02d}.pth", map_location=device))
model.eval()

# Perform predictions using the model
with torch.no_grad():  # No need to compute gradients during inference
    prediction = model(data)

# Calculate Mean Squared Error (MSE) for the sample
loss_sample = pd.DataFrame(loss(data, prediction).numpy(), columns=preprocessing.selection).mean(axis=1)

# Apply selection based on the Nth percentile of reconstruction error
percentile = 95
cut = np.percentile(loss_sample, percentile)
mjj_cut = mjj[loss_sample > cut]
print(f'    post cut stat : {mjj_cut.size}')
print(f'    selection threshold : {cut:.4f}')

# Compute signal efficiency and updated S/B ratio after the cut
labels_cut = labels[loss_sample > cut]
sig_eff = mjj_cut[labels_cut == 1].size / mjj[labels == 1].size
print(f'    signal efficiency : {sig_eff}')
sbr = mjj_cut[labels_cut == 1].size / mjj_cut[labels_cut == 0].size
print(f'    new S/B : {100 * sbr:.2f}%')

#######################################################################################################
############################################# BumpHunter Analysis #####################################

# Initialize BumpHunter with specified parameters
BH = bh.BumpHunter1D(
    rang=(2700, 5000),
    width_min=1,
    width_max=6,
    width_step=1,
    scan_step=1,
    npe=40000,
    bins=40,
    nworker=4,
    use_sideband=True,
    sideband_width=4,
    seed=500
)

# Perform BumpHunter scan on the data
BH.bump_scan(mjj_cut, mjj)

# Create directory for saving BumpHunter results
folder = f"figs/{signal}/{(int(pct * 10) % 100):02d}/BumpHunter"
if not os.path.exists(folder):
    os.makedirs(folder)

# Save bump information to a text file
filename = f"{folder}/bump_info.txt"
with open(filename, "w") as file:
    file.write(BH.bump_info(mjj_cut))

# Plot and save BumpHunter results
BH.plot_tomography(mjj, filename=f"{folder}/tomography.png")
BH.plot_bump(mjj_cut, mjj, filename=f"{folder}/bump.png")
BH.plot_stat(show_Pval=True, filename=f"{folder}/BHstat.png")
