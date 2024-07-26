#######################################################################################################
"""
Created on Jul 05 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import os
import numpy as np
import pandas as pd
from bumphunter_1dim import BumpHunter1D

import torch

from autoencoder import AutoEncoder, loss
from preprocessing import Preprocessor
from main import parse_args

####################################### GPU or CPU running ###########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#######################################################################################################
####################################### Data Preprocessing ###########################################

_, signal, pct = parse_args()
preprocessing = Preprocessor()
sample_scaled, bkg_scaled, sig_scaled = preprocessing.get_scaled_data()
weights_sample = preprocessing.get_weights()
mjj_sample, mjj_bkg, mjj_sig = preprocessing.get_mass()
labels = preprocessing.get_labels()

sbr = labels[labels==1].size / labels[labels==0].size
print(f'Original S/B = {100 * sbr:.3f}%')

#######################################################################################################
######################################## Data Preprocessing ###########################################

data = torch.from_numpy(sample_scaled[:100000].values).float().to(device)
mjj = mjj_sample[:100000]
labels = labels[:100000]

#######################################################################################################
############################################# Analysis ################################################

# Latent space dimension (embedding)
input_dim = preprocessing.selection.size

# Load Model
model = AutoEncoder(input_dim = input_dim).to(device)
model.load_state_dict(torch.load(f"models/model_parameters_{signal}_{(int(pct * 1000) % 100):02d}.pth", map_location=device))
model.eval()

# Predictions
with torch.no_grad(): # no need to compute gradients here
    prediction = model(data)

# MSE 
loss_sample = pd.DataFrame(loss(data, prediction).numpy(), columns=preprocessing.selection).mean(axis=1)

# Do the selection at Nth percentile
percentile = 95
cut = np.percentile(loss_sample, percentile)
mjj_cut = mjj[loss_sample > cut]
print(f'    post cut stat : {mjj_cut.size}')
print(f'    selec threshold : {cut:.4f}')

# Compute signal efficiency (based on truth) if possible
labels_cut = labels[loss_sample > cut]
sig_eff =  mjj_cut[labels_cut == 1].size / mjj[labels == 1].size
print(f'    sig_eff : {sig_eff}')
sbr = mjj_cut[labels_cut == 1].size / mjj_cut[labels_cut == 0].size
print(f'    new S/B : {100 * sbr:.2f}%')


#######################################################################################################
############################################# Analysis ################################################

BH = BumpHunter1D(
    rang=(2700, 5000),
    width_min = 1,
    width_max = 6,
    width_step = 1,
    scan_step = 1,
    npe = 40000,
    bins = 40,
    nworker = 4,
    use_sideband = True,
    sideband_width = 4,
    seed = 500
)

# Do the BH scan
BH.bump_scan(mjj_cut, mjj)

folder = f"figs/BumpHunter/{percentile}"

if not os.path.exists(folder):
    os.makedirs(folder)

filename = f"{folder}/bump_info.txt"
with open(filename, "w") as file:
    # Write the content to the file
    file.write(BH.bump_info(mjj_cut))

# Plot results
BH.plot_tomography(mjj, filename=f"{folder}/tomography.png")
BH.plot_bump(mjj_cut, mjj, filename=f"{folder}/bump.png")
BH.plot_stat(show_Pval=True, filename=f"{folder}/BHstat.png")
