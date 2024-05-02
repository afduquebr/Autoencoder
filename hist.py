#######################################################################################################
"""
Created on Apr 16 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler 
import torch
import os

from autoencoder import AutoEncoder
from main import main, parse_args

####################################### GPU or CPU running ###########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#######################################################################################################
####################################### Data Initialization ###########################################

path, scale, mid_dim, latent_dim = parse_args()
main()

if path == "local":
    path = "../GAN-AE/clustering-lhco/data"
elif path == "server": 
    path = "/AtlasDisk/user/duquebran/clustering-lhco/data"

bkg = pd.read_hdf(f"{path}/RnD_2j_scalars_bkg.h5")
sig1 = pd.read_hdf(f"{path}/RnD_2j_scalars_sig.h5")
sig2 = pd.read_hdf(f"{path}/RnD2_2j_scalars_sig.h5")

selection = pd.read_csv("dijet-selection.csv", header=None).values[:, 0]

bkg.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
sig1.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
sig2.replace([np.nan, -np.inf, np.inf], 0, inplace=True)

mass = 'mj1j2'
scope = [2700, 5000]

bkg = bkg[(bkg[mass] > scope[0]) & (bkg[mass] < scope[1])].reset_index()
sig1 = sig1[(sig1[mass] > scope[0]) & (sig1[mass] < scope[1])].reset_index()
sig2 = sig2[(sig2[mass] > scope[0]) & (sig2[mass] < scope[1])].reset_index()

mjj_bkg = bkg[mass].values
mjj_sig1 = sig1[mass].values
mjj_sig2 = sig2[mass].values

#######################################################################################################
######################################## Data Preprocessing ###########################################

if scale == "minmax":
    scaler = MinMaxScaler()
elif scale == "standard":
    scaler = StandardScaler()

sample_bkg = bkg[selection].sample(frac=1)
sample_sig1 = sig1[selection].sample(frac=1)
sample_sig2 = sig2[selection].sample(frac=1)

bkg_scaled = pd.DataFrame(scaler.fit_transform(sample_bkg), columns=selection)
sig1_scaled = pd.DataFrame(scaler.transform(sample_sig1), columns=selection)
sig2_scaled = pd.DataFrame(scaler.transform(sample_sig2), columns=selection)

if scale == "minmax":
    second_to_last_min = bkg_scaled.apply(lambda x: sorted(set(x))[-2] if len(set(x)) >= 2 else None)
    second_to_last_max = bkg_scaled.apply(lambda x: sorted(set(x))[1] if len(set(x)) >= 2 else None)
    for col in bkg_scaled.columns:
        min_val = second_to_last_min[col]
        bkg_scaled[col] = bkg_scaled[col].replace(0, min_val)
        sig1_scaled[col] = sig1_scaled[col].replace(0, min_val)
        sig2_scaled[col] = sig2_scaled[col].replace(0, min_val)
        max_val = second_to_last_max[col]
        bkg_scaled[col] = bkg_scaled[col].replace(1, max_val)
        sig1_scaled[col] = sig1_scaled[col].replace(1, max_val)
        sig2_scaled[col] = sig2_scaled[col].replace(1, max_val)

    scaler2 = MinMaxScaler()
    bkg_scaled = pd.DataFrame(scaler2.fit_transform(bkg_scaled.apply(lambda x: np.log(x / (1 - x)))), columns=selection)
    sig1_scaled = pd.DataFrame(scaler2.transform(sig1_scaled.apply(lambda x: np.log(x / (1 - x)))), columns=selection)
    sig2_scaled = pd.DataFrame(scaler2.transform(sig2_scaled.apply(lambda x: np.log(x / (1 - x)))), columns=selection)

#######################################################################################################
######################################## Data Rescaling ###########################################

train_bkg = bkg_scaled[(sig1_scaled.shape[0]):]
test_bkg = bkg_scaled[:(sig2_scaled.shape[0])]

train_bkg = torch.from_numpy(train_bkg.values).float().to(device)
test_bkg = torch.from_numpy(test_bkg.values).float().to(device)
test_sig1 = torch.from_numpy(sig1_scaled.values).float().to(device)
test_sig2 = torch.from_numpy(sig2_scaled.values).float().to(device)

#######################################################################################################
########################################## Histogram Analysis ############################################

# Latent space dimension (embedding)
input_dim = selection.size

# Load Model
model = AutoEncoder(input_dim = input_dim, mid_dim = mid_dim, latent_dim = latent_dim).to(device)
model.load_state_dict(torch.load(f"models/model_parameters_{scale}_{mid_dim}_{latent_dim}.pth", map_location=device))

# Predictions
with torch.no_grad(): # no need to compute gradients here
    predict_bkg = model(test_bkg)
    predict_sig1 = model(test_sig1)
    predict_sig2 = model(test_sig2)

#######################################################################################################
############################################# Histograms ##############################################

directory = f"figs/histograms/{scale}/mid_{mid_dim}_lat_{latent_dim}"

if not os.path.exists(directory):
    os.makedirs(directory)

nbins = 20
for i, column in enumerate(selection):
    fig, axes = plt.subplots(figsize=(8,6))
    axes.hist([test_bkg.cpu().numpy()[:,i]], nbins, density=0, histtype='step', label=['Background'], stacked=True, alpha=1)
    axes.hist([predict_bkg.cpu().numpy()[:,i]], nbins, density=0, histtype='step', label=['BKG prediction'], stacked=True, alpha=0.3)
    axes.hist([test_sig1.cpu().numpy()[:,i]], nbins, density=0, histtype='step', label=['Signal 1'], stacked=True, alpha=1)
    axes.hist([predict_sig1.cpu().numpy()[:,i]], nbins, density=0, histtype='step', label=['Signal 1 prediction'], stacked=True, alpha=0.3)
    axes.hist([test_sig2.cpu().numpy()[:,i]], nbins, density=0, histtype='step', label=['Signal 2'], stacked=True, alpha=1)
    axes.hist([predict_sig2.cpu().numpy()[:,i]], nbins, density=0, histtype='step', label=['Signal 2 prediction'], stacked=True, alpha=0.3)
    axes.set_xlabel(f"{column}")
    axes.set_ylabel("Events")
    axes.set_title(f"Prediction of {column}")
    axes.set_yscale("log")
    fig.legend()
    fig.savefig(f"{directory}/hist_{column}.png")
    plt.close()