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

bkg_scaled = pd.DataFrame(scaler.fit_transform(bkg[selection].sample(frac=1)), columns=selection)
sig1_scaled = pd.DataFrame(scaler.transform(sig1[selection].sample(frac=1)), columns=selection)
sig2_scaled = pd.DataFrame(scaler.transform(sig2[selection].sample(frac=1)), columns=selection)
train_bkg = bkg_scaled[(sig1_scaled.shape[0]):]
test_bkg = bkg_scaled[:(sig2_scaled.shape[0])]

train_bkg = torch.from_numpy(train_bkg.values).float()
test_bkg = torch.from_numpy(test_bkg.values).float()
test_sig1 = torch.from_numpy(sig1_scaled.values).float()
test_sig2 = torch.from_numpy(sig2_scaled.values).float()

#######################################################################################################
########################################## Histogram Analysis ############################################

# Latent space dimension (embedding)
input_dim = selection.size

# Load Model
model = AutoEncoder(input_dim = input_dim, mid_dim = mid_dim, latent_dim = latent_dim)
model.load_state_dict(torch.load(f"models/model_parameters_{scale}_{mid_dim}_{latent_dim}.pth"))

# Predictions
with torch.no_grad(): # no need to compute gradients here
    predict_bkg = model(test_bkg)
    predict_sig1 = model(test_sig1)
    predict_sig2 = model(test_sig2)

predict_bkg_df = pd.DataFrame(scaler.inverse_transform(predict_bkg.numpy()), columns=sig1[selection].columns)
predict_sig1_df = pd.DataFrame(scaler.inverse_transform(predict_sig1.numpy()), columns=sig1[selection].columns)
predict_sig2_df = pd.DataFrame(scaler.inverse_transform(predict_sig2.numpy()), columns=sig1[selection].columns)

#######################################################################################################
############################################# Histograms ##############################################

directory = f"figs/histograms/{scale}/mid_{mid_dim}_lat_{latent_dim}"

if not os.path.exists(directory):
    os.makedirs(directory)

nbins = 20
for i, column in enumerate(selection):
    fig, axes = plt.subplots(figsize=(8,6))
    axes.hist([test_bkg.numpy()[:,i]], nbins, density=0, histtype='bar', label=['Background'], stacked=True, alpha=1)
    axes.hist([predict_bkg.numpy()[:,i]], nbins, density=0, histtype='bar', label=['BKG prediction'], stacked=True, alpha=0.3)
    axes.hist([test_sig1.numpy()[:,i]], nbins, density=0, histtype='bar', label=['Signal 1'], stacked=True, alpha=1)
    axes.hist([predict_sig1.numpy()[:,i]], nbins, density=0, histtype='bar', label=['Signal 1 prediction'], stacked=True, alpha=0.3)
    axes.hist([test_sig2.numpy()[:,i]], nbins, density=0, histtype='bar', label=['Signal 2'], stacked=True, alpha=1)
    axes.hist([predict_sig2.numpy()[:,i]], nbins, density=0, histtype='bar', label=['Signal 2 prediction'], stacked=True, alpha=0.3)
    axes.set_xlabel(f"{column}")
    axes.set_ylabel("Events")
    axes.set_title(f"Prediction of {column}")
    axes.set_yscale("log")
    fig.legend()
    fig.savefig(f"{directory}/hist_{column}.png")
    plt.close()