#######################################################################################################
"""
Created on Jun 10 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import os
import matplotlib.pyplot as plt
import torch

from autoencoder import AutoEncoder
from preprocessing import Preprocessor
from main import parse_args

####################################### GPU or CPU running ###########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#######################################################################################################
####################################### Data Preprocessing ###########################################

_, signal, pct = parse_args()
preprocessing = Preprocessor()
sample_scaled, _, sig_scaled = preprocessing.get_scaled_data()

#######################################################################################################
######################################### Data Rescaling ##############################################

test_sample = torch.from_numpy(sample_scaled[:100000].values).float().to(device)
test_sig = torch.from_numpy(sig_scaled.values).float().to(device)

#######################################################################################################
########################################## Testing Analysis ############################################

# Latent space dimension (embedding)
input_dim = preprocessing.selection.size

# Load Model
model = AutoEncoder(input_dim = input_dim).to(device)
model.load_state_dict(torch.load(f"models/model_parameters_{signal}_{(int(pct * 1000) % 100):02d}.pth", map_location=device))
model.eval()

# Predictions
with torch.no_grad(): # no need to compute gradients here
    predict_sample = model(test_sample)
    predict_sig = model(test_sig)

#######################################################################################################
############################################# Histograms ##############################################

directory = f"figs/histograms/{signal}/{(int(pct * 1000) % 100):02d}"

if not os.path.exists(directory):
    os.makedirs(directory)

nbins = 20
for i, column in enumerate(preprocessing.selection):
    fig, axes = plt.subplots(figsize=(8,6))
    axes.hist([test_sample.cpu().numpy()[:,i]], nbins, density=0, histtype='step', label=['Background'], stacked=True, alpha=1)
    axes.hist([predict_sample.cpu().numpy()[:,i]], nbins, density=0, histtype='step', label=['BKG prediction'], stacked=True, alpha=0.3)
    axes.hist([test_sig.cpu().numpy()[:,i]], nbins, density=0, histtype='step', label=['Signal'], stacked=True, alpha=1)
    axes.hist([predict_sig.cpu().numpy()[:,i]], nbins, density=0, histtype='step', label=['Signal prediction'], stacked=True, alpha=0.3)
    axes.set_xlabel(f"{column}")
    axes.set_ylabel("Events")
    axes.set_title(f"Prediction of {column}")
    axes.set_yscale("log")
    fig.legend()
    fig.savefig(f"{directory}/hist_{column}.png")
    plt.close()