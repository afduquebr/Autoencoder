#######################################################################################################
"""
Created on Jun 10 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

path, signal, pct = parse_args()
main()

if path == "local":
    path = "../GAN-AE/clustering-lhco/data"
elif path == "server": 
    path = "/AtlasDisk/user/duquebran/clustering-lhco/data"

bkg = pd.read_hdf(f"{path}/RnD_2j_scalars_bkg.h5")
sig1 = pd.read_hdf(f"{path}/RnD_2j_scalars_sig.h5")
sig2 = pd.read_hdf(f"{path}/RnD2_2j_scalars_sig.h5")
bbox = pd.read_hdf(f"{path}/BBOX1_2j_scalars_sig.h5")

selection = pd.read_csv("dijet-selection.csv", header=None).values[:, 0]
smooth_cols = pd.read_csv("scale-selection.csv", header=None).values[:, 0]

bkg.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
sig1.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
sig2.replace([np.nan, -np.inf, np.inf], 0, inplace=True)
bbox.replace([np.nan, -np.inf, np.inf], 0, inplace=True)

bkg[["mass_1", "mass_2"]] = bkg[["mass_1", "mass_2"]].map(lambda x: max(x, 0))
sig1[["mass_1", "mass_2"]] = sig1[["mass_1", "mass_2"]].map(lambda x: max(x, 0))
sig2[["mass_1", "mass_2"]] = sig2[["mass_1", "mass_2"]].map(lambda x: max(x, 0))
bbox[["mass_1", "mass_2"]] = bbox[["mass_1", "mass_2"]].map(lambda x: max(x, 0))

mass = 'mj1j2'
scope = [2700, 5000]

bkg = bkg[(bkg[mass] > scope[0]) & (bkg[mass] < scope[1])].reset_index()
sig1 = sig1[(sig1[mass] > scope[0]) & (sig1[mass] < scope[1])].reset_index()
sig2 = sig2[(sig2[mass] > scope[0]) & (sig2[mass] < scope[1])].reset_index()
bbox = bbox[(bbox[mass] > scope[0]) & (bbox[mass] < scope[1])].reset_index()

# Mix signal or bbox with bkg

if signal != None:
    sample_sig = globals()[signal].sample(frac=1)
    sample = pd.concat([bkg, sample_sig[:int(pct * len(bkg))]]).sample(frac=1)
else:
    signal = "sig1"
    pct = 0
    sample_sig = sig1.sample(frac=1)
    sample = bkg.sample(frac=1)

mjj_sample = sample[mass].values
mjj_sig = sample_sig[mass].values

#######################################################################################################
############################################# Reweighting #############################################

Hc,Hb = np.histogram(mjj_sample,bins=500)
weights = np.array(Hc,dtype=float)
weights[weights > 0.0] = 1.0 / weights[weights > 0.0]
weights[weights == 0.0] = 1.0
weights = np.append(weights, weights[-1])
weights *= 1000.0 # To avoid very small weights
weights = weights[np.searchsorted(Hb, mjj_sample)]

#######################################################################################################
######################################## Data Preprocessing ###########################################

# Concatenate all datasets for the current column to find the global min and max
all_data = pd.concat([sample[selection], sample_sig[selection]])

for col in smooth_cols:
    first_positive = all_data[col][all_data[col] > 0].min()
    all_data.loc[all_data[col] <= 0, col] = first_positive

all_data.loc[:, smooth_cols] = all_data.loc[:, smooth_cols].apply(lambda x: np.log(x))

# Create a Scaler object with adjusted parameters for each column
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(all_data), columns=selection)

# Apply scaling to each dataset per column
sample_scaled = data_scaled.iloc[:len(sample)]
sig_scaled = data_scaled.iloc[len(sample):]

#######################################################################################################
######################################### Data Rescaling ##############################################

test_sample = torch.from_numpy(sample_scaled[:100000].values).float().to(device)
test_sig = torch.from_numpy(sig_scaled.values).float().to(device)
weights = torch.from_numpy(weights[:100000]).float().to(device)
mjj = torch.from_numpy(mjj_sample[:100000]).float().to(device)

#######################################################################################################
########################################## Testing Analysis ############################################

# Latent space dimension (embedding)
input_dim = selection.size

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

directory = f"figs/histograms/{signal}/{int(pct*100)}"

if not os.path.exists(directory):
    os.makedirs(directory)

nbins = 20
for i, column in enumerate(selection):
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